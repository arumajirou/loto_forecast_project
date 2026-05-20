from __future__ import annotations

import argparse
import os
import dataclasses
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from loto_forecast.infra.meta_store import (
    mark_model_run_end,
    upsert_model_run,
    write_log_run_history,
    write_resource_samples,
)
from loto_forecast.infra.monitoring import ResourceMonitor, generate_run_id
from .config import ResourcesConfig
from .context import start_run
from .db.postgres_copy import copy_dataframe_to_postgres
from .utils import safe_ident


@dataclass(frozen=True)
class ChronosExogSpec:
    profile: str = "local"
    env: str = "LOCAL"
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = "loto"
    password: str = ""
    database: str = "loto"

    source_schema: str = "dataset"
    source_table: str = "loto_y_ts"
    source_where: str | None = None

    target_schema: str = "exog"
    target_table: str = "chronos"
    if_exists: str = "append"
    only_missing: bool = True

    group_cols: tuple[str, ...] = ("loto", "ts_type")
    time_col: str = "ds"
    target_col: str = "y"
    source_row_id_column: str = "row_id"
    y_idx_order_column: str = "row_id"

    ds_start: str | None = None
    ds_end: str | None = None
    loto_filter: tuple[str, ...] = ()
    ts_type_filter: tuple[str, ...] = ()

    backend: str = "chronos_pipeline_auto"
    model_id: str = "amazon/chronos-bolt-small"
    model_name: str = "chronos"
    model_version: str = "1.0"
    embedding_dim: int = 256
    window_size: int = 128
    min_points: int = 16
    batch_size: int = 256
    normalize_method: str = "zscore"
    fill_method: str = "zero"

    parallel_workers: int = 4
    enable_gpu_compute: bool = True
    local_files_only: bool = True
    sampling_interval_sec: float = 1.0
    create_postgres_index: bool = True
    postgres_write_mode: str = "copy"
    postgres_copy_chunk_rows: int = 50000
    profile_stages: bool = False
    row_batch_size: int = 5000
    max_groups_per_batch: int = 64

    chronos_codegen_yaml: str = (
        "./docs/lib_docs/chronos-forecasting_scripts_evaluation_agg-relative-score_all_codegen.yaml"
    )


def _make_engine(spec: ChronosExogSpec) -> Engine:
    url = f"postgresql+psycopg2://{spec.user}:{spec.password}@{spec.host}:{spec.port}/{spec.database}"
    return create_engine(url, pool_pre_ping=True)


def _safe_table_ref(schema: str, table: str) -> str:
    return f'"{safe_ident(schema)}"."{safe_ident(table)}"'


def _parse_filter_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(vals)


def _summarize_codegen_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {"path": str(p), "exists": True, "error": str(e)}

    rows = payload.get("rows") if isinstance(payload, dict) else None
    n_rows = len(rows) if isinstance(rows, list) else int(payload.get("count", 0) or 0)
    modules = sorted({str(r.get("module", "")) for r in rows if isinstance(r, dict)}) if isinstance(rows, list) else []
    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": n_rows,
        "module_count": len([m for m in modules if m]),
    }


def _read_source(spec: ChronosExogSpec, engine: Engine) -> pd.DataFrame:
    table_ref = _safe_table_ref(spec.source_schema, spec.source_table)
    tcol = safe_ident(spec.time_col)
    sql = f"SELECT * FROM {table_ref}"

    where: list[str] = []
    params: dict[str, Any] = {}
    if spec.source_where:
        where.append(f"({spec.source_where})")
    if spec.ds_start:
        where.append(f'"{tcol}" >= :ds_start')
        params["ds_start"] = spec.ds_start
    if spec.ds_end:
        where.append(f'"{tcol}" <= :ds_end')
        params["ds_end"] = spec.ds_end
    if where:
        sql += " WHERE " + " AND ".join(where)

    df = pd.read_sql(text(sql), engine, params=params)
    if spec.loto_filter and "loto" in df.columns:
        df = df[df["loto"].astype(str).isin(spec.loto_filter)].copy()
    if spec.ts_type_filter and "ts_type" in df.columns:
        df = df[df["ts_type"].astype(str).isin(spec.ts_type_filter)].copy()
    return df.reset_index(drop=True)


def _validate_input(df: pd.DataFrame, spec: ChronosExogSpec) -> None:
    required = set(spec.group_cols) | {spec.time_col, spec.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in source: {sorted(missing)}")
    if spec.embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0")
    if spec.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if spec.min_points <= 0:
        raise ValueError("min_points must be > 0")
    if spec.min_points > spec.window_size:
        raise ValueError("min_points must be <= window_size")


def _stable_int64(payload: str) -> int:
    d = hashlib.sha1(payload.encode("utf-8")).digest()
    v = int.from_bytes(d[:8], byteorder="big", signed=False)
    v &= (1 << 63) - 1
    return 1 if v == 0 else v


def _resolve_row_id_candidates(spec: ChronosExogSpec) -> list[str]:
    src_row_id_col = f"{spec.source_table}_row_id"
    out: list[str] = []
    for c in [
        src_row_id_col,
        "loto_y_ts_row_id",
        spec.source_row_id_column,
        "row_id",
        "id",
        "unique_id",
    ]:
        if c and c not in out:
            out.append(c)
    return out


def _ensure_row_id_column(df: pd.DataFrame, spec: ChronosExogSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    chosen = None
    for c in _resolve_row_id_candidates(spec):
        if c in out.columns:
            chosen = c
            break

    if chosen is None:
        ids = []
        for i, row in out.reset_index(drop=True).iterrows():
            payload = "|".join(
                [
                    str(row.get("loto", "")),
                    str(row.get("ts_type", "")),
                    str(row.get(spec.time_col, "")),
                    str(row.get("unique_id", "")),
                    str(i + 1),
                ]
            )
            ids.append(_stable_int64(payload))
        out["loto_y_ts_row_id"] = np.asarray(ids, dtype=np.int64)
        return out, {"source_row_id_column": None, "row_id_strategy": "generated_hash"}

    numeric = pd.to_numeric(out[chosen], errors="coerce")
    invalid_mask = numeric.isna()
    row_ids = numeric.fillna(0).astype(np.int64).to_numpy(copy=True)

    if invalid_mask.any():
        for i in np.where(invalid_mask.to_numpy())[0]:
            row = out.iloc[i]
            payload = "|".join(
                [
                    str(chosen),
                    str(row.get(chosen, "")),
                    str(row.get("loto", "")),
                    str(row.get("ts_type", "")),
                    str(row.get(spec.time_col, "")),
                    str(i + 1),
                ]
            )
            row_ids[i] = _stable_int64(payload)

    out["loto_y_ts_row_id"] = row_ids
    return out, {"source_row_id_column": chosen, "row_id_strategy": "cast_or_hash"}


def _select_order_column(df: pd.DataFrame, spec: ChronosExogSpec) -> str | None:
    for c in [spec.y_idx_order_column, "updated_ts", "exec_ts", "row_id", "id", "unique_id"]:
        if c and c in df.columns:
            return c
    return None


def _build_history_windows(y: np.ndarray, window_size: int) -> np.ndarray:
    y2 = np.asarray(y, dtype=np.float32)
    n = y2.shape[0]
    pad = np.full((window_size,), np.nan, dtype=np.float32)
    x = np.concatenate([pad, y2], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size)
    return win[:n].astype(np.float32, copy=True)


def _preprocess_history(values: np.ndarray, fill_method: str, normalize_method: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)

    if fill_method == "zero":
        x = np.nan_to_num(x, nan=0.0)
    elif fill_method == "ffill":
        s = pd.to_numeric(pd.Series(x), errors="coerce")
        x = s.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    elif fill_method == "drop":
        x = x[~np.isnan(x)]
    else:
        raise ValueError(f"unsupported fill_method: {fill_method}")

    if x.size == 0:
        return x
    if normalize_method == "none":
        return x.astype(np.float32)
    if normalize_method != "zscore":
        raise ValueError(f"unsupported normalize_method: {normalize_method}")

    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        sd = 1.0
    return ((x - mu) / sd).astype(np.float32)


def _preprocess_windows_fast(
    windows: np.ndarray, fill_method: str, normalize_method: str, min_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if windows.size == 0:
        return windows, np.zeros((0,), dtype=bool)

    if fill_method == "zero" and normalize_method in {"zscore", "none"}:
        mask = np.isfinite(windows)
        arr = np.nan_to_num(windows, nan=0.0).astype(np.float32)
        keep = mask.sum(axis=1) >= int(min_points)
        if normalize_method == "none":
            return arr, keep

        den = np.clip(mask.sum(axis=1, keepdims=True).astype(np.float32), 1.0, None)
        mean = (arr * mask).sum(axis=1, keepdims=True) / den
        centered = (arr - mean) * mask
        std = np.sqrt((centered**2).sum(axis=1, keepdims=True) / den)
        arr = np.where(mask, (arr - mean) / (std + 1e-6), 0.0).astype(np.float32)
        return arr, keep

    # General path: supports ffill/drop.
    rows: list[np.ndarray] = []
    keep = np.zeros((windows.shape[0],), dtype=bool)
    for i in range(windows.shape[0]):
        h = _preprocess_history(windows[i], fill_method=fill_method, normalize_method=normalize_method)
        if h.size < int(min_points):
            continue
        keep[i] = True
        if h.size < windows.shape[1]:
            padded = np.zeros((windows.shape[1],), dtype=np.float32)
            padded[-h.size :] = h
            rows.append(padded)
        elif h.size > windows.shape[1]:
            rows.append(h[-windows.shape[1] :].astype(np.float32))
        else:
            rows.append(h.astype(np.float32))
    if not rows:
        return np.zeros((0, windows.shape[1]), dtype=np.float32), keep
    return np.stack(rows, axis=0).astype(np.float32), keep


def _fit_embedding_dim(emb: np.ndarray, embedding_dim: int) -> np.ndarray:
    if emb.shape[1] == embedding_dim:
        return emb.astype(np.float32)
    if emb.shape[1] > embedding_dim:
        return emb[:, :embedding_dim].astype(np.float32)
    pad = np.zeros((emb.shape[0], embedding_dim - emb.shape[1]), dtype=np.float32)
    return np.concatenate([emb.astype(np.float32), pad], axis=1)


def _pool_embedding_rows(obj: Any) -> np.ndarray:
    torch_module: Any | None
    try:
        import torch as _torch
    except Exception:  # pragma: no cover - torch missing is handled by caller fallback
        torch_module = None
    else:
        torch_module = _torch

    if torch_module is not None and isinstance(obj, torch_module.Tensor):
        t = obj.detach().float().cpu()
        if t.dim() == 1:
            return t.unsqueeze(0).numpy().astype(np.float32)
        if t.dim() == 2:
            return t.numpy().astype(np.float32)
        b = t.shape[0]
        return t.reshape(b, -1, t.shape[-1]).mean(dim=1).numpy().astype(np.float32)

    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return obj.reshape(1, -1).astype(np.float32)
        if obj.ndim == 2:
            return obj.astype(np.float32)
        b = obj.shape[0]
        return obj.reshape(b, -1, obj.shape[-1]).mean(axis=1).astype(np.float32)

    if isinstance(obj, list):
        mats: list[np.ndarray] = []
        for x in obj:
            m = _pool_embedding_rows(x)
            if m.shape[0] > 1:
                # For list of samples, reduce each sample to one vector.
                m = m.mean(axis=0, keepdims=True)
            mats.append(m)
        if not mats:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(mats, axis=0).astype(np.float32)

    raise RuntimeError(f"unsupported embedding output type: {type(obj)}")


class _ChronosForecastFeatureBackend:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)

    def encode_batch(self, windows: np.ndarray) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float32)
        if x.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        bsz, width = x.shape
        feats: list[np.ndarray] = [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.median(x, axis=1),
            x[:, -1],
        ]

        idx = np.arange(width, dtype=np.float32)
        idx_mean = idx.mean()
        idx_var = np.sum((idx - idx_mean) ** 2) + 1e-6
        x_mean = x.mean(axis=1, keepdims=True)
        cov = np.sum((x - x_mean) * (idx.reshape(1, -1) - idx_mean), axis=1)
        feats.append(cov / idx_var)

        max_k = max(2, self.embedding_dim // 2 + 1)
        for k in range(1, max_k):
            w = 2.0 * np.pi * float(k) * idx / float(max(width, 2))
            s = np.mean(x * np.sin(w).reshape(1, -1), axis=1)
            c = np.mean(x * np.cos(w).reshape(1, -1), axis=1)
            feats.extend([s, c])
            if len(feats) >= self.embedding_dim:
                break

        mat = np.stack(feats[: self.embedding_dim], axis=1).astype(np.float32)
        return _fit_embedding_dim(mat, self.embedding_dim)


class _ChronosPipelineBackend:
    def __init__(
        self,
        backend: str,
        model_id: str,
        embedding_dim: int,
        batch_size: int,
        window_size: int,
        enable_gpu: bool,
        local_files_only: bool,
    ) -> None:
        self.requested_backend = backend
        self.model_id = model_id
        self.embedding_dim = int(embedding_dim)
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.enable_gpu = bool(enable_gpu)
        self.local_files_only = bool(local_files_only)
        self.variant = backend
        self.pipeline: Any | None = None
        self._device: Any | None = None
        self._torch: Any | None = None
        self._load()

    def _load(self) -> None:
        import torch
        from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline

        self._torch = torch
        if self.enable_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        candidates: list[tuple[str, Any]]
        if self.requested_backend == "chronos2_pipeline":
            candidates = [("chronos2_pipeline", Chronos2Pipeline)]
        elif self.requested_backend == "chronos_bolt_pipeline":
            candidates = [("chronos_bolt_pipeline", ChronosBoltPipeline)]
        elif self.requested_backend == "chronos_pipeline_legacy":
            candidates = [("chronos_pipeline_legacy", ChronosPipeline)]
        elif self.requested_backend == "chronos_pipeline_auto":
            candidates = [
                ("chronos2_pipeline", Chronos2Pipeline),
                ("chronos_bolt_pipeline", ChronosBoltPipeline),
                ("chronos_pipeline_legacy", ChronosPipeline),
            ]
        else:
            raise ValueError(f"unsupported chronos backend: {self.requested_backend}")

        last_err: Exception | None = None
        for name, cls in candidates:
            kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
            if self._device is not None:
                kwargs["device_map"] = "cuda" if self._device.type == "cuda" else "cpu"
            try:
                self.pipeline = cls.from_pretrained(self.model_id, **kwargs)
                self.variant = name
                return
            except Exception as e1:
                last_err = e1
                try:
                    self.pipeline = cls.from_pretrained(self.model_id, local_files_only=self.local_files_only)
                    self.variant = name
                    return
                except Exception as e2:
                    last_err = e2
                    continue

        raise RuntimeError(f"failed to load chronos pipeline from model_id={self.model_id}; detail={last_err}")

    def encode_batch(self, windows: np.ndarray) -> np.ndarray:
        if windows.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self.pipeline is None:
            raise RuntimeError("chronos pipeline is not initialized")

        if self.variant == "chronos2_pipeline":
            emb_out, _ = self.pipeline.embed(
                windows,
                batch_size=int(self.batch_size),
                context_length=int(self.window_size),
            )
            pooled = _pool_embedding_rows(emb_out)
            return _fit_embedding_dim(pooled, self.embedding_dim)

        if self._torch is None:
            raise RuntimeError("torch is not available for chronos pipeline backend")
        t = self._torch.tensor(windows, dtype=self._torch.float32)
        if self._device is not None:
            t = t.to(self._device)
        emb_out, _ = self.pipeline.embed(t)
        pooled = _pool_embedding_rows(emb_out)
        return _fit_embedding_dim(pooled, self.embedding_dim)


class ChronosEmbedder:
    def __init__(self, spec: ChronosExogSpec) -> None:
        self.spec = spec
        self.requested_backend = spec.backend
        self.resolved_backend = spec.backend
        self.fallback_reason: str | None = None
        self.parallel_safe = True
        self._backend: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.requested_backend == "chronos_forecast_features":
            self._backend = _ChronosForecastFeatureBackend(self.spec.embedding_dim)
            self.resolved_backend = "chronos_forecast_features"
            self.parallel_safe = True
            return

        try:
            self._backend = _ChronosPipelineBackend(
                backend=self.requested_backend,
                model_id=self.spec.model_id,
                embedding_dim=self.spec.embedding_dim,
                batch_size=self.spec.batch_size,
                window_size=self.spec.window_size,
                enable_gpu=self.spec.enable_gpu_compute,
                local_files_only=self.spec.local_files_only,
            )
            self.resolved_backend = self._backend.variant
            self.parallel_safe = False
            return
        except Exception as e:
            self._backend = _ChronosForecastFeatureBackend(self.spec.embedding_dim)
            self.resolved_backend = "chronos_forecast_features"
            self.parallel_safe = True
            self.fallback_reason = str(e)

    def encode_batch(self, windows: np.ndarray) -> np.ndarray:
        emb = self._backend.encode_batch(windows)
        return _fit_embedding_dim(np.asarray(emb, dtype=np.float32), self.spec.embedding_dim)


def _target_key_set(spec: ChronosExogSpec, engine: Engine, src: pd.DataFrame) -> set[int] | None:
    if spec.if_exists == "replace" or not spec.only_missing:
        return None
    insp = inspect(engine)
    if not insp.has_table(spec.target_table, schema=spec.target_schema):
        return None

    t_ref = _safe_table_ref(spec.target_schema, spec.target_table)
    try:
        existing = pd.read_sql(text(f"SELECT loto_y_ts_row_id FROM {t_ref}"), engine)
    except Exception:
        return None
    if existing.empty:
        return None

    existing_ids = set(pd.to_numeric(existing["loto_y_ts_row_id"], errors="coerce").dropna().astype(np.int64).tolist())
    source_ids = set(src["loto_y_ts_row_id"].astype(np.int64).tolist())
    return source_ids - existing_ids


def _encode_group_fast(
    g: pd.DataFrame,
    spec: ChronosExogSpec,
    embedder: ChronosEmbedder,
    target_ids_arr: np.ndarray | None,
    order_col: str | None,
) -> tuple[pd.DataFrame, int, int]:
    sort_cols = [spec.time_col]
    if order_col is not None:
        sort_cols.append(order_col)
    ordered = g.sort_values(sort_cols).copy()

    y = pd.to_numeric(ordered[spec.target_col], errors="coerce").astype(float).to_numpy(dtype=np.float32)
    windows = _build_history_windows(y, spec.window_size)
    hist_counts = np.isfinite(windows).sum(axis=1)
    keep = hist_counts >= int(spec.min_points)

    if target_ids_arr is not None:
        rid = ordered["loto_y_ts_row_id"].astype(np.int64).to_numpy()
        keep = keep & np.isin(rid, target_ids_arr)

    idx = np.where(keep)[0]
    if idx.size == 0:
        cols = ["loto_y_ts_row_id", *spec.group_cols, spec.time_col, spec.target_col, "y_idx"]
        return pd.DataFrame(columns=cols), int((hist_counts < int(spec.min_points)).sum()), 0

    selected_windows = windows[idx]
    processed, keep_after = _preprocess_windows_fast(
        selected_windows,
        fill_method=spec.fill_method,
        normalize_method=spec.normalize_method,
        min_points=int(spec.min_points),
    )
    if keep_after.sum() == 0:
        cols = ["loto_y_ts_row_id", *spec.group_cols, spec.time_col, spec.target_col, "y_idx"]
        return pd.DataFrame(columns=cols), int((hist_counts < int(spec.min_points)).sum()), int(idx.size)

    final_idx = idx[np.where(keep_after)[0]]
    emb = embedder.encode_batch(processed)

    base_df = ordered.iloc[final_idx][["loto_y_ts_row_id", *spec.group_cols, spec.time_col, spec.target_col]].reset_index(drop=True).copy()
    y_idx_df = pd.DataFrame({"y_idx": final_idx.astype(np.int32)})
    emb_df = pd.DataFrame(
        emb.astype(np.float32, copy=False),
        columns=[f"hist_chronos_{i + 1}" for i in range(spec.embedding_dim)],
    )
    out = pd.concat([base_df, y_idx_df, emb_df], axis=1)

    skipped_short = int((hist_counts < int(spec.min_points)).sum())
    skipped_after = int(idx.size - keep_after.sum())
    return out, skipped_short, skipped_after


def build_chronos_exog_dataframe(
    source_df: pd.DataFrame,
    spec: ChronosExogSpec,
    target_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_input(source_df, spec)
    base, row_id_meta = _ensure_row_id_column(source_df, spec)
    base[spec.time_col] = pd.to_datetime(base[spec.time_col])
    keys = list(spec.group_cols)
    order_col = _select_order_column(base, spec)
    sort_cols = [*keys, spec.time_col]
    if order_col is not None and order_col not in sort_cols:
        sort_cols.append(order_col)
    base = base.sort_values(sort_cols).reset_index(drop=True)
    target_ids_arr = None if target_ids is None else np.fromiter(target_ids, dtype=np.int64)

    probe = ChronosEmbedder(spec)
    workers = max(1, int(spec.parallel_workers))
    if not probe.parallel_safe:
        workers = 1

    groups = [g for _, g in base.groupby(keys, sort=False)]
    parts: list[pd.DataFrame] = []
    skipped_short = 0
    skipped_after = 0

    if workers == 1 or len(groups) <= 1:
        embedder = probe
        for g in groups:
            part, s1, s2 = _encode_group_fast(g, spec, embedder, target_ids_arr, order_col)
            parts.append(part)
            skipped_short += s1
            skipped_after += s2
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_encode_group_fast, g, spec, ChronosEmbedder(spec), target_ids_arr, order_col) for g in groups
            ]
            for fut in as_completed(futs):
                part, s1, s2 = fut.result()
                parts.append(part)
                skipped_short += s1
                skipped_after += s2

    non_empty_parts = [p for p in parts if not p.empty]
    if non_empty_parts:
        out = pd.concat(non_empty_parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["loto_y_ts_row_id", *keys, spec.time_col, spec.target_col, "y_idx"])

    if not out.empty:
        out = out.sort_values(keys + [spec.time_col, "y_idx"]).reset_index(drop=True)

    cfg_hash = hashlib.sha256(
        json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    now_ts = datetime.now(timezone.utc)
    meta_df = pd.DataFrame(
        {
            "embedding_dim": np.full(len(out), int(spec.embedding_dim), dtype=np.int32),
            "model_name": np.full(len(out), spec.model_name, dtype=object),
            "model_version": np.full(len(out), spec.model_version, dtype=object),
            "config_hash": np.full(len(out), cfg_hash, dtype=object),
            "created_at": np.full(len(out), now_ts, dtype=object),
            "updated_at": np.full(len(out), now_ts, dtype=object),
        },
        index=out.index,
    )
    out = pd.concat([out, meta_df], axis=1)

    meta = {
        "requested_backend": spec.backend,
        "resolved_backend": probe.resolved_backend,
        "fallback_reason": probe.fallback_reason,
        "embedding_dim": spec.embedding_dim,
        "window_size": spec.window_size,
        "min_points": spec.min_points,
        "batch_size": spec.batch_size,
        "parallel_workers_requested": spec.parallel_workers,
        "parallel_workers_used": workers,
        "parallel_safe": probe.parallel_safe,
        "row_id_meta": row_id_meta,
        "target_ids_mode": "missing_only" if target_ids is not None else "all",
        "skipped_short_history": int(skipped_short),
        "skipped_after_preprocess": int(skipped_after),
    }
    return out, meta


def _write_table(engine: Engine, df: pd.DataFrame, spec: ChronosExogSpec) -> None:
    schema = safe_ident(spec.target_schema)
    table = safe_ident(spec.target_table)
    if spec.postgres_write_mode == "copy":
        copy_dataframe_to_postgres(
            engine,
            df,
            schema=schema,
            table=table,
            if_exists=spec.if_exists,
            chunk_rows=spec.postgres_copy_chunk_rows,
        )
    else:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        df.to_sql(table, engine, schema=schema, if_exists=spec.if_exists, index=False, method="multi", chunksize=5000)

    idx_stmts: list[str] = []
    if "loto_y_ts_row_id" in df.columns:
        idx_stmts.append(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_row_id" ON "{schema}"."{table}" ("loto_y_ts_row_id")'
        )
    cols = [c for c in [*spec.group_cols, spec.time_col] if c in df.columns]
    if cols:
        quoted = ", ".join([f'"{safe_ident(c)}"' for c in cols])
        idx_stmts.append(f'CREATE INDEX IF NOT EXISTS "idx_{table}_key_time" ON "{schema}"."{table}" ({quoted})')
    idx_stmts.append(
        f'CREATE INDEX IF NOT EXISTS "idx_{table}_model_meta" ON "{schema}"."{table}" ("model_name","model_version","config_hash")'
    )
    if spec.create_postgres_index:
        with engine.begin() as conn:
            for stmt in idx_stmts:
                conn.execute(text(stmt))


def run_chronos_exog_build(spec: ChronosExogSpec) -> dict[str, Any]:
    stage_timing_sec: dict[str, float] = {}
    t0 = time.perf_counter()
    engine = _make_engine(spec)
    run_id = generate_run_id("build_exog_chronos")
    monitor = ResourceMonitor(interval_sec=spec.sampling_interval_sec)
    run_status = "failed"
    error_message: str | None = None
    elapsed_started = time.perf_counter()
    source_rows: int | None = None
    written_rows: int | None = None
    target_name = f"{spec.target_schema}.{spec.target_table}"
    source_name = f"{spec.source_schema}.{spec.source_table}"
    codegen_summary = _summarize_codegen_yaml(spec.chronos_codegen_yaml)
    stage_timing_sec["setup"] = time.perf_counter() - t0

    cfg = ResourcesConfig(
        db_host=spec.host,
        db_port=spec.port,
        db_user=spec.user,
        db_password=spec.password,
        db_name=spec.database,
        schema="resources",
        namespace="chronos",
        table_naming="plain",
        app_name="chronos_exog_builder",
        env=spec.env,
        profile=spec.profile,
        command=f"resources.chronos_exog_pipeline {spec.source_schema}.{spec.source_table} -> {spec.target_schema}.{spec.target_table}",
        tags={
            "source": f"{spec.source_schema}.{spec.source_table}",
            "target": f"{spec.target_schema}.{spec.target_table}",
            "group_cols": list(spec.group_cols),
            "backend": spec.backend,
            "embedding_dim": spec.embedding_dim,
            "window_size": spec.window_size,
            "min_points": spec.min_points,
            "codegen": codegen_summary,
        },
        enable_gpu=spec.enable_gpu_compute,
        enable_sampling=True,
        sampling_interval_sec=spec.sampling_interval_sec,
        ensure_schema=True,
        parallel_snapshot_workers=max(1, spec.parallel_workers),
    )

    upsert_model_run(
        engine,
        run_id=run_id,
        model_name=spec.model_name,
        meta={
            "command": "build-exog-chronos",
            "source": source_name,
            "target": target_name,
            "backend": spec.backend,
        },
        library_name="resources",
        adapter_name="build_exog_chronos",
        status="running",
    )
    write_log_run_history(
        engine,
        run_id=run_id,
        event_type="build_start",
        status="running",
        model_name="chronos",
        library_name="resources",
        adapter_name="build_exog_chronos",
        dataset_name=target_name,
        message=f"build-exog-chronos started for {target_name}",
        payload={"source_table": source_name, "target_table": target_name, "backend": spec.backend},
    )
    monitor.start()
    try:
        with start_run(cfg) as run:
            run.attach_sqlalchemy_engine(engine)

            with run.span(stage_name="analyze_chronos_codegen", extra={"codegen_summary": codegen_summary}):
                pass

            t_extract = time.perf_counter()
            with run.span(stage_name="extract_source"):
                src = _read_source(spec, engine)
                src, row_meta = _ensure_row_id_column(src, spec)
            source_rows = int(len(src))
            stage_timing_sec["extract"] = time.perf_counter() - t_extract

            t_select = time.perf_counter()
            with run.span(stage_name="target_selection", rows_in=int(len(src)), extra={"only_missing": spec.only_missing}):
                target_ids = _target_key_set(spec, engine, src)
            stage_timing_sec["target_selection"] = time.perf_counter() - t_select

            t_build = time.perf_counter()
            with run.span(stage_name="build_chronos_embeddings", rows_in=int(len(src))):
                out, emb_meta = build_chronos_exog_dataframe(src, spec, target_ids=target_ids)
            stage_timing_sec["build"] = time.perf_counter() - t_build

            t_write = time.perf_counter()
            with run.span(
                stage_name="write_exog_chronos",
                rows_in=int(len(out)),
                rows_out=int(len(out)),
                extra={"embedding_meta": emb_meta, "row_meta": row_meta},
            ):
                _write_table(engine, out, spec)
            stage_timing_sec["write"] = time.perf_counter() - t_write

            run.set_counts(
                rows_target=int(len(src)),
                rows_written=int(len(out)),
                rows_failed=max(0, int(len(src) - len(out))),
            )
            written_rows = int(len(out))
            elapsed_sec = time.perf_counter() - elapsed_started
            write_log_run_history(
                engine,
                run_id=run_id,
                event_type="build_end",
                status="success",
                model_name="chronos",
                library_name="resources",
                adapter_name="build_exog_chronos",
                dataset_name=target_name,
                message=f"build-exog-chronos completed for {target_name}",
                payload={
                    "source_table": source_name,
                    "target_table": target_name,
                    "source_rows": source_rows,
                    "written_rows": written_rows,
                    "elapsed_sec": elapsed_sec,
                    "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
                    "backend": emb_meta.get("resolved_backend"),
                    "fallback_reason": emb_meta.get("fallback_reason"),
                },
            )

            return {
                "run_id": run_id,
                "source_rows": int(len(src)),
                "written_rows": int(len(out)),
                "target": f"{spec.target_schema}.{spec.target_table}",
                "embedding_meta": emb_meta,
                "row_id_meta": row_meta,
                "codegen_summary": codegen_summary,
                "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
            }
    except Exception as exc:
        error_message = str(exc)
        elapsed_sec = time.perf_counter() - elapsed_started
        write_log_run_history(
            engine,
            run_id=run_id,
            event_type="build_end",
            status="failed",
            model_name="chronos",
            library_name="resources",
            adapter_name="build_exog_chronos",
            dataset_name=target_name,
            message=f"build-exog-chronos failed for {target_name}",
            payload={
                "source_table": source_name,
                "target_table": target_name,
                "source_rows": source_rows,
                "written_rows": written_rows,
                "elapsed_sec": elapsed_sec,
                "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
                "error_message": error_message,
            },
        )
        raise
    finally:
        samples = monitor.stop()
        if samples:
            write_resource_samples(engine, run_id=run_id, samples=monitor.to_dicts())
        if error_message is None:
            run_status = "success"
        mark_model_run_end(engine, run_id=run_id, status=run_status, error_message=error_message)


def _parse_group_cols(raw: str) -> tuple[str, ...]:
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    if not cols:
        raise ValueError("group-cols must not be empty")
    return tuple(cols)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m resources.chronos_exog_pipeline")
    p.add_argument("--profile", default="local")
    p.add_argument("--env", default="LOCAL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5432)
    p.add_argument("--user", default="loto")
    p.set_defaults(password=os.environ.get("DB_PASSWORD", ""))
    p.add_argument("--database", default="loto")

    p.add_argument("--source-schema", default="dataset")
    p.add_argument("--source-table", default="loto_y_ts")
    p.add_argument("--source-where", default=None)
    p.add_argument("--ds-start", default=None)
    p.add_argument("--ds-end", default=None)
    p.add_argument("--loto-filter", default=None, help="CSV values")
    p.add_argument("--ts-type-filter", default=None, help="CSV values")

    p.add_argument("--target-schema", default="exog")
    p.add_argument("--target-table", default="chronos")
    p.add_argument("--if-exists", default="append", choices=["replace", "append", "fail"])
    p.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p.add_argument("--time-col", default="ds")
    p.add_argument("--target-col", default="y")
    p.add_argument("--source-row-id-column", default="row_id")
    p.add_argument("--y-idx-order-column", default="row_id")

    p.add_argument(
        "--backend",
        default="chronos_pipeline_auto",
        choices=[
            "chronos_pipeline_auto",
            "chronos2_pipeline",
            "chronos_bolt_pipeline",
            "chronos_pipeline_legacy",
            "chronos_forecast_features",
        ],
    )
    p.add_argument("--model-id", default="amazon/chronos-bolt-small")
    p.add_argument("--model-name", default="chronos")
    p.add_argument("--model-version", default="1.0")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--min-points", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--normalize-method", default="zscore", choices=["zscore", "none"])
    p.add_argument("--fill-method", default="zero", choices=["ffill", "zero", "drop"])
    p.add_argument("--parallel-workers", type=int, default=4)
    p.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p.add_argument(
        "--chronos-codegen-yaml",
        default="./docs/lib_docs/chronos-forecasting_scripts_evaluation_agg-relative-score_all_codegen.yaml",
    )

    args = p.parse_args()
    spec = ChronosExogSpec(
        profile=args.profile,
        env=args.env,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        source_schema=args.source_schema,
        source_table=args.source_table,
        source_where=args.source_where,
        target_schema=args.target_schema,
        target_table=args.target_table,
        if_exists=args.if_exists,
        only_missing=bool(args.only_missing),
        group_cols=_parse_group_cols(args.group_cols),
        time_col=args.time_col,
        target_col=args.target_col,
        source_row_id_column=args.source_row_id_column,
        y_idx_order_column=args.y_idx_order_column,
        ds_start=args.ds_start,
        ds_end=args.ds_end,
        loto_filter=_parse_filter_csv(args.loto_filter),
        ts_type_filter=_parse_filter_csv(args.ts_type_filter),
        backend=args.backend,
        model_id=args.model_id,
        model_name=args.model_name,
        model_version=args.model_version,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        min_points=args.min_points,
        batch_size=args.batch_size,
        normalize_method=args.normalize_method,
        fill_method=args.fill_method,
        parallel_workers=args.parallel_workers,
        enable_gpu_compute=bool(args.enable_gpu_compute),
        local_files_only=bool(args.local_files_only),
        sampling_interval_sec=args.sampling_interval_sec,
        chronos_codegen_yaml=args.chronos_codegen_yaml,
    )
    out = run_chronos_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
