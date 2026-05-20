from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .config import ResourcesConfig
from .context import start_run
from .db.postgres_copy import copy_dataframe_to_postgres
from .utils import safe_ident


@dataclass(frozen=True)
class TimesFMExogSpec:
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
    target_table: str = "timesfm"
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

    backend: str = "timesfm_forecast_features"
    model_id: str = "google/timesfm-2.5-200m-pytorch"
    model_name: str = "timesfm"
    model_version: str = "2.5"
    embedding_dim: int = 256
    window_size: int = 128
    min_points: int = 16
    batch_size: int = 64
    normalize_method: str = "zscore"
    fill_method: str = "ffill"

    parallel_workers: int = 4
    enable_gpu_compute: bool = True
    local_files_only: bool = True
    sampling_interval_sec: float = 1.0
    postgres_copy_strategy: str = "binary_copy"

    timesfm_codegen_yaml: str = "./docs/lib_docs/timesfm_all_codegen.yaml"


def _make_engine(spec: TimesFMExogSpec) -> Engine:
    url = f"postgresql+psycopg2://{spec.user}:{spec.password}@{spec.host}:{spec.port}/{spec.database}"
    return create_engine(url, pool_pre_ping=True)


def _safe_table_ref(schema: str, table: str) -> str:
    return f'"{safe_ident(schema)}"."{safe_ident(table)}"'


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


def _parse_filter_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(vals)


def _read_source(spec: TimesFMExogSpec, engine: Engine) -> pd.DataFrame:
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


def _validate_input(df: pd.DataFrame, spec: TimesFMExogSpec) -> None:
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


def _resolve_row_id_candidates(spec: TimesFMExogSpec) -> list[str]:
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


def _ensure_row_id_column(df: pd.DataFrame, spec: TimesFMExogSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    row_ids = numeric.fillna(0).astype(np.int64).to_numpy()

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


def _select_order_column(df: pd.DataFrame, spec: TimesFMExogSpec) -> str | None:
    for c in [spec.y_idx_order_column, "updated_ts", "exec_ts", "row_id", "id", "unique_id"]:
        if c and c in df.columns:
            return c
    return None


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


class _EmbeddingBackend:
    def encode(
        self, history_values: np.ndarray, history_ds: Sequence[str] | None = None, meta: dict[str, Any] | None = None
    ) -> np.ndarray:
        raise NotImplementedError


class _TimesfmForecastFeatureBackend(_EmbeddingBackend):
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)

    def encode(
        self, history_values: np.ndarray, history_ds: Sequence[str] | None = None, meta: dict[str, Any] | None = None
    ) -> np.ndarray:
        x = np.asarray(history_values, dtype=np.float32)
        if x.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        feats: list[float] = []
        feats.extend(
            [float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.median(x)), float(x[-1])]
        )
        if x.size > 1:
            idx = np.arange(x.size, dtype=np.float32)
            slope = float(np.polyfit(idx, x, 1)[0])
        else:
            slope = 0.0
        feats.append(slope)

        n = max(int(x.size), 2)
        idx = np.arange(x.size, dtype=np.float32)
        max_k = max(2, self.embedding_dim // 2 + 1)
        for k in range(1, max_k):
            w = 2.0 * np.pi * float(k) * idx / float(n)
            feats.append(float(np.mean(x * np.sin(w))))
            feats.append(float(np.mean(x * np.cos(w))))
            if len(feats) >= self.embedding_dim:
                break

        arr = np.asarray(feats[: self.embedding_dim], dtype=np.float32)
        if arr.size < self.embedding_dim:
            arr = np.pad(arr, (0, self.embedding_dim - arr.size)).astype(np.float32)
        return arr


class _TimesfmTransformersBackend(_EmbeddingBackend):
    def __init__(self, model_id: str, embedding_dim: int, enable_gpu: bool, local_files_only: bool) -> None:
        self.model_id = model_id
        self.embedding_dim = int(embedding_dim)
        self.enable_gpu = bool(enable_gpu)
        self.local_files_only = bool(local_files_only)
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._load()

    def _load(self) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch
        from transformers import AutoModel

        self._torch = torch
        if self.enable_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._model.to(self._device)
        self._model.eval()

    def encode(
        self, history_values: np.ndarray, history_ds: Sequence[str] | None = None, meta: dict[str, Any] | None = None
    ) -> np.ndarray:
        if self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("timesfm transformers backend not initialized")

        x = np.asarray(history_values, dtype=np.float32)
        if x.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        t = self._torch.from_numpy(x).unsqueeze(0).to(self._device)
        outputs = None
        with self._torch.no_grad():
            for kwargs in (
                {"input_values": t, "output_hidden_states": True},
                {"inputs": t, "output_hidden_states": True},
                {"x": t, "output_hidden_states": True},
                {"input_values": t},
                {"inputs": t},
                {"x": t},
            ):
                try:
                    outputs = self._model(**kwargs)
                    break
                except TypeError:
                    continue

        if outputs is None:
            raise RuntimeError("failed to run model forward")

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hs = getattr(outputs, "hidden_states", None)
            if hs:
                hidden = hs[-1]
        if hidden is None:
            if isinstance(outputs, tuple) and len(outputs) > 0:
                hidden = outputs[0]
            else:
                raise RuntimeError("hidden state not available from model output")

        pooled = hidden.mean(dim=1).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        if pooled.ndim > 1:
            pooled = pooled.reshape(-1)

        if pooled.size >= self.embedding_dim:
            return pooled[: self.embedding_dim]
        return np.pad(pooled, (0, self.embedding_dim - pooled.size)).astype(np.float32)


class TimesFMEmbedder:
    def __init__(self, spec: TimesFMExogSpec) -> None:
        self.spec = spec
        self.requested_backend = spec.backend
        self.resolved_backend = spec.backend
        self.fallback_reason: str | None = None
        self.backend: _EmbeddingBackend
        self._init_backend()

    def _init_backend(self) -> None:
        if self.requested_backend == "timesfm_forecast_features":
            self.backend = _TimesfmForecastFeatureBackend(self.spec.embedding_dim)
            self.resolved_backend = "timesfm_forecast_features"
            return

        if self.requested_backend == "timesfm_transformers":
            try:
                self.backend = _TimesfmTransformersBackend(
                    model_id=self.spec.model_id,
                    embedding_dim=self.spec.embedding_dim,
                    enable_gpu=self.spec.enable_gpu_compute,
                    local_files_only=self.spec.local_files_only,
                )
                self.resolved_backend = "timesfm_transformers"
                return
            except Exception as e:
                self.backend = _TimesfmForecastFeatureBackend(self.spec.embedding_dim)
                self.resolved_backend = "timesfm_forecast_features"
                self.fallback_reason = str(e)
                return

        raise ValueError(f"unsupported backend: {self.requested_backend}")

    def encode(self, history_values: np.ndarray, meta: dict[str, Any] | None = None) -> np.ndarray:
        vec = np.asarray(self.backend.encode(history_values, history_ds=None, meta=meta), dtype=np.float32)
        if vec.size == self.spec.embedding_dim:
            return vec
        if vec.size > self.spec.embedding_dim:
            return vec[: self.spec.embedding_dim]
        return np.pad(vec, (0, self.spec.embedding_dim - vec.size)).astype(np.float32)


def _target_key_set(spec: TimesFMExogSpec, engine: Engine, src: pd.DataFrame) -> set[int] | None:
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


def _encode_group(
    g: pd.DataFrame,
    spec: TimesFMExogSpec,
    embedder: TimesFMEmbedder,
    target_ids: set[int] | None,
    order_col: str | None,
) -> tuple[pd.DataFrame, int, int]:
    sort_cols = [spec.time_col]
    if order_col is not None:
        sort_cols.append(order_col)
    ordered = g.sort_values(sort_cols).copy()

    y = pd.to_numeric(ordered[spec.target_col], errors="coerce").astype(float).to_numpy(dtype=np.float32)
    rows: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_preprocess = 0

    for i in range(len(ordered)):
        row = ordered.iloc[i]
        rid = int(row["loto_y_ts_row_id"])
        if target_ids is not None and rid not in target_ids:
            continue

        hist = y[max(0, i - spec.window_size) : i]
        if hist.size < spec.min_points:
            skipped_short += 1
            continue

        hist2 = _preprocess_history(hist, fill_method=spec.fill_method, normalize_method=spec.normalize_method)
        if hist2.size < spec.min_points:
            skipped_preprocess += 1
            continue

        vec = embedder.encode(hist2, meta={"row_id": rid})
        out: dict[str, Any] = {
            "loto_y_ts_row_id": rid,
            spec.time_col: row[spec.time_col],
            spec.target_col: float(row[spec.target_col]) if pd.notna(row[spec.target_col]) else None,
            "y_idx": int(i),
        }
        for c in spec.group_cols:
            out[c] = row[c]
        if "unique_id" in ordered.columns and "unique_id" not in out:
            out["unique_id"] = row["unique_id"]

        for j in range(spec.embedding_dim):
            out[f"hist_timesfm_{j + 1}"] = float(vec[j])
        rows.append(out)

    if not rows:
        cols = ["loto_y_ts_row_id", *spec.group_cols, spec.time_col, spec.target_col, "y_idx"]
        return pd.DataFrame(columns=cols), skipped_short, skipped_preprocess
    return pd.DataFrame(rows), skipped_short, skipped_preprocess


def build_timesfm_exog_dataframe(
    source_df: pd.DataFrame,
    spec: TimesFMExogSpec,
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

    probe = TimesFMEmbedder(spec)
    resolved_backend = probe.resolved_backend
    fallback_reason = probe.fallback_reason
    workers = max(1, int(spec.parallel_workers))
    if resolved_backend == "timesfm_transformers":
        workers = 1

    groups = [g for _, g in base.groupby(keys, sort=False)]
    parts: list[pd.DataFrame] = []
    skipped_short = 0
    skipped_preprocess = 0

    if workers == 1 or len(groups) <= 1:
        embedder = probe
        for g in groups:
            part, s1, s2 = _encode_group(g, spec, embedder, target_ids, order_col)
            parts.append(part)
            skipped_short += s1
            skipped_preprocess += s2
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _encode_group,
                    g,
                    spec,
                    TimesFMEmbedder(spec),
                    target_ids,
                    order_col,
                )
                for g in groups
            ]
            for fut in as_completed(futs):
                part, s1, s2 = fut.result()
                parts.append(part)
                skipped_short += s1
                skipped_preprocess += s2

    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["loto_y_ts_row_id", *keys, spec.time_col, spec.target_col, "y_idx"])
    if not out.empty:
        out = out.sort_values(keys + [spec.time_col, "y_idx"]).reset_index(drop=True)

    cfg_hash = hashlib.sha256(
        json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    now_ts = datetime.now(timezone.utc)
    out["embedding_dim"] = int(spec.embedding_dim)
    out["model_name"] = spec.model_name
    out["model_version"] = spec.model_version
    out["config_hash"] = cfg_hash
    out["created_at"] = now_ts
    out["updated_at"] = now_ts

    meta = {
        "requested_backend": spec.backend,
        "resolved_backend": resolved_backend,
        "fallback_reason": fallback_reason,
        "embedding_dim": spec.embedding_dim,
        "window_size": spec.window_size,
        "min_points": spec.min_points,
        "row_id_meta": row_id_meta,
        "target_ids_mode": "missing_only" if target_ids is not None else "all",
        "skipped_short_history": int(skipped_short),
        "skipped_after_preprocess": int(skipped_preprocess),
    }
    return out, meta


def _write_table(engine: Engine, df: pd.DataFrame, spec: TimesFMExogSpec) -> None:
    schema = safe_ident(spec.target_schema)
    table = safe_ident(spec.target_table)

    write_df = df
    insp = inspect(engine)
    exists = insp.has_table(table, schema=schema)
    if exists and spec.if_exists == "append":
        cols_sql = text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
            """
        )
        existing_cols = pd.read_sql(cols_sql, engine, params={"schema": schema, "table": table})["column_name"].tolist()
        if existing_cols:
            for c in existing_cols:
                if c not in write_df.columns:
                    write_df[c] = np.nan
            write_df = write_df[[c for c in existing_cols if c in write_df.columns]]
    copy_dataframe_to_postgres(
        engine,
        write_df,
        schema=schema,
        table=table,
        if_exists=spec.if_exists,
        copy_strategy=spec.postgres_copy_strategy,
    )

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
    with engine.begin() as conn:
        for stmt in idx_stmts:
            conn.execute(text(stmt))


def run_timesfm_exog_build(spec: TimesFMExogSpec) -> dict[str, Any]:
    engine = _make_engine(spec)
    codegen_summary = _summarize_codegen_yaml(spec.timesfm_codegen_yaml)

    cfg = ResourcesConfig(
        db_host=spec.host,
        db_port=spec.port,
        db_user=spec.user,
        db_password=spec.password,
        db_name=spec.database,
        schema="resources",
        namespace="timesfm",
        table_naming="plain",
        app_name="timesfm_exog_builder",
        env=spec.env,
        profile=spec.profile,
        command=f"resources.timesfm_exog_pipeline {spec.source_schema}.{spec.source_table} -> {spec.target_schema}.{spec.target_table}",
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

    with start_run(cfg) as run:
        run.attach_sqlalchemy_engine(engine)

        with run.span(stage_name="analyze_timesfm_codegen", extra={"codegen_summary": codegen_summary}):
            pass

        with run.span(stage_name="extract_source"):
            src = _read_source(spec, engine)
            src, row_meta = _ensure_row_id_column(src, spec)

        with run.span(stage_name="target_selection", rows_in=int(len(src)), extra={"only_missing": spec.only_missing}):
            target_ids = _target_key_set(spec, engine, src)

        with run.span(stage_name="build_timesfm_embeddings", rows_in=int(len(src))):
            out, emb_meta = build_timesfm_exog_dataframe(src, spec, target_ids=target_ids)

        with run.span(
            stage_name="write_exog_timesfm",
            rows_in=int(len(out)),
            rows_out=int(len(out)),
            extra={"embedding_meta": emb_meta, "row_meta": row_meta},
        ):
            _write_table(engine, out, spec)

        run.set_counts(
            rows_target=int(len(src)),
            rows_written=int(len(out)),
            rows_failed=max(0, int(len(src) - len(out))),
        )

        return {
            "run_id": str(run.run_id),
            "source_rows": int(len(src)),
            "written_rows": int(len(out)),
            "target": f"{spec.target_schema}.{spec.target_table}",
            "embedding_meta": emb_meta,
            "row_id_meta": row_meta,
            "codegen_summary": codegen_summary,
        }


def _parse_group_cols(raw: str) -> tuple[str, ...]:
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    if not cols:
        raise ValueError("group-cols must not be empty")
    return tuple(cols)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m resources.timesfm_exog_pipeline")
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
    p.add_argument("--target-table", default="timesfm")
    p.add_argument("--if-exists", default="append", choices=["replace", "append", "fail"])
    p.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--group-cols", default="loto,ts_type")
    p.add_argument("--time-col", default="ds")
    p.add_argument("--target-col", default="y")
    p.add_argument("--source-row-id-column", default="row_id")
    p.add_argument("--y-idx-order-column", default="row_id")

    p.add_argument(
        "--backend", default="timesfm_forecast_features", choices=["timesfm_forecast_features", "timesfm_transformers"]
    )
    p.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    p.add_argument("--model-name", default="timesfm")
    p.add_argument("--model-version", default="2.5")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--min-points", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--normalize-method", default="zscore", choices=["zscore", "none"])
    p.add_argument("--fill-method", default="ffill", choices=["ffill", "zero", "drop"])
    p.add_argument("--parallel-workers", type=int, default=4)
    p.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p.add_argument("--postgres-copy-strategy", default="binary_copy", choices=["csv_buffer", "binary_copy", "psycopg3_row"])
    p.add_argument("--timesfm-codegen-yaml", default="./docs/lib_docs/timesfm_all_codegen.yaml")

    args = p.parse_args()
    spec = TimesFMExogSpec(
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
        postgres_copy_strategy=args.postgres_copy_strategy,
        timesfm_codegen_yaml=args.timesfm_codegen_yaml,
    )
    out = run_timesfm_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
