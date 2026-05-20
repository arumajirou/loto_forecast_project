from __future__ import annotations

import argparse
import os
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import inspect
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import ResourcesConfig
from .context import start_run
from .db.postgres_copy import copy_dataframe_to_postgres
from .utils import safe_ident


@dataclass(frozen=True)
class Uni2TSExogSpec:
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
    target_table: str = "uni2ts"
    if_exists: str = "replace"

    group_cols: tuple[str, ...] = ("loto", "unique_id", "ts_type")
    time_col: str = "ds"
    target_col: str = "y"

    context_length: int = 128
    embedding_dim: int = 256
    batch_size: int = 512
    parallel_workers: int = 4
    create_postgres_index: bool = True
    postgres_write_mode: str = "copy"
    postgres_copy_chunk_rows: int = 50000
    profile_stages: bool = False
    row_batch_size: int = 5000
    max_groups_per_batch: int = 64

    model_name: str = "uni2ts"
    model_version: str = "2.0.0"
    model_checkpoint: str | None = None
    local_files_only: bool = True

    enable_gpu_compute: bool = True
    sampling_interval_sec: float = 1.0

    uni2ts_codegen_yaml: str = "./docs/lib_docs/uni2ts_all_codegen.yaml"


def _make_engine(spec: Uni2TSExogSpec) -> Engine:
    url = f"postgresql+psycopg2://{spec.user}:{spec.password}@{spec.host}:{spec.port}/{spec.database}"
    return create_engine(url, pool_pre_ping=True)


def _safe_table_ref(schema: str, table: str) -> str:
    return f'"{safe_ident(schema)}"."{safe_ident(table)}"'


def _read_source(spec: Uni2TSExogSpec, engine: Engine) -> pd.DataFrame:
    table_ref = _safe_table_ref(spec.source_schema, spec.source_table)
    available_cols = {
        str(col["name"])
        for col in inspect(engine).get_columns(safe_ident(spec.source_table), schema=safe_ident(spec.source_schema))
    }
    selected_cols = [c for c in [*spec.group_cols, spec.time_col, spec.target_col] if c in available_cols]
    row_id_col = f"{spec.source_table}_row_id"
    for candidate in [row_id_col, "loto_y_ts_row_id", "row_id"]:
        if candidate in available_cols and candidate not in selected_cols:
            selected_cols.append(candidate)
    select_list = ", ".join(f'"{safe_ident(c)}"' for c in selected_cols)
    sql = f"SELECT {select_list} FROM {table_ref}"
    if spec.source_where:
        sql += f" WHERE {spec.source_where}"
    return pd.read_sql(sql, engine)


def _validate_input(df: pd.DataFrame, spec: Uni2TSExogSpec) -> None:
    required = set(spec.group_cols) | {spec.time_col, spec.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in source: {sorted(missing)}")


def _summarize_uni2ts_codegen(path: str) -> dict[str, Any]:
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


def _build_history_windows(y: np.ndarray, context_length: int) -> np.ndarray:
    n = len(y)
    windows = np.full((n, context_length), np.nan, dtype=np.float32)
    for i in range(n):
        end = i  # use history only (exclude current)
        start = max(0, end - context_length)
        hist = y[start:end]
        if len(hist) > 0:
            windows[i, -len(hist) :] = hist
    return windows


class Uni2TSEmbedder:
    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        batch_size: int,
        checkpoint: str | None,
        local_files_only: bool,
        enable_gpu: bool,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)
        self.checkpoint = checkpoint
        self.local_files_only = bool(local_files_only)
        self.enable_gpu = bool(enable_gpu)

        self.backend = "fallback"
        self.model: Any | None = None
        self.patch_size: int | None = None
        self.d_model: int | None = None

        self.device = "cpu"
        if self.enable_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
            except Exception:
                pass

        self._try_load_uni2ts_module()

    def _try_load_uni2ts_module(self) -> None:
        if not self.checkpoint:
            return
        try:
            from uni2ts.model.moirai2.module import Moirai2Module

            model = Moirai2Module.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            model.to(self.device)
            model.eval()

            in_features = int(model.in_proj.in_features)
            d_model = int(model.in_proj.out_features)
            patch = max(1, in_features // 2)

            self.model = model
            self.patch_size = patch
            self.d_model = d_model
            self.backend = "uni2ts_moirai2"
        except Exception:
            self.model = None
            self.backend = "fallback"

    def _fit_to_embedding_dim(self, emb: np.ndarray) -> np.ndarray:
        if emb.shape[1] == self.embedding_dim:
            return emb
        if emb.shape[1] > self.embedding_dim:
            return emb[:, : self.embedding_dim]

        pad = np.zeros((emb.shape[0], self.embedding_dim - emb.shape[1]), dtype=np.float32)
        return np.concatenate([emb, pad], axis=1)

    def _fallback_encode(self, windows: np.ndarray) -> np.ndarray:
        arr = windows.copy().astype(np.float32)
        mask = np.isfinite(arr).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        mean = np.sum(arr * mask, axis=1, keepdims=True) / (np.sum(mask, axis=1, keepdims=True) + 1e-6)
        centered = (arr - mean) * mask
        std = np.sqrt(np.sum(centered**2, axis=1, keepdims=True) / (np.sum(mask, axis=1, keepdims=True) + 1e-6))
        z = centered / (std + 1e-6)

        fft = np.fft.rfft(z, axis=1)
        feat = np.concatenate([z, np.real(fft), np.imag(fft), mask], axis=1).astype(np.float32)

        rs = np.random.RandomState(42)
        proj = rs.normal(0.0, 1.0 / np.sqrt(feat.shape[1]), size=(feat.shape[1], self.embedding_dim)).astype(np.float32)
        emb = feat @ proj
        return emb.astype(np.float32)

    def _uni2ts_encode(self, windows: np.ndarray) -> np.ndarray:
        import torch
        from uni2ts.model.moirai2.module import packed_causal_attention_mask

        assert self.model is not None
        assert self.patch_size is not None

        n = windows.shape[0]
        p = int(self.patch_size)
        L = windows.shape[1]
        need = ((L + p - 1) // p) * p

        arr = windows
        if need > L:
            pad = np.full((n, need - L), np.nan, dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=1)

        out_batches: list[np.ndarray] = []
        for s in range(0, n, self.batch_size):
            e = min(n, s + self.batch_size)
            x = arr[s:e]

            B = x.shape[0]
            S = x.shape[1] // p
            target_np = x.reshape(B, S, p)
            obs_np = np.isfinite(target_np)
            target_np = np.nan_to_num(target_np, nan=0.0)

            target = torch.tensor(target_np, dtype=torch.float32, device=self.device)
            observed_mask = torch.tensor(obs_np, dtype=torch.bool, device=self.device)
            sample_id = torch.zeros((B, S), dtype=torch.long, device=self.device)
            time_id = torch.arange(S, device=self.device).unsqueeze(0).repeat(B, 1)
            variate_id = torch.zeros((B, S), dtype=torch.long, device=self.device)
            prediction_mask = torch.zeros((B, S), dtype=torch.bool, device=self.device)

            with torch.no_grad():
                loc, scale = self.model.scaler(
                    target,
                    observed_mask * ~prediction_mask.unsqueeze(-1),
                    sample_id,
                    variate_id,
                )
                scaled_target = (target - loc) / scale
                input_tokens = torch.cat([scaled_target, observed_mask.to(torch.float32)], dim=-1)
                reprs = self.model.in_proj(input_tokens)
                reprs = self.model.encoder(
                    reprs,
                    packed_causal_attention_mask(sample_id, time_id),
                    time_id=time_id,
                    var_id=variate_id,
                )
                emb = reprs.mean(dim=1).detach().cpu().numpy().astype(np.float32)

            out_batches.append(self._fit_to_embedding_dim(emb))

        return np.concatenate(out_batches, axis=0)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self._fallback_encode(windows)
        try:
            return self._uni2ts_encode(windows)
        except Exception:
            self.backend = "fallback"
            return self._fallback_encode(windows)


def _embed_group(g: pd.DataFrame, spec: Uni2TSExogSpec, embedder: Uni2TSEmbedder) -> pd.DataFrame:
    keys = list(spec.group_cols)
    ordered = g.sort_values(spec.time_col).copy()
    y = pd.to_numeric(ordered[spec.target_col], errors="coerce").astype(float).values

    windows = _build_history_windows(y, spec.context_length)
    emb = embedder.encode(windows)

    row_id_col = f"{spec.source_table}_row_id"
    base_cols = [*keys, spec.time_col, spec.target_col]
    if row_id_col in ordered.columns:
        base_cols = [row_id_col, *base_cols]

    base_df = ordered[base_cols].copy()

    emb_cols = [f"hist_uni2ts_{i + 1}" for i in range(spec.embedding_dim)]
    emb_df = pd.DataFrame(
        emb.astype(np.float32, copy=False),
        index=base_df.index,
        columns=emb_cols,
    )

    y_idx_df = pd.DataFrame(
        {"y_idx": np.arange(len(base_df), dtype=np.int32)},
        index=base_df.index,
    )

    out = pd.concat([base_df, emb_df, y_idx_df], axis=1)
    return out

def build_uni2ts_exog_dataframe(df: pd.DataFrame, spec: Uni2TSExogSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_input(df, spec)

    base = df.copy()
    base[spec.time_col] = pd.to_datetime(base[spec.time_col])
    keys = list(spec.group_cols)
    base = base.sort_values(keys + [spec.time_col]).reset_index(drop=True)

    embedder = Uni2TSEmbedder(
        embedding_dim=spec.embedding_dim,
        context_length=spec.context_length,
        batch_size=spec.batch_size,
        checkpoint=spec.model_checkpoint,
        local_files_only=spec.local_files_only,
        enable_gpu=spec.enable_gpu_compute,
    )

    groups = [g for _, g in base.groupby(keys, sort=False)]
    workers = max(1, int(spec.parallel_workers))
    if spec.model_checkpoint:
        workers = 1

    parts: list[pd.DataFrame] = []
    if workers == 1 or len(groups) <= 1:
        for g in groups:
            parts.append(_embed_group(g, spec, embedder))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for g in groups:
                futs.append(
                    ex.submit(
                        lambda gg: _embed_group(
                            gg,
                            spec,
                            Uni2TSEmbedder(
                                embedding_dim=spec.embedding_dim,
                                context_length=spec.context_length,
                                batch_size=spec.batch_size,
                                checkpoint=spec.model_checkpoint,
                                local_files_only=spec.local_files_only,
                                enable_gpu=spec.enable_gpu_compute,
                            ),
                        ),
                        g,
                    )
                )
            for fut in as_completed(futs):
                parts.append(fut.result())

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(keys + [spec.time_col]).reset_index(drop=True)

    row_id_col = f"{spec.source_table}_row_id"
    if row_id_col not in out.columns:
        out.insert(0, row_id_col, np.arange(1, len(out) + 1, dtype=np.int64))

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

    out = pd.concat([out, meta_df], axis=1).copy()

    meta = {
        "backend": embedder.backend,
        "embedding_dim": spec.embedding_dim,
        "context_length": spec.context_length,
        "checkpoint": spec.model_checkpoint,
        "local_files_only": spec.local_files_only,
    }

    return out, meta

def _write_table(engine: Engine, df: pd.DataFrame, spec: Uni2TSExogSpec) -> None:
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
        df.to_sql(table, engine, schema=schema, if_exists=spec.if_exists, index=False, method="multi", chunksize=10000)

    idx_cols = [c for c in [*spec.group_cols, spec.time_col] if c in df.columns]
    if spec.create_postgres_index and idx_cols:
        idx_name = safe_ident(f"idx_{table}_key_time")
        cols = ", ".join([f'"{safe_ident(c)}"' for c in idx_cols])
        with engine.begin() as conn:
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{schema}"."{table}" ({cols})'))


def run_uni2ts_exog_build(spec: Uni2TSExogSpec) -> dict[str, Any]:
    stage_timing_sec: dict[str, float] = {}
    t0 = time.perf_counter()
    engine = _make_engine(spec)
    codegen_summary = _summarize_uni2ts_codegen(spec.uni2ts_codegen_yaml)
    stage_timing_sec["setup"] = time.perf_counter() - t0

    cfg = ResourcesConfig(
        db_host=spec.host,
        db_port=spec.port,
        db_user=spec.user,
        db_password=spec.password,
        db_name=spec.database,
        schema="resources",
        namespace="uni2ts",
        table_naming="plain",
        app_name="uni2ts_exog_builder",
        env=spec.env,
        profile=spec.profile,
        command=f"resources.uni2ts_exog_pipeline {spec.source_schema}.{spec.source_table} -> {spec.target_schema}.{spec.target_table}",
        tags={
            "source": f"{spec.source_schema}.{spec.source_table}",
            "target": f"{spec.target_schema}.{spec.target_table}",
            "group_cols": list(spec.group_cols),
            "embedding_dim": spec.embedding_dim,
            "context_length": spec.context_length,
            "uni2ts_codegen": codegen_summary,
        },
        enable_gpu=spec.enable_gpu_compute,
        enable_sampling=True,
        sampling_interval_sec=spec.sampling_interval_sec,
        ensure_schema=True,
        parallel_snapshot_workers=max(1, spec.parallel_workers),
    )

    with start_run(cfg) as run:
        run.attach_sqlalchemy_engine(engine)

        with run.span(stage_name="analyze_uni2ts_codegen", extra={"codegen_summary": codegen_summary}):
            pass

        t_extract = time.perf_counter()
        with run.span(stage_name="extract_source"):
            src = _read_source(spec, engine)
        stage_timing_sec["extract"] = time.perf_counter() - t_extract

        t_build = time.perf_counter()
        with run.span(stage_name="build_uni2ts_embeddings", rows_in=int(len(src))):
            out, emb_meta = build_uni2ts_exog_dataframe(src, spec)
        stage_timing_sec["build"] = time.perf_counter() - t_build

        t_write = time.perf_counter()
        with run.span(
            stage_name="write_exog_uni2ts",
            rows_in=int(len(out)),
            rows_out=int(len(out)),
            extra={"embedding_meta": emb_meta},
        ):
            _write_table(engine, out, spec)
        stage_timing_sec["write"] = time.perf_counter() - t_write

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
            "codegen_summary": codegen_summary,
            "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
        }


def _parse_group_cols(raw: str) -> tuple[str, ...]:
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    if not cols:
        raise ValueError("group-cols must not be empty")
    return tuple(cols)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m resources.uni2ts_exog_pipeline")
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

    p.add_argument("--target-schema", default="exog")
    p.add_argument("--target-table", default="uni2ts")
    p.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])

    p.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p.add_argument("--time-col", default="ds")
    p.add_argument("--target-col", default="y")

    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--parallel-workers", type=int, default=4)

    p.add_argument("--model-name", default="uni2ts")
    p.add_argument("--model-version", default="2.0.0")
    p.add_argument("--model-checkpoint", default=None)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p.add_argument("--uni2ts-codegen-yaml", default="./docs/lib_docs/uni2ts_all_codegen.yaml")

    args = p.parse_args()
    spec = Uni2TSExogSpec(
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
        group_cols=_parse_group_cols(args.group_cols),
        time_col=args.time_col,
        target_col=args.target_col,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        parallel_workers=args.parallel_workers,
        model_name=args.model_name,
        model_version=args.model_version,
        model_checkpoint=args.model_checkpoint,
        local_files_only=bool(args.local_files_only),
        enable_gpu_compute=bool(args.enable_gpu_compute),
        sampling_interval_sec=args.sampling_interval_sec,
        uni2ts_codegen_yaml=args.uni2ts_codegen_yaml,
    )

    out = run_uni2ts_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
