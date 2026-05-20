from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shlex
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _stable_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _slug(text_: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text_ or "")).strip("_").lower() or "x"


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None


def _query_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _table_exists(engine: Engine, schema: str, table: str) -> bool:
    try:
        df = _query_df(
            engine,
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_name = :table
            LIMIT 1
            """,
            {"schema": str(schema), "table": str(table)},
        )
        return not df.empty
    except Exception:
        return False


def build_train_combo_signature(model_name: str, horizon: int, params_obj: dict[str, Any]) -> str:
    payload = {
        "model_name": str(model_name or "").strip(),
        "horizon": max(1, int(horizon or 1)),
        "params_json": dict(params_obj or {}),
    }
    raw = _stable_json_dumps(payload)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def make_param_based_run_id(model_name: str, horizon: int, params_obj: dict[str, Any]) -> str:
    p = dict(params_obj or {})
    parts = [
        str(model_name or ""),
        str(p.get("backend", "")),
        str(p.get("loss_name", "")),
        str(p.get("valid_loss_name", "(none)") if p.get("valid_loss_name") is not None else "(none)"),
        str(p.get("search_alg_name", "")),
        str(p.get("dataset_schema", "")),
        str(p.get("dataset_table", "")),
        str(p.get("group_by_mode", "")),
        str(p.get("target_loto", "all") or "all"),
        str(p.get("target_unique_id", "all") or "all"),
        str(p.get("target_ts_type", "all") or "all"),
        f"h{int(max(1, int(horizon or 1)))}",
    ]
    head = "_".join([_slug(x) for x in parts if str(x).strip()])
    head = re.sub(r"_+", "_", head).strip("_")[:84] or "combo"
    sig = build_train_combo_signature(model_name, horizon, p)[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{head}_{ts}_{sig}"


def _extract_signature_prefix_from_run_id(run_id: str) -> str | None:
    raw = str(run_id or "").strip().lower()
    if not raw:
        return None
    m = re.search(r"_([0-9a-f]{10,40})$", raw)
    if not m:
        return None
    return m.group(1)[:10]


def _load_completed_prefixes_from_log(engine: Engine) -> set[str]:
    if not _table_exists(engine, "log", "run_history"):
        return set()
    has_error_event = _table_exists(engine, "log", "error_event")
    sql = """
        WITH latest AS (
          SELECT DISTINCT ON (run_id)
            run_id, status
          FROM log.run_history
          WHERE NULLIF(TRIM(COALESCE(run_id, '')), '') IS NOT NULL
          ORDER BY run_id, event_ts DESC
        )
        SELECT l.run_id
        FROM latest l
        WHERE LOWER(COALESCE(l.status, '')) = 'success'
    """
    if has_error_event:
        sql += """
          AND NOT EXISTS (
            SELECT 1
            FROM log.error_event e
            WHERE e.run_id = l.run_id
          )
        """
    try:
        df = _query_df(engine, sql)
    except Exception:
        return set()
    out: set[str] = set()
    if df.empty:
        return out
    for run_id in df.get("run_id", pd.Series(dtype=str)).astype(str).tolist():
        prefix = _extract_signature_prefix_from_run_id(run_id)
        if prefix:
            out.add(prefix)
    return out


def load_completed_combo_index(engine: Engine | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"signatures": set(), "prefixes": set()}
    if engine is None:
        return out
    if _table_exists(engine, "model", "nf_automodel"):
        sql_primary = """
            SELECT run_id, model_name, horizon, params_json, diagnostics_json
            FROM model.nf_automodel
            WHERE LOWER(COALESCE(status, '')) = 'success'
              AND NULLIF(TRIM(COALESCE(error_message, '')), '') IS NULL
        """
        sql_fallback = """
            SELECT run_id, model_name, horizon, params_json, diagnostics_json
            FROM model.nf_automodel
            WHERE LOWER(COALESCE(status, '')) = 'success'
        """
        try:
            df = _query_df(engine, sql_primary)
        except Exception:
            try:
                df = _query_df(engine, sql_fallback)
            except Exception:
                df = pd.DataFrame()
        if not df.empty:
            for row in df.to_dict(orient="records"):
                diag_obj = _parse_json_like(row.get("diagnostics_json"))
                if isinstance(diag_obj, dict):
                    sig = str(diag_obj.get("combo_signature", "")).strip().lower()
                    if sig:
                        out["signatures"].add(sig)
                params_obj = _parse_json_like(row.get("params_json"))
                if isinstance(params_obj, dict):
                    try:
                        sig_fallback = build_train_combo_signature(
                            str(row.get("model_name", "")),
                            int(row.get("horizon", 1) or 1),
                            params_obj,
                        )
                        out["signatures"].add(sig_fallback.lower())
                    except Exception:
                        pass
                run_id_v = str(row.get("run_id", "")).strip()
                run_id_prefix = _extract_signature_prefix_from_run_id(run_id_v)
                if run_id_prefix:
                    out["prefixes"].add(run_id_prefix)
    out["prefixes"].update(_load_completed_prefixes_from_log(engine))
    return out


def is_combo_signature_completed(combo_signature: str, completed_index: dict[str, set[str]] | None) -> bool:
    if not completed_index:
        return False
    sig = str(combo_signature or "").strip().lower()
    if not sig:
        return False
    signatures = completed_index.get("signatures", set())
    prefixes = completed_index.get("prefixes", set())
    return bool((sig in signatures) or (len(sig) >= 10 and sig[:10] in prefixes))


def _build_script_lines(commands: Sequence[str], cwd: Path, stop_on_error: bool) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(cwd.resolve()))}",
        "",
    ]
    if stop_on_error:
        lines.extend([str(c) for c in commands])
        return lines
    lines.append("set +e")
    total = len(commands)
    for i, one in enumerate(commands, start=1):
        lines.append(f'echo "[{i}/{total}] start"')
        lines.append(str(one))
        lines.append('rc=$?; if [ "$rc" -ne 0 ]; then echo "[WARN] command failed rc=$rc"; fi')
        lines.append("")
    return lines


def write_bash_script(
    commands: Sequence[str],
    *,
    out_dir: Path,
    cwd: Path,
    stop_on_error: bool,
    file_stem: str = "nf_combo_batch",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = out_dir / f"{file_stem}_{ts}.sh"
    lines = _build_script_lines(commands, cwd=cwd, stop_on_error=bool(stop_on_error))
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with contextlib.suppress(Exception):
        os.chmod(script_path, 0o755)
    return script_path


def _split_commands(commands: Sequence[str], split_count: int) -> list[list[str]]:
    seq = [str(c) for c in commands if str(c).strip()]
    if not seq:
        return []
    parts = max(1, min(50, int(split_count or 1), len(seq)))
    base = len(seq) // parts
    rem = len(seq) % parts
    out: list[list[str]] = []
    start = 0
    for i in range(parts):
        sz = base + (1 if i < rem else 0)
        if sz <= 0:
            continue
        out.append(seq[start : start + sz])
        start += sz
    return out


def write_split_bash_scripts(
    commands: Sequence[str],
    *,
    split_count: int,
    out_dir: Path,
    cwd: Path,
    stop_on_error: bool,
    file_stem: str = "nf_combo_batch",
) -> list[Path]:
    chunks = _split_commands(commands, split_count)
    if not chunks:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(chunks)
    out_paths: list[Path] = []
    for idx, chunk in enumerate(chunks, start=1):
        part_path = out_dir / f"{file_stem}_{ts}_part{idx:02d}_of_{total:02d}.sh"
        lines = _build_script_lines(chunk, cwd=cwd, stop_on_error=bool(stop_on_error))
        part_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with contextlib.suppress(Exception):
            os.chmod(part_path, 0o755)
        out_paths.append(part_path)
    return out_paths


def build_split_tab_launch_command(split_paths: Sequence[Path | str]) -> str:
    paths = [str(Path(p).resolve()) for p in split_paths if str(p).strip()]
    if not paths:
        return ""
    first = paths[0]
    glob_pattern = re.sub(r"_part\d+_of_\d+\.sh$", "_part*.sh", first)
    if glob_pattern == first:
        parent = str(Path(first).parent.resolve())
        glob_pattern = str(Path(parent) / "*.sh")
    return "\n".join(
        [
            f"for file in {glob_pattern}; do",
            '    gnome-terminal --tab -- bash -c "bash \\"$file\\"; exec bash"',
            "done",
        ]
    )


def write_split_tab_launcher_script(
    split_paths: Sequence[Path | str],
    *,
    out_dir: Path,
    file_stem: str = "nf_combo_batch_launch_tabs",
) -> Path | None:
    cmd = build_split_tab_launch_command(split_paths)
    if not cmd.strip():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_path = out_dir / f"{file_stem}_{ts}.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        cmd,
        "",
    ]
    launcher_path.write_text("\n".join(lines), encoding="utf-8")
    with contextlib.suppress(Exception):
        os.chmod(launcher_path, 0o755)
    return launcher_path
