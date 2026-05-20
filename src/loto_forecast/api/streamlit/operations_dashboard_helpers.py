from __future__ import annotations

import difflib
import hashlib
import html
import importlib.util
import json
import re
import shlex
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy.engine import Engine

NF_OPTUNA_SEARCH_ALGS = {"RandomSampler", "TPESampler", "CmaEsSampler", "NSGAIISampler"}
NF_RAY_SEARCH_ALGS = {"BasicVariantGenerator", "OptunaSearch", "HyperOptSearch", "BayesOptSearch"}
OPTIONAL_TRAIN_CORE_AXIS_KEYS = {"backend", "loss", "valid_loss", "search_alg"}
OPTIONAL_TRAIN_NONE_TOKEN = "None"
MODEL_MIN_H_RULES: dict[str, int] = {
    "AutoNBEATS": 2,
    "AutoNBEATSx": 2,
    "AutoAutoformer": 2,
    "Autoformer": 2,
    "AutoInformer": 2,
    "AutoFEDformer": 2,
    "AutoVanillaTransformer": 2,
    "AutoTimeMixer": 2,
    "AutoTimesNet": 2,
    "AutoTimeXer": 2,
    "AutoPatchTST": 2,
    "AutoRMoK": 2,
    "AutoStemGNN": 2,
}
HORIZON_AUTO_TOKEN = "(auto)"
HORIZON_AUTO_TOKENS = {HORIZON_AUTO_TOKEN.lower(), "auto", "__auto__"}
SUPPORTED_TEXT_SUFFIXES = {".json", ".csv", ".yaml", ".yml", ".md", ".html", ".htm", ".mmd"}
DIFF_TEXT_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {
    ".py",
    ".sql",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
}
DATASET_INPUT_BACKEND_MAP: dict[str, list[str]] = {
    "db_table": ["pandas", "polars", "dask", "spark", "ray"],
    "db_sql": ["pandas", "polars", "dask", "spark", "ray"],
    "csv": ["pandas", "polars", "dask", "spark", "ray"],
    "parquet": ["pandas", "polars", "dask", "spark", "ray"],
    "json": ["pandas", "polars", "dask", "spark", "ray"],
}
DATAFRAME_BACKEND_OPTIONS = ["pandas", "polars", "dask", "spark", "ray"]


def normalize_optional_train_core_value(value: Any) -> str | None:
    if value is None:
        return None
    sv = str(value).strip()
    if not sv or sv in {OPTIONAL_TRAIN_NONE_TOKEN, "(none)", "None", "null", "NULL"}:
        return None
    return sv


def decode_optional_train_core_choice(axis_key: str, value: Any) -> Any:
    if str(axis_key) in OPTIONAL_TRAIN_CORE_AXIS_KEYS:
        return normalize_optional_train_core_value(value)
    return value


def default_search_alg_for_backend(backend: Any, fallback: Any = None) -> str:
    backend_v = normalize_optional_train_core_value(backend)
    fallback_v = normalize_optional_train_core_value(fallback)
    if backend_v == "optuna":
        if fallback_v in NF_OPTUNA_SEARCH_ALGS:
            return str(fallback_v)
        return "RandomSampler"
    if backend_v == "ray":
        if fallback_v in NF_RAY_SEARCH_ALGS:
            return str(fallback_v)
        return "BasicVariantGenerator"
    return str(fallback_v or "")


def module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def available_dataframe_backends() -> list[str]:
    out: list[str] = ["pandas"]
    if module_exists("polars"):
        out.append("polars")
    if module_exists("dask.dataframe") or module_exists("dask"):
        out.append("dask")
    if module_exists("pyspark"):
        out.append("spark")
    if module_exists("ray") and module_exists("ray.data"):
        out.append("ray")
    return [x for x in DATAFRAME_BACKEND_OPTIONS if x in set(out)]


def dataset_loader_support_df() -> pd.DataFrame:
    def _backends_for(method: str) -> str:
        vals = list(DATASET_INPUT_BACKEND_MAP.get(str(method), []))
        return " / ".join(vals) if vals else "(none)"

    available = set(available_dataframe_backends())

    def _enabled_backends_for(method: str) -> str:
        vals = [b for b in DATASET_INPUT_BACKEND_MAP.get(str(method), []) if b in available]
        return " / ".join(vals) if vals else "(none)"

    rows = [
        {
            "input_method": "db_table",
            "input_desc": "DBテーブル(schema/table)を直接読み込み",
            "supported_backends(理論)": _backends_for("db_table"),
            "enabled_backends(現環境)": _enabled_backends_for("db_table"),
            "note": "dataset_schema / dataset_table / dataset_where を使用（db読込は内部でpandasフォールバック）",
        },
        {
            "input_method": "db_sql",
            "input_desc": "任意SQLで読み込み",
            "supported_backends(理論)": _backends_for("db_sql"),
            "enabled_backends(現環境)": _enabled_backends_for("db_sql"),
            "note": "dataset_sql を使用（db読込は内部でpandasフォールバック）",
        },
        {
            "input_method": "csv",
            "input_desc": "CSVファイル読み込み",
            "supported_backends(理論)": _backends_for("csv"),
            "enabled_backends(現環境)": _enabled_backends_for("csv"),
            "note": "dataset_path を使用",
        },
        {
            "input_method": "parquet",
            "input_desc": "Parquetファイル読み込み",
            "supported_backends(理論)": _backends_for("parquet"),
            "enabled_backends(現環境)": _enabled_backends_for("parquet"),
            "note": "dataset_path を使用",
        },
        {
            "input_method": "json",
            "input_desc": "JSON/JSONLファイル読み込み",
            "supported_backends(理論)": _backends_for("json"),
            "enabled_backends(現環境)": _enabled_backends_for("json"),
            "note": "dataset_path を使用",
        },
    ]
    return pd.DataFrame(rows)


def supported_backends_for_input_method(input_method: str) -> list[str]:
    method = str(input_method or "").strip()
    available = set(available_dataframe_backends())
    vals = [b for b in DATASET_INPUT_BACKEND_MAP.get(method, []) if b in available]
    if not vals:
        return [b for b in DATAFRAME_BACKEND_OPTIONS if b in available] or ["pandas"]
    return vals


def is_supported_backend_for_input_method(input_method: str, dataframe_backend: str) -> bool:
    backend = str(dataframe_backend or "").strip()
    if backend not in DATAFRAME_BACKEND_OPTIONS:
        return False
    allowed = supported_backends_for_input_method(str(input_method))
    return backend in allowed


def safe_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
        raise ValueError(f"unsafe identifier: {name}")
    return f'"{name}"'


def csv_nonempty_list(raw_v: Any) -> list[str]:
    if raw_v is None:
        return []
    if isinstance(raw_v, list):
        return [str(x).strip() for x in raw_v if str(x).strip()]
    sv = str(raw_v).strip()
    if not sv:
        return []
    return [x.strip() for x in sv.split(",") if x.strip()]


def group_mode_unique_id_validation_error(group_mode_v: Any, unique_id_v: Any) -> str | None:
    mode = str(group_mode_v or "").strip()
    if mode != "loto_unique_id_ts_type":
        return None
    if csv_nonempty_list(unique_id_v):
        return None
    return "学習単位 候補=loto_unique_id_ts_type のため unique_id 候補は必須です（None/空文字は不可）。"


def safe_tail(text_value: object, max_lines: int = 30) -> str:
    text = str(text_value or "")
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    return "\n".join(lines[-max_lines:])


def stable_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def query_cache_key(engine: Engine, sql: str, params: dict[str, Any] | None) -> str:
    base = "|".join([str(engine.url), str(sql).strip(), stable_json_dumps(params or {})])
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()


def slug(text_: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text_).strip("_").lower() or "x"


def db_connection_payload(host: str, port: int, user: str, password: str, database: str) -> dict[str, Any]:
    return {
        "host": str(host),
        "port": int(port),
        "user": str(user),
        "password": str(password),
        "database": str(database),
    }


def format_bytes(size: float | int | None) -> str:
    if size is None:
        return "n/a"
    n = float(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    return f"{n:.2f} {units[idx]}"


def nf_lab_ui_state_persistable_key(key: str, prefixes: tuple[str, ...]) -> bool:
    key_v = str(key)
    if not (key_v.startswith("nf_lab_") or any(key_v.startswith(pref) for pref in prefixes)):
        return False
    if key_v.startswith("nf_lab_bottom_combo_build_"):
        return False
    if key_v.startswith("nf_lab_hint_btn_") or "_btn_" in key_v or key_v.endswith("_btn"):
        return False
    if key_v.endswith("_result") or key_v.endswith("_df") or key_v.endswith("_snapshot"):
        return False
    if "_export_" in key_v or key_v.endswith("_export_bash"):
        return False
    if "download" in key_v:
        return False
    if key_v.startswith("nf_lab_run_") or key_v.startswith("nf_lab_apply_") or key_v.startswith("nf_lab_copy_"):
        return False
    if key_v.endswith("_show_manual"):
        return False
    return "_copy_" not in key_v


def nf_lab_ui_state_storage_key(host: str, port: int, user: str, database: str, app_name: str, scope: str) -> str:
    return f"{app_name}:{scope}:{str(host).strip()}:{int(port)}:{str(user).strip()}:{str(database).strip()}"


def read_nf_lab_ui_state_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(loaded, dict) and isinstance(loaded.get("payload"), dict):
        return dict(loaded.get("payload") or {})
    return loaded if isinstance(loaded, dict) else {}


def collect_nf_lab_ui_state_payload(
    session_state: Any,
    *,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_key, value in dict(session_state).items():
        key = str(raw_key)
        if not nf_lab_ui_state_persistable_key(key, prefixes):
            continue
        if isinstance(value, (pd.DataFrame, np.ndarray, set, bytes)):
            continue
        try:
            json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            continue
        payload[key] = value
    return payload


def merge_nf_lab_ui_state_payload(
    file_payload: dict[str, Any],
    db_payload: dict[str, Any],
    *,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    merged_payload: dict[str, Any] = {}
    merged_payload.update(dict(file_payload or {}))
    merged_payload.update(dict(db_payload or {}))
    return {
        str(key): value
        for key, value in merged_payload.items()
        if nf_lab_ui_state_persistable_key(str(key), prefixes)
    }


def normalize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    def _is_na(v: Any) -> bool:
        try:
            return bool(pd.isna(v))
        except Exception:
            return False

    def _fix_scalar(v: Any) -> Any:
        if v is None or _is_na(v):
            return None
        if isinstance(v, uuid.UUID):
            return str(v)
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(v, ensure_ascii=False, default=str)
        if isinstance(v, (bytes, bytearray, memoryview, np.bytes_)):
            b = v if isinstance(v, bytes) else bytes(v)
            try:
                return b.decode("utf-8")
            except Exception:
                return b.hex()
        if isinstance(v, (date, datetime, pd.Timestamp)):
            try:
                return v.isoformat()
            except Exception:
                return str(v)
        return v

    for col in out.columns:
        if out[col].dtype != "object":
            continue
        s = out[col].map(_fix_scalar)
        non_null = [x for x in s.tolist() if x is not None]
        if not non_null:
            out[col] = s
            continue

        types = {type(x) for x in non_null}
        numeric_types = (int, float, bool, np.integer, np.floating, np.bool_)
        if all(issubclass(t, numeric_types) for t in types):
            out[col] = pd.to_numeric(s, errors="coerce")
            continue

        if len(types) > 1:
            out[col] = s.astype("string")
        else:
            out[col] = s

    return out


def parameter_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(getattr(value, "name", "") or "").strip()
    return str(value or "").strip()


def is_valid_search_alg_for_backend(backend: Any, search_alg: Any) -> bool:
    backend_v = normalize_optional_train_core_value(backend)
    search_v = normalize_optional_train_core_value(search_alg)
    if backend_v is None or search_v is None:
        return True
    if backend_v == "optuna":
        return search_v in NF_OPTUNA_SEARCH_ALGS
    if backend_v == "ray":
        return search_v in NF_RAY_SEARCH_ALGS
    return False


def validate_train_combo_choice(model: Any, backend: Any, valid_loss: Any, search_alg: Any) -> str | None:
    model_v = str(model or "").strip()
    backend_v = normalize_optional_train_core_value(backend)
    search_v = normalize_optional_train_core_value(search_alg)
    if model_v == "AutoHINT" and backend_v is not None and backend_v != "ray":
        return "AutoHINT requires backend=ray"
    if not is_valid_search_alg_for_backend(backend_v, search_v):
        return f"invalid search_alg for {backend_v or 'backend'}"
    return None


def parse_horizon_axis_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return max(1, int(value))
    sv = str(value).strip()
    if not sv:
        return None
    if sv.lower() in HORIZON_AUTO_TOKENS:
        return None
    try:
        return max(1, int(sv))
    except Exception:
        try:
            fv = float(sv)
            if float(fv).is_integer():
                return max(1, int(fv))
        except Exception:
            pass
    return None


def resolve_model_horizon(model: str, horizon: int | None) -> tuple[int | None, str | None]:
    model_v = str(model or "").strip()
    if horizon is None:
        return None, None
    h_v = max(1, int(horizon))
    min_h = MODEL_MIN_H_RULES.get(model_v)
    if min_h is not None and int(h_v) < int(min_h):
        return int(min_h), f"{model_v} は h>={int(min_h)} が必要なため h={int(h_v)} を h={int(min_h)} に自動補正しました。"
    return int(h_v), None


def parse_json_like(value: Any) -> Any | None:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw[0] not in "{[":
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None
    return None


def flatten_json_like_value(
    value: Any,
    prefix: str,
    max_depth: int = 3,
    max_list_items: int = 4,
    max_list_len: int = 2000,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _scalar(v: Any) -> bool:
        return isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None

    def _walk(obj: Any, pfx: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{pfx}.{k}" if pfx else str(k)
                if isinstance(v, (dict, list)):
                    _walk(v, key, depth + 1)
                elif _scalar(v):
                    out[key] = v
            return
        if isinstance(obj, list):
            out[f"{pfx}.__len__"] = int(len(obj))
            if not obj:
                return
            if all(_scalar(x) for x in obj):
                for i, x in enumerate(obj[: max(1, int(max_list_items))]):
                    out[f"{pfx}[{i}]"] = x
                if len(obj) > int(max_list_items):
                    out[f"{pfx}.__truncated__"] = True
                if len(obj) <= max_list_len and all(
                    (x is None) or isinstance(x, (int, float, bool, np.integer, np.floating)) for x in obj
                ):
                    arr = np.asarray([np.nan if x is None else float(x) for x in obj], dtype=float)
                    out[f"{pfx}.__mean__"] = float(np.nanmean(arr))
                    out[f"{pfx}.__std__"] = float(np.nanstd(arr))
                    out[f"{pfx}.__min__"] = float(np.nanmin(arr))
                    out[f"{pfx}.__max__"] = float(np.nanmax(arr))
                return
            if all(isinstance(x, dict) for x in obj):
                for i, item in enumerate(obj[: max(1, int(max_list_items))]):
                    _walk(item, f"{pfx}[{i}]", depth + 1)
                if len(obj) > int(max_list_items):
                    out[f"{pfx}.__truncated__"] = True
                return
            out[f"{pfx}.__json__"] = stable_json_dumps(obj)

    _walk(value, prefix, 0)
    return out


def flatten_json_columns(
    df: pd.DataFrame,
    json_cols: dict[str, str],
    max_depth: int = 3,
    max_list_items: int = 4,
    max_new_cols_per_source: int = 120,
    max_list_len: int = 2000,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col, prefix in json_cols.items():
        if col not in out.columns:
            continue
        recs: list[dict[str, Any]] = []
        key_counts: Counter[str] = Counter()
        for v in out[col].tolist():
            obj = parse_json_like(v)
            row = (
                flatten_json_like_value(
                    obj,
                    prefix=prefix,
                    max_depth=int(max_depth),
                    max_list_items=int(max_list_items),
                    max_list_len=int(max_list_len),
                )
                if obj is not None
                else {}
            )
            if row:
                key_counts.update(row.keys())
            recs.append(row)
        if recs and key_counts:
            selected_keys = [k for k, _ in key_counts.most_common(max(1, int(max_new_cols_per_source)))]
            trimmed = [{k: rec.get(k) for k in selected_keys} for rec in recs]
            out = pd.concat([out, pd.DataFrame.from_records(trimmed, index=out.index)], axis=1)
    return out


def expand_semistructured_columns(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    max_depth: int = 3,
    max_list_items: int = 4,
    max_new_cols_per_source: int = 120,
    max_list_len: int = 2000,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = [c for c in (target_cols or list(df.columns)) if c in df.columns]
    selected: list[str] = []
    for col in cols:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        sample = series.head(100).tolist()
        if any(parse_json_like(v) is not None for v in sample):
            selected.append(col)
    if not selected:
        return df.copy()
    return flatten_json_columns(
        df,
        json_cols={c: c for c in selected},
        max_depth=int(max_depth),
        max_list_items=int(max_list_items),
        max_new_cols_per_source=int(max_new_cols_per_source),
        max_list_len=int(max_list_len),
    )


def bayesian_success_posterior(df: pd.DataFrame, group_col: str = "model_name") -> pd.DataFrame:
    if df.empty or "status" not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for g, gdf in df.groupby(group_col):
        s = int((gdf["status"].astype(str) == "success").sum())
        f = int((gdf["status"].astype(str) == "failed").sum())
        n = s + f
        if n == 0:
            continue
        a = 1 + s
        b = 1 + f
        mean = a / float(a + b)
        var = (a * b) / float(((a + b) ** 2) * (a + b + 1))
        sd = float(np.sqrt(max(0.0, var)))
        rows.append(
            {
                group_col: str(g),
                "success": s,
                "failed": f,
                "n": n,
                "posterior_mean": float(mean),
                "posterior_ci_low_95": float(max(0.0, mean - 1.96 * sd)),
                "posterior_ci_high_95": float(min(1.0, mean + 1.96 * sd)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("posterior_mean", ascending=False).reset_index(drop=True)
    return out


def impact_contribution_rates(impact_df: pd.DataFrame) -> pd.DataFrame:
    if impact_df is None or impact_df.empty or "effect_abs" not in impact_df.columns:
        return pd.DataFrame()
    tmp = impact_df.copy()
    tmp["effect_abs"] = pd.to_numeric(tmp["effect_abs"], errors="coerce").fillna(0.0)
    den = float(tmp["effect_abs"].sum())
    if den <= 0:
        tmp["contribution_rate"] = 0.0
    else:
        tmp["contribution_rate"] = tmp["effect_abs"] / den
    return tmp.sort_values("contribution_rate", ascending=False).reset_index(drop=True)


def r2_for_features(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> float:
    if not feature_cols:
        return 0.0
    use = [c for c in feature_cols if c in df.columns]
    if not use or target_col not in df.columns:
        return 0.0
    tmp = df[use + [target_col]].copy()
    y = pd.to_numeric(tmp[target_col], errors="coerce")
    X = pd.get_dummies(tmp[use], drop_first=True)
    if X.empty:
        return 0.0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = y.astype(float)
    valid = y.notna()
    if int(valid.sum()) < max(8, X.shape[1] + 2):
        return 0.0
    Xv = X.loc[valid].to_numpy(dtype=float)
    yv = y.loc[valid].to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(Xv.shape[0]), Xv])
    try:
        beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    except Exception:
        return 0.0
    pred = Xd @ beta
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_res = float(np.sum((yv - pred) ** 2))
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


def approx_shapley_contrib(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    n_perm: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    cols = [c for c in candidate_cols if c in df.columns][:8]
    if target_col not in df.columns or len(cols) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    contrib = {c: 0.0 for c in cols}
    for _ in range(max(10, int(n_perm))):
        order = cols.copy()
        rng.shuffle(order)
        curr: list[str] = []
        prev_r2 = 0.0
        for c in order:
            curr.append(c)
            r2 = r2_for_features(df, target_col=target_col, feature_cols=curr)
            contrib[c] += max(0.0, r2 - prev_r2)
            prev_r2 = r2
    den = float(sum(contrib.values()))
    rows = [
        {
            "feature": c,
            "shapley_r2_contrib": float(v / max(1.0, n_perm)),
            "shapley_rate": float(v / den) if den > 0 else 0.0,
        }
        for c, v in contrib.items()
    ]
    return pd.DataFrame(rows).sort_values("shapley_rate", ascending=False).reset_index(drop=True)


def graph_centrality_from_impact(impact_df: pd.DataFrame) -> pd.DataFrame:
    if impact_df is None or impact_df.empty or "parameter" not in impact_df.columns:
        return pd.DataFrame()
    tmp = impact_df.copy()
    tmp["effect_abs"] = pd.to_numeric(tmp.get("effect_abs"), errors="coerce").fillna(0.0)
    tmp["is_significant"] = tmp.get("is_significant", False).astype(bool)
    total = max(1, int(tmp.shape[0]))
    rows = [
        {
            "node": str(r.parameter),
            "weighted_degree": float(getattr(r, "effect_abs", 0.0)),
            "degree_centrality": float(1.0 / total),
            "bridge_score": float(getattr(r, "effect_abs", 0.0))
            * (1.2 if bool(getattr(r, "is_significant", False)) else 1.0),
        }
        for r in tmp.itertuples(index=False)
    ]
    return pd.DataFrame(rows).sort_values("bridge_score", ascending=False).reset_index(drop=True)


def causal_proxy_ate(df: pd.DataFrame, target_col: str, treatment_col: str) -> dict[str, Any]:
    if target_col not in df.columns or treatment_col not in df.columns:
        return {"ok": False, "reason": "missing_cols"}
    tmp = df[[target_col, treatment_col, "model_name", "horizon", "dataset_rows", "feature_cols"]].copy()
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    t_raw = tmp[treatment_col]
    if pd.api.types.is_numeric_dtype(t_raw):
        med = pd.to_numeric(t_raw, errors="coerce").median()
        tmp["_treat"] = (pd.to_numeric(t_raw, errors="coerce") >= med).astype(float)
    else:
        top = t_raw.astype(str).value_counts().index.tolist()
        if not top:
            return {"ok": False, "reason": "no_treatment_values"}
        tmp["_treat"] = (t_raw.astype(str) == top[0]).astype(float)
    tmp = tmp.dropna(subset=[target_col])
    if int(tmp.shape[0]) < 20 or tmp["_treat"].nunique() < 2:
        return {"ok": False, "reason": "insufficient_data"}
    X = tmp[["_treat", "model_name", "horizon", "dataset_rows", "feature_cols"]].copy()
    X = pd.get_dummies(X, columns=["model_name"], drop_first=True)
    for c in ["horizon", "dataset_rows", "feature_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = tmp[target_col].astype(float)
    Xd = np.column_stack([np.ones(X.shape[0]), X.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(Xd, y.to_numpy(dtype=float), rcond=None)
    pred = Xd @ beta
    resid = y.to_numpy(dtype=float) - pred
    dof = max(1, Xd.shape[0] - Xd.shape[1])
    sigma2 = float(np.sum(resid**2) / dof)
    try:
        cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
        se = float(np.sqrt(max(1e-12, cov[1, 1])))
    except Exception:
        se = float("nan")
    coef = float(beta[1])
    z = coef / se if np.isfinite(se) and se > 0 else np.nan
    return {
        "ok": True,
        "n": int(tmp.shape[0]),
        "treatment_col": str(treatment_col),
        "ate_proxy": coef,
        "std_error": se,
        "z_score": float(z) if np.isfinite(z) else None,
        "ci_low_95": float(coef - 1.96 * se) if np.isfinite(se) else None,
        "ci_high_95": float(coef + 1.96 * se) if np.isfinite(se) else None,
    }


def dump_selector(mode: str, schemas: list[str], tables: list[tuple[str, str]]) -> dict[str, Any]:
    mode_v = str(mode)
    if mode_v == "tables":
        table_refs = sorted({f"{s}.{t}" for s, t in tables})
        return {"mode": "tables", "schemas": [], "tables": table_refs}
    schema_refs = sorted({str(s) for s in schemas if str(s)})
    return {"mode": "schemas", "schemas": schema_refs, "tables": []}


def dump_selector_flags(selector: dict[str, Any]) -> str:
    mode = str(selector.get("mode", "schemas"))
    if mode == "tables":
        return " ".join([f"--table {shlex.quote(str(t))}" for t in selector.get("tables", [])])
    return " ".join([f"--schema {shlex.quote(str(s))}" for s in selector.get("schemas", [])])


def safe_read_json_file(path: Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return {}
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def artifact_file_stats(root: Path) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    ext_counts: dict[str, int] = {}
    total_bytes = 0
    if not base.exists() or not base.is_dir():
        return {"path": str(base), "exists": False, "file_count": 0, "total_bytes": 0, "ext_counts": {}, "files": []}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        size_b = int(p.stat().st_size)
        total_bytes += size_b
        ext = p.suffix.lower() or "<no_ext>"
        ext_counts[ext] = int(ext_counts.get(ext, 0)) + 1
        files.append({"path": str(p.relative_to(base)), "size_bytes": size_b, "size_human": format_bytes(size_b)})
    return {
        "path": str(base),
        "exists": True,
        "file_count": int(len(files)),
        "total_bytes": int(total_bytes),
        "ext_counts": ext_counts,
        "files": files,
    }


def has_model_artifacts(run_dir: Path) -> bool:
    base = Path(run_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return False
    has_cfg = (base / "configuration.pkl").exists()
    has_alias = (base / "alias_to_model.pkl").exists()
    has_checkpoint = any([any(base.glob("*.ckpt")), any(base.glob("*.pth")), any(base.glob("*.pt"))])
    if has_cfg and has_alias:
        return True
    return has_checkpoint


def has_analysis_bundle(run_dir: Path) -> bool:
    base = Path(run_dir).expanduser().resolve()
    if not has_model_artifacts(base):
        return False
    return bool((base / "forecast.parquet").exists() and (base / "evaluation.json").exists())


def read_text_file(path: Path, max_chars: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def scan_supported_files(root: Path, max_files: int = 2000) -> list[Path]:
    out: list[Path] = []
    if not root.exists() or not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if p.is_file() and p.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
            out.append(p)
    return out


def scan_diff_files(root: Path, max_files: int = 2500) -> list[Path]:
    out: list[Path] = []
    if not root.exists() or not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if p.is_file() and p.suffix.lower() in DIFF_TEXT_SUFFIXES:
            out.append(p)
    return out


def unified_diff_text(
    left_text: str,
    right_text: str,
    left_name: str,
    right_name: str,
    context_lines: int = 3,
    ignore_ws: bool = False,
) -> str:
    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    if ignore_ws:
        left_lines = [ln.rstrip() for ln in left_lines]
        right_lines = [ln.rstrip() for ln in right_lines]
    diff = difflib.unified_diff(left_lines, right_lines, fromfile=left_name, tofile=right_name, lineterm="", n=int(context_lines))
    return "\n".join(diff)


def summarize_supported_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    size = path.stat().st_size if path.exists() else 0
    result: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "size_bytes": int(size),
        "size_human": format_bytes(size),
        "preview": "",
        "meta": {},
    }
    if suffix == ".csv":
        try:
            df = pd.read_csv(path, nrows=300)
            result["meta"] = {"rows_preview": int(len(df)), "columns": df.columns.tolist()}
            result["preview"] = df.head(20).to_csv(index=False)
            return result
        except Exception as e:
            result["meta"] = {"error": str(e)}
            result["preview"] = read_text_file(path, 4000)
            return result

    text_content = read_text_file(path, 120_000)
    result["preview"] = text_content[:8000]
    if suffix == ".json":
        try:
            obj = json.loads(text_content)
            if isinstance(obj, dict):
                result["meta"] = {"top_keys": list(obj.keys())[:20], "key_count": len(obj)}
            elif isinstance(obj, list):
                result["meta"] = {"items": len(obj)}
        except Exception as e:
            result["meta"] = {"error": str(e)}
    elif suffix in {".yaml", ".yml"}:
        try:
            obj = yaml.safe_load(text_content)
            if isinstance(obj, dict):
                result["meta"] = {"top_keys": list(obj.keys())[:20], "key_count": len(obj)}
            elif isinstance(obj, list):
                result["meta"] = {"items": len(obj)}
        except Exception as e:
            result["meta"] = {"error": str(e)}
    elif suffix in {".md", ".html", ".htm", ".mmd"}:
        result["meta"] = {"chars": len(text_content)}
    return result


def compile_directory_payload(root: Path, files: list[Path]) -> dict[str, Any]:
    summaries = [summarize_supported_file(p) for p in files]
    suffix_counts = Counter([p.suffix.lower() for p in files])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "files": summaries,
    }


def scan_markdown_files(roots: list[Path], max_files: int = 2000) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in sorted(root.rglob("*.md")):
            if len(out) >= max_files:
                return out
            if p.is_file():
                out.append(p)
    return out


def compile_markdown_bundle(roots: list[Path], files: list[Path], max_chars_per_file: int = 120_000) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    toc_lines = ["# Compiled Markdown Documents", ""]
    body_lines: list[str] = []
    for i, p in enumerate(files, start=1):
        text_value = read_text_file(p, max_chars=max_chars_per_file)
        rel = str(p)
        for root in roots:
            try:
                rel = str(p.relative_to(root))
                break
            except Exception:
                continue
        toc_lines.extend([f"{i}. `{rel}`", ""])
        body_lines.extend([f"## [{i}] {rel}", "", text_value, ""])
        items.append(
            {"index": i, "path": str(p), "relative_path": rel, "chars": len(text_value), "lines": len(text_value.splitlines()), "preview": text_value[:3000]}
        )
    compiled_md = "\n".join(toc_lines + ["---", ""] + body_lines)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(r) for r in roots],
        "file_count": len(files),
        "documents": items,
        "compiled_markdown": compiled_md,
    }


def compiled_to_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Directory Compile Report",
        "",
        f"- root: {bundle.get('root', '')}",
        f"- generated_at: {bundle.get('generated_at', '')}",
        f"- file_count: {bundle.get('file_count', 0)}",
        "",
        "## Suffix Counts",
        "",
        "| suffix | count |",
        "|---|---:|",
    ]
    for k, v in bundle.get("suffix_counts", {}).items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Files"])
    for f in bundle.get("files", []):
        lines.append(f"### {f.get('path', '')}")
        lines.append(f"- suffix: {f.get('suffix', '')} / size: {f.get('size_human', '')}")
        meta = f.get("meta", {})
        if meta:
            lines.append(f"- meta: `{json.dumps(meta, ensure_ascii=False)}`")
        lines.append("")
        preview = str(f.get("preview", ""))
        if preview:
            lines.extend(["```text", preview[:1000], "```", ""])
    return "\n".join(lines)


def compiled_to_html(bundle: dict[str, Any]) -> str:
    rows = [{"path": f.get("path", ""), "suffix": f.get("suffix", ""), "size_human": f.get("size_human", ""), "meta": json.dumps(f.get("meta", {}), ensure_ascii=False)} for f in bundle.get("files", [])]
    df = pd.DataFrame(rows)
    table_html = df.to_html(index=False, escape=True)
    root = html.escape(str(bundle.get("root", "")))
    generated = html.escape(str(bundle.get("generated_at", "")))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Directory Compile Report</title>
<style>
body{{font-family:Arial,sans-serif;padding:16px;}}
table{{border-collapse:collapse;width:100%;font-size:13px;}}
th,td{{border:1px solid #ddd;padding:6px;}}
th{{background:#f5f5f5;}}
</style></head>
<body>
<h1>Directory Compile Report</h1>
<p>root: {root}</p>
<p>generated_at: {generated}</p>
{table_html}
</body></html>"""


def compiled_to_format(bundle: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(bundle, ensure_ascii=False, indent=2), "application/json", "json"
    if fmt == "yaml":
        return yaml.safe_dump(bundle, allow_unicode=True, sort_keys=False), "application/x-yaml", "yaml"
    if fmt == "csv":
        rows = [
            {
                "path": f.get("path", ""),
                "suffix": f.get("suffix", ""),
                "size_bytes": f.get("size_bytes", 0),
                "size_human": f.get("size_human", ""),
                "meta": json.dumps(f.get("meta", {}), ensure_ascii=False),
            }
            for f in bundle.get("files", [])
        ]
        return pd.DataFrame(rows).to_csv(index=False), "text/csv", "csv"
    if fmt == "md":
        return compiled_to_markdown(bundle), "text/markdown", "md"
    if fmt == "html":
        return compiled_to_html(bundle), "text/html", "html"
    raise ValueError(f"unsupported format: {fmt}")


def module_name_from_path(path: Path, root: Path, prefix: str) -> str:
    rel = path.relative_to(root).with_suffix("")
    return f"{prefix}.{'.'.join(rel.parts)}" if rel.parts else prefix


def resolve_from_import(base_module: str, level: int, module_name: str | None) -> str:
    parts = base_module.split(".")
    parent = parts[:-1]
    if level > 0:
        parent = parent[: max(0, len(parent) - level + 1)]
    mod_parts = module_name.split(".") if module_name else []
    merged = parent + mod_parts
    return ".".join([p for p in merged if p])
