from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import html
import inspect
import itertools
import json
import os
import re
import selectors
import shlex
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

from loto_forecast.observability import (
    OBSERVABILITY_ROOT,
    build_observability_snapshot,
    load_recent_events,
    record_event,
    summarize_observability,
)

STREAMLIT_DIR = Path(__file__).resolve().parent
API_ROOT = STREAMLIT_DIR.parent
PACKAGE_ROOT = API_ROOT.parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

for _path in (STREAMLIT_DIR, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dashboard_arg_utils import command_argument_table as _build_command_argument_table  # noqa: E402
from dashboard_db_admin_panel import render_db_admin_panel  # noqa: E402
from nf_lab_cockpit import render_neuralforecast_cockpit  # noqa: E402
from dashboard_nf_resource_analytics_panel import render_nf_resource_analytics_panel  # noqa: E402
from dashboard_nf_runid_panel import render_runid_integrated_panel  # noqa: E402
from dashboard_train_combo_utils import (  # noqa: E402
    build_split_tab_launch_command,
    build_train_combo_signature,
    is_combo_signature_completed,
    load_completed_combo_index,
    make_param_based_run_id,
    write_bash_script,
    write_split_bash_scripts,
    write_split_tab_launcher_script,
)
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from scipy import stats as spstats

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from loto_forecast.api.streamlit import operations_dashboard_helpers as dashboard_helpers  # noqa: E402
from loto_forecast.api.streamlit.ui.notifications import (  # noqa: E402
    StreamlitNotificationAdapter,
    drain_notifications,
    render_notification_center,
)
from loto_forecast.api.streamlit.ui.presets import (  # noqa: E402
    apply_pending_nf_preset,
    available_nf_presets,
    consume_active_nf_preset,
    queue_nf_preset,
)
from loto_forecast.api.streamlit.ui.wizard import build_nf_wizard_state  # noqa: E402
from loto_forecast.application.nf_combo_engine import ComboContext, evaluate_train_combinations  # noqa: E402
from loto_forecast.application.notification_events import (  # noqa: E402
    NotificationChannel,
    NotificationEventKind,
    NotificationSeverity,
    build_notification_event,
)
from loto_forecast.application.notification_service import (  # noqa: E402
    NotificationService,
    NotificationServiceConfig,
)
from loto_forecast.config.settings import settings  # noqa: E402
from loto_forecast.infrastructure.notifications.beep_notifier import BeepNotifier  # noqa: E402
from loto_forecast.infrastructure.notifications.email_notifier import EmailNotifier  # noqa: E402
from loto_forecast.infrastructure.notifications.mock_message_notifier import MockMessageNotifier  # noqa: E402
from resources.utils import detect_execution_environment  # noqa: E402

SCHEMAS = ("dataset", "exog", "resources", "meta", "model")
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

EXTERNAL_TARGETS = {
    "trend": Path("/mnt/e/env/ts/lib_ana/src/trend"),
    "timesfm": Path("/mnt/e/env/ts/lib_ana/src/model/timesfm"),
}

DASHBOARD_LOG_DIR = PROJECT_ROOT / "logs" / "dashboard"
NF_LAB_UI_STATE_PATH = PROJECT_ROOT / "artifacts" / "ui_state" / "nf_lab_ui_state.json"
NF_LAB_UI_STATE_DB_TABLE = "log.ui_state_snapshot"
NF_LAB_UI_STATE_APP_NAME = "operations_dashboard"
NF_LAB_UI_STATE_SCOPE = "nf_lab_fixed_grid"
NF_LAB_UI_STATE_PREFIXES = (
    "nf_lab_train_",
    "nf_lab_axis_fixed_",
    "nf_lab_axis_pool_",
    "nf_lab_axis_bool_",
    "nf_lab_axis_extra_",
    "nf_lab_combo_expand_",
    "nf_lab_combo_candidates_",
    "nf_lab_bottom_combo_",
    "nf_lab_meta_",
    "nf_lab_hint_open_",
)
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
DATASET_INPUT_METHOD_OPTIONS = ["db_table", "db_sql", "csv", "parquet", "json"]
DATAFRAME_BACKEND_OPTIONS = ["pandas", "polars", "dask", "spark", "ray"]
DATASET_INPUT_BACKEND_MAP: dict[str, list[str]] = {
    "db_table": ["pandas", "polars", "dask", "spark", "ray"],
    "db_sql": ["pandas", "polars", "dask", "spark", "ray"],
    "csv": ["pandas", "polars", "dask", "spark", "ray"],
    "parquet": ["pandas", "polars", "dask", "spark", "ray"],
    "json": ["pandas", "polars", "dask", "spark", "ray"],
}


_normalize_optional_train_core_value = dashboard_helpers.normalize_optional_train_core_value
_decode_optional_train_core_choice = dashboard_helpers.decode_optional_train_core_choice
_default_search_alg_for_backend = dashboard_helpers.default_search_alg_for_backend
_module_exists = dashboard_helpers.module_exists
_available_dataframe_backends = dashboard_helpers.available_dataframe_backends


_QUERY_CACHE: dict[str, dict[str, Any]] = {}
_QUERY_CACHE_LOCK = threading.Lock()
_QUERY_CACHE_STATS: dict[str, Any] = {
    "hits": 0,
    "misses": 0,
    "queries": 0,
    "slow_queries": 0,
    "last_query_ms": 0.0,
}

warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning,
)

STATUS_COLOR_MAP: dict[str, str] = {
    "ready": "#0f766e",
    "pending": "#d97706",
    "success": "#0f766e",
    "failed": "#b91c1c",
    "running": "#2563eb",
    "unknown": "#64748b",
}
NF_LAB_STEP_STATUS_ORDER = ["ready", "pending", "unknown"]
PASSWORD_ENV_VAR_NAME = "DB_PASSWORD"
APP_META_DESCRIPTION = (
    "ロト予測の運用・実行結果・分析・検定・可視化・成果物管理を統合確認する "
    "Streamlit ダッシュボード。"
)
NOTIFICATION_AUDIT_LOG_PATH = PROJECT_ROOT / "artifacts" / "logs" / "notification_audit.jsonl"
TREE_SKIP_NAMES = {
    ".git",
    ".claude",
    ".codex",
    ".agents",
    ".backup",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
}
TREE_SKIP_TOP_LEVEL_NAMES = {
    "artifacts",
    "build",
    "htmlcov",
    "logs",
}
_NOTIFICATION_SERVICE: NotificationService | None = None


def _get_notification_service() -> NotificationService:
    global _NOTIFICATION_SERVICE
    if _NOTIFICATION_SERVICE is None:
        _NOTIFICATION_SERVICE = NotificationService(
            [
                StreamlitNotificationAdapter(),
                BeepNotifier(),
                EmailNotifier(),
                MockMessageNotifier(),
            ],
            config=NotificationServiceConfig(
                dedup_window_sec=30.0,
                rate_limit_sec=1.0,
                audit_log_path=NOTIFICATION_AUDIT_LOG_PATH,
            ),
        )
    return _NOTIFICATION_SERVICE


def _sanitize_nf_train_widget_state(
    session_state: Any,
    *,
    model_choices: Sequence[str],
    backend_options: Sequence[str],
    loss_options: Sequence[str],
    search_options: Sequence[str],
    dataset_input_method_options: Sequence[str],
    dataframe_backend_options: Sequence[str],
    dataset_schema_options: Sequence[str] | None = None,
    dataset_table_options: Sequence[str] | None = None,
    default_dataset_schema: str = "",
    default_dataset_table: str = "",
) -> None:
    if model_choices:
        model_value = str(session_state.get("nf_lab_train_model", "") or "")
        if model_value not in set(str(item) for item in model_choices):
            session_state["nf_lab_train_model"] = str(model_choices[0])

    if backend_options:
        backend_value = str(session_state.get("nf_lab_train_backend", "") or "")
        if backend_value not in set(str(item) for item in backend_options):
            session_state["nf_lab_train_backend"] = str(backend_options[0])

    if loss_options:
        loss_value = str(session_state.get("nf_lab_train_loss_name", "") or "")
        if loss_value not in set(str(item) for item in loss_options):
            session_state["nf_lab_train_loss_name"] = str(loss_options[0])

    if search_options:
        search_value = str(session_state.get("nf_lab_train_search_alg_choice", "") or "")
        if search_value not in set(str(item) for item in search_options):
            session_state["nf_lab_train_search_alg_choice"] = str(search_options[0])

    if dataset_input_method_options:
        input_method_value = str(session_state.get("nf_lab_train_dataset_input_method", "") or "")
        if input_method_value not in set(str(item) for item in dataset_input_method_options):
            session_state["nf_lab_train_dataset_input_method"] = str(dataset_input_method_options[0])

    if dataframe_backend_options:
        backend_value = str(session_state.get("nf_lab_train_dataframe_backend", "") or "")
        if backend_value not in set(str(item) for item in dataframe_backend_options):
            session_state["nf_lab_train_dataframe_backend"] = str(dataframe_backend_options[0])

    if dataset_schema_options:
        schema_value = str(session_state.get("nf_lab_train_dataset_schema", "") or "")
        allowed_schemas = [str(item) for item in dataset_schema_options]
        fallback_schema = str(default_dataset_schema or allowed_schemas[0])
        if schema_value not in set(allowed_schemas):
            session_state["nf_lab_train_dataset_schema"] = fallback_schema

    if dataset_table_options:
        table_value = str(session_state.get("nf_lab_train_dataset_table", "") or "")
        allowed_tables = [str(item) for item in dataset_table_options]
        fallback_table = str(default_dataset_table or allowed_tables[0])
        if table_value not in set(allowed_tables):
            session_state["nf_lab_train_dataset_table"] = fallback_table


def _build_combo_reason_rows(combo_eval: Any) -> list[dict[str, Any]]:
    reason_rows = getattr(combo_eval, "reason_rows", [])
    rows: list[dict[str, Any]] = []
    if isinstance(reason_rows, list) and reason_rows:
        for item in reason_rows:
            rows.append(
                {
                    "reason_code": str(getattr(item, "reason_code", "")),
                    "reason": str(getattr(item, "reason_ja", "")),
                    "count": int(getattr(item, "count", 0) or 0),
                }
            )
    elif hasattr(combo_eval, "reason_summary") and hasattr(combo_eval, "excluded_combinations"):
        reason_summary = getattr(combo_eval, "reason_summary", {})
        excluded_combinations = getattr(combo_eval, "excluded_combinations", [])
        if isinstance(reason_summary, dict):
            for reason_code, count in sorted(reason_summary.items(), key=lambda item: (-int(item[1]), item[0])):
                sample = next(
                    (
                        str(getattr(item, "reason_ja", reason_code))
                        for item in excluded_combinations
                        if str(getattr(item, "reason_code", "")) == str(reason_code)
                    ),
                    str(reason_code),
                )
                rows.append({"reason_code": str(reason_code), "reason": sample, "count": int(count)})
    return rows


def _render_zero_combo_diagnostics(combo_eval: Any, *, label: str) -> None:
    reason_rows = _build_combo_reason_rows(combo_eval)
    st.warning(f"{label}: 有効な組合せが 0 件です。")
    if reason_rows:
        top_reason = reason_rows[0]
        st.markdown(f"原因: {top_reason['reason']}")
        st.markdown(
            "影響: 有効候補を作成できないため、`有効候補をすべて実行` と meta 反映後の全件実行は開始されません。"
        )
        suggestions = getattr(combo_eval, "fix_suggestions", []) or []
        if suggestions:
            st.markdown("対処: " + " / ".join(str(item) for item in suggestions))
        with st.expander("除外理由ランキング", expanded=False):
            _show_df(pd.DataFrame(reason_rows), hide_index=True)
    else:
        st.markdown("原因: 候補軸が空、または入力条件が不足しています。")
        st.markdown("影響: 有効候補を作成できません。")
        st.markdown("対処: 候補設定、dataset 条件、backend/search_alg の整合性を見直してください。")


def _render_combo_reason_summary(reason_rows: list[dict[str, Any]]) -> None:
    if not reason_rows:
        return
    summary_parts: list[str] = []
    for row in reason_rows[:5]:
        reason = str(row.get("reason", "")).strip()
        count = int(row.get("count", 0) or 0)
        if reason:
            summary_parts.append(f"{reason} ({count}件)")
    if summary_parts:
        st.markdown("除外理由サマリ: " + " / ".join(summary_parts))


def _normalize_fit_kwargs_for_horizon(fit_kwargs: dict[str, Any], horizon: int | None) -> dict[str, Any]:
    normalized = dict(fit_kwargs or {})
    try:
        horizon_int = int(horizon) if horizon is not None else None
    except Exception:
        horizon_int = None
    if horizon_int is None or horizon_int <= 0:
        return normalized
    try:
        val_size = int(normalized.get("val_size", 0) or 0)
    except Exception:
        return normalized
    if val_size != 0 and val_size < horizon_int:
        normalized["val_size"] = int(horizon_int)
    return normalized


def _beep_metadata(severity: NotificationSeverity) -> dict[str, Any]:
    tone = {
        NotificationSeverity.SUCCESS: "success",
        NotificationSeverity.FAILURE: "failure",
        NotificationSeverity.WARNING: "warning",
        NotificationSeverity.RUNNING: "running",
    }[severity]
    return {"enabled": True, "volume": 0.35, "tone": tone}


def _default_next_actions(action: str, status: str) -> list[str]:
    low_action = str(action).lower()
    if status == "failed":
        return ["stderr と json tail を確認する", "入力不足か保存先を修正する", "同じ導線で再実行する"]
    if "train" in low_action:
        return ["予測/評価へ進む", "必要なら save/load を確認する"]
    if "predict" in low_action:
        return ["evaluate で精度を確認する", "結果生成物を確認する"]
    if "save" in low_action:
        return ["load で復元確認する", "生成物パスを確認する"]
    if "load" in low_action:
        return ["predict_insample で再現確認する", "分析へ進む"]
    return ["結果サマリを確認する", "次の推奨ステップへ進む"]


def _publish_notification(
    *,
    kind: NotificationEventKind,
    severity: NotificationSeverity,
    title: str,
    message: str,
    action: str,
    status: str,
    command_summary: str = "",
    error_summary: str = "",
    artifact_paths: list[str] | None = None,
    next_actions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = build_notification_event(
        event_id=str(uuid.uuid4()),
        kind=kind,
        severity=severity,
        title=title,
        message=message,
        action=action,
        status=status,
        command_summary=command_summary,
        error_summary=error_summary,
        artifact_paths=artifact_paths or [],
        next_actions=next_actions or _default_next_actions(action, status),
        metadata={**(metadata or {}), "beep": _beep_metadata(severity)},
        channels=(
            NotificationChannel.SCREEN,
            NotificationChannel.BEEP,
            NotificationChannel.EMAIL,
            NotificationChannel.MESSAGE,
        ),
    )
    summary = _get_notification_service().publish(event)
    payload = event.to_payload()
    payload["metadata"]["dispatch_deliveries"] = summary.deliveries
    render_notification_center([payload])
    return payload


def _notification_action_name(title: str, command: str) -> str:
    title_v = str(title or "").strip() or str(command or "").strip()
    parts = title_v.replace("NF ", "").replace("  ", " ").split()
    return str(parts[-1] if parts else title_v).strip().lower() or "operation"


_dataset_loader_support_df = dashboard_helpers.dataset_loader_support_df
_supported_backends_for_input_method = dashboard_helpers.supported_backends_for_input_method
_is_supported_backend_for_input_method = dashboard_helpers.is_supported_backend_for_input_method
_safe_ident = dashboard_helpers.safe_ident
_csv_nonempty_list = dashboard_helpers.csv_nonempty_list
_group_mode_unique_id_validation_error = dashboard_helpers.group_mode_unique_id_validation_error
_safe_tail = dashboard_helpers.safe_tail


_stable_json_dumps = dashboard_helpers.stable_json_dumps


def _dashboard_event_log_path() -> Path:
    DASHBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts_day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return DASHBOARD_LOG_DIR / f"events_{ts_day}.jsonl"


def _log_dashboard_event(event_type: str, payload: dict[str, Any], level: str = "INFO") -> None:
    try:
        enabled = bool(st.session_state.get("ui_enable_event_log", True))
    except Exception:
        enabled = True
    if not enabled:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": str(level),
        "event_type": str(event_type),
        "payload": payload,
    }
    try:
        p = _dashboard_event_log_path()
        with p.open("a", encoding="utf-8") as f:
            f.write(_stable_json_dumps(rec) + "\n")
    except Exception:
        return


_query_cache_key = dashboard_helpers.query_cache_key


def _query_cache_stats_snapshot() -> dict[str, Any]:
    with _QUERY_CACHE_LOCK:
        return {
            **dict(_QUERY_CACHE_STATS),
            "entries": int(len(_QUERY_CACHE)),
        }


def _clear_query_cache() -> None:
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE.clear()
        _QUERY_CACHE_STATS["hits"] = 0
        _QUERY_CACHE_STATS["misses"] = 0
        _QUERY_CACHE_STATS["queries"] = 0
        _QUERY_CACHE_STATS["slow_queries"] = 0
        _QUERY_CACHE_STATS["last_query_ms"] = 0.0


_slug = dashboard_helpers.slug


_db_connection_payload = dashboard_helpers.db_connection_payload


_format_bytes = dashboard_helpers.format_bytes


def _inject_modern_theme() -> None:
    st.markdown(
        """
<style>
@font-face {
  font-family: "OpsNotoSansJP";
  src: url("/app/static/fonts/NotoSansJP-VF.ttf") format("truetype");
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
}
:root {
  --ops-bg-a: #f6fafc;
  --ops-bg-b: #eef5fb;
  --ops-ink: #10273d;
  --ops-accent: #0f766e;
  --ops-accent-soft: #dff6f2;
  --ops-warn: #8a4b0f;
}
.stApp {
  background:
    radial-gradient(1200px 380px at 8% -12%, #d8eefc 0%, rgba(216,238,252,0.15) 45%, transparent 72%),
    radial-gradient(980px 320px at 100% 0%, #d8f5e9 0%, rgba(216,245,233,0.15) 40%, transparent 70%),
    linear-gradient(180deg, var(--ops-bg-a), var(--ops-bg-b));
  color: var(--ops-ink);
  font-family:
    "OpsNotoSansJP",
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "BIZ UDGothic",
    "Hiragino Sans",
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS PGothic",
    "Segoe UI",
    system-ui,
    sans-serif;
}
.stAppViewContainer,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetric"],
.stApp [data-baseweb="input"],
.stApp [data-baseweb="select"] {
  font-family:
    "OpsNotoSansJP",
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "BIZ UDGothic",
    "Hiragino Sans",
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS PGothic",
    "Segoe UI",
    system-ui,
    sans-serif !important;
}
.stApp,
.stApp button,
.stApp input,
.stApp textarea,
.stApp select,
.stApp label,
.stApp p,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
  font-family:
    "OpsNotoSansJP",
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "BIZ UDGothic",
    "Hiragino Sans",
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS PGothic",
    "Segoe UI",
    system-ui,
    sans-serif !important;
}
[data-testid="stToolbar"],
#MainMenu,
header[data-testid="stHeader"] {
  display: none !important;
  visibility: hidden;
  height: 0;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top: 0.75rem;
}
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stButton,
[data-testid="stSidebar"] .stExpander {
  margin-bottom: 0.45rem;
}
[data-testid="stSidebar"] .ops-sidebar-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--ops-ink);
  margin: 0 0 0.9rem 0;
}
h1 {
  letter-spacing: -0.04em;
  line-height: 0.95;
}
[data-testid="stMetric"] {
  border: 1px solid #d7e6f4;
  border-radius: 10px;
  background: #ffffffc2;
  padding: 8px 10px;
}
[data-baseweb="tab-list"] button {
  border-radius: 10px 10px 0 0 !important;
  border: 1px solid #dce6f2 !important;
  border-bottom: 0 !important;
  background: #f7fbff !important;
}
[data-baseweb="tab-list"] button[aria-selected="true"] {
  background: #e8f8f4 !important;
  color: #0b5f58 !important;
  font-weight: 700 !important;
}
.stCodeBlock {
  border-radius: 10px !important;
  border: 1px solid #dce6f2 !important;
}
@media (max-width: 900px) {
  h1 {
    font-size: clamp(2.2rem, 8vw, 3.4rem) !important;
    line-height: 1.03;
  }
  [data-testid="stAppViewContainer"] {
    padding-left: 0.9rem;
    padding-right: 0.9rem;
  }
  [data-testid="stMetric"] {
    padding: 0.7rem 0.85rem;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _document_metadata_html(
    *,
    description: str = APP_META_DESCRIPTION,
    lang: str = "ja",
) -> str:
    description_json = json.dumps(description)
    lang_json = json.dumps(lang)
    return f"""
<script>
(() => {{
  const doc = window.parent?.document || document;
  const ensureMeta = (name, content) => {{
    let el = doc.head.querySelector(`meta[name="${{name}}"]`);
    if (!el) {{
      el = doc.createElement("meta");
      el.name = name;
      doc.head.appendChild(el);
    }}
    el.content = content;
  }};
  doc.documentElement.lang = {lang_json};
  ensureMeta("description", {description_json});
  doc.title = "ロト予測 運用ダッシュボード";
}})();
</script>
"""


def _inject_document_metadata() -> None:
    components.html(_document_metadata_html(), height=0)


_normalize_df_for_streamlit = dashboard_helpers.normalize_df_for_streamlit


def _show_df(df: pd.DataFrame, *, hide_index: bool = True, height: int | None = None) -> None:
    kwargs: dict[str, Any] = {"hide_index": hide_index, "width": "stretch"}
    if height is not None:
        kwargs["height"] = int(height)
    st.dataframe(_normalize_df_for_streamlit(df), **kwargs)


def _normalize_status_value(
    value: Any,
    *,
    allowed: list[str] | tuple[str, ...] | None = None,
    aliases: dict[str, str] | None = None,
    default: str = "unknown",
) -> str:
    normalized_default = str(default).strip().lower() or "unknown"
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "<na>", "nat"}:
        text = normalized_default
    if aliases:
        text = str(aliases.get(text, text)).strip().lower() or normalized_default
    if allowed:
        allowed_set = {str(item).strip().lower() for item in allowed if str(item).strip()}
        if text not in allowed_set:
            return normalized_default
    return text or normalized_default


def _normalize_status_series(
    values: pd.Series | list[Any] | tuple[Any, ...],
    *,
    allowed: list[str] | tuple[str, ...] | None = None,
    aliases: dict[str, str] | None = None,
    default: str = "unknown",
) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="object")
    return series.map(lambda value: _normalize_status_value(value, allowed=allowed, aliases=aliases, default=default))


def _present_category_order(values: pd.Series | list[Any] | tuple[Any, ...], preferred: list[str]) -> list[str]:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="object")
    seen = {str(v).strip() for v in series.dropna().astype(str).tolist() if str(v).strip()}
    ordered = [item for item in preferred if item in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def _build_categorical_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    orientation: str = "v",
    color_map: dict[str, str] | None = None,
    color_order: list[str] | None = None,
    barmode: str = "relative",
    height: int | None = None,
    hover_fields: list[str] | None = None,
) -> go.Figure:
    work = df.copy()
    work[color] = _normalize_status_series(work[color], default="unknown") if color == "status" else (
        work[color].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    )
    work = work.dropna(subset=[x, y])
    figure = go.Figure()
    ordered_groups = color_order or _present_category_order(work[color], [])
    if not ordered_groups:
        ordered_groups = _present_category_order(work[color], sorted(work[color].unique().tolist()))
    for group_name in ordered_groups:
        group_df = work[work[color] == group_name].copy()
        if group_df.empty:
            continue
        customdata = None
        hovertemplate = f"{color}=%{{fullData.name}}<br>"
        if hover_fields:
            custom_cols: list[pd.Series] = []
            for idx, field in enumerate(hover_fields):
                if field in group_df.columns:
                    custom_cols.append(group_df[field].astype(str))
                    hovertemplate += f"{field}=%{{customdata[{idx}]}}<br>"
            if custom_cols:
                customdata = np.column_stack(custom_cols)
        hovertemplate += f"{x}=%{{x}}<br>{y}=%{{y}}<extra></extra>"
        if orientation == "h":
            hovertemplate = hovertemplate.replace(f"{x}=%{{x}}<br>{y}=%{{y}}", f"{y}=%{{y}}<br>{x}=%{{x}}")
        figure.add_trace(
            go.Bar(
                x=group_df[x] if orientation != "h" else group_df[y],
                y=group_df[y] if orientation != "h" else group_df[x],
                name=str(group_name),
                orientation=orientation,
                marker_color=(color_map or {}).get(str(group_name), STATUS_COLOR_MAP.get(str(group_name), "#64748b")),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
    figure.update_layout(title=title, barmode=barmode)
    if height is not None:
        figure.update_layout(height=int(height))
    return figure


def _build_categorical_scatter_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    color_map: dict[str, str] | None = None,
    color_order: list[str] | None = None,
    hover_fields: list[str] | None = None,
) -> go.Figure:
    work = df.copy()
    work[color] = _normalize_status_series(work[color], default="unknown") if color == "status" else (
        work[color].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    )
    work = work.dropna(subset=[x, y])
    figure = go.Figure()
    ordered_groups = color_order or _present_category_order(work[color], [])
    if not ordered_groups:
        ordered_groups = _present_category_order(work[color], sorted(work[color].unique().tolist()))
    for group_name in ordered_groups:
        group_df = work[work[color] == group_name].copy()
        if group_df.empty:
            continue
        customdata = None
        hovertemplate = f"{color}=%{{fullData.name}}<br>{x}=%{{x}}<br>{y}=%{{y}}"
        if hover_fields:
            custom_cols: list[pd.Series] = []
            for idx, field in enumerate(hover_fields):
                if field in group_df.columns:
                    custom_cols.append(group_df[field].astype(str))
                    hovertemplate += f"<br>{field}=%{{customdata[{idx}]}}"
            if custom_cols:
                customdata = np.column_stack(custom_cols)
        hovertemplate += "<extra></extra>"
        figure.add_trace(
            go.Scatter(
                x=group_df[x],
                y=group_df[y],
                mode="markers",
                name=str(group_name),
                marker={"color": (color_map or {}).get(str(group_name), STATUS_COLOR_MAP.get(str(group_name), "#64748b"))},
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
    figure.update_layout(title=title)
    return figure


def _build_categorical_histogram_figure(
    df: pd.DataFrame,
    *,
    x: str,
    color: str,
    title: str,
    color_map: dict[str, str] | None = None,
    color_order: list[str] | None = None,
    nbinsx: int | None = None,
) -> go.Figure:
    work = df.copy()
    work[color] = _normalize_status_series(work[color], default="unknown") if color == "status" else (
        work[color].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    )
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work = work.dropna(subset=[x])
    figure = go.Figure()
    ordered_groups = color_order or _present_category_order(work[color], [])
    if not ordered_groups:
        ordered_groups = _present_category_order(work[color], sorted(work[color].unique().tolist()))
    for group_name in ordered_groups:
        group_df = work[work[color] == group_name]
        if group_df.empty:
            continue
        figure.add_trace(
            go.Histogram(
                x=group_df[x],
                name=str(group_name),
                marker_color=(color_map or {}).get(str(group_name), STATUS_COLOR_MAP.get(str(group_name), "#64748b")),
                nbinsx=nbinsx,
                opacity=0.8,
            )
        )
    figure.update_layout(title=title, barmode="overlay")
    return figure


def _nf_lab_ui_state_persistable_key(key: str) -> bool:
    return dashboard_helpers.nf_lab_ui_state_persistable_key(key, NF_LAB_UI_STATE_PREFIXES)


def _nf_lab_ui_state_storage_key(host: str, port: int, user: str, database: str) -> str:
    return dashboard_helpers.nf_lab_ui_state_storage_key(
        host=host,
        port=int(port),
        user=user,
        database=database,
        app_name=NF_LAB_UI_STATE_APP_NAME,
        scope=NF_LAB_UI_STATE_SCOPE,
    )


def _nf_lab_ui_state_file_path(database: str) -> Path:
    db_slug = _slug(str(database).strip() or "default")
    return NF_LAB_UI_STATE_PATH.with_name(f"{NF_LAB_UI_STATE_PATH.stem}_{db_slug}{NF_LAB_UI_STATE_PATH.suffix}")


def _read_nf_lab_ui_state_file(path: Path) -> dict[str, Any]:
    return dashboard_helpers.read_nf_lab_ui_state_file(path)


def _collect_nf_lab_ui_state_payload() -> dict[str, Any]:
    return dashboard_helpers.collect_nf_lab_ui_state_payload(
        st.session_state,
        prefixes=NF_LAB_UI_STATE_PREFIXES,
    )


def _ensure_nf_lab_ui_state_table(engine: Engine) -> None:
    ddls = [
        "CREATE SCHEMA IF NOT EXISTS log",
        f"""
        CREATE TABLE IF NOT EXISTS {NF_LAB_UI_STATE_DB_TABLE} (
          state_key TEXT PRIMARY KEY,
          app_name TEXT NOT NULL,
          scope TEXT NOT NULL,
          db_identity TEXT NOT NULL,
          state_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
          state_hash TEXT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        f"CREATE INDEX IF NOT EXISTS idx_log_ui_state_updated_at ON {NF_LAB_UI_STATE_DB_TABLE}(updated_at DESC)",
    ]
    with engine.begin() as conn:
        for ddl in ddls:
            conn.execute(text(ddl))


def _load_nf_lab_ui_state_from_db(engine: Engine, state_key: str) -> dict[str, Any]:
    try:
        _ensure_nf_lab_ui_state_table(engine)
        sql = text(
            f"""
            SELECT state_json
            FROM {NF_LAB_UI_STATE_DB_TABLE}
            WHERE state_key = :state_key
            LIMIT 1
            """
        )
        with engine.connect() as conn:
            row = conn.execute(sql, {"state_key": str(state_key)}).mappings().first()
        if not row:
            return {}
        payload = row.get("state_json")
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, str):
            loaded = json.loads(payload)
            return loaded if isinstance(loaded, dict) else {}
        return {}
    except Exception:
        return {}


def _save_nf_lab_ui_state_to_db(
    engine: Engine,
    *,
    state_key: str,
    db_identity: str,
    payload: dict[str, Any],
    payload_hash: str,
) -> None:
    _ensure_nf_lab_ui_state_table(engine)
    sql = text(
        f"""
        INSERT INTO {NF_LAB_UI_STATE_DB_TABLE} (
          state_key, app_name, scope, db_identity, state_json, state_hash, updated_at
        ) VALUES (
          :state_key, :app_name, :scope, :db_identity, CAST(:state_json AS jsonb), :state_hash, now()
        )
        ON CONFLICT (state_key)
        DO UPDATE SET
          app_name = EXCLUDED.app_name,
          scope = EXCLUDED.scope,
          db_identity = EXCLUDED.db_identity,
          state_json = EXCLUDED.state_json,
          state_hash = EXCLUDED.state_hash,
          updated_at = now()
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "state_key": str(state_key),
                "app_name": NF_LAB_UI_STATE_APP_NAME,
                "scope": NF_LAB_UI_STATE_SCOPE,
                "db_identity": str(db_identity),
                "state_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                "state_hash": str(payload_hash),
            },
        )


def _load_nf_lab_ui_state_once(
    engine: Engine | None = None,
    *,
    host: str = "",
    port: int = 0,
    user: str = "",
    database: str = "",
) -> None:
    state_key = _nf_lab_ui_state_storage_key(host=host, port=int(port), user=user, database=database)
    loaded_key_flag = "_nf_lab_ui_state_loaded_key"
    if str(st.session_state.get(loaded_key_flag, "")) == str(state_key):
        return

    file_path = _nf_lab_ui_state_file_path(database=database)
    legacy_path = NF_LAB_UI_STATE_PATH
    file_payload = _read_nf_lab_ui_state_file(file_path)
    if not file_payload and file_path != legacy_path:
        file_payload = _read_nf_lab_ui_state_file(legacy_path)
    db_payload = _load_nf_lab_ui_state_from_db(engine, state_key=state_key) if engine is not None else {}

    merged_payload = dashboard_helpers.merge_nf_lab_ui_state_payload(
        file_payload,
        db_payload,
        prefixes=NF_LAB_UI_STATE_PREFIXES,
    )

    try:
        for k, v in merged_payload.items():
            key = str(k)
            if not _nf_lab_ui_state_persistable_key(key):
                continue
            st.session_state[key] = v
    except Exception:
        pass

    st.session_state[loaded_key_flag] = str(state_key)
    st.session_state["_nf_lab_ui_state_loaded"] = True


def _save_nf_lab_ui_state(
    engine: Engine | None = None,
    *,
    host: str = "",
    port: int = 0,
    user: str = "",
    database: str = "",
) -> None:
    if not bool(st.session_state.get("_nf_lab_ui_state_loaded", False)):
        return

    payload = _collect_nf_lab_ui_state_payload()
    state_key = _nf_lab_ui_state_storage_key(host=host, port=int(port), user=user, database=database)
    payload_hash = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8", errors="ignore")).hexdigest()
    hash_state_key = f"_nf_lab_ui_state_saved_hash_{_slug(state_key)}"
    if str(st.session_state.get(hash_state_key, "")) == payload_hash:
        return

    try:
        p = _nf_lab_ui_state_file_path(database=database)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass

    if engine is not None:
        try:
            db_identity = f"{str(host).strip()}:{int(port)}:{str(user).strip()}:{str(database).strip()}"
            _save_nf_lab_ui_state_to_db(
                engine,
                state_key=state_key,
                db_identity=db_identity,
                payload=payload,
                payload_hash=payload_hash,
            )
        except Exception:
            pass

    st.session_state[hash_state_key] = payload_hash
    st.session_state["_nf_lab_ui_state_saved_hash"] = payload_hash


def _tail_lines(path: Path, n: int = 250) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def _tree_lines(root: Path, max_depth: int = 3, max_entries: int = 500) -> str:
    if not root.exists() or not root.is_dir():
        return f"{root} (not found)"

    lines: list[str] = [root.name + "/"]
    count = 1
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        parts = rel.parts
        if not parts:
            continue
        if parts[0] in TREE_SKIP_TOP_LEVEL_NAMES:
            continue
        if any(part.startswith(".") or part in TREE_SKIP_NAMES for part in parts):
            continue
        if count >= max_entries:
            lines.append("... (truncated)")
            break
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        suffix = "/" if p.is_dir() else ""
        lines.append(f"{indent}- {rel.name}{suffix}")
        count += 1
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def _engine_for_dsn(dsn: str) -> Engine:
    return create_engine(dsn, pool_pre_ping=True)


def _dsn(host: str, port: int, user: str, password: str, database: str) -> str:
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def _resolve_effective_password(password_input: str) -> str:
    raw = str(password_input or "").strip()
    if raw:
        return raw
    return str(getattr(settings, "db_password", "") or "")


def _masked_secret(secret: str, *, min_width: int = 8) -> str:
    return "*" * max(min_width, len(str(secret or "")))


def _safe_db_env_assignment(password_env_var: str = PASSWORD_ENV_VAR_NAME) -> str:
    return f"{password_env_var}=${{{password_env_var}}}"


def _safe_db_cli_flags(
    host: str,
    port: int,
    user: str,
    database: str,
    *,
    password_env_var: str = PASSWORD_ENV_VAR_NAME,
) -> str:
    def _quote(value: Any) -> str:
        return shlex.quote(str(value))

    return " ".join(
        [
            f"--host {_quote(host)}",
            f"--port {int(port)}",
            f"--user {_quote(user)}",
            f"--database {_quote(database)}",
            _safe_db_env_assignment(password_env_var),
        ]
    )


def _try_connect(engine: Engine) -> tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, ""
    except Exception as e:
        return False, str(e)


def _query_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    query_sql = str(sql).strip()
    query_params = dict(params or {})
    lower_sql = query_sql.lower()
    is_select = lower_sql.startswith("select") or lower_sql.startswith("with")
    try:
        cache_enabled = bool(st.session_state.get("ui_enable_query_cache", True))
        cache_ttl = float(st.session_state.get("ui_query_cache_ttl_sec", 15))
        cache_max_entries = int(st.session_state.get("ui_query_cache_max_entries", 256))
        slow_ms_threshold = float(st.session_state.get("ui_slow_query_ms", 800))
    except Exception:
        cache_enabled = True
        cache_ttl = 15.0
        cache_max_entries = 256
        slow_ms_threshold = 800.0

    cache_key: str | None = None
    now = time.time()
    if cache_enabled and is_select:
        cache_key = _query_cache_key(engine=engine, sql=query_sql, params=query_params)
        with _QUERY_CACHE_LOCK:
            cached = _QUERY_CACHE.get(cache_key)
            if cached is not None and (now - float(cached.get("ts", 0.0))) <= cache_ttl:
                _QUERY_CACHE_STATS["hits"] = int(_QUERY_CACHE_STATS.get("hits", 0)) + 1
                return cached["df"].copy(deep=False)
            _QUERY_CACHE_STATS["misses"] = int(_QUERY_CACHE_STATS.get("misses", 0)) + 1

    t0 = time.perf_counter()
    with engine.connect() as conn:
        out_df = pd.read_sql(text(query_sql), conn, params=query_params if query_params else None)
    elapsed_ms = float((time.perf_counter() - t0) * 1000.0)

    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE_STATS["queries"] = int(_QUERY_CACHE_STATS.get("queries", 0)) + 1
        _QUERY_CACHE_STATS["last_query_ms"] = elapsed_ms
        if elapsed_ms >= slow_ms_threshold:
            _QUERY_CACHE_STATS["slow_queries"] = int(_QUERY_CACHE_STATS.get("slow_queries", 0)) + 1

    if cache_key is not None:
        with _QUERY_CACHE_LOCK:
            if len(_QUERY_CACHE) >= max(10, cache_max_entries):
                oldest_key = min(_QUERY_CACHE.items(), key=lambda kv: float(kv[1].get("ts", 0.0)))[0]
                _QUERY_CACHE.pop(oldest_key, None)
            _QUERY_CACHE[cache_key] = {"ts": now, "df": out_df.copy(deep=False)}

    if elapsed_ms >= slow_ms_threshold:
        _log_dashboard_event(
            event_type="slow_query",
            level="WARN",
            payload={
                "elapsed_ms": elapsed_ms,
                "sql_head": query_sql[:240],
                "params": query_params,
                "rows": int(out_df.shape[0]),
            },
        )
    return out_df


def _read_sql_direct(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(str(sql)), conn, params=dict(params or {}))


def _parallel_query_frames(
    engine: Engine,
    query_specs: dict[str, tuple[str, dict[str, Any] | None]],
    max_workers: int = 4,
) -> dict[str, pd.DataFrame]:
    if not query_specs:
        return {}
    workers = max(1, min(int(max_workers), max(1, len(query_specs))))
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_key = {ex.submit(_read_sql_direct, engine, spec[0], spec[1]): key for key, spec in query_specs.items()}
        for fut in as_completed(fut_to_key):
            key = fut_to_key[fut]
            try:
                out[key] = fut.result()
            except Exception as e:
                out[key] = pd.DataFrame({"error": [str(e)]})
    return out


def _gpu_runtime_info() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cupy_available": bool(CUPY_AVAILABLE),
        "torch_available": bool(TORCH_AVAILABLE),
        "cuda_available": False,
        "device_name": None,
    }
    if TORCH_AVAILABLE:
        try:
            out["cuda_available"] = bool(torch.cuda.is_available())
            if out["cuda_available"]:
                out["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return out


def _corr_matrix_fast(df: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    num = df.apply(pd.to_numeric, errors="coerce")
    if num.shape[1] <= 1:
        return pd.DataFrame()
    if use_gpu and CUPY_AVAILABLE:
        try:
            arr = num.to_numpy(dtype=float)
            if arr.size == 0:
                return pd.DataFrame()
            arr_gpu = cp.asarray(arr)
            mask = cp.isnan(arr_gpu)
            col_mean = cp.nanmean(arr_gpu, axis=0)
            row_idx, col_idx = cp.where(mask)
            arr_gpu[row_idx, col_idx] = col_mean[col_idx]
            corr = cp.corrcoef(arr_gpu, rowvar=False)
            return pd.DataFrame(cp.asnumpy(corr), index=num.columns, columns=num.columns)
        except Exception:
            pass
    return num.corr(numeric_only=True)


def _existing_tables(engine: Engine) -> set[tuple[str, str]]:
    df = _query_df(
        engine,
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('dataset', 'exog', 'resources', 'meta', 'model', 'log')
          AND table_type='BASE TABLE'
        """,
    )
    return {(str(r.table_schema), str(r.table_name)) for r in df.itertuples(index=False)}


def _table_catalog(engine: Engine) -> pd.DataFrame:
    df = _query_df(
        engine,
        """
        SELECT
          n.nspname AS table_schema,
          c.relname AS table_name,
          COALESCE(s.n_live_tup::bigint, c.reltuples::bigint, 0) AS est_rows,
          pg_total_relation_size(c.oid) AS total_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE c.relkind='r'
          AND n.nspname IN ('dataset', 'exog', 'resources', 'meta', 'model', 'log')
        ORDER BY n.nspname, c.relname
        """,
    )
    if not df.empty:
        df["table_size"] = df["total_bytes"].apply(_format_bytes)
    return df


def _table_columns(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    return _query_df(
        engine,
        """
        SELECT
          column_name,
          data_type,
          is_nullable,
          ordinal_position
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """,
        {"schema": schema, "table": table},
    )


def _sample_table(engine: Engine, schema: str, table: str, limit: int) -> pd.DataFrame:
    sql = f"SELECT * FROM {_safe_ident(schema)}.{_safe_ident(table)} LIMIT {int(limit)}"
    return _query_df(engine, sql)


def _exact_count(engine: Engine, schema: str, table: str) -> int:
    sql = f"SELECT COUNT(*)::bigint AS cnt FROM {_safe_ident(schema)}.{_safe_ident(table)}"
    df = _query_df(engine, sql)
    return int(df.iloc[0]["cnt"])


def _schema_tables(engine: Engine, schema: str) -> list[str]:
    df = _query_df(
        engine,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """,
        {"schema": str(schema).strip()},
    )
    if df.empty or "table_name" not in df.columns:
        return []
    return [str(x).strip() for x in df["table_name"].astype(str).tolist() if str(x).strip()]


def _schema_table_counts(engine: Engine, schema: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table in _schema_tables(engine, schema):
        try:
            cnt = _exact_count(engine, schema, table)
            rows.append({"schema": str(schema), "table": str(table), "row_count": int(cnt)})
        except Exception as e:
            rows.append({"schema": str(schema), "table": str(table), "row_count": None, "error": str(e)})
    return pd.DataFrame(rows)


def _truncate_schema_tables(engine: Engine, schema: str) -> dict[str, Any]:
    schema_v = str(schema or "").strip()
    if not schema_v:
        raise ValueError("schema is required")
    tables = _schema_tables(engine, schema_v)
    if not tables:
        return {"schema": schema_v, "tables": [], "truncated": 0}
    table_refs = ", ".join([f"{_safe_ident(schema_v)}.{_safe_ident(t)}" for t in tables])
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_refs} RESTART IDENTITY CASCADE"))
    return {"schema": schema_v, "tables": tables, "truncated": len(tables)}


def _yaml_summary(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"exists": False, "path": str(p)}
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {"exists": True, "path": str(p), "error": str(e)}

    rows = payload.get("rows")
    row_count = int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0
    out: dict[str, Any] = {
        "exists": True,
        "path": str(p),
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": row_count,
    }
    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
        if "module" in df.columns:
            out["top_modules"] = (
                df["module"].astype(str).value_counts().head(12).rename_axis("module").reset_index(name="count")
            )
        if "top_group" in df.columns:
            out["top_groups"] = (
                df["top_group"].astype(str).value_counts().head(12).rename_axis("top_group").reset_index(name="count")
            )
    return out


def _looks_like_shell_command(text_value: str) -> bool:
    raw = str(text_value or "").strip()
    if not raw:
        return False
    first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    if not first:
        return False
    if first.startswith(("```", "{", "[", "<", "http://", "https://", "# ")):
        return False
    if first.startswith("--"):
        return False
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first):
        return True
    command_heads = (
        "python ",
        "bash ",
        "sh ",
        "streamlit ",
        "conda ",
        "pip ",
        "uv ",
        "pytest ",
        "make ",
        "git ",
        "docker ",
        "./",
        "/",
    )
    if first.startswith(command_heads):
        return True
    return " " in first and re.match(r"^[A-Za-z0-9_./:-]+(\s+.+)$", first) is not None


def _resolve_copy_cwd(cwd: str | Path | None) -> str | Path | None:
    if cwd is not None:
        return cwd
    try:
        raw = str(st.session_state.get("ui_copy_cwd", "")).strip()
    except Exception:
        raw = ""
    if not raw:
        return None
    return Path(raw).expanduser()


def _prepend_cd_for_copy(text_value: str, cwd: str | Path | None) -> str:
    text = str(text_value or "")
    resolved_cwd = _resolve_copy_cwd(cwd)
    if resolved_cwd is None or not _looks_like_shell_command(text):
        return text
    cwd_s = str(Path(str(resolved_cwd)).expanduser())
    if not cwd_s.strip():
        return text
    lines = str(text).splitlines()
    first_nonempty = next((ln.strip() for ln in lines if str(ln).strip()), "")
    if first_nonempty.startswith("cd "):
        return text
    cd_line = f"cd {shlex.quote(cwd_s)}"
    return cd_line if not str(text).strip() else f"{cd_line}\n{text}"


def _truncate_arg_preview(value: Any, max_len: int = 220) -> str:
    text_v = str(value)
    if len(text_v) > int(max_len):
        return text_v[: max_len - 3] + "..."
    return text_v


def _command_argument_table(command: str) -> pd.DataFrame:
    return _build_command_argument_table(command)


def _render_command_preview(
    command: str,
    *,
    copy_key: str,
    copy_label: str,
    cwd: str | Path | None = None,
    show_arg_table: bool = True,
) -> None:
    st.code(command, language="bash")
    _render_copy_button(command, key=copy_key, label=copy_label, cwd=cwd)
    if show_arg_table:
        with st.expander("引数の設定一覧表", expanded=False):
            arg_df = _command_argument_table(command)
            if arg_df.empty:
                st.info("引数を解析できませんでした。")
            else:
                _show_df(arg_df, hide_index=True)


def _render_copy_button(
    text_value: str,
    key: str,
    label: str = "Copy To Clipboard",
    cwd: str | Path | None = None,
) -> None:
    button_id = f"copy_{_slug(key)}"
    status_id = f"status_{_slug(key)}"
    copy_text = _prepend_cd_for_copy(text_value, cwd)
    js_text = json.dumps(copy_text)
    html_code = f"""
    <button id="{button_id}" style="padding:6px 10px;border:1px solid #888;border-radius:6px;cursor:pointer;">
      {html.escape(label)}
    </button>
    <span id="{status_id}" style="margin-left:8px;font-size:12px;color:#3c7;"></span>
    <script>
      const btn = document.getElementById("{button_id}");
      const status = document.getElementById("{status_id}");
      btn.onclick = async () => {{
        try {{
          await navigator.clipboard.writeText({js_text});
          status.textContent = "copied";
          setTimeout(() => status.textContent = "", 1500);
        }} catch(e) {{
          status.textContent = "copy failed";
        }}
      }};
    </script>
    """
    components.html(html_code, height=42)


def _render_read_aloud(text_value: str, key: str) -> None:
    play_id = f"play_{_slug(key)}"
    stop_id = f"stop_{_slug(key)}"
    status_id = f"speech_status_{_slug(key)}"
    js_text = json.dumps(text_value[:12000])
    html_code = f"""
    <div>
      <button id="{play_id}" style="padding:6px 10px;border:1px solid #888;border-radius:6px;cursor:pointer;">Read Aloud</button>
      <button id="{stop_id}" style="padding:6px 10px;border:1px solid #888;border-radius:6px;cursor:pointer;margin-left:8px;">Stop</button>
      <span id="{status_id}" style="margin-left:8px;font-size:12px;color:#3c7;"></span>
    </div>
    <script>
      const text = {js_text};
      const status = document.getElementById("{status_id}");
      document.getElementById("{play_id}").onclick = () => {{
        if (!('speechSynthesis' in window)) {{
          status.textContent = "speech api unavailable";
          return;
        }}
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "ja-JP";
        u.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        status.textContent = "speaking...";
      }};
      document.getElementById("{stop_id}").onclick = () => {{
        if ('speechSynthesis' in window) {{
          window.speechSynthesis.cancel();
        }}
        status.textContent = "stopped";
      }};
    </script>
    """
    components.html(html_code, height=54)


def _render_mermaid(mermaid_code: str, key: str, height: int = 460) -> None:
    cid = f"mmd_{_slug(key)}"
    code_json = json.dumps(mermaid_code)
    html_code = f"""
    <div id="{cid}" class="mermaid"></div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: false, securityLevel: "loose", theme: "default" }});
      const el = document.getElementById("{cid}");
      el.textContent = {code_json};
      mermaid.run({{ nodes: [el] }});
    </script>
    """
    components.html(html_code, height=height, scrolling=True)


def _render_rich_content(content: str, ext: str, key: str) -> None:
    ext = ext.lower().lstrip(".")
    if ext == "json":
        try:
            st.json(json.loads(content))
            return
        except Exception:
            st.code(content)
            return
    if ext in {"yaml", "yml"}:
        try:
            st.json(yaml.safe_load(content))
            return
        except Exception:
            st.code(content)
            return
    if ext == "md":
        st.markdown(content)
        return
    if ext == "html":
        components.html(content, height=460, scrolling=True)
        return
    if ext == "mmd":
        _render_mermaid(content, key=f"rich_{key}", height=460)
        return
    if ext == "csv":
        try:
            df = pd.read_csv(StringIO(content))
            _show_df(df.head(500))
            return
        except Exception:
            st.code(content)
            return
    st.code(content)


def _run_shell_command(command: str, cwd: Path, timeout_sec: int = 600) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": -9,
            "stdout": (e.stdout or ""),
            "stderr": (
                f"timeout after {timeout_sec}s\n"
                + (e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
            ),
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": traceback.format_exc(),
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }


def _tail_join(lines: list[str], max_lines: int = 40) -> str:
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)) :])


def _try_parse_json_tail(text_value: str) -> dict[str, Any] | list[Any] | None:
    raw = str(text_value or "").strip()
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, (dict, list)):
            return loaded
    except Exception:
        pass

    starts = [i for i, ch in enumerate(raw) if ch == "{"]
    for i in reversed(starts[-500:]):
        candidate = raw[i:].strip()
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            continue
    return None


def _api_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        base = "http://127.0.0.1:8000"
    p = str(path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    return base + p


def _http_json_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url=str(url),
        method=str(method or "GET").upper(),
        data=data,
        headers=headers,
    )

    status: int | None = None
    raw = ""
    err: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=float(max(1.0, timeout_sec))) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = int(getattr(e, "code", 500))
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        err = f"http_error_{status}"
    except Exception as e:
        err = str(e)

    data_obj: Any = None
    if raw:
        try:
            data_obj = json.loads(raw)
        except Exception:
            data_obj = None

    ok = bool(err is None and status is not None and 200 <= int(status) < 300)
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    return {
        "ok": ok,
        "status": status,
        "error": err,
        "elapsed_ms": elapsed_ms,
        "data": data_obj,
        "raw": raw,
    }


def _api_get_json(
    base_url: str,
    path: str,
    timeout_sec: float = 10.0,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = _api_url(base_url, path)
    if params:
        qp = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None},
            doseq=True,
        )
        if qp:
            url += f"?{qp}"
    return _http_json_request("GET", url, payload=None, timeout_sec=timeout_sec)


def _api_post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    return _http_json_request(
        "POST",
        _api_url(base_url, path),
        payload=payload,
        timeout_sec=timeout_sec,
    )


def _parse_json_dict_input(raw: str, default: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    text_raw = str(raw or "").strip()
    if not text_raw:
        return dict(default or {}), None
    try:
        obj = json.loads(text_raw)
    except Exception as e:
        return None, f"json parse error: {e}"
    if not isinstance(obj, dict):
        return None, "json must be object/dict"
    return dict(obj), None


_parameter_name = dashboard_helpers.parameter_name


def _estimate_script_step_progress(stdout_lines: list[str], stderr_lines: list[str]) -> tuple[str, float] | None:
    step_re = re.compile(
        r"^\[(?P<step>\d+)/(?P<total>\d+)\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+"
        r"(?P<state>START|RUNNING|DONE|FAIL)?\s*(?P<label>.*)$"
    )
    done_re = re.compile(r"^\[done\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
    lines = stdout_lines[-400:] + stderr_lines[-120:]
    for raw in reversed(lines):
        line = str(raw or "").strip()
        if not line:
            continue
        if done_re.search(line):
            return ("pipeline finished", 0.95)
        m = step_re.search(line)
        if m:
            step = max(1, int(m.group("step")))
            total = max(step, int(m.group("total")))
            state = str(m.group("state") or "").strip().lower()
            label = str(m.group("label") or "").strip() or "running"
            base = (step - 1) / float(total)
            if state == "start":
                ratio = min(0.94, base + 0.03)
                stage = f"step {step}/{total} start: {label}"
            elif state == "running":
                ratio = min(0.94, base + 0.06)
                stage = f"step {step}/{total} running: {label}"
            elif state == "done":
                ratio = min(0.94, (step / float(total)) - 0.01)
                stage = f"step {step}/{total} done: {label}"
            elif state == "fail":
                ratio = min(0.94, (step / float(total)))
                stage = f"step {step}/{total} failed: {label}"
            else:
                ratio = min(0.94, base + 0.02)
                stage = f"step {step}/{total}: {label}"
            return stage, ratio
    return None


def _extract_last_run_id(text_value: str) -> str | None:
    raw = str(text_value or "")
    if not raw.strip():
        return None
    # Prefer explicit JSON keys first.
    json_re = re.compile(r'"run_id"\s*:\s*"([^"]+)"')
    for m in reversed(list(json_re.finditer(raw[-120_000:]))):
        rid = str(m.group(1) or "").strip()
        if rid:
            return rid
    # Fallback to known run-id pattern produced by meta-automodel-run.
    id_re = re.compile(r"\bcfg\d+_d\d+_t\d+_\d{8}_\d{6}\b")
    ids = id_re.findall(raw[-120_000:])
    if ids:
        return str(ids[-1])
    return None


def _extract_execution_evidence(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    raw = "\n".join([str(stdout_text or ""), str(stderr_text or "")])
    run_ids: list[str] = []
    run_ids.extend(re.findall(r"\bcfg\d+_d\d+_t\d+_\d{8}_\d{6}\b", raw))
    run_ids.extend(re.findall(r'"run_id"\s*:\s*"([^"]+)"', raw))
    run_ids = list(dict.fromkeys([str(x).strip() for x in run_ids if str(x).strip()]))
    success_lines = len(re.findall(r"\[meta-automodel-run\]\s+success\b", raw))
    failed_lines = len(re.findall(r"\[meta-automodel-run\]\s+failed\b", raw))
    task_lines = len(re.findall(r"\[meta-automodel-run\].*task=\d+/\d+", raw))
    traceback_count = raw.count("Traceback (most recent call last):")
    error_lines = len(re.findall(r"\b(ERROR|Error|Exception|Traceback)\b", raw))
    executed_confirmed = bool(run_ids or success_lines > 0 or failed_lines > 0 or task_lines > 0)
    return {
        "executed_confirmed": executed_confirmed,
        "run_id_count": int(len(run_ids)),
        "task_log_count": int(task_lines),
        "success_log_count": int(success_lines),
        "failed_log_count": int(failed_lines),
        "traceback_count": int(traceback_count),
        "error_line_count": int(error_lines),
        "sample_run_ids": run_ids[:10],
    }


_is_valid_search_alg_for_backend = dashboard_helpers.is_valid_search_alg_for_backend
_validate_train_combo_choice = dashboard_helpers.validate_train_combo_choice
_parse_horizon_axis_value = dashboard_helpers.parse_horizon_axis_value
_resolve_model_horizon = dashboard_helpers.resolve_model_horizon


def _validate_model_runtime_prerequisites(model: str, horizon: int | None = None) -> str | None:
    model_v = str(model or "").strip()
    if model_v == "AutoxLSTM":
        try:
            __import__("xlstm")
        except Exception:
            return "AutoxLSTM は `xlstm` 依存が未導入のため実行対象外です。"
    return None


def _extract_train_result_summary(stdout_text: str, stderr_text: str) -> dict[str, Any] | None:
    raw = "\n".join([str(stdout_text or ""), str(stderr_text or "")])
    payload = _try_parse_json_tail(raw)
    if isinstance(payload, dict):
        keys = {
            "status",
            "run_id",
            "model_name",
            "artifact_path",
            "artifact_exists",
            "meta_exists",
            "log_path",
            "error",
        }
        if any(k in payload for k in keys):
            return {
                "status": str(payload.get("status", "")).strip() or None,
                "run_id": str(payload.get("run_id", "")).strip() or None,
                "model_name": str(payload.get("model_name", "")).strip() or None,
                "artifact_path": str(payload.get("artifact_path", "")).strip() or None,
                "artifact_exists": bool(payload.get("artifact_exists", False))
                if "artifact_exists" in payload
                else None,
                "meta_exists": bool(payload.get("meta_exists", False)) if "meta_exists" in payload else None,
                "log_path": str(payload.get("log_path", "")).strip() or None,
                "error": str(payload.get("error", "")).strip() or None,
            }

    # Fallback: parse key fields from noisy progress logs where JSON-tail extraction can fail.
    compact = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw)
    compact = compact.replace("\r", "\n")
    status_m = re.findall(r'"status"\s*:\s*"(success|failed)"', compact, flags=re.IGNORECASE)
    run_id_m = re.findall(r'"run_id"\s*:\s*"([^"]+)"', compact)
    model_m = re.findall(r'"model_name"\s*:\s*"([^"]+)"', compact)
    art_path_m = re.findall(r'"artifact_path"\s*:\s*"([^"]+)"', compact)
    log_path_m = re.findall(r'"log_path"\s*:\s*"([^"]+)"', compact)
    art_exists_m = re.findall(r'"artifact_exists"\s*:\s*(true|false)', compact, flags=re.IGNORECASE)
    meta_exists_m = re.findall(r'"meta_exists"\s*:\s*(true|false)', compact, flags=re.IGNORECASE)
    err_m = re.findall(r'"error"\s*:\s*"([^"]+)"', compact)
    if not any([status_m, run_id_m, model_m, art_path_m, log_path_m, err_m]):
        return None
    return {
        "status": (str(status_m[-1]).lower() if status_m else None),
        "run_id": (str(run_id_m[-1]).strip() if run_id_m else None),
        "model_name": (str(model_m[-1]).strip() if model_m else None),
        "artifact_path": (str(art_path_m[-1]).strip() if art_path_m else None),
        "artifact_exists": (str(art_exists_m[-1]).lower() == "true") if art_exists_m else None,
        "meta_exists": (str(meta_exists_m[-1]).lower() == "true") if meta_exists_m else None,
        "log_path": (str(log_path_m[-1]).strip() if log_path_m else None),
        "error": (str(err_m[-1]).strip() if err_m else None),
    }


def _build_train_combo_signature(model_name: str, horizon: int, params_obj: dict[str, Any]) -> str:
    return build_train_combo_signature(model_name=model_name, horizon=horizon, params_obj=params_obj)


def _make_param_based_run_id(model_name: str, horizon: int, params_obj: dict[str, Any]) -> str:
    return make_param_based_run_id(model_name=model_name, horizon=horizon, params_obj=params_obj)


def _load_completed_combo_signatures(engine: Engine | None) -> dict[str, set[str]]:
    return load_completed_combo_index(engine)


def _is_combo_signature_completed(combo_signature: str, completed_index: dict[str, set[str]] | None) -> bool:
    return is_combo_signature_completed(combo_signature=combo_signature, completed_index=completed_index)


def _upsert_combo_run_result_log(
    engine: Engine | None,
    *,
    run_id: str,
    status: str,
    model_name: str,
    horizon: int,
    params_json: dict[str, Any],
    diagnostics_json: dict[str, Any],
    model_save_json: dict[str, Any],
    artifact_path: str | None,
    log_path: str | None,
    error_message: str | None,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    if engine is None or not str(run_id).strip():
        return
    q = text(
        """
        INSERT INTO model.nf_automodel (
          config_id, run_id, status, model_name, horizon,
          params_json, diagnostics_json, model_save_json,
          artifact_path, log_path, error_message, started_at, ended_at
        ) VALUES (
          NULL, :run_id, :status, :model_name, :horizon,
          CAST(:params_json AS jsonb), CAST(:diagnostics_json AS jsonb), CAST(:model_save_json AS jsonb),
          :artifact_path, :log_path, :error_message, :started_at, :ended_at
        )
        ON CONFLICT (run_id) DO UPDATE SET
          status = EXCLUDED.status,
          model_name = EXCLUDED.model_name,
          horizon = EXCLUDED.horizon,
          params_json = EXCLUDED.params_json,
          diagnostics_json = EXCLUDED.diagnostics_json,
          model_save_json = EXCLUDED.model_save_json,
          artifact_path = EXCLUDED.artifact_path,
          log_path = EXCLUDED.log_path,
          error_message = EXCLUDED.error_message,
          started_at = EXCLUDED.started_at,
          ended_at = EXCLUDED.ended_at
        """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "run_id": str(run_id).strip(),
                "status": ("success" if str(status).strip().lower() == "success" else "failed"),
                "model_name": str(model_name or "unknown"),
                "horizon": max(1, int(horizon or 1)),
                "params_json": json.dumps(dict(params_json or {}), ensure_ascii=False),
                "diagnostics_json": json.dumps(dict(diagnostics_json or {}), ensure_ascii=False),
                "model_save_json": json.dumps(dict(model_save_json or {}), ensure_ascii=False),
                "artifact_path": str(artifact_path or "") or None,
                "log_path": str(log_path or "") or None,
                "error_message": str(error_message or "")[:8000] or None,
                "started_at": started_at,
                "ended_at": ended_at,
            },
        )


def _estimate_command_stage(command: str, stdout_lines: list[str], stderr_lines: list[str]) -> tuple[str, float]:
    cmd = str(command).lower()
    tail = stdout_lines[-120:] + stderr_lines[-80:]
    text = "\n".join(tail).lower()
    script_progress = _estimate_script_step_progress(stdout_lines, stderr_lines)
    if script_progress and (
        "scripts/run_local_nf_" in cmd
        or "scripts/run_model_save_load_analyze.sh" in cmd
        or "scripts/run_fast_meta_pipeline.sh" in cmd
    ):
        return script_progress

    if "python -m loto_forecast.cli train" in cmd:
        all_text = "\n".join(stdout_lines[-1200:] + stderr_lines[-400:])
        lower_text = all_text.lower()
        if '"status": "success"' in lower_text:
            return "model artifact confirmed", 0.97
        if '"status": "failed"' in lower_text or "traceback (most recent call last):" in lower_text:
            return "training failed (see traceback)", 0.97
        if "predicting dataloader" in lower_text:
            return "generating final forecast outputs", 0.92
        fit_stop_count = lower_text.count("trainer.fit` stopped")
        if fit_stop_count > 0:
            return f"refit/eval stage ({fit_stop_count})", min(0.9, 0.82 + 0.02 * fit_stop_count)
        trial_finished = len(re.findall(r"\btrial\s+\d+\s+finished\b", lower_text))
        if trial_finished > 0:
            return f"hyperparameter trial finished x{trial_finished}", min(0.8, 0.38 + 0.14 * trial_finished)
        if "a new study created in memory" in lower_text:
            return "hyperparameter study started", 0.3
        if "fit start model=" in lower_text:
            return "neuralforecast fit started", 0.18
        if "gpu available" in lower_text:
            return "initializing trainer/gpu", 0.14
        return "preparing train runtime", 0.1

    if "meta-automodel-create" in cmd:
        if '"action"' in text and '"config_id"' in text:
            return "meta row upsert done", 0.92
        return "upserting meta config", 0.25

    if "meta-automodel-run" in cmd:
        if '"executed"' in text and '"success"' in text:
            return "collecting run summary", 0.92
        task_matches = re.findall(r"task=(\d+)/(\d+)", text)
        if task_matches:
            done_s, total_s = task_matches[-1]
            done = max(0, int(done_s))
            total = max(1, int(total_s))
            ratio = min(0.9, 0.45 + 0.45 * (done / float(total)))
            return f"training tasks {done}/{total}", ratio
        exog_matches = re.findall(r"\[exog\s+(\d+)/(\d+)\]", text)
        if exog_matches:
            done_s, total_s = exog_matches[-1]
            done = max(0, int(done_s))
            total = max(1, int(total_s))
            ratio = min(0.42, 0.08 + 0.34 * (done / float(total)))
            return f"building unified dataset (exog {done}/{total})", ratio
        if "unified filter applied" in text:
            return "filtered unified dataset", 0.44
        if "skip postgres persist" in text:
            return "skipping postgres persist (column limit)", 0.48
        if "writing postgres table" in text:
            return "writing unified dataset to postgres", 0.5
        if "building unified dataset" in text:
            return "building unified dataset", 0.2
        return "running recursive/param-grid tasks", 0.45

    if "run-table-pyspark" in cmd:
        if "execution backend selected: polars" in text:
            return "running polars pipeline", 0.5
        if "execution backend selected: dask" in text:
            return "running dask pipeline", 0.5
        if "execution backend selected: pandas" in text:
            return "running pandas pipeline", 0.5
        if '"fallback_engine"' in text:
            return "running pandas fallback", 0.75
        if '"ok": true' in text and '"outputs"' in text:
            return "writing output summary", 0.92
        if "setting default log level" in text or "spark" in text:
            return "starting spark runtime", 0.35
        if "classnotfoundexception" in text or "driver" in text:
            return "jdbc driver check / fallback detection", 0.55
        return "executing spark table pipeline", 0.25

    if "build-unified-dataset" in cmd:
        if '"row_count"' in text or '"output"' in text:
            return "writing unified dataset outputs", 0.88
        return "building unified dataset", 0.4

    if "model-save-load-analyze" in cmd:
        if '"analyze"' in text and '"file_count"' in text:
            return "collecting artifact analysis", 0.92
        if '"load"' in text:
            return "reloading saved model bundle", 0.7
        if '"save"' in text:
            return "saving model bundle", 0.45
        return "running save/load/analyze", 0.25

    if "db-init" in cmd:
        applied_count = text.count("applied:")
        if applied_count > 0:
            return f"applying sql files ({applied_count})", min(0.9, 0.2 + 0.15 * applied_count)
        return "initializing database schema", 0.2

    return "running command", 0.2


def _run_shell_command_live(
    command: str, cwd: Path, timeout_sec: int = 600, title: str = "Running Command"
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    action_name = _notification_action_name(title, command)
    _publish_notification(
        kind=NotificationEventKind.ACTION_CONFIRMED,
        severity=NotificationSeverity.RUNNING,
        title=f"{title} を受け付けました",
        message="通知処理を開始しました。次はコマンド起動と進捗監視に進みます。",
        action=action_name,
        status="accepted",
        command_summary=str(command),
    )
    _publish_notification(
        kind=NotificationEventKind.OPERATION_START,
        severity=NotificationSeverity.RUNNING,
        title=f"{title} を開始しました",
        message="処理を実行中です。完了まで待つか、進捗ログを確認してください。",
        action=action_name,
        status="running",
        command_summary=str(command),
    )
    if int(timeout_sec) >= 180:
        _publish_notification(
            kind=NotificationEventKind.LONG_RUNNING_START,
            severity=NotificationSeverity.RUNNING,
            title=f"{title} は長時間処理です",
            message="長時間処理として監視します。完了時に追加通知します。",
            action=action_name,
            status="running",
            command_summary=str(command),
        )
    _log_dashboard_event(
        "command_start",
        {"title": title, "command": str(command), "cwd": str(cwd), "timeout_sec": int(timeout_sec)},
    )
    progress = st.progress(0, text=f"{title}: initializing")
    status_box = st.empty()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    max_lines = 6000
    timed_out = False
    last_log_ts: float | None = None

    try:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception:
        progress.progress(1.0, text=f"{title}: failed to start")
        err = traceback.format_exc()
        status_box.error(err)
        _publish_notification(
            kind=NotificationEventKind.EXCEPTION,
            severity=NotificationSeverity.FAILURE,
            title=f"{title} の起動に失敗しました",
            message="本処理は開始できませんでした。stderr と環境設定を確認してください。",
            action=action_name,
            status="failed",
            command_summary=str(command),
            error_summary=err[:1000],
        )
        _log_dashboard_event(
            "command_start_failed",
            {"title": title, "command": str(command), "cwd": str(cwd), "error": err[:1000]},
            level="ERROR",
        )
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": err,
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

    selector = selectors.DefaultSelector()
    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    if proc.stderr is not None:
        selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    last_ui = 0.0
    timeout_sec = max(1, int(timeout_sec))
    while True:
        now = time.time()
        elapsed = now - started.timestamp()
        if elapsed > timeout_sec:
            timed_out = True
            proc.kill()
            break

        events = selector.select(timeout=0.2)
        for key, _ in events:
            stream_name = str(key.data)
            stream_obj: Any = key.fileobj
            line = str(stream_obj.readline())
            if line == "":
                with contextlib.suppress(Exception):
                    selector.unregister(key.fileobj)
                continue
            if stream_name == "stdout":
                stdout_lines.append(line.rstrip("\n"))
                if len(stdout_lines) > max_lines:
                    stdout_lines = stdout_lines[-max_lines:]
            else:
                stderr_lines.append(line.rstrip("\n"))
                if len(stderr_lines) > max_lines:
                    stderr_lines = stderr_lines[-max_lines:]
            last_log_ts = now

        if now - last_ui >= 0.35:
            stage_label, stage_hint = _estimate_command_stage(command, stdout_lines, stderr_lines)
            wall_ratio = min(0.92, max(0.01, elapsed / float(timeout_sec)))
            ratio = min(0.95, max(stage_hint, wall_ratio * 0.8))
            progress.progress(
                ratio,
                text=f"{title}: {stage_label} / elapsed={elapsed:.1f}s timeout={timeout_sec}s",
            )
            out_tail = _tail_join(stdout_lines, max_lines=10)
            err_tail = _tail_join(stderr_lines, max_lines=6)
            lag = (now - last_log_ts) if last_log_ts is not None else None
            lag_text = f"{lag:.1f}s ago" if lag is not None else "no logs yet"
            status_text = (
                f"stage: {stage_label}\n"
                f"cwd: {cwd}\n"
                f"stdout_lines={len(stdout_lines)} stderr_lines={len(stderr_lines)} last_log={lag_text}\n"
            )
            if out_tail:
                status_text += "\n[stdout tail]\n" + out_tail
            if err_tail:
                status_text += "\n\n[stderr tail]\n" + err_tail
            status_box.code(status_text[:8000], language="bash")
            last_ui = now

        if proc.poll() is not None and not events and len(selector.get_map()) > 0:
            for sk in list(selector.get_map().values()):
                stream_name = str(sk.data)
                try:
                    stream_read_obj: Any = sk.fileobj
                    tail = str(stream_read_obj.read())
                except Exception:
                    tail = ""
                if tail:
                    lines = tail.splitlines()
                    if stream_name == "stdout":
                        stdout_lines.extend(lines)
                        if len(stdout_lines) > max_lines:
                            stdout_lines = stdout_lines[-max_lines:]
                    else:
                        stderr_lines.extend(lines)
                        if len(stderr_lines) > max_lines:
                            stderr_lines = stderr_lines[-max_lines:]
                with contextlib.suppress(Exception):
                    selector.unregister(sk.fileobj)

        if proc.poll() is not None and len(selector.get_map()) == 0:
            break

    try:
        if proc.stdout is not None:
            rem_out = proc.stdout.read() or ""
            if rem_out:
                stdout_lines.extend(rem_out.splitlines())
        if proc.stderr is not None:
            rem_err = proc.stderr.read() or ""
            if rem_err:
                stderr_lines.extend(rem_err.splitlines())
    except Exception:
        pass

    proc_returncode = proc.poll()
    returncode = proc_returncode if proc_returncode is not None else -9
    ended = datetime.now(timezone.utc)
    ok = (returncode == 0) and (not timed_out)
    if timed_out:
        stderr_lines.append(f"timeout after {timeout_sec}s")

    progress.progress(1.0, text=f"{title}: {'success' if ok else 'failed'} (rc={returncode})")
    elapsed_sec = float((ended - started).total_seconds())
    _log_dashboard_event(
        "command_end",
        {
            "title": title,
            "command": str(command),
            "cwd": str(cwd),
            "ok": bool(ok),
            "returncode": int(returncode),
            "elapsed_sec": elapsed_sec,
            "stdout_lines": int(len(stdout_lines)),
            "stderr_lines": int(len(stderr_lines)),
        },
        level="INFO" if ok else "ERROR",
    )
    artifact_candidates = []
    for line in stdout_lines[-40:]:
        if "/" in line and ("artifacts" in line or "saved_models" in line):
            artifact_candidates.append(line.strip())
    artifact_candidates = artifact_candidates[:5]
    _publish_notification(
        kind=(
            NotificationEventKind.OPERATION_SUCCESS
            if ok
            else NotificationEventKind.OPERATION_FAILURE
        ),
        severity=NotificationSeverity.SUCCESS if ok else NotificationSeverity.FAILURE,
        title=f"{title} が{'完了' if ok else '失敗'}しました",
        message=(
            "結果サマリ、生成物、次の推奨操作を確認してください。"
            if ok
            else "本処理は失敗しました。原因、影響、対処を結果欄で確認してください。"
        ),
        action=action_name,
        status="success" if ok else "failed",
        command_summary=str(command),
        error_summary="\n".join(stderr_lines[-8:])[:1000],
        artifact_paths=artifact_candidates,
    )
    if int(timeout_sec) >= 180:
        _publish_notification(
            kind=NotificationEventKind.LONG_RUNNING_COMPLETE,
            severity=NotificationSeverity.SUCCESS if ok else NotificationSeverity.WARNING,
            title=f"{title} の長時間処理が終了しました",
            message="長時間処理の監視を終了しました。結果サマリを確認してください。",
            action=action_name,
            status="success" if ok else "warning",
            command_summary=str(command),
            error_summary="\n".join(stderr_lines[-8:])[:1000],
            artifact_paths=artifact_candidates,
        )
    with contextlib.suppress(Exception):
        selector.close()
    return {
        "ok": ok,
        "returncode": returncode,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_sec": elapsed_sec,
    }


def _run_command_sequence_live(
    commands: list[str],
    cwd: Path,
    timeout_sec_per_command: int = 600,
    stop_on_error: bool = True,
) -> list[dict[str, Any]]:
    _log_dashboard_event(
        "sequence_start",
        {
            "steps": int(len(commands)),
            "cwd": str(cwd),
            "timeout_sec_per_command": int(timeout_sec_per_command),
            "stop_on_error": bool(stop_on_error),
        },
    )
    results: list[dict[str, Any]] = []
    total = max(1, len(commands))
    overall = st.progress(0.0, text=f"sequence: 0/{total}")
    summary_box = st.empty()
    success_count = 0
    failed_count = 0
    for i, cmd in enumerate(commands, start=1):
        overall.progress((i - 1) / total, text=f"sequence: {i - 1}/{total}")
        summary_box.info(f"running {i}/{total} | success={success_count} failed={failed_count}")
        st.markdown(f"**[{i}/{total}]** `{cmd}`")
        res = _run_shell_command_live(
            cmd,
            cwd=cwd,
            timeout_sec=timeout_sec_per_command,
            title=f"step {i}/{total}",
        )
        res["command"] = cmd
        results.append(res)
        if bool(res.get("ok", False)):
            success_count += 1
        else:
            failed_count += 1
        if not res.get("ok", False) and stop_on_error:
            break
    overall.progress(1.0, text=f"sequence: done (success={success_count} failed={failed_count})")
    if failed_count > 0:
        summary_box.warning(f"sequence finished with failures | success={success_count} failed={failed_count}")
    else:
        summary_box.success(f"sequence finished successfully | steps={success_count}")
    _log_dashboard_event(
        "sequence_end",
        {"success_count": int(success_count), "failed_count": int(failed_count), "steps": int(len(results))},
        level="INFO" if failed_count == 0 else "WARN",
    )
    return results


def _insert_nf_automodel_fatal_log(
    engine: Engine | None,
    *,
    config_id: int | None,
    config_name: str,
    model_name: str,
    horizon: int,
    error_message: str,
) -> str | None:
    if engine is None:
        return None
    run_id = f"fatal_cfg{int(config_id) if config_id is not None else 0}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    sql = text(
        """
        INSERT INTO model.nf_automodel (
            config_id, run_id, status, model_name, horizon,
            params_json, diagnostics_json, error_message, started_at, ended_at
        ) VALUES (
            :config_id, :run_id, 'failed', :model_name, :horizon,
            CAST(:params_json AS jsonb), CAST(:diagnostics_json AS jsonb), :error_message, now(), now()
        )
        """
    )
    params_json = json.dumps({}, ensure_ascii=False)
    diagnostics_json = json.dumps(
        {
            "source": "operations_dashboard",
            "type": "meta_automodel_fatal",
            "config_name": str(config_name or ""),
        },
        ensure_ascii=False,
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "config_id": int(config_id) if config_id is not None else None,
                "run_id": run_id,
                "model_name": str(model_name or "unknown"),
                "horizon": max(1, int(horizon)),
                "params_json": params_json,
                "diagnostics_json": diagnostics_json,
                "error_message": str(error_message or "")[:8000],
            },
        )
    return run_id


def _run_meta_automodel_configs_live(
    *,
    configs: list[dict[str, Any]],
    ensure_db_init: bool,
    skip_existing_success: bool,
    engine: Engine | None,
) -> list[dict[str, Any]]:
    from loto_forecast.orchestration.meta_automodel import run_meta_automodel  # noqa: PLC0415

    targets = [dict(x) for x in configs if x.get("config_id") is not None]
    total = max(1, len(targets))
    p = st.progress(0.0, text=f"meta-automodel batch: 0/{total}")
    status_box = st.empty()
    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(targets, start=1):
        cfg_id_raw = cfg.get("config_id")
        if cfg_id_raw is None:
            continue
        cfg_id = int(cfg_id_raw)
        cfg_name = str(cfg.get("config_name", ""))
        model_name = str(cfg.get("model_name", "unknown"))
        horizon = int(cfg.get("horizon", 1) or 1)
        p.progress((i - 1) / total, text=f"meta-automodel batch: {i - 1}/{total}")
        status_box.info(f"running {i}/{total}: config_id={cfg_id} config_name={cfg_name}")
        started = datetime.now(timezone.utc)
        try:
            out = run_meta_automodel(
                config_id=cfg_id,
                limit=1,
                stop_on_error=False,
                ensure_db_init=bool(ensure_db_init and i == 1),
                skip_existing_success=bool(skip_existing_success),
            )
            rows.append(
                {
                    "config_id": cfg_id,
                    "config_name": cfg_name,
                    "ok": True,
                    "executed": int(out.get("executed", 0) or 0),
                    "success": int(out.get("success", 0) or 0),
                    "failed": int(out.get("failed", 0) or 0),
                    "skipped": int(out.get("skipped", 0) or 0),
                    "stopped_on_error": bool(out.get("stopped_on_error", False)),
                    "error": "",
                    "fatal_log_run_id": None,
                    "elapsed_sec": float((datetime.now(timezone.utc) - started).total_seconds()),
                }
            )
        except Exception as e:
            fatal_run_id: str | None = None
            try:
                fatal_run_id = _insert_nf_automodel_fatal_log(
                    engine,
                    config_id=cfg_id,
                    config_name=cfg_name,
                    model_name=model_name,
                    horizon=horizon,
                    error_message=str(e),
                )
            except Exception:
                fatal_run_id = None
            rows.append(
                {
                    "config_id": cfg_id,
                    "config_name": cfg_name,
                    "ok": False,
                    "executed": 0,
                    "success": 0,
                    "failed": 1,
                    "skipped": 0,
                    "stopped_on_error": False,
                    "error": str(e),
                    "fatal_log_run_id": fatal_run_id,
                    "elapsed_sec": float((datetime.now(timezone.utc) - started).total_seconds()),
                }
            )
    ok_n = int(sum(1 for r in rows if bool(r.get("ok", False))))
    ng_n = int(len(rows) - ok_n)
    p.progress(1.0, text=f"meta-automodel batch done: success={ok_n} failed={ng_n}")
    if ng_n > 0:
        status_box.warning(f"done with failures: success={ok_n} failed={ng_n}")
    else:
        status_box.success(f"done: all {ok_n} configs processed")
    return rows


def _run_commands_parallel(
    commands: list[str],
    cwd: Path,
    timeout_sec: int = 600,
    max_workers: int = 2,
) -> list[dict[str, Any]]:
    cmds = [str(c).strip() for c in commands if str(c).strip()]
    if not cmds:
        return []
    workers = max(1, min(int(max_workers), len(cmds)))
    _log_dashboard_event(
        "parallel_commands_start",
        {"count": int(len(cmds)), "max_workers": int(workers), "cwd": str(cwd)},
    )
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_shell_command, c, cwd, int(timeout_sec)): c for c in cmds}
        for fut in as_completed(futs):
            cmd = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "ok": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }
            res["command"] = cmd
            out.append(res)
    ok_n = int(sum(1 for r in out if bool(r.get("ok", False))))
    ng_n = int(len(out) - ok_n)
    _log_dashboard_event(
        "parallel_commands_end",
        {"success_count": ok_n, "failed_count": ng_n, "count": int(len(out))},
        level="INFO" if ng_n == 0 else "WARN",
    )
    return out


def _fetch_runner_live_status(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    config_name: str,
    row_limit: int = 30,
) -> dict[str, Any]:
    try:
        dsn = _dsn(host=host, port=int(port), user=user, password=password, database=database)
        engine = _engine_for_dsn(dsn)
        meta_df = _query_df(
            engine,
            """
            SELECT
              config_id, config_name, active, last_status, last_run_id, last_run_at,
              run_predict, run_evaluate, run_explain, run_save, run_load, run_analyze,
              output_schema, output_table, updated_at
            FROM meta.nf_automodel
            WHERE config_name = :config_name
            ORDER BY config_id DESC
            LIMIT 1
            """,
            {"config_name": str(config_name)},
        )
        model_df = _query_df(
            engine,
            """
            SELECT
              result_id, config_id, run_id, status, model_name, horizon,
              dataset_rows, feature_cols, model_store_path, error_message,
              started_at, ended_at, created_at
            FROM model.nf_automodel
            WHERE config_id = COALESCE((SELECT config_id FROM meta.nf_automodel WHERE config_name = :config_name ORDER BY config_id DESC LIMIT 1), -1)
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"config_name": str(config_name), "limit": int(max(1, row_limit))},
        )
        success_count = (
            int((model_df.get("status", pd.Series(dtype=str)).astype(str) == "success").sum())
            if not model_df.empty
            else 0
        )
        failed_count = (
            int((model_df.get("status", pd.Series(dtype=str)).astype(str) == "failed").sum())
            if not model_df.empty
            else 0
        )
        latest_model = model_df.iloc[0].to_dict() if not model_df.empty else {}
        latest_meta = meta_df.iloc[0].to_dict() if not meta_df.empty else {}
        return {
            "ok": True,
            "config_name": str(config_name),
            "meta": latest_meta,
            "model_rows": model_df,
            "latest_model": latest_model,
            "success_count": success_count,
            "failed_count": failed_count,
        }
    except Exception as e:
        return {
            "ok": False,
            "config_name": str(config_name),
            "error": str(e),
            "meta": {},
            "model_rows": pd.DataFrame(),
            "latest_model": {},
            "success_count": 0,
            "failed_count": 0,
        }


def _start_background_command(command: str, cwd: Path) -> dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "background"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"bg_{ts}.out.log"
    err_path = log_dir / f"bg_{ts}.err.log"
    try:
        out_f = out_path.open("w", encoding="utf-8")
        err_f = err_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
        return {
            "ok": True,
            "pid": int(proc.pid),
            "command": command,
            "cwd": str(cwd),
            "stdout_log": str(out_path),
            "stderr_log": str(err_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return {
            "ok": False,
            "pid": -1,
            "command": command,
            "cwd": str(cwd),
            "stdout_log": str(out_path),
            "stderr_log": str(err_path),
            "error": traceback.format_exc(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }


def _write_text_file(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        os.chmod(path, mode | 0o111)


def _snapshot_schema(engine: Engine, database: str) -> dict[str, Any]:
    tables_df = _table_catalog(engine)
    cols_df = _query_df(
        engine,
        """
        SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema IN ('dataset', 'exog', 'resources', 'meta', 'model')
        ORDER BY table_schema, table_name, ordinal_position
        """,
    )

    table_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tables_df.itertuples(index=False):
        table_meta[(str(row.table_schema), str(row.table_name))] = {
            "estimated_rows": int(row.est_rows) if pd.notna(row.est_rows) else 0,
            "table_size_bytes": int(row.total_bytes) if pd.notna(row.total_bytes) else 0,
            "table_size": str(row.table_size) if "table_size" in tables_df.columns else _format_bytes(row.total_bytes),
        }

    schemas_payload: list[dict[str, Any]] = []
    for schema in SCHEMAS:
        tables_payload: list[dict[str, Any]] = []
        schema_cols = cols_df[cols_df["table_schema"] == schema]
        for table in sorted(schema_cols["table_name"].unique().tolist()):
            cols = schema_cols[schema_cols["table_name"] == table]
            columns_payload = [
                {
                    "name": str(r.column_name),
                    "data_type": str(r.data_type),
                    "is_nullable": str(r.is_nullable),
                    "ordinal_position": int(r.ordinal_position),
                }
                for r in cols.itertuples(index=False)
            ]
            meta = table_meta.get((schema, table), {})
            tables_payload.append(
                {
                    "table": table,
                    "estimated_rows": int(meta.get("estimated_rows", 0)),
                    "table_size_bytes": int(meta.get("table_size_bytes", 0)),
                    "table_size": str(meta.get("table_size", "n/a")),
                    "column_count": len(columns_payload),
                    "columns": columns_payload,
                }
            )

        schemas_payload.append(
            {
                "schema": schema,
                "table_count": len(tables_payload),
                "tables": tables_payload,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": database,
        "schemas": schemas_payload,
    }


def _snapshot_flat_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schema_obj in snapshot.get("schemas", []):
        schema = str(schema_obj.get("schema", ""))
        for table_obj in schema_obj.get("tables", []):
            table = str(table_obj.get("table", ""))
            est_rows = int(table_obj.get("estimated_rows", 0))
            size = str(table_obj.get("table_size", "n/a"))
            cols = table_obj.get("columns", [])
            if not cols:
                rows.append(
                    {
                        "schema": schema,
                        "table": table,
                        "estimated_rows": est_rows,
                        "table_size": size,
                        "column_name": "",
                        "data_type": "",
                        "is_nullable": "",
                        "ordinal_position": "",
                    }
                )
                continue
            for c in cols:
                rows.append(
                    {
                        "schema": schema,
                        "table": table,
                        "estimated_rows": est_rows,
                        "table_size": size,
                        "column_name": str(c.get("name", "")),
                        "data_type": str(c.get("data_type", "")),
                        "is_nullable": str(c.get("is_nullable", "")),
                        "ordinal_position": int(c.get("ordinal_position", 0)),
                    }
                )
    return rows


def _snapshot_to_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        f"# Schema Snapshot ({snapshot.get('database', '')})",
        "",
        f"- generated_at: {snapshot.get('generated_at', '')}",
        "",
    ]
    for schema_obj in snapshot.get("schemas", []):
        lines.append(f"## {schema_obj.get('schema', '')}")
        lines.append(f"- table_count: {schema_obj.get('table_count', 0)}")
        lines.append("")
        for table_obj in schema_obj.get("tables", []):
            lines.append(f"### {table_obj.get('table', '')}")
            lines.append(
                f"- estimated_rows: {table_obj.get('estimated_rows', 0)} / size: {table_obj.get('table_size', 'n/a')}"
            )
            lines.append("")
            lines.append("| ordinal | column | data_type | nullable |")
            lines.append("|---:|---|---|---|")
            for c in table_obj.get("columns", []):
                lines.append(
                    f"| {c.get('ordinal_position', '')} | {c.get('name', '')} | {c.get('data_type', '')} | {c.get('is_nullable', '')} |"
                )
            lines.append("")
    return "\n".join(lines)


def _snapshot_to_html(snapshot: dict[str, Any]) -> str:
    flat = pd.DataFrame(_snapshot_flat_rows(snapshot))
    title = html.escape(f"Schema Snapshot - {snapshot.get('database', '')}")
    generated = html.escape(str(snapshot.get("generated_at", "")))
    table_html = flat.to_html(index=False, escape=True)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;padding:16px;}}
table{{border-collapse:collapse;width:100%;font-size:13px;}}
th,td{{border:1px solid #ddd;padding:6px;}}
th{{background:#f5f5f5;}}
</style></head>
<body>
<h1>{title}</h1>
<p>generated_at: {generated}</p>
{table_html}
</body></html>"""


def _snapshot_to_format(snapshot: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(snapshot, ensure_ascii=False, indent=2), "application/json", "json"
    if fmt == "yaml":
        return yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), "application/x-yaml", "yaml"
    if fmt == "csv":
        df = pd.DataFrame(_snapshot_flat_rows(snapshot))
        return df.to_csv(index=False), "text/csv", "csv"
    if fmt == "md":
        return _snapshot_to_markdown(snapshot), "text/markdown", "md"
    if fmt == "html":
        return _snapshot_to_html(snapshot), "text/html", "html"
    raise ValueError(f"unsupported format: {fmt}")


def _all_user_tables(engine: Engine) -> pd.DataFrame:
    return _query_df(
        engine,
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
        """,
    )


def _selected_schema_snapshot(
    engine: Engine,
    database: str,
    selected_tables: list[tuple[str, str]],
) -> dict[str, Any]:
    unique_tables = sorted({(str(s), str(t)) for s, t in selected_tables})
    if not unique_tables:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database": database,
            "schemas": [],
        }

    meta_df = _query_df(
        engine,
        """
        SELECT
          n.nspname AS table_schema,
          c.relname AS table_name,
          COALESCE(s.n_live_tup::bigint, c.reltuples::bigint, 0) AS est_rows,
          pg_total_relation_size(c.oid) AS total_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE c.relkind='r'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY n.nspname, c.relname
        """,
    )
    table_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in meta_df.itertuples(index=False):
        table_meta[(str(row.table_schema), str(row.table_name))] = {
            "estimated_rows": int(row.est_rows) if pd.notna(row.est_rows) else 0,
            "table_size_bytes": int(row.total_bytes) if pd.notna(row.total_bytes) else 0,
            "table_size": _format_bytes(row.total_bytes),
        }

    by_schema: dict[str, list[str]] = defaultdict(list)
    for schema, table in unique_tables:
        by_schema[schema].append(table)

    schemas_payload: list[dict[str, Any]] = []
    for schema in sorted(by_schema.keys()):
        table_payloads: list[dict[str, Any]] = []
        for table in sorted(set(by_schema[schema])):
            cols_df = _table_columns(engine, schema, table)
            columns_payload = [
                {
                    "name": str(r.column_name),
                    "data_type": str(r.data_type),
                    "is_nullable": str(r.is_nullable),
                    "ordinal_position": int(r.ordinal_position),
                }
                for r in cols_df.itertuples(index=False)
            ]
            meta = table_meta.get((schema, table), {})
            table_payloads.append(
                {
                    "table": table,
                    "estimated_rows": int(meta.get("estimated_rows", 0)),
                    "table_size_bytes": int(meta.get("table_size_bytes", 0)),
                    "table_size": str(meta.get("table_size", "n/a")),
                    "column_count": len(columns_payload),
                    "columns": columns_payload,
                }
            )
        schemas_payload.append(
            {
                "schema": schema,
                "table_count": len(table_payloads),
                "tables": table_payloads,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": database,
        "schemas": schemas_payload,
    }


_dump_selector = dashboard_helpers.dump_selector
_dump_selector_flags = dashboard_helpers.dump_selector_flags


def _build_backup_restore_bundle(
    engine: Engine,
    *,
    host: str,
    port: int,
    user: str,
    database: str,
    selected_schemas: list[str],
    selected_tables: list[tuple[str, str]],
) -> dict[str, str]:
    mode = "tables" if selected_tables else "schemas"
    selector = _dump_selector(mode=mode, schemas=selected_schemas, tables=selected_tables)
    selector_flags = _dump_selector_flags(selector)
    selector_clause = f" {selector_flags}" if selector_flags else ""
    conn_flags = (
        f"--host {shlex.quote(str(host))} "
        f"--port {int(port)} "
        f"--username {shlex.quote(str(user))} "
        f"--dbname {shlex.quote(str(database))}"
    )

    db_slug = _slug(database)
    dump_file = f"{db_slug}_data.dump"
    schema_file = f"{db_slug}_schema.sql"
    manifest_file = f"{db_slug}_manifest.json"

    snapshot_tables = list(selected_tables)
    if mode == "schemas":
        all_tables = _all_user_tables(engine)
        selected_schema_set = {str(x) for x in selected_schemas}
        snapshot_tables = [
            (str(r.table_schema), str(r.table_name))
            for r in all_tables.itertuples(index=False)
            if str(r.table_schema) in selected_schema_set
        ]
    selected_snapshot = _selected_schema_snapshot(engine, database=database, selected_tables=snapshot_tables)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "connection": {
            "host": str(host),
            "port": int(port),
            "user": str(user),
            "database": str(database),
        },
        "selector": selector,
        "execution_env": detect_execution_environment(),
        "snapshot": selected_snapshot,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)

    backup_script = f"""#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${{PGPASSWORD:-}}" ]]; then
  echo "PGPASSWORD is not set. export PGPASSWORD before backup." >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${{1:-./db_backup_{db_slug}_${{TS}}}}"
mkdir -p "$OUT_DIR"

pg_dump {conn_flags} --format=custom --no-owner --no-privileges{selector_clause} --file "$OUT_DIR/{dump_file}"
pg_dump {conn_flags} --schema-only --no-owner --no-privileges{selector_clause} --file "$OUT_DIR/{schema_file}"

cat > "$OUT_DIR/{manifest_file}" <<'JSON'
{manifest_json}
JSON

echo "backup files written under: $OUT_DIR"
"""

    restore_script = f"""#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${{PGPASSWORD:-}}" ]]; then
  echo "PGPASSWORD is not set. export PGPASSWORD before restore." >&2
  exit 1
fi

BACKUP_DIR="${{1:-./db_backup_{db_slug}}}"
SCHEMA_SQL="${{2:-$BACKUP_DIR/{schema_file}}}"
DATA_DUMP="${{3:-$BACKUP_DIR/{dump_file}}}"
MANIFEST_JSON="${{4:-$BACKUP_DIR/{manifest_file}}}"

if [[ -f "$MANIFEST_JSON" ]]; then
  echo "using manifest: $MANIFEST_JSON"
fi

psql --host {shlex.quote(str(host))} --port {int(port)} --username {shlex.quote(str(user))} --dbname {shlex.quote(str(database))} -v ON_ERROR_STOP=1 --file "$SCHEMA_SQL"
pg_restore --host {shlex.quote(str(host))} --port {int(port)} --username {shlex.quote(str(user))} --dbname {shlex.quote(str(database))} --clean --if-exists --no-owner --no-privileges "$DATA_DUMP"

echo "restore completed from: $BACKUP_DIR"
"""

    return {
        "manifest_json": manifest_json,
        "backup_script": backup_script,
        "restore_script": restore_script,
    }


_read_text_file = dashboard_helpers.read_text_file
_scan_supported_files = dashboard_helpers.scan_supported_files
_scan_diff_files = dashboard_helpers.scan_diff_files
_unified_diff_text = dashboard_helpers.unified_diff_text
_summarize_supported_file = dashboard_helpers.summarize_supported_file
_compile_directory_payload = dashboard_helpers.compile_directory_payload
_scan_markdown_files = dashboard_helpers.scan_markdown_files
_compile_markdown_bundle = dashboard_helpers.compile_markdown_bundle
_compiled_to_markdown = dashboard_helpers.compiled_to_markdown
_compiled_to_html = dashboard_helpers.compiled_to_html
_compiled_to_format = dashboard_helpers.compiled_to_format
_module_name_from_path = dashboard_helpers.module_name_from_path
_resolve_from_import = dashboard_helpers.resolve_from_import


def _analyze_python_codebase(src_root: Path, scripts_root: Path) -> dict[str, Any]:
    py_files: list[tuple[Path, str, Path]] = []
    if src_root.exists():
        for p in sorted(src_root.rglob("*.py")):
            py_files.append((p, "src", src_root))
    if scripts_root.exists():
        for p in sorted(scripts_root.rglob("*.py")):
            py_files.append((p, "scripts", scripts_root))

    modules: list[dict[str, Any]] = []
    funcs: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    edges: Counter[tuple[str, str]] = Counter()
    call_names: Counter[str] = Counter()

    internal_prefixes = ("loto_forecast", "resources", "scripts")

    for path, group, root in py_files:
        module_name = (
            _module_name_from_path(path, root, "scripts")
            if group == "scripts"
            else _module_name_from_path(path, root, "")
        )
        module_name = module_name.lstrip(".")
        source = path.read_text(encoding="utf-8", errors="replace")
        line_count = source.count("\n") + 1
        imports: set[str] = set()
        func_count = 0
        class_count = 0

        try:
            tree = ast.parse(source)
        except Exception:
            modules.append(
                {
                    "module": module_name,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "group": group,
                    "lines": int(line_count),
                    "functions": 0,
                    "classes": 0,
                    "imports": [],
                }
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith(internal_prefixes):
                        imports.add(name)
                        edges[(module_name, name)] += 1
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_from_import(module_name, int(node.level), node.module)
                if resolved.startswith(internal_prefixes):
                    imports.add(resolved)
                    edges[(module_name, resolved)] += 1
            elif isinstance(node, ast.FunctionDef):
                func_count += 1
                funcs.append(
                    {
                        "module": module_name,
                        "function": node.name,
                        "lineno": int(node.lineno),
                        "arg_count": len(node.args.args),
                    }
                )
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        if isinstance(sub.func, ast.Name):
                            call_names[sub.func.id] += 1
                        elif isinstance(sub.func, ast.Attribute):
                            call_names[sub.func.attr] += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
                classes.append(
                    {
                        "module": module_name,
                        "class": node.name,
                        "lineno": int(node.lineno),
                    }
                )

        modules.append(
            {
                "module": module_name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "group": group,
                "lines": int(line_count),
                "functions": int(func_count),
                "classes": int(class_count),
                "imports": sorted(list(imports)),
            }
        )

    edge_rows = [
        {"src": src, "dst": dst, "weight": int(weight)} for (src, dst), weight in edges.most_common() if src and dst
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module_count": len(modules),
        "function_count": len(funcs),
        "class_count": len(classes),
        "modules": modules,
        "functions": funcs,
        "classes": classes,
        "edges": edge_rows,
        "top_call_names": [{"name": k, "count": int(v)} for k, v in call_names.most_common(40)],
    }


def _analyze_python_project(root: Path, max_py_files: int = 2500) -> dict[str, Any]:
    py_files = sorted([p for p in root.rglob("*.py") if p.is_file()])[: max(1, int(max_py_files))]
    modules: list[dict[str, Any]] = []
    funcs: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    edges: Counter[tuple[str, str]] = Counter()
    call_names: Counter[str] = Counter()
    module_names: dict[Path, str] = {}
    for p in py_files:
        rel = p.relative_to(root).with_suffix("")
        mod = ".".join(rel.parts)
        module_names[p] = mod
    top_packages = {m.split(".")[0] for m in module_names.values() if m}

    for p in py_files:
        mod = module_names[p]
        source = p.read_text(encoding="utf-8", errors="replace")
        lines = source.count("\n") + 1
        imports: set[str] = set()
        fcnt = 0
        ccnt = 0
        try:
            tree = ast.parse(source)
        except Exception:
            modules.append(
                {
                    "module": mod,
                    "path": str(p),
                    "group": "external",
                    "lines": int(lines),
                    "functions": 0,
                    "classes": 0,
                    "imports": [],
                }
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name and name.split(".")[0] in top_packages:
                        imports.add(name)
                        edges[(mod, name)] += 1
            elif isinstance(node, ast.ImportFrom):
                if int(node.level) > 0:
                    resolved = _resolve_from_import(mod, int(node.level), node.module)
                else:
                    resolved = str(node.module or "")
                if resolved and resolved.split(".")[0] in top_packages:
                    imports.add(resolved)
                    edges[(mod, resolved)] += 1
            elif isinstance(node, ast.FunctionDef):
                fcnt += 1
                funcs.append(
                    {
                        "module": mod,
                        "function": node.name,
                        "lineno": int(node.lineno),
                        "arg_count": len(node.args.args),
                    }
                )
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        if isinstance(sub.func, ast.Name):
                            call_names[sub.func.id] += 1
                        elif isinstance(sub.func, ast.Attribute):
                            call_names[sub.func.attr] += 1
            elif isinstance(node, ast.ClassDef):
                ccnt += 1
                classes.append({"module": mod, "class": node.name, "lineno": int(node.lineno)})

        modules.append(
            {
                "module": mod,
                "path": str(p),
                "group": "external",
                "lines": int(lines),
                "functions": int(fcnt),
                "classes": int(ccnt),
                "imports": sorted(list(imports)),
            }
        )

    edge_rows = [{"src": src, "dst": dst, "weight": int(w)} for (src, dst), w in edges.most_common() if src and dst]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "module_count": len(modules),
        "function_count": len(funcs),
        "class_count": len(classes),
        "modules": modules,
        "functions": funcs,
        "classes": classes,
        "edges": edge_rows,
        "top_call_names": [{"name": k, "count": int(v)} for k, v in call_names.most_common(40)],
    }


@st.cache_data(show_spinner=False)
def _cached_external_analysis(root_path: str, max_py_files: int) -> dict[str, Any]:
    return _analyze_python_project(Path(root_path), max_py_files=max_py_files)


def _find_streamlit_apps(root: Path, max_files: int = 300) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.py")):
        if len(out) >= max_files:
            break
        name = p.name.lower()
        if name == "streamlit_app.py" or name == "app.py":
            out.append(p)
    return out


def _render_external_target(name: str, root: Path) -> None:
    st.markdown(f"### 外部ターゲット: {name}")
    st.code(str(root))
    if not root.exists() or not root.is_dir():
        st.error("対象ディレクトリが見つかりません。")
        return

    depth = st.slider("ツリー深度", min_value=2, max_value=8, value=4, step=1, key=f"ext_depth_{name}")
    st.code(_tree_lines(root, max_depth=depth, max_entries=900))

    max_files = st.slider(
        "走査ファイル上限",
        min_value=100,
        max_value=5000,
        value=1200,
        step=100,
        key=f"ext_max_files_{name}",
    )
    files = _scan_supported_files(root, max_files=max_files)
    st.write(f"検出テキスト系ファイル数: {len(files)}")
    if files:
        rows = [
            {
                "path": str(p.relative_to(root)),
                "suffix": p.suffix.lower(),
                "size_bytes": int(p.stat().st_size),
                "size_human": _format_bytes(p.stat().st_size),
            }
            for p in files
        ]
        files_df = pd.DataFrame(rows)
        _show_df(files_df.head(1500), hide_index=True)
        selected_rel = st.selectbox("プレビューファイル", files_df["path"].tolist(), index=0, key=f"ext_preview_{name}")
        selected_path = root / selected_rel
        summary = _summarize_supported_file(selected_path)
        st.json({"meta": summary.get("meta", {}), "size": summary.get("size_human", "")})
        preview = str(summary.get("preview", ""))
        if selected_path.suffix.lower() == ".csv":
            try:
                _show_df(pd.read_csv(selected_path, nrows=300), hide_index=True)
            except Exception:
                _render_rich_content(preview[:50000], ext=selected_path.suffix, key=f"ext_prev_{name}")
        else:
            _render_rich_content(preview[:50000], ext=selected_path.suffix, key=f"ext_prev_{name}")
        _render_read_aloud(preview, key=f"ext_speech_{name}")

        if st.button("ターゲットを集約コンパイル", key=f"compile_ext_{name}"):
            st.session_state[f"compiled_ext_{name}"] = _compile_directory_payload(root, files)

        bundle = st.session_state.get(f"compiled_ext_{name}")
        if bundle:
            st.markdown("**集約サマリ**")
            st.json(
                {
                    "root": bundle.get("root"),
                    "generated_at": bundle.get("generated_at"),
                    "file_count": bundle.get("file_count"),
                    "suffix_counts": bundle.get("suffix_counts"),
                }
            )
            fmt = st.selectbox(
                "出力形式",
                ["json", "csv", "yaml", "md", "html"],
                index=0,
                key=f"ext_fmt_{name}",
            )
            content, mime, ext = _compiled_to_format(bundle, fmt)
            _render_export_controls(content, mime, ext, filename_base=f"{name}_compiled", key=f"{name}_compiled_export")
    else:
        st.info("対応ファイルが見つかりません。")

    st.markdown("**Pythonコード解析**")
    max_py = st.slider(
        "Python解析上限ファイル数", min_value=50, max_value=4000, value=1500, step=50, key=f"ext_py_{name}"
    )
    analysis = _cached_external_analysis(str(root), int(max_py))
    st.json(
        {
            "generated_at": analysis.get("generated_at"),
            "module_count": analysis.get("module_count"),
            "function_count": analysis.get("function_count"),
            "class_count": analysis.get("class_count"),
            "edge_count": len(analysis.get("edges", [])),
        }
    )
    mods = pd.DataFrame(analysis.get("modules", []))
    if not mods.empty:
        _show_df(mods.head(1200), hide_index=True)
    st.markdown("**Mermaid（モジュール依存）**")
    flow = _module_edges_mermaid(analysis.get("edges", []))
    _render_mermaid(flow, key=f"ext_mermaid_{name}", height=420)
    with st.expander("Mermaidコード"):
        st.code(flow, language="mermaid")
        _render_copy_button(flow, key=f"copy_ext_mermaid_{name}", label="Copy Mermaid")
    if PLOTLY_AVAILABLE:
        sankey = _edge_sankey(analysis.get("edges", []))
        if sankey is not None:
            st.plotly_chart(sankey, width="stretch")
        sun_mod = _sunburst_modules(analysis.get("modules", []))
        if sun_mod is not None:
            st.plotly_chart(sun_mod, width="stretch")
        sun_dir = _sunburst_directory(root, max_depth=4)
        if sun_dir is not None:
            st.plotly_chart(sun_dir, width="stretch")

    st.markdown("**Streamlit起動候補**")
    app_files = _find_streamlit_apps(root)
    if app_files:
        app_rel = st.selectbox(
            "streamlitアプリ",
            [str(p.relative_to(root)) for p in app_files],
            index=0,
            key=f"ext_app_sel_{name}",
        )
        app_path = root / app_rel
        app_port = st.number_input(
            "streamlitポート",
            min_value=1,
            max_value=65535,
            value=8520 if name == "trend" else 8521,
            step=1,
            key=f"ext_app_port_{name}",
        )
        cmd = f"streamlit run {shlex.quote(str(app_path))} --server.port {int(app_port)}"
        st.code(cmd + " &", language="bash")
        _render_copy_button(cmd + " &", key=f"copy_ext_app_cmd_{name}", label="起動コマンドをコピー")
        if st.button("バックグラウンド起動", key=f"launch_ext_app_{name}"):
            launch = _start_background_command(cmd, cwd=root)
            st.session_state[f"ext_launch_{name}"] = launch
        if f"ext_launch_{name}" in st.session_state:
            st.json(st.session_state[f"ext_launch_{name}"])
    else:
        st.info("streamlit起動候補が見つかりません。")


def _render_external_targets() -> None:
    st.subheader("外部ターゲット: trend / timesfm")
    st.caption(
        "`/mnt/e/env/ts/lib_ana/src/trend` と `/mnt/e/env/ts/lib_ana/src/model/timesfm` の内容解析・可視化・起動を行います。"
    )
    sub = st.tabs(["trend", "timesfm"])
    with sub[0]:
        _render_external_target("trend", EXTERNAL_TARGETS["trend"])
    with sub[1]:
        _render_external_target("timesfm", EXTERNAL_TARGETS["timesfm"])


def _module_edges_mermaid(edges: list[dict[str, Any]], max_edges: int = 140) -> str:
    use_edges = edges[:max_edges]
    ids: dict[str, str] = {}
    lines = ["flowchart LR"]
    idx = 0
    for e in use_edges:
        src = str(e.get("src", ""))
        dst = str(e.get("dst", ""))
        if not src or not dst:
            continue
        if src not in ids:
            idx += 1
            ids[src] = f"N{idx}"
            lines.append(f'{ids[src]}["{src}"]')
        if dst not in ids:
            idx += 1
            ids[dst] = f"N{idx}"
            lines.append(f'{ids[dst]}["{dst}"]')
        w = int(e.get("weight", 1))
        lines.append(f"{ids[src]} -->|{w}| {ids[dst]}")
    if len(lines) == 1:
        lines.append('A["No edges"]')
    return "\n".join(lines)


def _default_sequence_mermaid() -> str:
    return """sequenceDiagram
actor User
participant Dash as Streamlit Dashboard
participant CLI as loto_forecast.cli
participant Exog as resources.exog_pipeline
participant Res as resources.context
participant DB as PostgreSQL

User->>Dash: Open dashboard / inspect status
User->>CLI: build-exog / train / grid-run
CLI->>Exog: run_exog_build(spec)
Exog->>Res: start_run(ResourcesConfig)
Res->>DB: insert resources.run / stage_span / resource_metric
Exog->>DB: read dataset.* / write exog.*
DB-->>Dash: query tables, metrics, logs
Dash-->>User: visualize runs, schema, artifacts
"""


def _edge_sankey(edges: list[dict[str, Any]]):
    if not PLOTLY_AVAILABLE:
        return None
    if not edges:
        return None
    nodes: list[str] = []
    node_index: dict[str, int] = {}
    src_idx: list[int] = []
    dst_idx: list[int] = []
    values: list[int] = []
    for e in edges[:220]:
        src = str(e.get("src", ""))
        dst = str(e.get("dst", ""))
        if not src or not dst:
            continue
        if src not in node_index:
            node_index[src] = len(nodes)
            nodes.append(src)
        if dst not in node_index:
            node_index[dst] = len(nodes)
            nodes.append(dst)
        src_idx.append(node_index[src])
        dst_idx.append(node_index[dst])
        values.append(int(e.get("weight", 1)))
    if not src_idx:
        return None
    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": nodes, "pad": 10, "thickness": 12},
                link={"source": src_idx, "target": dst_idx, "value": values},
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=520)
    return fig


def _sunburst_modules(modules: list[dict[str, Any]]):
    if not PLOTLY_AVAILABLE:
        return None
    rows: list[dict[str, Any]] = []
    for m in modules:
        module_name = str(m.get("module", ""))
        parts = module_name.split(".")
        if len(parts) == 1:
            parts = [parts[0], "root"]
        rows.append(
            {
                "level1": parts[0] if len(parts) >= 1 else "root",
                "level2": parts[1] if len(parts) >= 2 else "root",
                "level3": parts[2] if len(parts) >= 3 else "root",
                "module": module_name,
                "lines": int(m.get("lines", 1)),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return px.sunburst(df, path=["level1", "level2", "level3", "module"], values="lines")


def _sunburst_directory(root: Path, max_depth: int = 4):
    if not PLOTLY_AVAILABLE:
        return None
    rows: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        parts = list(rel.parts)
        while len(parts) < max_depth:
            parts.append("__")
        rows.append(
            {
                "l1": parts[0],
                "l2": parts[1] if max_depth >= 2 else "__",
                "l3": parts[2] if max_depth >= 3 else "__",
                "l4": parts[3] if max_depth >= 4 else "__",
                "path": str(rel),
                "bytes": int(p.stat().st_size),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return px.sunburst(df, path=["l1", "l2", "l3", "l4", "path"], values="bytes")


def _render_export_controls(content: str, mime: str, ext: str, filename_base: str, key: str) -> None:
    st.download_button(
        "Download",
        data=content.encode("utf-8"),
        file_name=f"{filename_base}.{ext}",
        mime=mime,
        key=f"download_{_slug(key)}",
    )
    _render_copy_button(content, key=f"copy_{key}", label="Copy")
    st.markdown("**Rich Preview**")
    _render_rich_content(content, ext=ext, key=f"preview_{key}")
    with st.expander("Raw text preview"):
        st.text_area("Raw", value=content[:12000], height=220, key=f"preview_{_slug(key)}")
    _render_read_aloud(content, key=f"speech_{key}")


@st.cache_data(show_spinner=False)
def _cli_command_catalog() -> list[dict[str, Any]]:
    try:
        from loto_forecast.cli import build_parser

        parser = build_parser()
    except Exception:
        return []
    sub_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub_action = action
            break
    if sub_action is None:
        return []

    helps: dict[str, str] = {}
    for ca in getattr(sub_action, "_choices_actions", []):
        helps[ca.dest] = ca.help or ""

    catalog: list[dict[str, Any]] = []
    for cmd_name, sub in sorted(sub_action.choices.items()):
        args_meta: list[dict[str, Any]] = []
        for act in sub._actions:
            if act.dest in {"help", "func", "cmd"}:
                continue
            option_strings = [s for s in act.option_strings if s.startswith("--")]
            primary = option_strings[0] if option_strings else act.dest
            is_bool_optional = isinstance(act, argparse.BooleanOptionalAction)
            positive_opt = next((s for s in option_strings if not s.startswith("--no-")), primary)
            negative_opt = next((s for s in option_strings if s.startswith("--no-")), "")
            arg_kind = "text"
            if isinstance(act, (argparse._StoreTrueAction, argparse._StoreFalseAction)) or is_bool_optional:
                arg_kind = "bool"
            elif act.choices is not None:
                arg_kind = "choice"
            elif act.type is int:
                arg_kind = "int"
            elif act.type is float:
                arg_kind = "float"

            default = None if act.default is argparse.SUPPRESS else act.default
            args_meta.append(
                {
                    "dest": act.dest,
                    "flag": primary,
                    "required": bool(getattr(act, "required", False)),
                    "default": default,
                    "kind": arg_kind,
                    "help": (act.help or "").strip(),
                    "choices": list(act.choices) if act.choices is not None else [],
                    "option_strings": option_strings,
                    "is_bool_optional": is_bool_optional,
                    "positive_opt": positive_opt,
                    "negative_opt": negative_opt,
                }
            )

        catalog.append(
            {
                "command": cmd_name,
                "help": helps.get(cmd_name, "") or (sub.description or ""),
                "arguments": args_meta,
            }
        )
    return catalog


def _build_cli_command(
    command_name: str,
    arguments: list[dict[str, Any]],
    values: dict[str, Any],
    include_default_values: bool,
) -> str:
    parts: list[str] = ["python", "-m", "loto_forecast.cli", command_name]
    for arg in arguments:
        dest = str(arg["dest"])
        kind = str(arg["kind"])
        required = bool(arg["required"])
        default = arg.get("default")
        value = values.get(dest, default)
        flag = str(arg["flag"])

        if kind == "bool":
            val_bool = bool(value)
            def_bool = bool(default) if default is not None else False
            if not include_default_values and not required and val_bool == def_bool:
                continue
            if bool(arg.get("is_bool_optional", False)):
                parts.append(str(arg["positive_opt"]) if val_bool else str(arg["negative_opt"]))
            else:
                if val_bool:
                    parts.append(flag)
            continue

        if value in (None, ""):
            if required:
                continue
            if not include_default_values:
                continue

        if not include_default_values and not required and default is not None and value == default:
            continue
        parts.extend([flag, str(value)])
    return " ".join(shlex.quote(p) for p in parts)


def _shell_script_content(command: str, cwd: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(cwd))}",
            command,
            "",
        ]
    )


def _python_script_content(command: str, cwd: Path) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "import subprocess",
            "",
            f"CWD = {cwd.as_posix()!r}",
            f"CMD = {command!r}",
            "",
            "print(f'$ {CMD}')",
            "subprocess.run(['/bin/bash', '-lc', CMD], check=True, cwd=CWD)",
            "",
        ]
    )


def _command_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "command": "db-init",
            "purpose": "DBスキーマ/テーブル初期化",
            "key_inputs": "接続情報",
            "main_effect": "schema作成、初期テーブル定義",
        },
        {
            "command": "build-unified-dataset",
            "purpose": "学習用統合データセット生成",
            "key_inputs": "base/hist/exog schema-table",
            "main_effect": "dataset.* に統合テーブル出力",
        },
        {
            "command": "meta-automodel-create",
            "purpose": "AutoModel設定の登録・更新",
            "key_inputs": "model_name, h, auto_*",
            "main_effect": "meta.nf_automodel に設定保存",
        },
        {
            "command": "meta-automodel-run",
            "purpose": "登録設定を実行",
            "key_inputs": "config_id or limit",
            "main_effect": "model.nf_automodel と artifacts を更新",
        },
        {
            "command": "run-table-pyspark",
            "purpose": "PostgreSQL->Spark(または代替エンジン)変換",
            "key_inputs": "source_sql, target_table, execution_backend",
            "main_effect": "DB/Parquetへ変換結果を出力",
        },
        {
            "command": "model-save-load-analyze",
            "purpose": "保存済みモデルの save/load/analyze",
            "key_inputs": "run_id, save_path, run_save/run_load/run_analyze",
            "main_effect": "saved_models 配下へモデルバンドル保存",
        },
    ]


def _command_arg_hint(command_name: str, arg: dict[str, Any]) -> dict[str, str]:
    dest = str(arg.get("dest", ""))
    kind = str(arg.get("kind", "text"))
    default = arg.get("default")
    low = dest.lower()

    meaning = "引数値"
    effect = "コマンド実行時に該当オプションとして渡されます。"
    guidance = ""

    if low in {"host", "port", "user", "password", "database"}:
        meaning = "DB接続情報"
        effect = "接続先PostgreSQLが決まります。誤ると取得/書込先が変わります。"
        guidance = "本番/検証DBの取り違え防止のため、実行前に再確認してください。"
    elif "schema" in low or "table" in low:
        meaning = "対象テーブル指定"
        effect = "読み取り元/書き込み先テーブルを決めます。"
        guidance = "replaceモード時は上書き影響を確認してください。"
    elif "source_sql" in low:
        meaning = "抽出SQL"
        effect = "処理対象行が直接決まります。"
        guidance = "WHEREで範囲を絞ると速度・安全性が向上します。"
    elif "transform_sql" in low:
        meaning = "変換SQL"
        effect = "source結果に対する列変換/加工ロジックになります。"
        guidance = "`{{source}}` プレースホルダを使って中間結果を参照します。"
    elif low in {"output_if_exists"}:
        meaning = "出力競合時の挙動"
        effect = "replace/append/fail のどれで処理するかを決めます。"
        guidance = "再実行時は replace、追記運用は append を選択します。"
    elif low in {"execution_backend"}:
        meaning = "実行エンジン選択"
        effect = "auto/polars/pandas/dask/spark の実行経路を切替えます。"
        guidance = "ローカル検証は pandas/polars、本番大量処理は spark 推奨です。"
    elif low in {"auto_backend"}:
        meaning = "HPOバックエンド"
        effect = "AutoModel探索を optuna / ray のどちらで行うかを決めます。"
        guidance = "単機で軽量なら optuna、分散探索なら ray を選択します。"
    elif low in {"auto_num_samples", "num_samples"}:
        meaning = "探索試行回数"
        effect = "値を増やすほど精度改善余地は増えますが、時間と計算コストが増えます。"
        guidance = "まず 10-30 で検証し、必要時に増やしてください。"
    elif low in {"auto_cpus", "cpus", "auto_gpus", "gpus"}:
        meaning = "探索時リソース上限"
        effect = "並列度やGPU使用量を制御します。"
        guidance = "他ジョブと競合しない値に調整してください。"
    elif low in {"auto_search_alg", "search_alg_name"}:
        meaning = "探索アルゴリズム"
        effect = "ハイパーパラメータ探索のサンプリング戦略を変更します。"
        guidance = "既定は BasicVariantGenerator です。"
    elif low in {"recursive_depth"}:
        meaning = "再帰実行深度"
        effect = "組み合わせ探索/再帰実行の深さを決めます。"
        guidance = "大きい値は指数的にタスク数が増える可能性があります。"
    elif low in {"max_tasks"}:
        meaning = "最大タスク数制限"
        effect = "探索組み合わせ数の上限をかけます。"
        guidance = "0 は無制限です。まず小さめで検証してください。"
    elif low.startswith("run_"):
        meaning = "実行ON/OFFフラグ"
        effect = "predict/evaluate/explain/save/load/analyze 各工程の実行有無を制御します。"
        guidance = "原因切り分け時は工程を段階的にONにしてください。"
    elif low in {"save_dataset", "save_overwrite"}:
        meaning = "保存挙動"
        effect = "データセット同梱や既存保存先上書き可否を制御します。"
        guidance = "容量最小化は save_dataset=false、再実行は overwrite=true が一般的です。"
    elif low in {"load_check_predict", "insample_step_size"}:
        meaning = "load後の疎通検証"
        effect = "predict_insample を実行して保存物の再利用可否を確認します。"
        guidance = "本番前に少なくとも1回は有効化を推奨します。"
    elif low in {"model_name", "auto_cls_model"}:
        meaning = "モデル種別"
        effect = "学習器の構造と外生変数対応が決まります。"
        guidance = "外生変数対応表(F/H/S)と合わせて選択してください。"
    elif low in {"h", "horizon", "auto_h"}:
        meaning = "予測地平"
        effect = "先読みステップ数が決まり、難易度と必要特徴量が変化します。"
        guidance = "地平を長くするほど誤差増大が起きやすくなります。"
    elif low in {"model_params_json", "auto_config_json", "param_space_json"}:
        meaning = "JSON設定"
        effect = "モデル詳細設定・探索空間を上書きします。"
        guidance = "JSON構文エラー時は実行失敗します。キー名はCLI仕様に一致させてください。"
    elif low in {"param_mode_json"}:
        meaning = "固定/変動フラグ設定"
        effect = "mode=fixed は固定値、mode=vary はグリッド探索候補として扱います。"
        guidance = '例: {"lr":{"mode":"vary","values":[0.001,0.0005]}}'
    elif low in {"ensure_db_init"}:
        meaning = "メタテーブル初期化フラグ"
        effect = "meta-automodel実行前に必要DDL(db-init相当)を自動適用します。"
        guidance = "初回環境やDDL差分がある環境ではON推奨です。"
    elif low in {"unified_filter_json", "unified_group_cols_json"}:
        meaning = "統合データ抽出条件"
        effect = "学習対象の系列・粒度を直接決定します。"
        guidance = "group cols は一意系列キーを構成する列を指定してください。"
    elif low in {"config_id", "config_name"}:
        meaning = "実行設定識別子"
        effect = "どの meta 設定を参照するかが決まります。"
        guidance = "config_id 指定時は limit より優先されます。"
    else:
        meaning = "CLI引数"
        effect = "該当コマンドの動作パラメータとして使われます。"
        guidance = ""

    if kind == "bool":
        guidance = (guidance + " " if guidance else "") + "boolはON/OFFで実行フローを切替えます。"
    if default is not None:
        guidance = (guidance + " " if guidance else "") + f"既定値={default}"

    return {
        "flag": str(arg.get("flag", "")),
        "dest": dest,
        "kind": kind,
        "required": "yes" if bool(arg.get("required", False)) else "no",
        "meaning": meaning,
        "effect": effect,
        "guidance": guidance,
    }


def _render_command_result_block(session_key: str, title: str, key_prefix: str) -> None:
    res = st.session_state.get(session_key)
    if not isinstance(res, dict) or not res:
        return
    st.markdown(f"**{title}**")
    stdout_text = str(res.get("stdout", ""))
    stderr_text = str(res.get("stderr", ""))
    combined_text = "\n".join([stdout_text, stderr_text])
    parsed = _try_parse_json_tail(combined_text)
    train_summary = _extract_train_result_summary(stdout_text, stderr_text)
    rid = _extract_last_run_id(combined_text)
    evidence = _extract_execution_evidence(stdout_text, stderr_text)
    ok = bool(res.get("ok", False))
    rc = int(res.get("returncode", -1))
    has_trace = int(evidence.get("traceback_count", 0) or 0) > 0
    executed = bool(evidence.get("executed_confirmed", False))
    verdict = "成功" if ok else "失敗"
    verdict_reason = f"returncode={rc}"
    if ok and (executed or rid):
        verdict_reason += " / 実行証跡あり"
    elif ok:
        verdict_reason += " / 実行証跡は弱い"
    if has_trace:
        verdict_reason += " / traceback検出"
    if ok:
        st.success(f"実行判定: {verdict} ({verdict_reason})")
    else:
        st.error(f"実行判定: {verdict} ({verdict_reason})")
        hint_rows: list[str] = []
        low_txt = combined_text.lower()
        if "no model found in directory" in low_txt:
            hint_rows.append(
                "`No model found in directory`。`source-path` が run ディレクトリ（例: `artifacts/<run_id>`）か確認してください。"
            )
        if "duplicate key value violates unique constraint" in low_txt and "forecast_pkey" in low_txt:
            hint_rows.append(
                "`model.forecast` に同一 `(run_id, unique_id, ds)` が既に存在します。新しい `run_id` を使うか、既存レコードを整理してください。"
            )
        if "filenotfounderror" in low_txt and "saved_models" in low_txt:
            hint_rows.append(
                "`save-path` が未作成です。先に save を実行するか、`save-path` を存在するディレクトリに変更してください。"
            )
        if "you need to have a stored dataset to save it" in low_txt:
            hint_rows.append(
                "`save_dataset=True` のままでは保存できません。`--no-save-dataset` で再実行するか、学習時に dataset 同梱保存を有効化してください。"
            )
        if hint_rows:
            st.warning("\n".join([f"- {m}" for m in hint_rows]))
    if isinstance(train_summary, dict):
        status_v = str(train_summary.get("status") or "").lower()
        model_v = str(train_summary.get("model_name") or "-")
        run_v = str(train_summary.get("run_id") or rid or "-")
        artifact_ok = train_summary.get("artifact_exists")
        meta_ok = train_summary.get("meta_exists")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("train status", status_v or "-")
        c2.metric("model", model_v)
        c3.metric("run_id", run_v)
        c4.metric("artifact", "yes" if artifact_ok is True else ("no" if artifact_ok is False else "-"))
        c5, c6 = st.columns(2)
        c5.metric("meta record", "yes" if meta_ok is True else ("no" if meta_ok is False else "-"))
        c6.metric("returncode", int(rc))
        if status_v == "success" and artifact_ok is True:
            st.success("モデル生成判定: 成功（artifact を確認）")
        elif status_v == "success":
            st.warning("モデル生成判定: 成功ステータスだが artifact 確認情報が不足")
        elif status_v == "failed":
            st.error("モデル生成判定: 失敗")
            if train_summary.get("error"):
                st.error(str(train_summary.get("error")))
        if train_summary.get("artifact_path"):
            st.caption(f"artifact_path: `{train_summary.get('artifact_path')}`")
        if train_summary.get("log_path"):
            st.caption(f"log_path: `{train_summary.get('log_path')}`")
    if rid:
        st.info(f"run_id: `{rid}`")
    elif ok:
        st.warning("run_id を検出できませんでした。stdout/json tail を確認してください。")
    st.json(
        {
            "ok": ok,
            "returncode": rc,
            "started_at": str(res.get("started_at", "")),
            "ended_at": str(res.get("ended_at", "")),
            "elapsed_sec": float(res.get("elapsed_sec", 0.0) or 0.0),
            "resolved_run_id": rid,
            "parsed_json_tail_available": parsed is not None,
            "execution_evidence": evidence,
        }
    )
    out_tab, err_tab, json_tab = st.tabs(["stdout", "stderr", "json tail"])
    with out_tab:
        st.code(str(res.get("stdout", ""))[:120000], language="bash")
    with err_tab:
        st.code(str(res.get("stderr", ""))[:120000], language="bash")
    with json_tab:
        if parsed is None:
            st.info("JSONを末尾から復元できませんでした。")
        else:
            st.json(parsed)
    _render_copy_button(str(res.get("stdout", "")), key=f"{key_prefix}_stdout_copy", label="stdoutをコピー")


def _model_ops_preflight_checks(
    *,
    run_id: str,
    source_path: str,
    save_path: str,
    run_save: bool,
    run_load: bool,
) -> tuple[pd.DataFrame, list[str]]:
    rid = str(run_id or "").strip()
    src = Path(str(source_path or "")).expanduser().resolve()
    save = Path(str(save_path or "")).expanduser().resolve()
    src_candidate = (src / rid).resolve() if rid and src.name != rid else src
    load_target = save if (run_save or str(save_path or "").strip()) else src
    load_candidate = (load_target / rid).resolve() if rid and load_target.name != rid else load_target

    rows = [
        {"check": "source path exists", "ok": bool(src.exists()), "detail": str(src)},
        {
            "check": "source has model artifacts",
            "ok": bool(_has_model_artifacts(src)),
            "detail": "configuration.pkl / alias_to_model.pkl / *.ckpt",
        },
        {
            "check": "source path looks like base dir",
            "ok": bool(src_candidate.exists() and src_candidate != src),
            "detail": str(src_candidate),
        },
        {
            "check": "source/<run_id> has artifacts",
            "ok": bool(src_candidate.exists() and _has_model_artifacts(src_candidate)),
            "detail": str(src_candidate),
        },
        {"check": "save parent exists", "ok": bool(save.parent.exists()), "detail": str(save.parent)},
        {"check": "save path exists", "ok": bool(save.exists()), "detail": str(save)},
        {"check": "load target exists", "ok": bool(load_target.exists()), "detail": str(load_target)},
        {
            "check": "load target has artifacts",
            "ok": bool(_has_model_artifacts(load_target)),
            "detail": str(load_target),
        },
        {
            "check": "load target/<run_id> has artifacts",
            "ok": bool(load_candidate.exists() and _has_model_artifacts(load_candidate)),
            "detail": str(load_candidate),
        },
    ]

    hints: list[str] = []
    if rid and src.exists() and not _has_model_artifacts(src) and _has_model_artifacts(src_candidate):
        hints.append(
            f"`source-path` がベースディレクトリです。`{src_candidate}` を指定するか、そのまま run_id 解決を使ってください。"
        )
    if run_load and not run_save:
        if not load_target.exists() and not load_candidate.exists():
            hints.append(
                "`run-load` 単独実行では `save-path` が既存の保存済みモデルディレクトリである必要があります。先に `run-save` を実行してください。"
            )
        if load_target.exists() and (not _has_model_artifacts(load_target)) and _has_model_artifacts(load_candidate):
            hints.append(
                f"`save-path` はベースディレクトリです。`{load_candidate}` を指定すると load 成功率が上がります。"
            )
    return pd.DataFrame(rows), hints


def _render_nf_lifecycle_lab(
    engine: Engine | None,
    tables: set[tuple[str, str]],
    row_limit: int,
    sample_limit: int,
    host: str,
    port: int,
    user: str,
    database: str,
) -> None:
    _load_nf_lab_ui_state_once(
        engine=engine,
        host=host,
        port=int(port),
        user=user,
        database=database,
    )

    # --- helpers: always define to avoid UnboundLocalError (branch-dependent defs) ---
    def _decode_exog_axis_value(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        sv = str(value).strip()
        if (not sv) or sv in {"(none)", "None", "null", "NULL"}:
            return []
        try:
            loaded = json.loads(sv)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in sv.split(",") if x.strip()]

    st.subheader("NeuralForecast 実行・検証ラボ")
    st.caption(
        "メタ確認、学習/再学習/予測/cross_validation/predict_insample/save/load を分離実行し、効果検証とproxy因果分析まで実施します。"
    )
    st.info(
        "\n".join(
            [
                "初見向け最短ルート: `メタテーブル確認` -> `学習(train)` -> `予測/評価`。",
                "`学習(train)` はまず `かんたん` モードで `学習データ選択` と `backend/num_samples/loss/search_alg` のみ設定してください。",
                "コマンド実行前に下部の `実行前チェック` で `エラーなし` を確認してから `Run` を押してください。",
            ]
        )
    )
    step_rows = [
        {
            "step": "1. メタテーブル確認",
            "ready": bool(st.session_state.get("nf_lab_db_init_result", {}).get("ok")),
            "checkpoint": "db-init 実行成功",
        },
        {
            "step": "2. 学習(train)",
            "ready": bool(st.session_state.get("nf_lab_train_result", {}).get("ok")),
            "checkpoint": "train 実行成功",
        },
        {
            "step": "3. 再学習(retrain)",
            "ready": bool(st.session_state.get("nf_lab_retrain_result", {}).get("ok")),
            "checkpoint": "retrain 実行成功",
        },
        {
            "step": "4. 予測/評価",
            "ready": bool(st.session_state.get("nf_lab_predict_result", {}).get("ok"))
            and bool(st.session_state.get("nf_lab_evaluate_result", {}).get("ok")),
            "checkpoint": "predict + evaluate 成功",
        },
        {
            "step": "5. CV/Insample",
            "ready": bool(st.session_state.get("nf_lab_cv_result", {}).get("ok"))
            or bool(st.session_state.get("nf_lab_ins_result", {}).get("ok")),
            "checkpoint": "cross_validation もしくは predict_insample 成功",
        },
        {
            "step": "6. 保存/ロード",
            "ready": bool(st.session_state.get("nf_lab_save_result", {}).get("ok"))
            and bool(st.session_state.get("nf_lab_load_result", {}).get("ok")),
            "checkpoint": "save + load 成功",
        },
        {
            "step": "7. 効果検証・因果",
            "ready": bool(("model", "nf_automodel") in tables),
            "checkpoint": "model.nf_automodel に分析対象あり",
        },
        {
            "step": "8. Run-ID統合分析",
            "ready": bool(st.session_state.get("nf_lab_runid_sel", "")),
            "checkpoint": "run-id を選択して統合分析表示",
        },
    ]
    completed_steps = int(sum(1 for r in step_rows if bool(r.get("ready", False))))
    total_steps = int(len(step_rows))
    pending_steps = int(max(0, total_steps - completed_steps))
    first_pending = next((r for r in step_rows if not bool(r.get("ready", False))), None)
    n1, n2, n3 = st.columns(3)
    n1.metric("完了ステップ", f"{completed_steps}/{total_steps}")
    n2.metric("未完了", pending_steps)
    n3.metric("到達率", f"{(completed_steps / total_steps * 100.0):.0f}%")
    if first_pending:
        st.info(f"次の推奨操作: **{first_pending['step']}**（目標: {first_pending['checkpoint']}）")
    else:
        st.success("全ステップの前提条件が満たされています。解析/効果検証に進めます。")
    with st.expander("操作ステップ進捗（アフォーダンス表示）", expanded=True):
        step_df = pd.DataFrame(step_rows)
        step_df["status"] = _normalize_status_series(
            step_df["ready"].map(lambda x: "ready" if bool(x) else "pending"),
            allowed=NF_LAB_STEP_STATUS_ORDER,
            default="pending",
        )
        _show_df(step_df[["step", "status", "checkpoint"]], hide_index=True)
        if PLOTLY_AVAILABLE and not step_df.empty:
            vis_df = step_df.copy()
            vis_df["order"] = np.arange(1, len(vis_df) + 1)
            vis_df["progress"] = (
                vis_df["status"]
                .map({"ready": 1.0, "pending": 0.35, "unknown": 0.1})
                .fillna(0.1)
                .astype(float)
            )
            fig = _build_categorical_bar_figure(
                vis_df,
                x="progress",
                y="step",
                orientation="h",
                color="status",
                color_map=STATUS_COLOR_MAP,
                color_order=_present_category_order(vis_df["status"], NF_LAB_STEP_STATUS_ORDER),
                title="実行ステップ進捗",
                height=330,
            )
            fig.update_layout(xaxis_range=[0, 1], xaxis_title="progress")
            st.plotly_chart(fig, width="stretch")
    with st.expander("初回ナビゲーション（どこから触るか）", expanded=False):
        nav_sections = ["最短3ステップ", "タブの役割", "よくある失敗"]
        nav_section = st.selectbox(
            "初回ナビメニュー",
            nav_sections,
            index=0,
            key="nf_lab_nav_sub_select",
        )
        if nav_section == "最短3ステップ":
            st.markdown(
                "\n".join(
                    [
                        "1. `メタテーブル確認` で `db-init` 実行と `meta/model` テーブルの行数確認",
                        "2. `学習(train)` で `かんたん` モードのまま `Run train` 実行",
                        "3. `予測/評価` で `predict` -> `evaluate` を順に実行し、結果JSONを確認",
                    ]
                )
            )
        if nav_section == "タブの役割":
            guide_rows = [
                {"tab": "メタテーブル確認", "何をするか": "DDL初期化とテーブル状態確認", "目安時間": "1-2分"},
                {"tab": "学習(train)", "何をするか": "モデル学習(必須)", "目安時間": "数分-数十分"},
                {"tab": "再学習(retrain)", "何をするか": "既存runの条件を再利用", "目安時間": "数分-数十分"},
                {"tab": "予測/評価", "何をするか": "推論と評価を個別実行", "目安時間": "1-5分"},
                {"tab": "CV/Insample", "何をするか": "汎化・再現性の検証", "目安時間": "数分"},
                {"tab": "保存/ロード", "何をするか": "成果物保存と復元疎通", "目安時間": "1-3分"},
                {"tab": "効果検証・因果", "何をするか": "寄与候補/ATE proxy確認", "目安時間": "1-3分"},
                {"tab": "Run-ID統合分析", "何をするか": "run単位で整合性/リソース/精度を統合監査", "目安時間": "1-5分"},
            ]
            _show_df(pd.DataFrame(guide_rows), hide_index=True)
        if nav_section == "よくある失敗":
            st.markdown(
                "\n".join(
                    [
                        '- `params-json` に `"h": ""` のような空文字を入れると型エラーになります。',
                        "- `backend=optuna` なのに `search_alg=BasicVariantGenerator` を指定すると不整合です。",
                        "- `AutoHINT` は `valid_loss` が必須で、`backend=ray` を推奨します。",
                    ]
                )
            )
    with st.expander("実行手順 / パラメータ補助情報", expanded=False):
        h_sections = ["実行手順", "主要パラメータ", "search_alg 対応"]
        h_section = st.selectbox(
            "補助情報メニュー",
            h_sections,
            index=0,
            key="nf_lab_help_sub_select",
        )
        if h_section == "実行手順":
            st.markdown(
                "\n".join(
                    [
                        "1. `メタテーブル確認` で `db-init` と `meta/model` の状態を確認",
                        "2. `学習(train)` で model/backend/loss/search_alg と学習データ条件を設定",
                        "3. `予測/評価` で `predict` と `evaluate` を独立実行して差分確認",
                        "4. `CV/Insample` で汎化と再現性（predict_insample）を検証",
                        "5. `保存/ロード` で save/load を分離確認し、最後に一括検証",
                        "6. `効果検証・因果` で寄与候補/ATE代理を確認",
                        "7. `Runner` で `param_mode_json` を使い fixed/vary 切替で網羅グリッド実行",
                    ]
                )
            )
        if h_section == "主要パラメータ":
            p_help = pd.DataFrame(
                [
                    {
                        "parameter": "backend",
                        "meaning": "HPO実行基盤(optuna/ray)",
                        "effect": "探索アルゴリズムと実行方式が変わる",
                        "recommended": "単機検証はoptuna, 分散はray",
                    },
                    {
                        "parameter": "num_samples",
                        "meaning": "探索試行回数",
                        "effect": "増やすほど精度改善余地↑/時間↑",
                        "recommended": "10-50から開始",
                    },
                    {
                        "parameter": "loss / valid_loss",
                        "meaning": "学習損失/検証損失",
                        "effect": "モデル選択基準が変わる",
                        "recommended": "MAE基準で開始",
                    },
                    {
                        "parameter": "search_alg",
                        "meaning": "探索器",
                        "effect": "探索効率・探索の偏りが変わる",
                        "recommended": "optuna=TPE, ray=BasicVariantGenerator",
                    },
                    {
                        "parameter": "cpus / gpus",
                        "meaning": "試行時の計算資源",
                        "effect": "速度・競合が変わる",
                        "recommended": "初期は自動最大値ON",
                    },
                    {
                        "parameter": "h_mode=unique_id_count",
                        "meaning": "hをunique_id一意数で自動決定",
                        "effect": "系列数に追従した地平設定",
                        "recommended": "系列単位実験で有効",
                    },
                    {
                        "parameter": "param_mode_json",
                        "meaning": "固定/変動フラグ",
                        "effect": "fixedは固定値・varyはparam_spaceに展開",
                        "recommended": "まず2-3パラメータのみvary",
                    },
                ]
            )
            _show_df(p_help, hide_index=True)
        if h_section == "search_alg 対応":
            st.markdown("`backend=optuna`: `RandomSampler`, `TPESampler`, `CmaEsSampler`, `NSGAIISampler`")
            st.markdown("`backend=ray`: `BasicVariantGenerator`, `OptunaSearch`, `HyperOptSearch`, `BayesOptSearch`")
            st.caption("`search_alg_name(override)` を空にすると、上の `search_alg` 選択値を使用します。")

    def _bool_opt(name: str, value: bool) -> str:
        return f"--{name}" if bool(value) else f"--no-{name}"

    def _json_text(raw: str, expect: str, label: str, errors: list[str]) -> str | None:
        text_raw = str(raw or "").strip()
        if not text_raw:
            return None
        try:
            obj = json.loads(text_raw)
        except Exception as e:
            errors.append(f"{label}: JSON parse error: {e}")
            return None
        if expect == "dict" and not isinstance(obj, dict):
            errors.append(f"{label}: JSON object(dict)が必要です。")
            return None
        if expect == "list" and not isinstance(obj, list):
            errors.append(f"{label}: JSON array(list)が必要です。")
            return None
        return json.dumps(obj, ensure_ascii=False)

    def _nf_signature_rows(
        method_name: str,
        defaults: dict[str, Any],
        overrides: dict[str, Any],
        forced: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        merged = dict(defaults)
        merged.update(dict(overrides or {}))
        merged.update(dict(forced or {}))
        rows: list[dict[str, Any]] = []
        for k, default_v in defaults.items():
            v = merged.get(k, default_v)
            source = "forced" if forced and k in forced else ("override" if k in overrides else "default")
            if isinstance(v, (dict, list)):
                v_text = _truncate_arg_preview(_stable_json_dumps(v))
            else:
                v_text = _truncate_arg_preview(v)
            rows.append(
                {
                    "method": method_name,
                    "arg": str(k),
                    "value": v_text,
                    "source": source,
                }
            )
        extra_keys = sorted([k for k in overrides if k not in defaults])
        for k in extra_keys:
            v = overrides.get(k)
            if isinstance(v, (dict, list)):
                v_text = _truncate_arg_preview(_stable_json_dumps(v))
            else:
                v_text = _truncate_arg_preview(v)
            rows.append(
                {
                    "method": method_name,
                    "arg": str(k),
                    "value": v_text,
                    "source": "override(extra)",
                }
            )
        return pd.DataFrame(rows)

    def _render_help_hint(
        key: str,
        brief: str,
        detail: str,
        button_label: str = "ヒント",
    ) -> None:
        st.caption(f"簡易: {brief}")
        flag_key = f"nf_lab_hint_open_{_slug(key)}"
        button_key = f"nf_lab_hint_btn_{_slug(key)}"
        st.session_state.pop(button_key, None)
        if st.button(button_label, key=button_key):
            st.session_state[flag_key] = not bool(st.session_state.get(flag_key, False))
        if bool(st.session_state.get(flag_key, False)):
            st.info(detail)

    def _render_tab_playbook(
        *,
        purpose: str,
        required_inputs: list[str],
        outputs: list[str],
        steps: list[str],
    ) -> None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**目的**")
            st.write(purpose)
        with c2:
            st.markdown("**必要入力**")
            for item in required_inputs:
                st.write(f"- {item}")
        with c3:
            st.markdown("**主な出力**")
            for item in outputs:
                st.write(f"- {item}")
        st.caption("手順: " + " -> ".join([str(x) for x in steps]))

    def _json_preset_input(
        label: str,
        key_prefix: str,
        presets: dict[str, Any],
        default_name: str,
        height: int = 90,
    ) -> str:
        preset_names = list(presets.keys())
        if default_name not in preset_names:
            preset_names = [default_name] + [p for p in preset_names if p != default_name]
        p_name = st.selectbox(
            f"{label} preset",
            preset_names,
            index=(preset_names.index(default_name) if default_name in preset_names else 0),
            key=f"{key_prefix}_preset",
        )
        preset_obj = presets.get(p_name, {})
        preset_text = json.dumps(preset_obj, ensure_ascii=False)
        _render_copy_button(
            preset_text,
            key=f"{key_prefix}_preset_copy",
            label=f"{label} sample copy",
        )
        return st.text_area(
            f"{label} (dict)",
            value=str(st.session_state.get(f"{key_prefix}_text", preset_text)),
            height=height,
            key=f"{key_prefix}_text",
        )

    if "nf_lab_cwd" not in st.session_state:
        st.session_state["nf_lab_cwd"] = str(PROJECT_ROOT)
    lab_cwd = Path(st.text_input("lab 実行ディレクトリ(cwd)", key="nf_lab_cwd")).expanduser()
    lab_timeout_sec = st.slider(
        "lab タイムアウト(秒)", min_value=30, max_value=7200, value=1200, step=30, key="nf_lab_timeout"
    )

    run_dirs = _discover_run_directories(require_model_artifacts=True)
    run_id_options = [p.name for p in run_dirs]
    run_id_to_dir = {p.name: p for p in run_dirs}

    meta_df = pd.DataFrame()
    model_df = pd.DataFrame()
    if engine is not None and ("meta", "nf_automodel") in tables:
        try:
            meta_cols_df = _table_columns(engine, "meta", "nf_automodel")
            meta_cols = set(meta_cols_df["column_name"].astype(str).tolist()) if not meta_cols_df.empty else set()
            cfg_col = "id" if "id" in meta_cols else ("config_id" if "config_id" in meta_cols else None)
            order_col = (
                "updated_at" if "updated_at" in meta_cols else ("created_at" if "created_at" in meta_cols else None)
            )
            select_cols = [
                c
                for c in [
                    cfg_col,
                    "config_name",
                    "active",
                    "priority",
                    "created_at",
                    "updated_at",
                    "model_name",
                    "horizon",
                ]
                if c and c in meta_cols
            ]
            if not select_cols:
                select_cols = sorted(list(meta_cols))
            order_sql = f" ORDER BY {_safe_ident(order_col)} DESC NULLS LAST" if order_col else ""
            meta_df = _query_df(
                engine,
                f"""
                SELECT {", ".join([_safe_ident(c) for c in select_cols])}
                FROM "meta"."nf_automodel"
                {order_sql}
                LIMIT :limit
                """,
                {"limit": int(max(100, row_limit))},
            )
            if cfg_col and cfg_col in meta_df.columns and "config_id" not in meta_df.columns:
                meta_df = meta_df.rename(columns={cfg_col: "config_id"})
        except Exception as e:
            st.warning(f"meta.nf_automodel 取得失敗: {e}")
    if engine is not None and ("model", "nf_automodel") in tables:
        try:
            model_cols_df = _table_columns(engine, "model", "nf_automodel")
            model_cols = set(model_cols_df["column_name"].astype(str).tolist()) if not model_cols_df.empty else set()
            order_col = (
                "updated_at" if "updated_at" in model_cols else ("created_at" if "created_at" in model_cols else None)
            )
            select_cols = [
                c
                for c in [
                    "run_id",
                    "config_id",
                    "model_name",
                    "horizon",
                    "status",
                    "created_at",
                    "updated_at",
                    "metrics_json",
                    "params_json",
                    "exog_json",
                    "artifact_path",
                    "model_store_path",
                ]
                if c in model_cols
            ]
            if not select_cols:
                select_cols = sorted(list(model_cols))
            order_sql = f" ORDER BY {_safe_ident(order_col)} DESC NULLS LAST" if order_col else ""
            model_df = _query_df(
                engine,
                f"""
                SELECT {", ".join([_safe_ident(c) for c in select_cols])}
                FROM "model"."nf_automodel"
                {order_sql}
                LIMIT :limit
                """,
                {"limit": int(max(200, row_limit * 2))},
            )
        except Exception as e:
            st.warning(f"model.nf_automodel 取得失敗: {e}")

    if not model_df.empty and "run_id" in model_df.columns:
        db_ids = [str(x) for x in model_df["run_id"].astype(str).tolist() if str(x).strip()]
        run_id_options = list(dict.fromkeys(db_ids + run_id_options))

    st.caption(
        "クイック移動: `メタテーブル確認` | `学習(train)` | `再学習(retrain)` | `予測/評価` | "
        "`CV/Insample` | `保存/ロード` | `効果検証・因果` | `Run-ID統合分析` | `リソース解析` | `DB管理/ER`"
    )
    lab_sections = [
        "メタテーブル確認",
        "学習(train)",
        "再学習(retrain)",
        "予測/評価",
        "CV/Insample",
        "保存/ロード",
        "効果検証・因果",
        "Run-ID統合分析",
        "リソース解析",
        "DB管理/ER",
    ]
    lab_section = st.selectbox(
        "NeuralForecast 実行・検証ラボ メニュー",
        lab_sections,
        index=0,
        key="nf_lab_section_select",
    )

    if lab_section == "メタテーブル確認":
        st.caption("meta/modelテーブルを同時に確認し、設定値反映の差異やrun実績を監査します。")
        with st.expander("メタテーブル作成(db-init)", expanded=False):
            cmd_db_init = "python -m loto_forecast.cli db-init"
            _render_command_preview(
                cmd_db_init,
                copy_key="nf_lab_copy_db_init",
                copy_label="Copy db-init command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            if st.button("Run db-init", key="nf_lab_run_db_init"):
                if not lab_cwd.exists() or not lab_cwd.is_dir():
                    st.error("lab cwd が有効なディレクトリではありません。")
                else:
                    st.session_state["nf_lab_db_init_result"] = _run_shell_command_live(
                        cmd_db_init,
                        cwd=lab_cwd,
                        timeout_sec=lab_timeout_sec,
                        title="NF db-init",
                    )
            _render_command_result_block("nf_lab_db_init_result", "db-init 実行結果", "nf_lab_db_init")
        with st.expander("schema:log 確認/初期化", expanded=False):
            if engine is None:
                st.info("DB未接続のため確認できません。")
            else:
                try:
                    log_df = _schema_table_counts(engine, "log")
                    if log_df.empty:
                        st.info("schema:log に BASE TABLE が見つかりません。先に db-init を実行してください。")
                    else:
                        _show_df(log_df, hide_index=True)
                except Exception as e:
                    st.error(f"schema:log 確認失敗: {e}")
                log_reset_confirm = st.checkbox(
                    "schema:log の全テーブルを初期化(TRUNCATE)する",
                    value=False,
                    key="nf_lab_log_reset_confirm",
                )
                if st.button("schema:log テーブル初期化", key="nf_lab_log_reset_btn", disabled=not log_reset_confirm):
                    try:
                        reset_out = _truncate_schema_tables(engine, "log")
                        _clear_query_cache()
                        st.success(f"schema:log 初期化完了: {int(reset_out.get('truncated', 0))} table(s)")
                        after_df = _schema_table_counts(engine, "log")
                        if not after_df.empty:
                            _show_df(after_df, hide_index=True)
                    except Exception as e:
                        st.error(f"schema:log 初期化失敗: {e}")
        if engine is None:
            st.info("DB未接続のためメタテーブル確認は利用できません。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("meta rows", int(meta_df.shape[0]))
            c2.metric("model rows", int(model_df.shape[0]))
            c3.metric(
                "run_id(unique)",
                int(model_df["run_id"].nunique()) if "run_id" in model_df.columns and not model_df.empty else 0,
            )
            fail_n = (
                int((model_df.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "failed").sum())
                if not model_df.empty
                else 0
            )
            total_n = int(model_df.shape[0])
            c4.metric("failed rate", f"{(fail_n / total_n * 100.0):.1f}%" if total_n > 0 else "n/a")

            meta_sub = st.selectbox(
                "メタ表示メニュー",
                ["meta.nf_automodel", "model.nf_automodel(展開)"],
                index=0,
                key="nf_lab_meta_sub_select",
            )
            if meta_sub == "meta.nf_automodel":
                if meta_df.empty:
                    st.info("meta.nf_automodel は空です。")
                else:
                    _show_df(meta_df.head(max(50, row_limit)), hide_index=True)
            if meta_sub == "model.nf_automodel(展開)":
                if model_df.empty:
                    st.info("model.nf_automodel は空です。")
                else:
                    expanded = _expand_semistructured_columns(
                        model_df.copy(),
                        max_depth=3,
                        max_list_items=5,
                        max_new_cols_per_source=120,
                    )
                    _show_df(expanded.head(max(50, row_limit)), hide_index=True)

    if lab_section == "学習(train)":
        st.caption(
            "train を直接実行します。model選択連動デフォルト、全パラメータ選択、学習データ条件、h自動算出を設定できます。"
        )
        st.info(
            "初回は `かんたんモード` で、`学習データ選択` と `backend/num_samples/loss/search_alg` だけ設定してください。"
        )
        st.caption(
            "タブの違い: `全パラメータ選択`=学習データ+モデル設定 / "
            "`NeuralForecast runtime`=fit/predict/save等の実行挙動 / `exog lists`=外生変数列の役割指定 / "
            "`固定/網羅メタ反映`=固定・可変の組合せを meta テーブルへ反映"
        )
        d_sections = ["全パラメータ選択", "NeuralForecast runtime", "exog lists", "固定/網羅メタ反映"]
        d_section = st.selectbox(
            "学習(train) サブメニュー",
            d_sections,
            index=0,
            key="nf_lab_train_sub_select",
        )
        if d_section == "全パラメータ選択":
            applied_preset_values = apply_pending_nf_preset(st.session_state)
            applied_preset_source, applied_preset_runtime_values = consume_active_nf_preset(st.session_state)
            preset_applied_this_run = bool(applied_preset_values)
            with st.expander("このタブの操作順（初見向け）", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "1. `model` を選択",
                            "2. `backend / num_samples / loss / valid_loss / search_alg` を設定",
                            "3. `学習データ選択` で schema/table/group/h を決定",
                            "4. 必要なら `NeuralForecast runtime` と `exog lists` を編集",
                            "5. `実行前チェック` がエラーなしであることを確認して `Run train`",
                        ]
                    )
                )
                st.caption(
                    "優先順位: CLIオプション(`--search-alg-name` など) > params-jsonの同名キー。"
                    " 画面では不整合が出ないよう自動で同じ値に揃えます。"
                )
            train_ui_mode = st.radio(
                "操作モード",
                ["かんたん", "標準", "詳細"],
                horizontal=True,
                index=0,
                key="nf_lab_train_ui_mode",
            )
            preset_defs = available_nf_presets(int(settings.default_horizon))
            preset_col1, preset_col2, preset_col3 = st.columns([2, 1, 1])
            selected_preset = preset_col1.selectbox(
                "おすすめプリセット",
                list(preset_defs.keys()),
                index=(
                    list(preset_defs.keys()).index("おすすめ設定を自動入力")
                    if "おすすめ設定を自動入力" in preset_defs
                    else 0
                ),
                key="nf_lab_train_recommended_preset",
            )
            if preset_col2.button("最短で試す", key="nf_lab_apply_quick_preset"):
                queue_nf_preset(st.session_state, preset_defs["最短で試す"], source="quick")
                _publish_notification(
                    kind=NotificationEventKind.ACTION_CONFIRMED,
                    severity=NotificationSeverity.SUCCESS,
                    title="最短プリセットを適用しました",
                    message="必須項目を埋めやすい初期値へ更新しました。次はデータ選択と実行前チェックです。",
                    action="preset",
                    status="success",
                )
                st.rerun()
            if preset_col3.button("おすすめ設定を自動入力", key="nf_lab_apply_recommended_preset"):
                queue_nf_preset(st.session_state, preset_defs[selected_preset], source=selected_preset)
                _publish_notification(
                    kind=NotificationEventKind.ACTION_CONFIRMED,
                    severity=NotificationSeverity.SUCCESS,
                    title="おすすめ設定を反映しました",
                    message="推奨初期値を反映しました。次は unique_id と ts_type を確認してください。",
                    action="preset",
                    status="success",
                )
                st.rerun()
            if applied_preset_values:
                st.caption(
                    "プリセット反映済み: "
                    + ", ".join([f"{k}={applied_preset_values[k]}" for k in sorted(applied_preset_values.keys())[:6]])
                )
            try:
                from loto_forecast.models.neuralforecast_model import (  # noqa: PLC0415
                    AUTO_MODEL_NAMES,
                    get_model_exog_support,
                )
                from loto_forecast.models.registry import _resolve_model_class, get_adapter  # noqa: PLC0415

                model_choices = sorted(list(AUTO_MODEL_NAMES))
                adapter = get_adapter("neuralforecast_auto")
            except Exception:
                model_choices = ["AutoNHITS", "AutoNBEATSx", "AutoTFT", "AutoMLP", "AutoDLinear", "AutoPatchTST"]
                adapter = None
                get_model_exog_support = None  # type: ignore[assignment]
                _resolve_model_class = None  # type: ignore[assignment]

            max_cpu = max(1, int(os.cpu_count() or 1))
            max_gpu = 0
            if TORCH_AVAILABLE:
                try:
                    max_gpu = int(torch.cuda.device_count())
                except Exception:
                    max_gpu = 0

            top1, top2, top3 = st.columns([2, 1, 1])
            _sanitize_nf_train_widget_state(
                st.session_state,
                model_choices=model_choices,
                backend_options=["ray"] if str(st.session_state.get("nf_lab_train_model", "")) == "AutoHINT" else ["optuna", "ray"],
                loss_options=["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "HUBER"],
                search_options=(
                    ["RandomSampler", "TPESampler", "CmaEsSampler", "NSGAIISampler"]
                    if str(st.session_state.get("nf_lab_train_backend", "optuna")) == "optuna"
                    else ["BasicVariantGenerator", "OptunaSearch", "HyperOptSearch", "BayesOptSearch"]
                ),
                dataset_input_method_options=DATASET_INPUT_METHOD_OPTIONS,
                dataframe_backend_options=_supported_backends_for_input_method(
                    str(st.session_state.get("nf_lab_train_dataset_input_method", "db_table") or "db_table")
                )
                or DATAFRAME_BACKEND_OPTIONS,
            )
            model_select_kwargs: dict[str, Any] = {"key": "nf_lab_train_model"}
            if "nf_lab_train_model" not in st.session_state:
                model_select_kwargs["index"] = model_choices.index("AutoNHITS") if "AutoNHITS" in model_choices else 0
            tr_model = top1.selectbox("model", model_choices, **model_select_kwargs)
            auto_apply_defaults = top2.toggle(
                "モデル変更時にデフォルト自動反映", value=True, key="nf_lab_train_auto_apply_defaults"
            )
            use_max_resources = top3.toggle("cpus/gpus 自動最大値", value=True, key="nf_lab_train_use_max_resources")

            model_support = {"futr": False, "hist": False, "stat": False}
            if get_model_exog_support is not None:
                try:
                    model_support = dict(get_model_exog_support(tr_model))
                except Exception:
                    model_support = {"futr": False, "hist": False, "stat": False}

            accepted_params: list[str] = []
            required_model_params: list[str] = []
            reserved_param_specs: dict[str, dict[str, Any]] = {}
            if adapter is not None:
                try:
                    v = adapter.validate(model_name=tr_model, model_params={})
                    accepted_params = sorted(list(v.get("accepted_params", [])))
                    required_model_params = sorted(list(v.get("required_model_params", [])))
                    reserved_param_specs = dict(v.get("reserved_param_specs", {}))
                except Exception:
                    accepted_params = []
                    required_model_params = []
                    reserved_param_specs = {}

            blocked_manual_params = {"h", "loss", "valid_loss", "search_alg"}
            accepted_params_safe = [
                param_name
                for param_name in (_parameter_name(p) for p in accepted_params)
                if param_name and param_name not in blocked_manual_params
            ]

            sig_defaults: dict[str, Any] = {}
            if _resolve_model_class is not None:
                try:
                    cls = _resolve_model_class(tr_model)
                    sig = inspect.signature(cls.__init__)
                    for p in sig.parameters.values():
                        if p.name == "self" or p.kind not in {
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        }:
                            continue
                        if p.name in {"h", "loss", "valid_loss"}:
                            continue
                        if p.default is inspect._empty:
                            continue
                        if isinstance(p.default, (str, int, float, bool, dict, list)) or p.default is None:
                            sig_defaults[p.name] = p.default
                except Exception:
                    sig_defaults = {}

            default_param_values: dict[str, Any] = {
                "backend": "optuna",
                "num_samples": 10,
                "seed": 1,
                "search_alg_name": "BasicVariantGenerator",
                "cpus": max_cpu,
                "gpus": max_gpu,
                "refit_with_val": False,
                "verbose": False,
                "strict_exog": bool(
                    model_support.get("futr") or model_support.get("hist") or model_support.get("stat")
                ),
                "run_cross_validation": False,
                "local_scaler_type": None,
                "local_static_scaler_type": None,
                "nf_fit_kwargs": {},
                "nf_predict_kwargs": {"h": settings.default_horizon},
                "nf_cross_validation_kwargs": {"n_windows": 3, "step_size": 1, "refit": False},
                "nf_save_kwargs": {"save_dataset": False, "overwrite": True},
                "nf_load_kwargs": {},
                "nf_predict_insample_kwargs": {"step_size": 1},
                "futr_exog_list": [],
                "hist_exog_list": [],
                "stat_exog_list": [],
            }
            default_param_values.update(sig_defaults)
            for req in required_model_params:
                if req in default_param_values:
                    continue
                if req == "input_size":
                    default_param_values[req] = -1
                elif req in {"hidden_size", "encoder_hidden_size", "decoder_hidden_size"}:
                    default_param_values[req] = 32
                elif req in {"n_head"}:
                    default_param_values[req] = 4
                elif req in {"max_steps"}:
                    default_param_values[req] = 500
                elif req in {"learning_rate"}:
                    default_param_values[req] = 1e-3
                elif req in {"batch_size"}:
                    default_param_values[req] = 32
                elif req in {"windows_batch_size"}:
                    default_param_values[req] = 128
                else:
                    default_param_values[req] = 1

            model_changed = st.session_state.get("nf_lab_train_model_last") != tr_model
            st.session_state["nf_lab_train_model_last"] = tr_model
            if model_changed and auto_apply_defaults and not preset_applied_this_run:
                st.session_state["nf_lab_train_search_alg_override"] = ""
                st.session_state["nf_lab_train_refit"] = bool(default_param_values.get("refit_with_val", False))
                st.session_state["nf_lab_train_verbose"] = bool(default_param_values.get("verbose", False))
                st.session_state["nf_lab_train_strict_exog"] = bool(default_param_values.get("strict_exog", True))
                st.session_state["nf_lab_train_run_cv"] = bool(default_param_values.get("run_cross_validation", False))
                st.session_state["nf_lab_train_cpus"] = int(max_cpu)
                st.session_state["nf_lab_train_gpus"] = int(max_gpu)
                st.session_state["nf_lab_train_params_json"] = "{}"
                base_selected = [p for p in ["backend", "num_samples", "seed"] if p in accepted_params_safe]
                base_selected.extend(
                    [p for p in required_model_params if p in accepted_params_safe and p not in base_selected]
                )
                st.session_state["nf_lab_train_selected_params"] = base_selected
            elif preset_applied_this_run:
                if "nf_lab_train_search_alg_choice" in applied_preset_runtime_values:
                    st.session_state["nf_lab_train_search_alg_choice"] = str(
                        applied_preset_runtime_values["nf_lab_train_search_alg_choice"]
                    )
                if "nf_lab_train_run_cv" in applied_preset_runtime_values:
                    st.session_state["nf_lab_train_run_cv"] = bool(applied_preset_runtime_values["nf_lab_train_run_cv"])

            st.caption(
                f"モデル外生変数対応: F={bool(model_support.get('futr'))} / H={bool(model_support.get('hist'))} / S={bool(model_support.get('stat'))}"
            )
            if preset_applied_this_run:
                st.caption(
                    "プリセットソース: "
                    + str(applied_preset_source or "manual")
                    + " / 主な反映値: "
                    + ", ".join([f"{k}={applied_preset_values[k]}" for k in sorted(applied_preset_values.keys())[:8]])
                )

            # Dedicated AutoModel controls for frequently tuned runtime/search params.
            st.markdown("**主要パラメータ（縦並び）**")
            _render_help_hint(
                "model",
                "使用するAutoModel種別です。",
                "モデル構造・外生変数対応(F/H/S)・必要なハイパーパラメータが変わります。"
                " 初回は AutoNHITS か AutoDLinear を推奨します。",
            )
            backend_options = ["ray"] if str(tr_model) == "AutoHINT" else ["optuna", "ray"]
            tr_backend = st.selectbox("backend", backend_options, index=0, key="nf_lab_train_backend")
            _render_help_hint(
                "backend",
                "探索エンジン(optuna/ray)を選びます。",
                "optuna は単機で軽量、ray は並列分散向きです。 search_alg の選択候補は backend に連動します。",
            )
            tr_num_samples = int(
                st.number_input(
                    "num_samples",
                    min_value=1,
                    max_value=20000,
                    value=10,
                    step=1,
                    key="nf_lab_train_num_samples",
                )
            )
            _render_help_hint(
                "num_samples",
                "探索試行回数です。",
                "値を増やすと探索品質は上がる可能性がありますが、実行時間とコストが増えます。"
                " まずは 10-30 で検証し、必要時に増やしてください。",
            )
            loss_opts = ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "HUBER"]
            tr_loss = st.selectbox("loss", loss_opts, index=0, key="nf_lab_train_loss_name")
            _render_help_hint(
                "loss",
                "学習時に最小化する指標です。",
                "MAE は外れ値に比較的頑健、MSE/RMSE は大きな誤差を強く罰します。"
                " 迷った場合は MAE から開始してください。",
            )
            valid_loss_opts = [str(tr_loss)]
            tr_valid_loss = str(tr_loss)
            st.text_input("valid_loss", value=tr_valid_loss, key="nf_lab_train_valid_loss_name_locked", disabled=True)
            _render_help_hint(
                "valid_loss",
                "探索時のモデル選択に使う検証指標です。",
                "運用ルールで `loss` と同じ値に固定されます。",
            )
            search_opts = (
                ["RandomSampler", "TPESampler", "CmaEsSampler", "NSGAIISampler"]
                if tr_backend == "optuna"
                else ["BasicVariantGenerator", "OptunaSearch", "HyperOptSearch", "BayesOptSearch"]
            )
            tr_search = st.selectbox("search_alg", search_opts, index=0, key="nf_lab_train_search_alg_choice")
            _render_help_hint(
                "search_alg",
                "探索アルゴリズムです。",
                "optuna なら TPESampler が実運用で使いやすく、ray なら BasicVariantGenerator が安全な初期値です。",
            )
            scaler_opts = ["(none)", "standard", "robust", "robust-iqr", "minmax", "boxcox"]
            tr_local_scaler_type = st.selectbox(
                "local_scaler_type (= local_static_scaler_type)",
                scaler_opts,
                index=0,
                key="nf_lab_train_local_scaler_type",
            )
            freq_default = str(st.session_state.get("nf_lab_train_freq", settings.freq) or settings.freq)
            freq_options = ["D", "W", "W-WED", "W-THU", "M", "MS", "Q", "QS", "Y", "YS"]
            if freq_default not in freq_options:
                freq_options = [freq_default] + [x for x in freq_options if x != freq_default]
            tr_freq = st.selectbox(
                "freq (NeuralForecast)",
                freq_options,
                index=(freq_options.index(freq_default) if freq_default in freq_options else 0),
                key="nf_lab_train_freq",
            )
            _render_help_hint(
                "local_scaler_type",
                "NeuralForecast の系列スケーラ設定です。",
                "`local_scaler_type` と `local_static_scaler_type` は常に同一値で実行されます。",
            )
            if str(tr_model) == "AutoHINT":
                st.info("AutoHINT は backend=ray が必須です。config に reconciliation を含めてください。")

        # Dataset / group / filter controls
        tr_param_builder_errors: list[str] = []
        tr_param_builder_obj: dict[str, Any] = {}
        tr_params_raw = str(st.session_state.get("nf_lab_train_params_json", "{}"))
        axis_plan: dict[str, dict[str, Any]] = {}
        combo_rows: list[dict[str, Any]] = []
        combo_df: pd.DataFrame = pd.DataFrame()
        dataset_col_names: list[str] = []
        futr_prefixed_cols: list[str] = []
        hist_prefixed_cols: list[str] = []
        stat_prefixed_cols: list[str] = []
        exog_prefix_cache: dict[str, dict[str, list[str]]] = {}
        try:
            from loto_forecast.models.neuralforecast_model import (  # noqa: PLC0415
                AUTO_MODEL_NAMES,
                get_model_exog_support,
            )

            model_choices = sorted(list(AUTO_MODEL_NAMES))
        except Exception:
            model_choices = ["AutoNHITS", "AutoNBEATSx", "AutoTFT", "AutoMLP", "AutoDLinear", "AutoPatchTST"]
            get_model_exog_support = None  # type: ignore[assignment]

        max_cpu = max(1, int(os.cpu_count() or 1))
        max_gpu = 0
        if TORCH_AVAILABLE:
            try:
                max_gpu = int(torch.cuda.device_count())
            except Exception:
                max_gpu = 0
        train_ui_mode = str(st.session_state.get("nf_lab_train_ui_mode", "かんたん"))
        use_max_resources = bool(st.session_state.get("nf_lab_train_use_max_resources", True))
        tr_model = str(st.session_state.get("nf_lab_train_model", model_choices[0] if model_choices else "AutoNHITS"))
        if model_choices and tr_model not in model_choices:
            tr_model = str(model_choices[0])
        tr_backend = str(st.session_state.get("nf_lab_train_backend", "optuna"))
        if tr_backend not in {"optuna", "ray"}:
            tr_backend = "optuna"
        try:
            tr_num_samples = int(st.session_state.get("nf_lab_train_num_samples", 10))
        except Exception:
            tr_num_samples = 10
        tr_num_samples = max(1, int(tr_num_samples))
        tr_loss = str(st.session_state.get("nf_lab_train_loss_name", "MAE") or "MAE")
        if tr_loss not in {"MAE", "MSE", "RMSE", "MAPE", "SMAPE", "HUBER"}:
            tr_loss = "MAE"
        tr_valid_loss = str(tr_loss)
        search_opts = (
            ["RandomSampler", "TPESampler", "CmaEsSampler", "NSGAIISampler"]
            if tr_backend == "optuna"
            else ["BasicVariantGenerator", "OptunaSearch", "HyperOptSearch", "BayesOptSearch"]
        )
        tr_search = str(st.session_state.get("nf_lab_train_search_alg_choice", search_opts[0]) or search_opts[0])
        if tr_search not in search_opts:
            tr_search = str(search_opts[0])
        tr_local_scaler_type = str(st.session_state.get("nf_lab_train_local_scaler_type", "(none)"))
        tr_freq = str(st.session_state.get("nf_lab_train_freq", settings.freq) or settings.freq)
        tr_nf_fit_raw = str(st.session_state.get("nf_lab_train_nf_fit", "{}"))
        tr_nf_predict_raw = str(st.session_state.get("nf_lab_train_nf_predict", "{}"))
        tr_nf_cv_raw = str(st.session_state.get("nf_lab_train_nf_cv", "{}"))
        tr_nf_save_raw = str(st.session_state.get("nf_lab_train_nf_save", "{}"))
        tr_nf_load_raw = str(st.session_state.get("nf_lab_train_nf_load", "{}"))
        tr_nf_ins_raw = str(st.session_state.get("nf_lab_train_nf_ins", "{}"))
        tr_futr_raw = str(st.session_state.get("nf_lab_train_futr", "[]"))
        tr_hist_raw = str(st.session_state.get("nf_lab_train_hist", "[]"))
        tr_stat_raw = str(st.session_state.get("nf_lab_train_stat", "[]"))
        tr_exog_auto = bool(st.session_state.get("nf_lab_train_exog_auto", True))
        dataset_schema_options = sorted(list({s for s, _ in tables if s in {"dataset", "exog"}})) if tables else []
        if not dataset_schema_options:
            dataset_schema_options = [settings.db_schema]
        tr_dataset_schema = str(st.session_state.get("nf_lab_train_dataset_schema", settings.db_schema))
        if tr_dataset_schema not in dataset_schema_options:
            tr_dataset_schema = (
                str(settings.db_schema)
                if str(settings.db_schema) in dataset_schema_options
                else str(dataset_schema_options[0])
            )
        dataset_table_options = sorted([t for s, t in tables if s == tr_dataset_schema]) if tables else []
        tr_dataset_table = str(
            st.session_state.get(
                "nf_lab_train_dataset_table",
                st.session_state.get("nf_lab_train_dataset_table_text", settings.db_table),
            )
        )
        if dataset_table_options and tr_dataset_table not in dataset_table_options:
            tr_dataset_table = (
                str(settings.db_table)
                if str(settings.db_table) in dataset_table_options
                else str(dataset_table_options[0])
            )
        tr_dataset_where = str(st.session_state.get("nf_lab_train_dataset_where", ""))
        tr_dataset_input_method = str(
            st.session_state.get("nf_lab_train_dataset_input_method", "db_table") or "db_table"
        )
        if tr_dataset_input_method not in DATASET_INPUT_METHOD_OPTIONS:
            tr_dataset_input_method = "db_table"
        tr_backend_candidates = _supported_backends_for_input_method(str(tr_dataset_input_method))
        tr_dataframe_backend = str(st.session_state.get("nf_lab_train_dataframe_backend", DATAFRAME_BACKEND_OPTIONS[0]))
        if tr_dataframe_backend not in tr_backend_candidates:
            tr_dataframe_backend = (
                str(tr_backend_candidates[0]) if tr_backend_candidates else str(DATAFRAME_BACKEND_OPTIONS[0])
            )
        tr_dataset_path = str(st.session_state.get("nf_lab_train_dataset_path", ""))
        tr_dataset_sql = str(st.session_state.get("nf_lab_train_dataset_sql", ""))
        group_mode_label_state = str(
            st.session_state.get("nf_lab_train_group_mode_label", "loto+unique_id+ts_type ごと")
        )
        group_mode = "loto_unique_id_ts_type" if group_mode_label_state.startswith("loto+unique_id") else "loto_ts_type"
        loto_vals: list[str] = []
        uid_vals: list[str] = []
        ts_type_vals: list[str] = []
        tr_loto = (
            [str(x) for x in st.session_state.get("nf_lab_train_loto", []) if str(x).strip()]
            if isinstance(st.session_state.get("nf_lab_train_loto", []), list)
            else []
        )
        tr_uid = (
            [str(x) for x in st.session_state.get("nf_lab_train_uid", []) if str(x).strip()]
            if isinstance(st.session_state.get("nf_lab_train_uid", []), list)
            else []
        )
        tr_ts_type = (
            [str(x) for x in st.session_state.get("nf_lab_train_ts_type", []) if str(x).strip()]
            if isinstance(st.session_state.get("nf_lab_train_ts_type", []), list)
            else []
        )
        auto_h_by_uid = bool(st.session_state.get("nf_lab_train_auto_h_by_uid", True))
        try:
            manual_h = int(st.session_state.get("nf_lab_train_h", settings.default_horizon))
        except Exception:
            manual_h = int(settings.default_horizon)
        manual_h = max(1, min(3650, int(manual_h)))
        uid_count = len(tr_uid) if tr_uid else 0
        effective_h = int(uid_count) if auto_h_by_uid and uid_count > 0 else int(manual_h)
        if effective_h <= 0:
            effective_h = 1
        if d_section == "全パラメータ選択":
            dataset_schema_options = sorted(list({s for s, _ in tables if s in {"dataset", "exog"}})) if tables else []
            if not dataset_schema_options:
                dataset_schema_options = [settings.db_schema]
            pending_schema_value = str(st.session_state.get("nf_lab_train_dataset_schema", settings.db_schema) or settings.db_schema)
            if pending_schema_value not in dataset_schema_options:
                pending_schema_value = str(settings.db_schema) if str(settings.db_schema) in dataset_schema_options else str(dataset_schema_options[0])
            dataset_table_options = sorted([t for s, t in tables if s == pending_schema_value]) if tables else []
            default_table = (
                settings.db_table
                if pending_schema_value == settings.db_schema
                else (dataset_table_options[0] if dataset_table_options else "")
            )
            _sanitize_nf_train_widget_state(
                st.session_state,
                model_choices=model_choices,
                backend_options=backend_options,
                loss_options=loss_opts,
                search_options=search_opts,
                dataset_input_method_options=DATASET_INPUT_METHOD_OPTIONS,
                dataframe_backend_options=tr_backend_candidates if tr_backend_candidates else DATAFRAME_BACKEND_OPTIONS,
                dataset_schema_options=dataset_schema_options,
                dataset_table_options=dataset_table_options,
                default_dataset_schema=pending_schema_value,
                default_dataset_table=default_table,
            )
            tr_dataset_schema = st.selectbox(
                "dataset schema",
                dataset_schema_options,
                index=(
                    dataset_schema_options.index(settings.db_schema)
                    if settings.db_schema in dataset_schema_options
                    else 0
                ),
                key="nf_lab_train_dataset_schema",
            )
            _render_help_hint(
                "dataset_schema",
                "学習元データのスキーマです。",
                "通常は `dataset` を選択します。テーブル参照先が変わるため、誤ると別データで学習されます。",
            )
            dataset_table_options = sorted([t for s, t in tables if s == tr_dataset_schema]) if tables else []
            default_table = (
                settings.db_table
                if tr_dataset_schema == settings.db_schema
                else (dataset_table_options[0] if dataset_table_options else "")
            )
            if dataset_table_options:
                _sanitize_nf_train_widget_state(
                    st.session_state,
                    model_choices=model_choices,
                    backend_options=backend_options,
                    loss_options=loss_opts,
                    search_options=search_opts,
                    dataset_input_method_options=DATASET_INPUT_METHOD_OPTIONS,
                    dataframe_backend_options=tr_backend_candidates if tr_backend_candidates else DATAFRAME_BACKEND_OPTIONS,
                    dataset_schema_options=dataset_schema_options,
                    dataset_table_options=dataset_table_options,
                    default_dataset_schema=tr_dataset_schema,
                    default_dataset_table=default_table,
                )
            if not dataset_table_options:
                tr_dataset_table = st.text_input(
                    "dataset table", value=default_table, key="nf_lab_train_dataset_table_text"
                )
            else:
                tr_dataset_table = st.selectbox(
                    "dataset table",
                    dataset_table_options,
                    index=(dataset_table_options.index(default_table) if default_table in dataset_table_options else 0),
                    key="nf_lab_train_dataset_table",
                )
            _render_help_hint(
                "dataset_table",
                "学習対象テーブルです。",
                "`loto_y_ts` は原系列、`*_unified` は特徴量統合済み系列です。"
                " 初回は列構成が単純なテーブルから試すと失敗原因を切り分けやすくなります。",
            )
            tr_dataset_where = st.text_input("dataset where SQL (optional)", value="", key="nf_lab_train_dataset_where")
            source_c1, source_c2 = st.columns(2)
            tr_dataset_input_method = source_c1.selectbox(
                "dataset input method",
                DATASET_INPUT_METHOD_OPTIONS,
                index=0,
                key="nf_lab_train_dataset_input_method",
            )
            tr_backend_candidates = _supported_backends_for_input_method(str(tr_dataset_input_method))
            tr_backend_default = str(
                st.session_state.get("nf_lab_train_dataframe_backend", DATAFRAME_BACKEND_OPTIONS[0])
            )
            if tr_backend_default not in tr_backend_candidates:
                tr_backend_default = (
                    str(tr_backend_candidates[0]) if tr_backend_candidates else DATAFRAME_BACKEND_OPTIONS[0]
                )
            tr_dataframe_backend = source_c2.selectbox(
                "dataframe backend",
                tr_backend_candidates if tr_backend_candidates else DATAFRAME_BACKEND_OPTIONS,
                index=(
                    (tr_backend_candidates.index(tr_backend_default))
                    if (tr_backend_candidates and tr_backend_default in tr_backend_candidates)
                    else 0
                ),
                key="nf_lab_train_dataframe_backend",
            )
            tr_dataset_path = st.text_input(
                "dataset path (csv/parquet/json 用)",
                value=str(st.session_state.get("nf_lab_train_dataset_path", "")),
                key="nf_lab_train_dataset_path",
            )
            tr_dataset_sql = st.text_area(
                "dataset SQL (db_sql 用)",
                value=str(st.session_state.get("nf_lab_train_dataset_sql", "")),
                height=80,
                key="nf_lab_train_dataset_sql",
            )
            st.caption(
                "読み込み方式が `db_table` 以外の場合、schema/table の代わりに "
                "`dataset path` または `dataset SQL` を使用します。"
            )
            if tr_backend_candidates:
                st.caption(f"選択中 input method の backend 候補: {', '.join(tr_backend_candidates)}")
            available_backends = _available_dataframe_backends()
            unavailable_backends = [b for b in DATAFRAME_BACKEND_OPTIONS if b not in set(available_backends)]
            if unavailable_backends:
                st.caption(
                    "現環境で未導入のbackend: " + ", ".join(unavailable_backends) + " (必要なら環境へ追加インストール)"
                )
            with st.expander("DataFrame候補と読み込み対応表", expanded=False):
                _show_df(_dataset_loader_support_df(), hide_index=True)
            group_mode_label = st.selectbox(
                "学習単位",
                ["loto+unique_id+ts_type ごと", "loto+ts_type ごと (unique_id集約)"],
                index=0,
                key="nf_lab_train_group_mode_label",
            )
            _render_help_hint(
                "group_mode",
                "系列の分割単位です。",
                "`loto+unique_id+ts_type` は系列を細かく分割、`loto+ts_type` は unique_id をまとめて扱います。"
                " 分割粒度により学習サンプル数と h の意味が変わります。",
            )
            group_mode = "loto_unique_id_ts_type" if group_mode_label.startswith("loto+unique_id") else "loto_ts_type"

            loto_vals = []
            uid_vals = []
            ts_type_vals = []
            if engine is not None and tr_dataset_table and str(tr_dataset_input_method) == "db_table":
                try:
                    cols_df = _table_columns(engine, tr_dataset_schema, tr_dataset_table)
                    cols = set(cols_df["column_name"].astype(str).tolist()) if not cols_df.empty else set()
                    dataset_col_names = sorted(list(cols))
                    futr_prefixed_cols = [c for c in dataset_col_names if str(c).startswith("feat_")]
                    hist_prefixed_cols = [c for c in dataset_col_names if str(c).startswith("hist_")]
                    stat_prefixed_cols = [c for c in dataset_col_names if str(c).startswith("stat_")]
                except Exception:
                    cols = set()
                    dataset_col_names = []
                    futr_prefixed_cols = []
                    hist_prefixed_cols = []
                    stat_prefixed_cols = []

                def _load_distinct_values(col_name: str, limit: int = 300) -> list[str]:
                    if col_name not in cols:
                        return []
                    q = (
                        f"SELECT DISTINCT {_safe_ident(col_name)} AS v "
                        f"FROM {_safe_ident(tr_dataset_schema)}.{_safe_ident(tr_dataset_table)} "
                        f"WHERE {_safe_ident(col_name)} IS NOT NULL "
                        f"ORDER BY {_safe_ident(col_name)} LIMIT :limit"
                    )
                    d = _query_df(engine, q, {"limit": int(limit)})
                    return [str(x) for x in d["v"].astype(str).tolist()] if not d.empty and "v" in d.columns else []

                loto_vals = _load_distinct_values("loto")
                uid_vals = _load_distinct_values(settings.id_col)
                ts_type_vals = _load_distinct_values("ts_type")

            tr_loto = st.multiselect("loto", loto_vals, default=[], key="nf_lab_train_loto")
            uid_disabled_by_group_mode = str(group_mode) == "loto_ts_type"
            if uid_disabled_by_group_mode:
                uid_state = st.session_state.get("nf_lab_train_uid")
                if isinstance(uid_state, list) and uid_state:
                    st.session_state["nf_lab_train_uid"] = []
            tr_uid = st.multiselect(
                "unique_id",
                uid_vals,
                default=[],
                key="nf_lab_train_uid",
                disabled=uid_disabled_by_group_mode,
            )
            if uid_disabled_by_group_mode:
                tr_uid = []
                st.caption("学習単位 候補=loto_ts_type のため unique_id 候補は自動で None 扱いになります。")
            tr_ts_type = st.multiselect("ts_type", ts_type_vals, default=[], key="nf_lab_train_ts_type")
            _render_help_hint(
                "group_filters",
                "学習対象の系列フィルタです。",
                "未選択は全件対象、選択すると条件一致系列のみ対象になります。"
                " まずは `loto=bingo5`, `unique_id=N1`, `ts_type=raw` のように狭めると検証しやすいです。",
            )

            auto_h_by_uid = st.toggle("hをunique_id一意数に自動設定", value=True, key="nf_lab_train_auto_h_by_uid")
            manual_h = int(
                st.number_input(
                    "manual h",
                    min_value=1,
                    max_value=3650,
                    value=max(1, int(settings.default_horizon)),
                    step=1,
                    key="nf_lab_train_h",
                    disabled=auto_h_by_uid,
                )
            )

            uid_count = len(tr_uid) if tr_uid else 0
            if (
                uid_count <= 0
                and engine is not None
                and tr_dataset_table
                and str(tr_dataset_input_method) == "db_table"
            ):
                try:
                    where_parts = [f"{_safe_ident(settings.id_col)} IS NOT NULL"]
                    if tr_dataset_where.strip():
                        where_parts.append(f"({tr_dataset_where.strip()})")

                    def _quote_list(vals: list[str]) -> str:
                        return ", ".join(["'" + str(v).replace("'", "''") + "'" for v in vals])

                    if tr_loto:
                        where_parts.append(f"{_safe_ident('loto')} IN ({_quote_list(tr_loto)})")
                    if tr_uid:
                        where_parts.append(f"{_safe_ident(settings.id_col)} IN ({_quote_list(tr_uid)})")
                    if tr_ts_type:
                        where_parts.append(f"{_safe_ident('ts_type')} IN ({_quote_list(tr_ts_type)})")
                    q_uid = (
                        f"SELECT COUNT(DISTINCT {_safe_ident(settings.id_col)}) AS uid_cnt "
                        f"FROM {_safe_ident(tr_dataset_schema)}.{_safe_ident(tr_dataset_table)} "
                        f"WHERE {' AND '.join(where_parts)}"
                    )
                    uid_df = _query_df(engine, q_uid)
                    uid_count = int(uid_df.iloc[0]["uid_cnt"]) if not uid_df.empty else 0
                except Exception:
                    uid_count = 0

            effective_h = int(uid_count) if auto_h_by_uid and uid_count > 0 else int(manual_h)
            st.metric("effective h", int(effective_h))
            if auto_h_by_uid and uid_count <= 0:
                st.info("unique_id数を取得できないため manual h を使用します。")

        def _prefixed_exog_cols_for_table(schema_v: str, table_v: str) -> dict[str, list[str]]:
            cache_key = f"{schema_v}.{table_v}"
            if cache_key in exog_prefix_cache:
                cached = exog_prefix_cache[cache_key]
                return {
                    "futr_exog": list(cached.get("futr_exog", [])),
                    "hist_exog": list(cached.get("hist_exog", [])),
                    "stat_exog": list(cached.get("stat_exog", [])),
                }
            cols_local: list[str] = []
            if str(schema_v) == str(tr_dataset_schema) and str(table_v) == str(tr_dataset_table) and dataset_col_names:
                cols_local = list(dataset_col_names)
            elif engine is not None and str(table_v).strip():
                try:
                    cols_df_local = _table_columns(engine, schema_v, table_v)
                    cols_local = (
                        [str(c) for c in cols_df_local["column_name"].astype(str).tolist()]
                        if not cols_df_local.empty
                        else []
                    )
                except Exception:
                    cols_local = []
            out = {
                "futr_exog": [c for c in cols_local if str(c).startswith("feat_")],
                "hist_exog": [c for c in cols_local if str(c).startswith("hist_")],
                "stat_exog": [c for c in cols_local if str(c).startswith("stat_")],
            }
            exog_prefix_cache[cache_key] = {
                "futr_exog": list(out["futr_exog"]),
                "hist_exog": list(out["hist_exog"]),
                "stat_exog": list(out["stat_exog"]),
            }
            return out

        if d_section == "全パラメータ選択":
            sample_catalog: dict[str, Any] = {
                "model": "AutoNHITS",
                "backend": "optuna",
                "num_samples": 10,
                "loss": "MAE",
                "valid_loss": "MAE",
                "search_alg": "TPESampler",
                "alias": "nhits_bingo5_baseline",
                "callbacks": [],
                "nf_fit_kwargs": {"val_size": 28, "verbose": False},
                "nf_predict_kwargs": {"h": int(effective_h)},
                "nf_cross_validation_kwargs": {"n_windows": 3, "step_size": 1, "refit": False, "h": int(effective_h)},
                "nf_save_kwargs": {"save_dataset": False, "overwrite": True},
                "nf_load_kwargs": {"verbose": False},
            }
            with st.expander("サンプルコピー（初見向け）", expanded=False):
                st.caption("各パラメータの最小サンプルをコピーして、そのまま入力に流用できます。")
                sample_key = st.selectbox(
                    "サンプル対象パラメータ",
                    list(sample_catalog.keys()),
                    index=0,
                    key="nf_lab_train_sample_key",
                )
                sample_value = sample_catalog.get(sample_key)
                sample_text = (
                    json.dumps(sample_value, ensure_ascii=False)
                    if isinstance(sample_value, (dict, list, bool, int, float))
                    else str(sample_value)
                )
                st.code(sample_text, language="json")
                _render_copy_button(
                    sample_text, key=f"nf_lab_train_sample_copy_{sample_key}", label=f"{sample_key} sample copy"
                )

            selected_params: list[str] = []
            if str(train_ui_mode) == "かんたん":
                st.success(
                    "かんたんモードでは危険な上書き（`h/loss/valid_loss/search_alg`）を防ぐため、"
                    "個別パラメータ上書きを無効化しています。"
                )
                tr_params_raw = st.text_area(
                    "base params-json (必要時のみ。通常は空のまま)",
                    value=tr_params_raw,
                    height=110,
                    key="nf_lab_train_params_json",
                )
            else:
                selected_default = list(st.session_state.get("nf_lab_train_selected_params", []))
                if not selected_default:
                    selected_default = [p for p in ["backend", "num_samples", "seed"] if p in accepted_params_safe]
                selected_params = st.multiselect(
                    "選択パラメータ (model + reserved)",
                    options=accepted_params_safe,
                    default=[p for p in selected_default if p in accepted_params_safe],
                    key="nf_lab_train_selected_params",
                )
                if blocked_manual_params:
                    st.caption("自動管理パラメータは選択対象外: " + ", ".join(sorted(list(blocked_manual_params))))

                for raw_param in selected_params:
                    param_name = _parameter_name(raw_param)
                    spec = dict(reserved_param_specs.get(param_name, {}))
                    type_hint = str(spec.get("type", "") or "").strip()
                    default_val = default_param_values.get(param_name)
                    wk = f"nf_lab_train_param_{_slug(tr_model)}_{_slug(param_name)}"
                    if model_changed and auto_apply_defaults and wk not in st.session_state:
                        if isinstance(default_val, (dict, list)):
                            st.session_state[wk] = json.dumps(default_val, ensure_ascii=False)
                        else:
                            st.session_state[wk] = default_val

                    c_left, c_right = st.columns([2, 2])
                    with c_left:
                        if type_hint == "bool" or isinstance(default_val, bool):
                            bool_v = st.toggle(
                                param_name, value=bool(default_val) if default_val is not None else False, key=wk
                            )
                            tr_param_builder_obj[param_name] = bool(bool_v)
                        elif type_hint.startswith("int") or isinstance(default_val, int):
                            min_v = 1 if type_hint == "int>0" else 0 if type_hint == "int>=0" else -1_000_000
                            int_default = int(default_val) if default_val is not None else max(min_v, 0)
                            int_v = int(st.number_input(param_name, min_value=min_v, value=int_default, step=1, key=wk))
                            tr_param_builder_obj[param_name] = int_v
                        elif isinstance(default_val, float):
                            float_default = float(default_val)
                            float_v = float(
                                st.number_input(param_name, value=float_default, step=0.1, format="%.6f", key=wk)
                            )
                            tr_param_builder_obj[param_name] = float_v
                        elif type_hint in {"dict", "list"} or isinstance(default_val, (dict, list)):
                            json_default = json.dumps(
                                default_val if default_val is not None else ({} if type_hint == "dict" else []),
                                ensure_ascii=False,
                            )
                            raw_v = st.text_area(param_name, value=json_default, height=76, key=wk)
                            try:
                                obj = json.loads(str(raw_v or "").strip() or ("{}" if type_hint == "dict" else "[]"))
                                if type_hint == "dict" and not isinstance(obj, dict):
                                    raise ValueError("dict expected")
                                if type_hint == "list" and not isinstance(obj, list):
                                    raise ValueError("list expected")
                                tr_param_builder_obj[param_name] = obj
                            except Exception as e:
                                tr_param_builder_errors.append(f"{param_name}: {e}")
                        else:
                            text_default = "" if default_val is None else str(default_val)
                            allowed_vals = spec.get("allowed") if isinstance(spec.get("allowed"), list) else []
                            if allowed_vals and all(isinstance(x, (str, int, float, bool)) for x in allowed_vals):
                                allowed_opts = [str(x) for x in allowed_vals]
                                default_opt = str(default_val) if default_val is not None else allowed_opts[0]
                                if default_opt not in allowed_opts:
                                    allowed_opts = [default_opt] + allowed_opts
                                sel = st.selectbox(
                                    param_name, allowed_opts, index=allowed_opts.index(default_opt), key=wk
                                )
                                tr_param_builder_obj[param_name] = sel
                            else:
                                raw_v = st.text_input(param_name, value=text_default, key=wk)
                                if str(raw_v).strip() == "" and param_name not in required_model_params:
                                    pass
                                elif type_hint == "str":
                                    tr_param_builder_obj[param_name] = str(raw_v)
                                else:
                                    parsed: Any = str(raw_v)
                                    try:
                                        parsed = json.loads(str(raw_v))
                                    except Exception:
                                        try:
                                            parsed = ast.literal_eval(str(raw_v))
                                        except Exception:
                                            parsed = str(raw_v)
                                    tr_param_builder_obj[param_name] = parsed
                    with c_right:
                        st.caption(f"type: {type_hint or 'model_param'}")
                        if spec.get("allowed"):
                            st.caption(f"allowed: {spec.get('allowed')}")
                        if param_name in required_model_params:
                            st.caption("required: yes")

                if tr_param_builder_errors:
                    st.error("\n".join(tr_param_builder_errors))
                with st.expander("選択パラメータ(JSON preview)", expanded=False):
                    st.json(tr_param_builder_obj)
                with st.expander("選択したパラメータの組み合わせ表", expanded=False):
                    st.caption(
                        "各パラメータを `展開ON` にすると候補配列の総当たりを表形式で表示します。"
                        " `展開OFF` は現在値で固定です。"
                    )
                    combo_axis_values: dict[str, list[Any]] = {}
                    combo_axis_errors: list[str] = []

                    def _to_combo_cell(v: Any) -> Any:
                        if isinstance(v, (dict, list, tuple, set)):
                            return _stable_json_dumps(v)
                        return v

                    for raw_param in selected_params:
                        param_name = _parameter_name(raw_param)
                        cur_v = tr_param_builder_obj.get(param_name, default_param_values.get(param_name))
                        cur_cell = _to_combo_cell(cur_v)
                        col_expand, col_values = st.columns([1, 3])
                        expand = col_expand.toggle(
                            f"{param_name} 展開",
                            value=False,
                            key=f"nf_lab_combo_expand_{_slug(tr_model)}_{_slug(param_name)}",
                        )
                        if not expand:
                            combo_axis_values[param_name] = [cur_cell]
                            col_values.caption(f"固定: `{cur_cell}`")
                            continue

                        default_candidates = list(cur_v) if isinstance(cur_v, list) and len(cur_v) > 0 else [cur_cell]
                        raw_candidates = col_values.text_input(
                            f"{param_name} 候補(JSON array)",
                            value=json.dumps(default_candidates, ensure_ascii=False),
                            key=f"nf_lab_combo_candidates_{_slug(tr_model)}_{_slug(param_name)}",
                        )
                        try:
                            parsed = json.loads(str(raw_candidates).strip() or "[]")
                            if not isinstance(parsed, list) or len(parsed) == 0:
                                raise ValueError("non-empty JSON array required")
                            combo_axis_values[param_name] = [_to_combo_cell(x) for x in parsed]
                        except Exception as e:
                            combo_axis_errors.append(f"{param_name}: {e}")

                    if combo_axis_errors:
                        st.error("\n".join(combo_axis_errors))
                    else:
                        total_combos = 1
                        for vals in combo_axis_values.values():
                            total_combos *= max(1, len(vals))
                        st.metric("理論組み合わせ数", int(total_combos))
                        show_limit = int(
                            st.number_input(
                                "表示上限行数",
                                min_value=10,
                                max_value=20000,
                                value=1000,
                                step=10,
                                key=f"nf_lab_combo_show_limit_{_slug(tr_model)}",
                            )
                        )
                        combo_rows_preview: list[dict[str, Any]] = []
                        if combo_axis_values:
                            keys = list(combo_axis_values.keys())
                            for i, combo_vals in enumerate(
                                itertools.product(*[combo_axis_values[k] for k in keys]), start=1
                            ):
                                if i > show_limit:
                                    break
                                combo_rows_preview.append({k: v for k, v in zip(keys, combo_vals, strict=False)})
                        combo_preview_df = pd.DataFrame(combo_rows_preview)
                        if combo_preview_df.empty:
                            st.info("表示対象の組み合わせがありません。")
                        else:
                            _show_df(combo_preview_df, hide_index=True)
                            st.download_button(
                                "Download selected_param_combinations.csv",
                                data=combo_preview_df.to_csv(index=False),
                                file_name="selected_param_combinations.csv",
                                mime="text/csv",
                                key=f"nf_lab_combo_download_{_slug(tr_model)}",
                            )

                tr_params_raw = st.text_area(
                    "base params-json (manual merge)",
                    value=tr_params_raw,
                    height=110,
                    key="nf_lab_train_params_json",
                )

        if d_section == "NeuralForecast runtime":
            _render_help_hint(
                "nf_runtime",
                "NeuralForecast本体のAPI呼び出し引数です。",
                "学習(train)で model_params とは別に、`fit/predict/cross_validation/save/load/predict_insample` の"
                " 実行挙動だけを個別に上書きするための領域です。",
            )
            st.caption(
                "ここで設定した値は `meta.json -> nf_runtime_kwargs` に保存され、後続の predict/evaluate/load でも再利用されます。"
            )
            # Keep command/meta runtime h consistent with the training horizon to avoid hidden mismatch.
            st.session_state["nf_lab_train_runtime_sync_h"] = True
            st.toggle(
                "effective h を predict/cross_validation kwargs の h に同期（固定）",
                key="nf_lab_train_runtime_sync_h",
                disabled=True,
            )
            st.caption(
                "`--h` と `--nf-predict-kwargs-json.h` / `--nf-cross-validation-kwargs-json.h` は常に同じ値で出力されます。"
            )
            tr_nf_fit_raw = _json_preset_input(
                "nf-fit-kwargs-json",
                "nf_lab_train_nf_fit",
                presets={
                    "minimal": {},
                    "with_val": {"val_size": 28, "verbose": False},
                    "reuse_init_models": {"use_init_models": True},
                    "all_fit_keys": {
                        "static_df": None,
                        "val_size": 0,
                        "use_init_models": False,
                        "verbose": False,
                        "id_col": "unique_id",
                        "time_col": "ds",
                        "target_col": "y",
                        "distributed_config": None,
                        "prediction_intervals": None,
                    },
                },
                default_name="minimal",
                height=90,
            )
            tr_nf_predict_raw = _json_preset_input(
                "nf-predict-kwargs-json",
                "nf_lab_train_nf_predict",
                presets={
                    "h_only": {"h": int(effective_h)},
                    "quantiles": {"h": int(effective_h), "quantiles": [0.1, 0.5, 0.9]},
                    "levels": {"h": int(effective_h), "level": [80, 90]},
                },
                default_name="h_only",
                height=90,
            )
            tr_nf_cv_raw = _json_preset_input(
                "nf-cross-validation-kwargs-json",
                "nf_lab_train_nf_cv",
                presets={
                    "basic": {"n_windows": 3, "step_size": 1, "refit": False, "h": int(effective_h)},
                    "refit_each": {"n_windows": 3, "step_size": 1, "refit": True, "h": int(effective_h)},
                    "rolling_test": {"test_size": 56, "step_size": 1, "refit": False, "h": int(effective_h)},
                },
                default_name="basic",
                height=90,
            )
            tr_nf_save_raw = _json_preset_input(
                "nf-save-kwargs-json",
                "nf_lab_train_nf_save",
                presets={
                    "light": {"save_dataset": False, "overwrite": True},
                    "full": {"save_dataset": True, "overwrite": True},
                    "safe_no_overwrite": {"save_dataset": False, "overwrite": False},
                },
                default_name="light",
                height=90,
            )
            tr_nf_load_raw = _json_preset_input(
                "nf-load-kwargs-json",
                "nf_lab_train_nf_load",
                presets={
                    "minimal": {},
                    "verbose_on": {"verbose": True},
                },
                default_name="minimal",
                height=80,
            )
            tr_nf_ins_raw = _json_preset_input(
                "nf-predict-insample-kwargs-json",
                "nf_lab_train_nf_ins",
                presets={
                    "basic": {"step_size": 1},
                    "with_levels": {"step_size": 1, "level": [80, 90]},
                },
                default_name="basic",
                height=80,
            )

        if d_section == "exog lists":
            _render_help_hint(
                "exog_lists",
                "外生変数を時点役割ごとに指定します。",
                "futr=将来時点でも既知の列(例: カレンダー)、hist=過去にのみ既知の列、"
                "stat=系列に紐づく静的属性です。モデルのF/H/S対応と一致しない指定は失敗要因になります。",
            )
            tr_exog_auto = st.toggle(
                "exog lists をテーブルから自動取得（推奨）",
                value=bool(st.session_state.get("nf_lab_train_exog_auto", True)),
                key="nf_lab_train_exog_auto",
            )
            exog_cols: list[str] = []
            if engine is not None and str(tr_dataset_table).strip():
                try:
                    ex_cols_df = _table_columns(engine, tr_dataset_schema, tr_dataset_table)
                    exog_cols = (
                        [str(c) for c in ex_cols_df["column_name"].astype(str).tolist()] if not ex_cols_df.empty else []
                    )
                    exog_cols = [
                        c
                        for c in exog_cols
                        if c not in {settings.id_col, settings.time_col, settings.target_col, "loto", "ts_type"}
                    ]
                except Exception:
                    exog_cols = []
            futr_pick = st.multiselect(
                "futr exog columns (dropdown)",
                exog_cols,
                default=[],
                key="nf_lab_train_futr_pick",
                disabled=tr_exog_auto,
            )
            hist_pick = st.multiselect(
                "hist exog columns (dropdown)",
                exog_cols,
                default=[],
                key="nf_lab_train_hist_pick",
                disabled=tr_exog_auto,
            )
            stat_pick = st.multiselect(
                "stat exog columns (dropdown)",
                exog_cols,
                default=[],
                key="nf_lab_train_stat_pick",
                disabled=tr_exog_auto,
            )
            if st.button("選択列をJSONへ反映", key="nf_lab_apply_exog_pick", disabled=tr_exog_auto):
                st.session_state["nf_lab_train_futr"] = json.dumps(futr_pick, ensure_ascii=False)
                st.session_state["nf_lab_train_hist"] = json.dumps(hist_pick, ensure_ascii=False)
                st.session_state["nf_lab_train_stat"] = json.dumps(stat_pick, ensure_ascii=False)
            if tr_exog_auto:
                st.info("自動取得ON: コマンドでは exog-list 引数を省略し、train実行時にテーブル列から自動推定します。")
            tr_futr_raw = st.text_area(
                "futr-exog-list-json (list)", value="[]", height=80, key="nf_lab_train_futr", disabled=tr_exog_auto
            )
            tr_hist_raw = st.text_area(
                "hist-exog-list-json (list)", value="[]", height=80, key="nf_lab_train_hist", disabled=tr_exog_auto
            )
            tr_stat_raw = st.text_area(
                "stat-exog-list-json (list)", value="[]", height=80, key="nf_lab_train_stat", disabled=tr_exog_auto
            )

        if d_section == "固定/網羅メタ反映":
            st.markdown("**固定/網羅メタ反映**")
            st.caption(
                "チェックON=固定値、OFF=候補集合を総当たりして対応表を作成します。"
                " `meta.nf_automodel` へ一括登録し、後続の `meta-automodel-run` で実行できます。"
            )

            def _first_or_empty(values: list[str]) -> str:
                return str(values[0]) if values else ""

            tr_nf_fit_obj_for_meta: dict[str, Any] = {}
            try:
                parsed_fit_for_meta = json.loads(str(tr_nf_fit_raw).strip() or "{}")
                if isinstance(parsed_fit_for_meta, dict):
                    tr_nf_fit_obj_for_meta = dict(parsed_fit_for_meta)
            except Exception:
                tr_nf_fit_obj_for_meta = {}

            def _encode_fit_axis_value(value: Any) -> Any:
                if value is None:
                    return "(none)"
                if isinstance(value, (dict, list)):
                    return _stable_json_dumps(value)
                return value

            def _decode_fit_axis_value(axis_key: str, value: Any) -> Any:
                if axis_key == "fit_val_size":
                    try:
                        return int(value)
                    except Exception:
                        return 0
                if axis_key in {"fit_use_init_models", "fit_verbose"}:
                    return bool(value)
                if axis_key in {"fit_id_col", "fit_time_col", "fit_target_col"}:
                    return str(value)
                if axis_key in {"fit_static_df", "fit_distributed_config", "fit_prediction_intervals"}:
                    sv = str(value).strip() if value is not None else ""
                    if sv in {"", "(none)", "None", "null", "NULL"}:
                        return None
                    try:
                        loaded = json.loads(sv)
                        return loaded
                    except Exception:
                        return sv
                return value

            fit_axis_to_fit_key: dict[str, str] = {
                "fit_static_df": "static_df",
                "fit_val_size": "val_size",
                "fit_use_init_models": "use_init_models",
                "fit_verbose": "verbose",
                "fit_id_col": "id_col",
                "fit_time_col": "time_col",
                "fit_target_col": "target_col",
                "fit_distributed_config": "distributed_config",
                "fit_prediction_intervals": "prediction_intervals",
            }

            def _fit_kwargs_from_axis_source(src: dict[str, Any]) -> dict[str, Any]:
                out = dict(tr_nf_fit_obj_for_meta)
                for axis_key, fit_key in fit_axis_to_fit_key.items():
                    if axis_key not in src:
                        continue
                    out[fit_key] = _decode_fit_axis_value(axis_key, src.get(axis_key))
                return out

            def _decode_exog_axis_value_local(value: Any) -> list[str]:
                if value is None:
                    return []
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
                sv = str(value).strip()
                if not sv or sv in {"(none)", "None", "null", "NULL"}:
                    return []
                try:
                    loaded = json.loads(sv)
                    if isinstance(loaded, list):
                        return [str(x).strip() for x in loaded if str(x).strip()]
                except Exception:
                    pass
                return [x.strip() for x in sv.split(",") if x.strip()]

            detected_exog_current = (
                _prefixed_exog_cols_for_table(str(tr_dataset_schema), str(tr_dataset_table))
                if str(tr_dataset_input_method) == "db_table"
                else {"futr_exog": [], "hist_exog": [], "stat_exog": []}
            )
            detected_futr_exog = list(detected_exog_current.get("futr_exog", []))
            detected_hist_exog = list(detected_exog_current.get("hist_exog", []))
            detected_stat_exog = list(detected_exog_current.get("stat_exog", []))

            fit_static_auto = detected_hist_exog if detected_hist_exog else None
            fit_static_current = tr_nf_fit_obj_for_meta.get("static_df", fit_static_auto)
            if fit_static_current is None and fit_static_auto:
                fit_static_current = fit_static_auto
            fit_static_choices = ["(none)"]
            if detected_hist_exog:
                for hist_col in detected_hist_exog:
                    fit_static_choices.append(_encode_fit_axis_value([hist_col]))
                fit_static_choices.append(_encode_fit_axis_value(detected_hist_exog))
            fit_static_existing = _encode_fit_axis_value(tr_nf_fit_obj_for_meta.get("static_df"))
            if fit_static_existing not in fit_static_choices:
                fit_static_choices.append(fit_static_existing)
            fit_static_choices = list(dict.fromkeys(fit_static_choices))

            fit_id_choices = (
                list(dataset_col_names)
                if dataset_col_names
                else [str(tr_nf_fit_obj_for_meta.get("id_col", "unique_id"))]
            )
            if str(tr_nf_fit_obj_for_meta.get("id_col", "unique_id")) not in fit_id_choices:
                fit_id_choices = [str(tr_nf_fit_obj_for_meta.get("id_col", "unique_id"))] + fit_id_choices

            fit_time_choices = (
                list(dataset_col_names) if dataset_col_names else [str(tr_nf_fit_obj_for_meta.get("time_col", "ds"))]
            )
            if str(tr_nf_fit_obj_for_meta.get("time_col", "ds")) not in fit_time_choices:
                fit_time_choices = [str(tr_nf_fit_obj_for_meta.get("time_col", "ds"))] + fit_time_choices

            fit_target_choices = (
                list(dataset_col_names) if dataset_col_names else [str(tr_nf_fit_obj_for_meta.get("target_col", "y"))]
            )
            if str(tr_nf_fit_obj_for_meta.get("target_col", "y")) not in fit_target_choices:
                fit_target_choices = [str(tr_nf_fit_obj_for_meta.get("target_col", "y"))] + fit_target_choices

            fit_distributed_choices = ["(none)", "{}", '{"backend":"ray"}']
            fit_dist_existing = _encode_fit_axis_value(tr_nf_fit_obj_for_meta.get("distributed_config"))
            if fit_dist_existing not in fit_distributed_choices:
                fit_distributed_choices.append(fit_dist_existing)

            fit_pi_choices = ["(none)", "{}", '{"method":"conformal","level":[80,90]}']
            fit_pi_existing = _encode_fit_axis_value(tr_nf_fit_obj_for_meta.get("prediction_intervals"))
            if fit_pi_existing not in fit_pi_choices:
                fit_pi_choices.append(fit_pi_existing)

            exog_mapping_rows: list[dict[str, Any]] = []
            for model_name in list(model_choices):
                support_one = (
                    dict(get_model_exog_support(str(model_name)))
                    if get_model_exog_support is not None
                    else {"futr": False, "hist": False, "stat": False}
                )
                exog_mapping_rows.append(
                    {
                        "selected": bool(str(model_name) == str(tr_model)),
                        "model": str(model_name),
                        "supports_futr_exog(F)": bool(support_one.get("futr", False)),
                        "supports_hist_exog(H)": bool(support_one.get("hist", False)),
                        "supports_stat_exog(S)": bool(support_one.get("stat", False)),
                        "futr_exog": detected_futr_exog,
                        "hist_exog": detected_hist_exog,
                        "stat_exog": detected_stat_exog,
                    }
                )
            st.markdown("**モデルと外生変数の対応表**")
            _show_df(pd.DataFrame(exog_mapping_rows), hide_index=True)
            st.markdown("**データ読み込み方式とDataFrame対応表**")
            _show_df(_dataset_loader_support_df(), hide_index=True)

            # (固定/網羅メタ反映) dataset table から loto/unique_id/ts_type 候補を自動抽出
            if engine is not None and str(tr_dataset_input_method) == "db_table" and str(tr_dataset_table).strip():
                try:
                    _ax_cols_df = _table_columns(engine, str(tr_dataset_schema), str(tr_dataset_table))
                    _ax_cols = (
                        set(_ax_cols_df["column_name"].astype(str).tolist())
                        if (
                            _ax_cols_df is not None
                            and (not _ax_cols_df.empty)
                            and ("column_name" in _ax_cols_df.columns)
                        )
                        else set()
                    )
                except Exception:
                    _ax_cols = set()

                def _load_distinct_axis_values(col_name: str, limit: int = 500) -> list[str]:
                    if col_name not in _ax_cols:
                        return []
                    where_parts = [f"{_safe_ident(col_name)} IS NOT NULL"]
                    if str(tr_dataset_where).strip():
                        where_parts.append(f"({str(tr_dataset_where).strip()})")
                    where_sql = " AND ".join(where_parts)
                    q = (
                        f"SELECT DISTINCT {_safe_ident(col_name)} AS v "
                        f"FROM {_safe_ident(str(tr_dataset_schema))}.{_safe_ident(str(tr_dataset_table))} "
                        f"WHERE {where_sql} "
                        f"ORDER BY {_safe_ident(col_name)} "
                        f"LIMIT :limit"
                    )
                    d = _query_df(engine, q, {"limit": int(limit)})
                    if d is None or d.empty or "v" not in d.columns:
                        return []
                    return [str(x).strip() for x in d["v"].astype(str).tolist() if str(x).strip()]

                loto_vals = _load_distinct_axis_values("loto")
                uid_vals = _load_distinct_axis_values(str(settings.id_col))
                ts_type_vals = _load_distinct_axis_values("ts_type")

            ax1, ax2, ax3, ax4 = st.columns(4)
            loto_extra_raw = ax1.text_input("loto 候補追加(CSV,任意)", value="", key="nf_lab_axis_extra_loto")
            uid_extra_raw = ax2.text_input("unique_id 候補追加(CSV,任意)", value="", key="nf_lab_axis_extra_uid")
            ts_type_extra_raw = ax3.text_input("ts_type 候補追加(CSV,任意)", value="", key="nf_lab_axis_extra_ts_type")
            horizon_extra_raw = ax4.text_input("horizon 候補追加(CSV,任意)", value="", key="nf_lab_axis_extra_horizon")

            def _csv_values_for_axis(raw: str) -> list[str]:
                return [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]

            def _csv_int_values_for_axis(raw: str) -> list[int]:
                out: list[int] = []
                for token in _csv_values_for_axis(raw):
                    try:
                        iv = int(token)
                    except Exception:
                        continue
                    if iv > 0:
                        out.append(int(iv))
                return out

            loto_axis_choices = list(dict.fromkeys(list(loto_vals) + _csv_values_for_axis(loto_extra_raw)))
            uid_axis_choices = list(dict.fromkeys(list(uid_vals) + _csv_values_for_axis(uid_extra_raw)))
            ts_type_axis_choices = list(dict.fromkeys(list(ts_type_vals) + _csv_values_for_axis(ts_type_extra_raw)))
            horizon_axis_choices = [
                int(x)
                for x in dict.fromkeys(
                    [1, 2, 3, 5, 7, 14, 28, int(manual_h), int(effective_h)]
                    + _csv_int_values_for_axis(horizon_extra_raw)
                )
                if int(x) > 0
            ]

            # Ensure backend_options is defined in every code path (avoid UnboundLocalError)
            backend_options = ["ray"] if str(tr_model) == "AutoHINT" else ["optuna", "ray"]
            if str(tr_backend) not in backend_options:
                tr_backend = str(backend_options[0])

            # Ensure loss_opts/valid_loss_opts are defined in every code path (avoid UnboundLocalError)
            loss_opts = ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "HUBER"]
            _loss_current = str(locals().get("tr_loss", st.session_state.get("nf_lab_train_loss_name", "MAE")))
            tr_loss = _loss_current if _loss_current in loss_opts else str(loss_opts[0])
            valid_loss_opts = [str(tr_loss)]
            tr_valid_loss = str(tr_loss)

            num_samples_axis_choices = [1] + list(range(10, 201, 10))
            fit_val_size_current = max(1, int(tr_nf_fit_obj_for_meta.get("val_size", 1) or 1), int(effective_h))
            fit_val_size_axis_choices = sorted(
                {1, int(fit_val_size_current), int(effective_h), *list(range(10, 101, 10))}
            )

            def _nearest_axis_int_choice(current: Any, options: list[int], fallback: int) -> int:
                if not options:
                    return int(fallback)
                try:
                    current_int = int(current)
                except Exception:
                    current_int = int(fallback)
                if current_int in options:
                    return int(current_int)
                return int(min(options, key=lambda x: abs(int(x) - current_int)))

            axis_defs: list[dict[str, Any]] = [
                {"key": "model", "label": "model", "current": str(tr_model), "choices": list(model_choices)},
                {
                    "key": "backend",
                    "label": "backend",
                    "current": str(tr_backend),
                    "choices": [OPTIONAL_TRAIN_NONE_TOKEN] + list(backend_options),
                },
                {
                    "key": "num_samples",
                    "label": "num_samples",
                    "current": int(tr_num_samples),
                    "choices": num_samples_axis_choices,
                },
                {
                    "key": "loss",
                    "label": "loss",
                    "current": str(tr_loss),
                    "choices": [OPTIONAL_TRAIN_NONE_TOKEN] + list(loss_opts),
                },
                {
                    "key": "valid_loss",
                    "label": "valid_loss",
                    "current": str(tr_valid_loss),
                    "choices": [OPTIONAL_TRAIN_NONE_TOKEN] + list(valid_loss_opts),
                },
                {
                    "key": "search_alg",
                    "label": "search_alg",
                    "current": str(tr_search),
                    "choices": [OPTIONAL_TRAIN_NONE_TOKEN]
                    + [
                        "RandomSampler",
                        "TPESampler",
                        "CmaEsSampler",
                        "NSGAIISampler",
                        "BasicVariantGenerator",
                        "OptunaSearch",
                        "HyperOptSearch",
                        "BayesOptSearch",
                    ],
                },
                {
                    "key": "dataset_schema",
                    "label": "dataset schema",
                    "current": str(tr_dataset_schema),
                    "choices": list(dataset_schema_options),
                },
                {
                    "key": "dataset_table",
                    "label": "dataset table",
                    "current": str(tr_dataset_table),
                    "choices": list(dataset_table_options) if dataset_table_options else [str(tr_dataset_table)],
                },
                {
                    "key": "dataset_input_method",
                    "label": "dataset input",
                    "current": str(tr_dataset_input_method),
                    "choices": list(DATASET_INPUT_METHOD_OPTIONS),
                },
                {
                    "key": "dataframe_backend",
                    "label": "dataframe backend",
                    "current": str(tr_dataframe_backend),
                    "choices": list(_supported_backends_for_input_method(str(tr_dataset_input_method))),
                },
                {
                    "key": "group_mode",
                    "label": "学習単位",
                    "current": str(group_mode),
                    "choices": ["loto_unique_id_ts_type", "loto_ts_type"],
                },
                {
                    "key": "loto",
                    "label": "loto",
                    "current": _first_or_empty(list(tr_loto)),
                    "choices": loto_axis_choices,
                },
                {
                    "key": "unique_id",
                    "label": "unique_id",
                    "current": _first_or_empty(list(tr_uid)),
                    "choices": uid_axis_choices,
                },
                {
                    "key": "ts_type",
                    "label": "ts_type",
                    "current": _first_or_empty(list(tr_ts_type)),
                    "choices": ts_type_axis_choices,
                },
                {
                    "key": "horizon",
                    "label": "horizon",
                    "current": HORIZON_AUTO_TOKEN,
                    "choices": [HORIZON_AUTO_TOKEN] + horizon_axis_choices,
                },
                {
                    "key": "futr_exog",
                    "label": "futr_exog",
                    "current": _stable_json_dumps(detected_futr_exog),
                    "choices": [_stable_json_dumps(detected_futr_exog)],
                },
                {
                    "key": "hist_exog",
                    "label": "hist_exog",
                    "current": _stable_json_dumps(detected_hist_exog),
                    "choices": [_stable_json_dumps(detected_hist_exog)],
                },
                {
                    "key": "stat_exog",
                    "label": "stat_exog",
                    "current": _stable_json_dumps(detected_stat_exog),
                    "choices": [_stable_json_dumps(detected_stat_exog)],
                },
                {
                    "key": "freq",
                    "label": "freq",
                    "current": str(tr_freq),
                    "choices": list(
                        dict.fromkeys([str(tr_freq), "D", "W", "W-WED", "W-THU", "M", "MS", "Q", "QS", "Y", "YS"])
                    ),
                },
                {
                    "key": "local_scaler_type",
                    "label": "local_scaler_type",
                    "current": str(tr_local_scaler_type),
                    "choices": ["(none)", "standard", "robust", "robust-iqr", "minmax", "boxcox"],
                },
                {
                    "key": "fit_val_size",
                    "label": "fit.val_size",
                    "current": int(fit_val_size_current),
                    "choices": fit_val_size_axis_choices,
                },
                {
                    "key": "fit_static_df",
                    "label": "fit.static_df",
                    "current": _encode_fit_axis_value(fit_static_current),
                    "choices": fit_static_choices,
                },
                {
                    "key": "fit_use_init_models",
                    "label": "fit.use_init_models",
                    "current": bool(tr_nf_fit_obj_for_meta.get("use_init_models", False)),
                    "choices": [False, True],
                },
                {
                    "key": "fit_verbose",
                    "label": "fit.verbose",
                    "current": bool(tr_nf_fit_obj_for_meta.get("verbose", False)),
                    "choices": [False, True],
                },
                {
                    "key": "fit_id_col",
                    "label": "fit.id_col",
                    "current": str(tr_nf_fit_obj_for_meta.get("id_col", "unique_id")),
                    "choices": fit_id_choices,
                },
                {
                    "key": "fit_time_col",
                    "label": "fit.time_col",
                    "current": str(tr_nf_fit_obj_for_meta.get("time_col", "ds")),
                    "choices": fit_time_choices,
                },
                {
                    "key": "fit_target_col",
                    "label": "fit.target_col",
                    "current": str(tr_nf_fit_obj_for_meta.get("target_col", "y")),
                    "choices": fit_target_choices,
                },
                {
                    "key": "fit_distributed_config",
                    "label": "fit.distributed_config",
                    "current": _encode_fit_axis_value(tr_nf_fit_obj_for_meta.get("distributed_config")),
                    "choices": fit_distributed_choices,
                },
                {
                    "key": "fit_prediction_intervals",
                    "label": "fit.prediction_intervals",
                    "current": _encode_fit_axis_value(tr_nf_fit_obj_for_meta.get("prediction_intervals")),
                    "choices": fit_pi_choices,
                },
            ]

            axis_plan = {}
            for axis in axis_defs:
                k = str(axis["key"])
                fixed_default = True
                col_a, col_b = st.columns([1, 3])
                fixed_flag = col_a.checkbox(f"{axis['label']} 固定", value=fixed_default, key=f"nf_lab_axis_fixed_{k}")
                current_value = axis["current"]
                choices = list(
                    dict.fromkeys([current_value] + [x for x in axis["choices"] if str(x) != str(current_value)])
                )
                if fixed_flag:
                    col_b.caption(f"固定値: `{current_value}`")
                    axis_plan[k] = {"fixed": True, "values": [current_value]}
                else:
                    if k == "num_samples":
                        ns_options = list(num_samples_axis_choices)
                        ns_default = _nearest_axis_int_choice(current_value, ns_options, int(tr_num_samples))
                        ns_selected = col_b.select_slider(
                            "num_samples 候補(単一選択)",
                            options=ns_options,
                            value=ns_default,
                            key="nf_lab_axis_pool_num_samples_single",
                        )
                        axis_plan[k] = {"fixed": False, "values": [int(ns_selected)]}
                        continue
                    if k == "fit_val_size":
                        fit_vs_options = list(fit_val_size_axis_choices)
                        fit_vs_default = _nearest_axis_int_choice(current_value, fit_vs_options, 1)
                        fit_vs_selected = col_b.select_slider(
                            "fit.val_size 候補(単一選択)",
                            options=fit_vs_options,
                            value=fit_vs_default,
                            key="nf_lab_axis_pool_fit_val_size_single",
                        )
                        axis_plan[k] = {"fixed": False, "values": [int(fit_vs_selected)]}
                        continue
                    if k in {"fit_use_init_models", "fit_verbose"}:
                        b1, b2 = col_b.columns(2)
                        include_false = b1.toggle("False", value=True, key=f"nf_lab_axis_bool_false_{k}")
                        include_true = b2.toggle("True", value=True, key=f"nf_lab_axis_bool_true_{k}")
                        bool_vals: list[bool] = []
                        if include_false:
                            bool_vals.append(False)
                        if include_true:
                            bool_vals.append(True)
                        if not bool_vals:
                            bool_vals = [bool(current_value)]
                        axis_plan[k] = {"fixed": False, "values": bool_vals}
                        continue
                    picked = col_b.multiselect(
                        f"{axis['label']} 候補",
                        options=choices,
                        default=[current_value] if current_value in choices else (choices[:1] if choices else []),
                        key=f"nf_lab_axis_pool_{k}",
                    )
                    axis_plan[k] = {
                        "fixed": False,
                        "values": list(picked) if picked else ([current_value] if str(current_value).strip() else []),
                    }

            axis_input_spec = (
                axis_plan.get("dataset_input_method", {})
                if isinstance(axis_plan.get("dataset_input_method"), dict)
                else {}
            )
            input_vals_raw = axis_input_spec.get("values", []) if isinstance(axis_input_spec, dict) else []
            input_vals = [
                str(v).strip() for v in (input_vals_raw if isinstance(input_vals_raw, list) else []) if str(v).strip()
            ]
            if input_vals:
                filtered_input_vals: list[str] = []
                dropped_input_vals: list[str] = []
                for m in input_vals:
                    if m == "db_table":
                        if str(tr_dataset_table).strip():
                            filtered_input_vals.append(m)
                        else:
                            dropped_input_vals.append(m)
                    elif m == "db_sql":
                        if str(tr_dataset_sql).strip():
                            filtered_input_vals.append(m)
                        else:
                            dropped_input_vals.append(m)
                    elif m in {"csv", "parquet", "json"}:
                        if str(tr_dataset_path).strip():
                            filtered_input_vals.append(m)
                        else:
                            dropped_input_vals.append(m)
                    else:
                        filtered_input_vals.append(m)
                if filtered_input_vals and dropped_input_vals:
                    axis_plan["dataset_input_method"] = {**axis_input_spec, "values": filtered_input_vals}
                    st.info("dataset入力条件不足のため input method 候補から自動除外: " + ", ".join(dropped_input_vals))

            axis_keys = [str(a["key"]) for a in axis_defs]
            combo_axes = {k: axis_plan[k]["values"] for k in axis_keys}

            def _normalize_meta_combo_row(row: dict[str, Any], context: ComboContext) -> dict[str, Any]:
                normalized = dict(row)
                normalized["backend"] = _decode_optional_train_core_choice("backend", normalized.get("backend"))
                normalized["loss"] = _decode_optional_train_core_choice("loss", normalized.get("loss"))
                valid_loss_raw = _decode_optional_train_core_choice("valid_loss", normalized.get("valid_loss"))
                normalized["search_alg"] = _decode_optional_train_core_choice("search_alg", normalized.get("search_alg"))
                normalized["valid_loss"] = None if valid_loss_raw is None else str(normalized.get("loss") or tr_loss)
                group_mode_v = str(normalized.get("group_mode") or group_mode)
                normalized["group_mode"] = group_mode_v
                uid_values_v = _csv_nonempty_list(normalized.get("unique_id"))
                if group_mode_v == "loto_ts_type":
                    normalized["unique_id"] = None
                elif uid_values_v:
                    normalized["unique_id"] = ",".join(uid_values_v)
                normalized["dataset_input_method"] = str(
                    normalized.get("dataset_input_method") or tr_dataset_input_method
                )
                normalized["dataframe_backend"] = str(normalized.get("dataframe_backend") or tr_dataframe_backend)
                schema_v = str(normalized.get("dataset_schema") or tr_dataset_schema)
                table_v = str(normalized.get("dataset_table") or context.dataset_table or tr_dataset_table)
                normalized["dataset_table"] = table_v
                if context.dataset_path:
                    normalized["dataset_path"] = context.dataset_path
                if context.dataset_sql:
                    normalized["dataset_sql"] = context.dataset_sql
                detected_exog_v = (
                    _prefixed_exog_cols_for_table(schema_v, table_v)
                    if normalized["dataset_input_method"] == "db_table"
                    else {"futr_exog": [], "hist_exog": [], "stat_exog": []}
                )
                futr_exog_v = _decode_exog_axis_value(normalized.get("futr_exog")) or list(
                    detected_exog_v.get("futr_exog", [])
                )
                hist_exog_v = _decode_exog_axis_value(normalized.get("hist_exog")) or list(
                    detected_exog_v.get("hist_exog", [])
                )
                stat_exog_v = _decode_exog_axis_value(normalized.get("stat_exog")) or list(
                    detected_exog_v.get("stat_exog", [])
                )
                model_v = str(normalized.get("model", ""))
                support_v = (
                    dict(get_model_exog_support(model_v))
                    if get_model_exog_support is not None
                    else {"futr": False, "hist": False, "stat": False}
                )
                exog_adjustments: list[str] = []
                if futr_exog_v and not bool(support_v.get("futr", False)):
                    exog_adjustments.append(f"drop futr_exog({len(futr_exog_v)})")
                    futr_exog_v = []
                if hist_exog_v and not bool(support_v.get("hist", False)):
                    exog_adjustments.append(f"drop hist_exog({len(hist_exog_v)})")
                    hist_exog_v = []
                if stat_exog_v and not bool(support_v.get("stat", False)):
                    exog_adjustments.append(f"drop stat_exog({len(stat_exog_v)})")
                    stat_exog_v = []
                normalized["futr_exog"] = futr_exog_v
                normalized["hist_exog"] = hist_exog_v
                normalized["stat_exog"] = stat_exog_v
                normalized["supports_futr_exog(F)"] = bool(support_v.get("futr", False))
                normalized["supports_hist_exog(H)"] = bool(support_v.get("hist", False))
                normalized["supports_stat_exog(S)"] = bool(support_v.get("stat", False))
                if exog_adjustments:
                    normalized["exog_adjustment"] = "; ".join(exog_adjustments)
                return normalized

            combo_eval = evaluate_train_combinations(
                combo_axes,
                context=ComboContext(
                    dataset_table=str(tr_dataset_table),
                    dataset_path=str(tr_dataset_path or "").strip(),
                    dataset_sql=str(tr_dataset_sql or "").strip(),
                    default_values={},
                ),
                row_normalizer=_normalize_meta_combo_row,
                runtime_validator=_validate_model_runtime_prerequisites,
            )
            combo_rows = list(combo_eval.valid_combinations)
            skipped_rows = [
                {**dict(item.values), "reason": item.reason_ja, "reason_code": item.reason_code}
                for item in combo_eval.excluded_combinations
            ]
            combo_df = pd.DataFrame(combo_rows)
            c_combo1, c_combo2, c_combo3 = st.columns(3)
            c_combo1.metric("理論組合せ数", int(combo_eval.theoretical_count))
            c_combo2.metric("自動除外後の有効件数", int(combo_df.shape[0]))
            c_combo3.metric("除外件数", int(len(skipped_rows)))
            if skipped_rows:
                st.warning(f"除外対象の設定組合せ: {len(skipped_rows)}")
                reason_rows = _build_combo_reason_rows(combo_eval)
                _render_combo_reason_summary(reason_rows)
                with st.expander("除外理由を表示", expanded=False):
                    if reason_rows:
                        _show_df(pd.DataFrame(reason_rows).sort_values("count", ascending=False), hide_index=True)
            if combo_eval.fix_suggestions:
                st.info("推奨修正: " + " / ".join(combo_eval.fix_suggestions))
            if combo_df.empty:
                _render_zero_combo_diagnostics(combo_eval, label="固定/網羅メタ反映")
            else:
                _show_df(combo_df.head(500), hide_index=True)

            meta_cfg_prefix = st.text_input(
                "meta config名プレフィックス",
                value=f"nf_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key="nf_lab_meta_cfg_prefix",
            )
            meta_priority = int(
                st.number_input(
                    "meta priority", min_value=1, max_value=999999, value=100, step=1, key="nf_lab_meta_priority"
                )
            )
            meta_limit = int(
                st.number_input(
                    "meta反映上限(0=全件)",
                    min_value=0,
                    max_value=5000,
                    value=100,
                    step=10,
                    key="nf_lab_meta_reflect_limit",
                )
            )
            dry_run_meta = st.toggle("meta反映 dry-run（DB未書込）", value=False, key="nf_lab_meta_reflect_dry_run")
            if dry_run_meta:
                st.warning("dry-run ON: 対応表はDBへ保存されません。実行用設定を残すには OFF にしてください。")
            st.info(
                "注意: `meta反映` は設定登録のみです。学習実行は下の `有効候補をすべて実行` か `meta-automodel-run` を使います。"
            )
            auto_run_after_reflect = st.toggle(
                "反映後に全件自動実行（エラー時も継続・DB記録）",
                value=True,
                key="nf_lab_meta_reflect_auto_run",
                disabled=dry_run_meta,
            )
            auto_run_ensure_db_init = st.toggle(
                "自動実行時に db-init を先頭で実施",
                value=False,
                key="nf_lab_meta_reflect_auto_run_db_init",
                disabled=dry_run_meta,
            )
            auto_run_skip_success = st.toggle(
                "自動実行: 既成功の組み合わせをスキップ",
                value=True,
                key="nf_lab_meta_reflect_auto_run_skip_success",
                disabled=dry_run_meta,
            )

            mode_json_preview: dict[str, Any] = {}
            axis_to_param_key = {
                "backend": "backend",
                "num_samples": "num_samples",
                "loss": "loss_name",
                "valid_loss": "valid_loss_name",
                "search_alg": "search_alg_name",
                "dataset_schema": "dataset_schema",
                "dataset_table": "dataset_table",
                "dataset_input_method": "dataset_input_method",
                "dataframe_backend": "dataframe_backend",
                "group_mode": "group_by_mode",
                "loto": "target_loto",
                "unique_id": "target_unique_id",
                "ts_type": "target_ts_type",
                "futr_exog": "futr_exog_list",
                "hist_exog": "hist_exog_list",
                "stat_exog": "stat_exog_list",
                "freq": "freq",
                "local_scaler_type": "local_scaler_type",
            }
            for axis in axis_defs:
                axis_key = str(axis["key"])
                param_key = axis_to_param_key.get(axis_key)
                if not param_key:
                    continue
                values_k = axis_plan.get(axis_key, {}).get("values", [])
                if axis_key in OPTIONAL_TRAIN_CORE_AXIS_KEYS:
                    values_k = [_decode_optional_train_core_choice(axis_key, v) for v in values_k]
                    values_k = [v for v in values_k if v is not None]
                if axis_key in {"futr_exog", "hist_exog", "stat_exog"}:
                    values_k = [_decode_exog_axis_value(v) for v in values_k]
                if axis_key in OPTIONAL_TRAIN_CORE_AXIS_KEYS and not values_k:
                    continue
                if axis_plan.get(axis_key, {}).get("fixed", True):
                    mode_json_preview[param_key] = {"mode": "fixed", "value": values_k[0] if values_k else None}
                else:
                    mode_json_preview[param_key] = {"mode": "vary", "values": values_k}
            if str(tr_dataset_path).strip():
                mode_json_preview["dataset_path"] = {"mode": "fixed", "value": str(tr_dataset_path).strip()}
            if str(tr_dataset_sql).strip():
                mode_json_preview["dataset_sql"] = {"mode": "fixed", "value": str(tr_dataset_sql).strip()}
            if str(tr_dataset_where).strip():
                mode_json_preview["dataset_where"] = {"mode": "fixed", "value": str(tr_dataset_where).strip()}
            if "local_scaler_type" in mode_json_preview:
                local_mode = dict(mode_json_preview.get("local_scaler_type") or {})
                if str(local_mode.get("mode")) == "vary":
                    mode_json_preview["local_static_scaler_type"] = {
                        "mode": "vary",
                        "values": list(local_mode.get("values", [])),
                    }
                else:
                    mode_json_preview["local_static_scaler_type"] = {
                        "mode": "fixed",
                        "value": local_mode.get("value"),
                    }
            fit_axis_keys = [k for k in fit_axis_to_fit_key if k in axis_plan]
            if fit_axis_keys:
                fit_kwargs_values: list[dict[str, Any]] = []
                for fit_vals in itertools.product(*[axis_plan[k]["values"] for k in fit_axis_keys]):
                    src = {k: v for k, v in zip(fit_axis_keys, fit_vals, strict=False)}
                    fit_kwargs_values.append(_fit_kwargs_from_axis_source(src))
                fit_kwargs_values_dedup = []
                fit_seen: set[str] = set()
                for one in fit_kwargs_values:
                    sk = _stable_json_dumps(one)
                    if sk in fit_seen:
                        continue
                    fit_seen.add(sk)
                    fit_kwargs_values_dedup.append(one)
                fit_all_fixed = all(bool(axis_plan[k]["fixed"]) for k in fit_axis_keys)
                if fit_kwargs_values_dedup:
                    if fit_all_fixed:
                        mode_json_preview["nf_fit_kwargs"] = {"mode": "fixed", "value": fit_kwargs_values_dedup[0]}
                    else:
                        mode_json_preview["nf_fit_kwargs"] = {"mode": "vary", "values": fit_kwargs_values_dedup}
            with st.expander("param_mode_json preview", expanded=False):
                st.json(mode_json_preview)
                _render_copy_button(
                    json.dumps(mode_json_preview, ensure_ascii=False),
                    key="nf_lab_param_mode_json_copy",
                    label="param_mode_json copy",
                )

            rows_to_apply = combo_rows[:meta_limit] if meta_limit > 0 else list(combo_rows)
            uid_count_cache: dict[str, int] = {}

            def _uid_count_for_combo(schema_v: str, table_v: str, loto_v: str, ts_type_v: str) -> int:
                cache_k = f"{schema_v}|{table_v}|{loto_v}|{ts_type_v}"
                if cache_k in uid_count_cache:
                    return uid_count_cache[cache_k]
                if engine is None:
                    return 1
                where_parts = [f"{_safe_ident(settings.id_col)} IS NOT NULL"]
                if str(loto_v).strip():
                    loto_esc = str(loto_v).replace("'", "''")
                    where_parts.append(f"{_safe_ident('loto')} = '{loto_esc}'")
                if str(ts_type_v).strip():
                    ts_type_esc = str(ts_type_v).replace("'", "''")
                    where_parts.append(f"{_safe_ident('ts_type')} = '{ts_type_esc}'")
                q = (
                    f"SELECT COUNT(DISTINCT {_safe_ident(settings.id_col)}) AS uid_cnt "
                    f"FROM {_safe_ident(schema_v)}.{_safe_ident(table_v)} "
                    f"WHERE {' AND '.join(where_parts)}"
                )
                out_df = _query_df(engine, q)
                cnt = int(out_df.iloc[0]["uid_cnt"]) if not out_df.empty else 1
                uid_count_cache[cache_k] = max(1, cnt)
                return max(1, cnt)

            base_params_for_meta: dict[str, Any] = {}
            try:
                parsed_base = json.loads(str(tr_params_raw).strip() or "{}")
                if isinstance(parsed_base, dict):
                    base_params_for_meta.update(parsed_base)
            except Exception:
                pass
            base_params_for_meta.update(tr_param_builder_obj)
            for k, v in list(base_params_for_meta.items()):
                if isinstance(v, str) and v.strip() == "":
                    base_params_for_meta.pop(k, None)

            meta_runtime_cpus = int(st.session_state.get("nf_lab_train_cpus", max_cpu))
            meta_runtime_gpus = int(st.session_state.get("nf_lab_train_gpus", max_gpu))
            meta_runtime_refit = bool(st.session_state.get("nf_lab_train_refit", False))
            meta_runtime_verbose = bool(st.session_state.get("nf_lab_train_verbose", False))

            payload_rows: list[dict[str, Any]] = []
            payload_build_errors: list[dict[str, Any]] = []
            for i, row in enumerate(rows_to_apply, start=1):
                try:
                    schema_v = str(row.get("dataset_schema") or tr_dataset_schema)
                    table_v = str(row.get("dataset_table") or tr_dataset_table)
                    dataset_input_method_v = str(row.get("dataset_input_method") or tr_dataset_input_method)
                    dataframe_backend_v = str(row.get("dataframe_backend") or tr_dataframe_backend)
                    payload_dataset_path_v: str | None = (
                        str(row.get("dataset_path") or tr_dataset_path or "").strip() or None
                    )
                    payload_dataset_sql_v: str | None = (
                        str(row.get("dataset_sql") or tr_dataset_sql or "").strip() or None
                    )
                    if dataset_input_method_v == "db_table" and not table_v.strip():
                        raise ValueError("dataset_input_method=db_table では dataset table が必須です")
                    if dataset_input_method_v == "db_sql" and not str(payload_dataset_sql_v or "").strip():
                        raise ValueError("dataset_input_method=db_sql では dataset SQL が必須です")
                    if dataset_input_method_v in {"csv", "parquet", "json"} and not str(payload_dataset_path_v or "").strip():
                        raise ValueError(f"dataset_input_method={dataset_input_method_v} では dataset path が必須です")
                    group_mode_v = str(row.get("group_mode") or group_mode)
                    loto_v = str(row.get("loto") or "")
                    uid_v = str(row.get("unique_id") or "").strip()
                    uid_group_error = _group_mode_unique_id_validation_error(group_mode_v, uid_v)
                    if uid_group_error:
                        raise ValueError(uid_group_error)
                    if group_mode_v == "loto_ts_type":
                        uid_v = ""
                    ts_type_v = str(row.get("ts_type") or "")
                    h_from_axis = _parse_horizon_axis_value(row.get("horizon"))
                    if h_from_axis is not None:
                        h_combo = int(h_from_axis)
                    elif auto_h_by_uid:
                        if dataset_input_method_v != "db_table":
                            h_combo = int(manual_h)
                        elif group_mode_v == "loto_unique_id_ts_type":
                            h_combo = 1 if uid_v else _uid_count_for_combo(schema_v, table_v, loto_v, ts_type_v)
                        else:
                            h_combo = _uid_count_for_combo(schema_v, table_v, loto_v, ts_type_v)
                    else:
                        h_combo = int(manual_h)
                    h_combo = max(1, int(h_combo))
                    h_combo_resolved, h_adjust_note = _resolve_model_horizon(str(row.get("model", tr_model)), int(h_combo))
                    h_combo = int(h_combo_resolved or 1)
                    h_mode_v = "fixed" if (h_from_axis is not None or not auto_h_by_uid) else "unique_id_count"
                    backend_meta_raw = _normalize_optional_train_core_value(row.get("backend"))
                    loss_meta_raw = _normalize_optional_train_core_value(row.get("loss"))
                    valid_loss_meta_raw = _normalize_optional_train_core_value(row.get("valid_loss"))
                    search_meta_raw = _normalize_optional_train_core_value(row.get("search_alg"))
                    effective_backend_meta = backend_meta_raw or str(tr_backend)
                    effective_loss_meta = loss_meta_raw or str(tr_loss)
                    effective_valid_loss_meta = (
                        effective_loss_meta if valid_loss_meta_raw is not None else effective_loss_meta
                    )
                    effective_search_meta = search_meta_raw or _default_search_alg_for_backend(
                        effective_backend_meta,
                        tr_search,
                    )

                    model_params_meta = dict(base_params_for_meta)
                    local_scaler_name = (
                        None
                        if str(row.get("local_scaler_type", tr_local_scaler_type)) == "(none)"
                        else str(row.get("local_scaler_type", tr_local_scaler_type))
                    )
                    fit_kwargs_meta = _normalize_fit_kwargs_for_horizon(_fit_kwargs_from_axis_source(row), int(h_combo))
                    detected_exog_meta = _prefixed_exog_cols_for_table(schema_v, table_v)
                    futr_exog_meta = _decode_exog_axis_value_local(row.get("futr_exog"))
                    hist_exog_meta = _decode_exog_axis_value_local(row.get("hist_exog"))
                    stat_exog_meta = _decode_exog_axis_value_local(row.get("stat_exog"))
                    if not futr_exog_meta:
                        futr_exog_meta = list(detected_exog_meta.get("futr_exog", []))
                    if not hist_exog_meta:
                        hist_exog_meta = list(detected_exog_meta.get("hist_exog", []))
                    if not stat_exog_meta:
                        stat_exog_meta = list(detected_exog_meta.get("stat_exog", []))
                    support_meta = (
                        dict(get_model_exog_support(str(row.get("model", tr_model))))
                        if get_model_exog_support is not None
                        else {"futr": False, "hist": False, "stat": False}
                    )
                    if futr_exog_meta and not bool(support_meta.get("futr", False)):
                        futr_exog_meta = []
                    if hist_exog_meta and not bool(support_meta.get("hist", False)):
                        hist_exog_meta = []
                    if stat_exog_meta and not bool(support_meta.get("stat", False)):
                        stat_exog_meta = []
                    prereq_err = _validate_model_runtime_prerequisites(str(row.get("model", tr_model)), int(h_combo))
                    if prereq_err:
                        raise ValueError(prereq_err)
                    model_params_meta.update(
                        {
                            "backend": str(row.get("backend")),
                            "num_samples": int(row.get("num_samples") or 0),
                            "loss_name": str(row.get("loss")),
                            "valid_loss_name": str(row.get("loss")),
                            "search_alg_name": str(row.get("search_alg")),
                            "freq": str(row.get("freq", tr_freq)),
                            "local_scaler_type": local_scaler_name,
                            "local_static_scaler_type": local_scaler_name,
                            "nf_fit_kwargs": fit_kwargs_meta,
                            "dataset_schema": schema_v,
                            "dataset_table": table_v,
                            "dataset_input_method": dataset_input_method_v,
                            "dataset_path": payload_dataset_path_v,
                            "dataset_sql": payload_dataset_sql_v,
                            "dataset_where": str(tr_dataset_where).strip() or None,
                            "dataframe_backend": dataframe_backend_v,
                            "group_by_mode": group_mode_v,
                            "target_loto": loto_v,
                            "target_unique_id": uid_v,
                            "target_ts_type": ts_type_v,
                            "h_mode": h_mode_v,
                            "futr_exog_list": futr_exog_meta,
                            "hist_exog_list": hist_exog_meta,
                            "stat_exog_list": stat_exog_meta,
                        }
                    )
                    if backend_meta_raw is not None:
                        model_params_meta["backend"] = str(backend_meta_raw)
                    else:
                        model_params_meta.pop("backend", None)
                    model_params_meta["num_samples"] = int(row.get("num_samples") or 0)
                    if loss_meta_raw is not None:
                        model_params_meta["loss_name"] = str(loss_meta_raw)
                    else:
                        model_params_meta.pop("loss_name", None)
                    if valid_loss_meta_raw is not None:
                        model_params_meta["valid_loss_name"] = str(effective_valid_loss_meta)
                    else:
                        model_params_meta.pop("valid_loss_name", None)
                    if search_meta_raw is not None:
                        model_params_meta["search_alg_name"] = str(search_meta_raw)
                    else:
                        model_params_meta.pop("search_alg_name", None)
                    for _blocked in ["h", "loss", "valid_loss", "search_alg"]:
                        model_params_meta.pop(_blocked, None)

                    cfg_name = f"{meta_cfg_prefix}_{i:04d}"
                    payload = {
                        "config_name": cfg_name,
                        "active": True,
                        "priority": int(meta_priority),
                        "base_schema": schema_v,
                        "base_table": table_v,
                        "output_schema": schema_v,
                        "output_table": f"{table_v}_unified",
                        "unified_filter_json": {
                            k: v
                            for k, v in {"loto": loto_v, "unique_id": uid_v, "ts_type": ts_type_v}.items()
                            if str(v).strip()
                        },
                        "unified_group_cols_json": ["loto", "unique_id", "ts_type"]
                        if group_mode_v == "loto_unique_id_ts_type"
                        else ["loto", "ts_type"],
                        "model_name": str(row.get("model")),
                        "horizon": int(h_combo),
                        "auto_cls_model": str(row.get("model")),
                        "auto_h": int(h_combo),
                        "auto_loss": str(effective_loss_meta),
                        "auto_valid_loss": str(effective_valid_loss_meta),
                        "auto_search_alg": str(effective_search_meta),
                        "auto_num_samples": int(row.get("num_samples") or 0),
                        "auto_cpus": int(meta_runtime_cpus),
                        "auto_gpus": int(meta_runtime_gpus),
                        "auto_refit_with_val": bool(meta_runtime_refit),
                        "auto_verbose": bool(meta_runtime_verbose),
                        "auto_alias": str(base_params_for_meta.get("alias") or "") or None,
                        "auto_backend": str(effective_backend_meta),
                        "auto_callbacks_json": (
                            base_params_for_meta.get("callbacks")
                            if isinstance(base_params_for_meta.get("callbacks"), list)
                            else []
                        ),
                        "model_params_json": model_params_meta,
                        "param_space_json": {},
                        "param_mode_json": mode_json_preview,
                        "run_predict": True,
                        "run_evaluate": True,
                        "run_explain": False,
                        "run_save": True,
                        "run_load": True,
                        "run_analyze": True,
                        "save_dataset": False,
                        "save_overwrite": True,
                        "note": (
                            "created from NF lab fixed/vary mapping"
                            + (f" | {h_adjust_note}" if str(h_adjust_note or "").strip() else "")
                        ),
                    }
                    payload_rows.append({"index": i, "config_name": cfg_name, "payload": payload})
                except Exception as e:
                    payload_build_errors.append({"index": i, "error": str(e), "row": row})

            if payload_build_errors:
                st.error(f"payload生成エラー: {len(payload_build_errors)}")
                _show_df(pd.DataFrame(payload_build_errors).head(100), hide_index=True)

            payload_skipped_executed: list[dict[str, Any]] = []
            if payload_rows:
                completed_index = _load_completed_combo_signatures(engine)
                filtered_rows: list[dict[str, Any]] = []
                for item in payload_rows:
                    payload = dict(item.get("payload", {}))
                    model_name_v = str(payload.get("model_name", "")).strip()
                    horizon_v = int(payload.get("horizon", 1) or 1)
                    params_v: dict[str, Any] = (
                        dict(payload["model_params_json"]) if isinstance(payload.get("model_params_json"), dict) else {}
                    )
                    sig_v = _build_train_combo_signature(model_name_v, horizon_v, params_v)
                    item["combo_signature"] = sig_v
                    if _is_combo_signature_completed(sig_v, completed_index):
                        payload_skipped_executed.append(
                            {
                                "config_name": str(item.get("config_name", "")),
                                "model_name": model_name_v,
                                "horizon": horizon_v,
                                "reason": "already_executed",
                            }
                        )
                        continue
                    filtered_rows.append(item)
                payload_rows = filtered_rows
                if payload_skipped_executed:
                    st.info(f"既実行のため meta 生成対象から除外: {len(payload_skipped_executed)}件")
                    with st.expander("既実行スキップ詳細", expanded=False):
                        _show_df(pd.DataFrame(payload_skipped_executed).head(300), hide_index=True)
                if not payload_rows:
                    st.success("対象組合せは全件実行済みです。metaテーブル生成/コマンド生成をスキップしました。")

            preview_rows: list[dict[str, Any]] = []
            diff_details: dict[str, dict[str, Any]] = {}
            if engine is not None and payload_rows:
                existing_df = _query_df(
                    engine,
                    """
                    SELECT *
                    FROM meta.nf_automodel
                    WHERE config_name LIKE :prefix
                    """,
                    {"prefix": f"{meta_cfg_prefix}_%"},
                )
                existing_map = {}
                if not existing_df.empty and "config_name" in existing_df.columns:
                    existing_map = {str(r.get("config_name")): dict(r) for r in existing_df.to_dict(orient="records")}
                compare_fields = [
                    "active",
                    "priority",
                    "base_schema",
                    "base_table",
                    "output_schema",
                    "output_table",
                    "unified_filter_json",
                    "unified_group_cols_json",
                    "model_name",
                    "horizon",
                    "auto_cls_model",
                    "auto_h",
                    "auto_loss",
                    "auto_valid_loss",
                    "auto_search_alg",
                    "auto_num_samples",
                    "auto_cpus",
                    "auto_gpus",
                    "auto_refit_with_val",
                    "auto_verbose",
                    "auto_alias",
                    "auto_backend",
                    "auto_callbacks_json",
                    "model_params_json",
                    "param_space_json",
                    "param_mode_json",
                    "run_predict",
                    "run_evaluate",
                    "run_explain",
                    "run_save",
                    "run_load",
                    "run_analyze",
                    "save_dataset",
                    "save_overwrite",
                    "note",
                ]
                for item in payload_rows:
                    cfg_name = str(item["config_name"])
                    payload = dict(item["payload"])
                    old_row = existing_map.get(cfg_name)
                    if old_row is None:
                        preview_rows.append(
                            {"config_name": cfg_name, "plan": "new", "changed_fields": "*", "config_id": None}
                        )
                        diff_details[cfg_name] = {"plan": "new", "changed": compare_fields, "old": {}, "new": payload}
                        continue
                    changed: list[str] = []
                    changes: list[dict[str, Any]] = []
                    for f in compare_fields:
                        old_v = old_row.get(f)
                        new_v = payload.get(f)
                        if _stable_json_dumps(old_v) != _stable_json_dumps(new_v):
                            changed.append(f)
                            changes.append({"field": f, "old": old_v, "new": new_v})
                    plan = "update" if changed else "unchanged"
                    preview_rows.append(
                        {
                            "config_name": cfg_name,
                            "plan": plan,
                            "changed_fields": ", ".join(changed[:12]) if changed else "",
                            "changed_count": len(changed),
                            "config_id": old_row.get("config_id"),
                        }
                    )
                    diff_details[cfg_name] = {"plan": plan, "changed": changed, "changes": changes}
            else:
                for item in payload_rows:
                    preview_rows.append(
                        {
                            "config_name": str(item["config_name"]),
                            "plan": "new",
                            "changed_fields": "*",
                            "config_id": None,
                        }
                    )

            st.markdown("**反映前差分プレビュー（既存config比較）**")
            preview_df = pd.DataFrame(preview_rows)
            if preview_df.empty:
                st.info("プレビュー対象がありません。")
            else:
                c_new = int((preview_df["plan"] == "new").sum()) if "plan" in preview_df.columns else 0
                c_upd = int((preview_df["plan"] == "update").sum()) if "plan" in preview_df.columns else 0
                c_same = int((preview_df["plan"] == "unchanged").sum()) if "plan" in preview_df.columns else 0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("対象", int(preview_df.shape[0]))
                p2.metric("new", c_new)
                p3.metric("update", c_upd)
                p4.metric("unchanged", c_same)
                _show_df(preview_df.head(1000), hide_index=True)

                detail_names = preview_df["config_name"].astype(str).tolist()
                detail_cfg = st.selectbox(
                    "差分詳細 config_name", detail_names, index=0, key="nf_lab_meta_diff_detail_cfg"
                )
                detail = diff_details.get(str(detail_cfg), {})
                if detail:
                    st.caption(f"plan: {detail.get('plan')}")
                    if detail.get("plan") == "new":
                        st.json(detail.get("new", {}))
                    else:
                        ch_df = pd.DataFrame(detail.get("changes", []))
                        if ch_df.empty:
                            st.info("変更差分なし（unchanged）。")
                        else:
                            _show_df(ch_df, hide_index=True)

            if st.button(
                "対応表を meta.nf_automodel に反映",
                key="nf_lab_apply_combo_to_meta",
                disabled=combo_df.empty or bool(payload_build_errors) or (not bool(payload_rows)),
            ):
                _publish_notification(
                    kind=NotificationEventKind.ACTION_CONFIRMED,
                    severity=NotificationSeverity.RUNNING,
                    title="メタ反映を受け付けました",
                    message="対応表の差分確認後に meta.nf_automodel への反映を開始します。",
                    action="meta_reflect",
                    status="accepted",
                    command_summary="meta.nf_automodel upsert",
                )
                if engine is None:
                    st.error("DB未接続のため meta 反映できません。")
                    _publish_notification(
                        kind=NotificationEventKind.OPERATION_FAILURE,
                        severity=NotificationSeverity.FAILURE,
                        title="メタ反映に失敗しました",
                        message="DB 未接続のため meta.nf_automodel を更新できません。",
                        action="meta_reflect",
                        status="failed",
                        command_summary="meta.nf_automodel upsert",
                        error_summary="DB未接続",
                    )
                else:
                    from loto_forecast.orchestration.meta_automodel import create_meta_automodel_config  # noqa: PLC0415

                    inserted: list[dict[str, Any]] = []
                    failed: list[dict[str, Any]] = []
                    payload_map = {str(item["config_name"]): item for item in payload_rows}
                    for r in preview_rows:
                        cfg_name = str(r.get("config_name", ""))
                        payload_item = payload_map.get(cfg_name)
                        if payload_item is None:
                            continue
                        payload = dict(payload_item["payload"])
                        try:
                            if dry_run_meta:
                                inserted.append(
                                    {
                                        "config_name": cfg_name,
                                        "dry_run": True,
                                        "plan": r.get("plan"),
                                        "config_id": None,
                                        "model_name": str(payload.get("model_name", "")),
                                        "horizon": int(payload.get("horizon", 1) or 1),
                                    }
                                )
                            else:
                                ret = create_meta_automodel_config(payload, upsert_by_name=True)
                                inserted.append(
                                    {
                                        "config_name": cfg_name,
                                        "config_id": ret.get("config_id"),
                                        "action": ret.get("action"),
                                        "plan": r.get("plan"),
                                        "model_name": str(payload.get("model_name", "")),
                                        "horizon": int(payload.get("horizon", 1) or 1),
                                    }
                                )
                        except Exception as e:
                            failed.append({"config_name": cfg_name, "error": str(e)})

                    st.success(
                        f"meta反映完了: success={len(inserted)} failed={len(failed)} dry_run={bool(dry_run_meta)}"
                    )
                    st.session_state["nf_lab_meta_reflect_inserted"] = inserted
                    if inserted:
                        _show_df(pd.DataFrame(inserted).head(300), hide_index=True)
                    if failed:
                        st.error("一部失敗が発生しました。")
                        _show_df(pd.DataFrame(failed).head(100), hide_index=True)
                        _publish_notification(
                            kind=NotificationEventKind.OPERATION_FAILURE,
                            severity=NotificationSeverity.WARNING,
                            title="メタ反映は一部失敗しました",
                            message="成功分は保存しましたが、一部設定は反映できませんでした。",
                            action="meta_reflect",
                            status="warning",
                            command_summary="meta.nf_automodel upsert",
                            error_summary=str(failed[:3]),
                        )
                    else:
                        _publish_notification(
                            kind=NotificationEventKind.OPERATION_SUCCESS,
                            severity=NotificationSeverity.SUCCESS,
                            title="メタ反映が完了しました",
                            message="対応表の反映が完了しました。次は auto-run か train 実行に進めます。",
                            action="meta_reflect",
                            status="success",
                            command_summary="meta.nf_automodel upsert",
                        )
                    if (not dry_run_meta) and bool(auto_run_after_reflect) and inserted:
                        targets = [x for x in inserted if x.get("config_id") is not None]
                        st.info(
                            f"自動実行開始: targets={len(targets)}（失敗時も継続、fatalは model.nf_automodel に記録）"
                        )
                        auto_rows = _run_meta_automodel_configs_live(
                            configs=targets,
                            ensure_db_init=bool(auto_run_ensure_db_init),
                            skip_existing_success=bool(auto_run_skip_success),
                            engine=engine,
                        )
                        st.session_state["nf_lab_meta_reflect_auto_run_result"] = auto_rows

            if "nf_lab_meta_reflect_auto_run_result" in st.session_state:
                auto_rows = st.session_state.get("nf_lab_meta_reflect_auto_run_result", [])
                auto_df = pd.DataFrame(auto_rows if isinstance(auto_rows, list) else [])
                if not auto_df.empty:
                    ok_n = int((auto_df["ok"] == True).sum())  # noqa: E712
                    ng_n = int((auto_df["ok"] == False).sum())  # noqa: E712
                    skipped_n = (
                        int(auto_df.get("skipped", pd.Series(dtype=int)).fillna(0).sum())
                        if "skipped" in auto_df.columns
                        else 0
                    )
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("auto-run success", ok_n)
                    a2.metric("auto-run failed", ng_n)
                    a3.metric("auto-run total", int(auto_df.shape[0]))
                    a4.metric("auto-run skipped", skipped_n)
                    if ng_n > 0:
                        st.warning("一部失敗あり: 失敗設定は DB (model.nf_automodel) に failed で記録されます。")
                    else:
                        st.success("全設定の自動実行が完了しました。")
                    _show_df(auto_df, hide_index=True)

        # core runtime args
        c1, c2, c3 = st.columns(3)
        if use_max_resources:
            st.session_state["nf_lab_train_cpus"] = int(max_cpu)
            st.session_state["nf_lab_train_gpus"] = int(max_gpu)
        tr_search_raw = c1.text_input(
            "search_alg_name(override: 空なら上の選択を使用)",
            value=st.session_state.get("nf_lab_train_search_alg_override", ""),
            key="nf_lab_train_search_alg_override",
        )
        tr_search = str(tr_search_raw).strip() or str(tr_search).strip()
        tr_cpus = int(
            c2.number_input(
                "cpus",
                min_value=0,
                max_value=max(1, max_cpu),
                value=int(st.session_state.get("nf_lab_train_cpus", max_cpu)),
                step=1,
                key="nf_lab_train_cpus",
                disabled=use_max_resources,
            )
        )
        tr_gpus = int(
            c3.number_input(
                "gpus",
                min_value=0,
                max_value=max(0, max_gpu),
                value=int(st.session_state.get("nf_lab_train_gpus", max_gpu)),
                step=1,
                key="nf_lab_train_gpus",
                disabled=use_max_resources,
            )
        )

        c4, c5, c6 = st.columns(3)
        tr_refit = c4.toggle(
            "refit_with_val", value=bool(st.session_state.get("nf_lab_train_refit", False)), key="nf_lab_train_refit"
        )
        tr_verbose = c5.toggle(
            "verbose", value=bool(st.session_state.get("nf_lab_train_verbose", False)), key="nf_lab_train_verbose"
        )
        tr_strict = c6.toggle(
            "strict_exog",
            value=bool(st.session_state.get("nf_lab_train_strict_exog", True)),
            key="nf_lab_train_strict_exog",
        )
        tr_run_cv = st.toggle(
            "run_cross_validation",
            value=bool(st.session_state.get("nf_lab_train_run_cv", False)),
            key="nf_lab_train_run_cv",
        )

        combo_cmd_row: dict[str, Any] = {}
        combo_cmd_enabled = st.toggle(
            "固定/網羅の組合せ行をコマンドへ反映",
            value=True,
            key="nf_lab_train_apply_combo_to_cmd",
            disabled=combo_df.empty,
        )
        if combo_cmd_enabled and not combo_df.empty:
            combo_row_idx = int(
                st.number_input(
                    "コマンドに反映する組合せ行番号 (1-based)",
                    min_value=1,
                    max_value=int(combo_df.shape[0]),
                    value=1,
                    step=1,
                    key="nf_lab_train_combo_row_idx",
                )
            )
            combo_cmd_row = dict(combo_df.iloc[combo_row_idx - 1].to_dict())
            st.caption(
                "反映中: "
                f"model={combo_cmd_row.get('model', tr_model)}, "
                f"backend={combo_cmd_row.get('backend', tr_backend)}, "
                f"loss={combo_cmd_row.get('loss', tr_loss)}, "
                f"valid_loss={combo_cmd_row.get('valid_loss', tr_valid_loss)}, "
                f"search_alg={combo_cmd_row.get('search_alg', tr_search)}, "
                f"horizon={combo_cmd_row.get('horizon', HORIZON_AUTO_TOKEN)}, "
                f"input={combo_cmd_row.get('dataset_input_method', tr_dataset_input_method)}, "
                f"df={combo_cmd_row.get('dataframe_backend', tr_dataframe_backend)}, "
                f"schema={combo_cmd_row.get('dataset_schema', tr_dataset_schema)}, "
                f"table={combo_cmd_row.get('dataset_table', tr_dataset_table)}"
            )

        def _to_csv_list(raw_v: Any) -> list[str]:
            if raw_v is None:
                return []
            if isinstance(raw_v, list):
                return [str(x).strip() for x in raw_v if str(x).strip()]
            sv = str(raw_v).strip()
            if not sv:
                return []
            return [x.strip() for x in sv.split(",") if x.strip()]

        cmd_model = str(combo_cmd_row.get("model", tr_model)) if combo_cmd_row else str(tr_model)
        cmd_backend_raw = _normalize_optional_train_core_value(combo_cmd_row.get("backend")) if combo_cmd_row else None
        cmd_backend = cmd_backend_raw or str(tr_backend)
        cmd_num_samples = (
            int(combo_cmd_row.get("num_samples", tr_num_samples)) if combo_cmd_row else int(tr_num_samples)
        )
        cmd_loss_raw = _normalize_optional_train_core_value(combo_cmd_row.get("loss")) if combo_cmd_row else None
        cmd_loss = cmd_loss_raw or str(tr_loss)
        cmd_valid_loss_raw = (
            _normalize_optional_train_core_value(combo_cmd_row.get("valid_loss")) if combo_cmd_row else None
        )
        cmd_valid_loss = str(cmd_loss) if cmd_valid_loss_raw is not None else str(cmd_loss)
        cmd_search_raw = (
            _normalize_optional_train_core_value(combo_cmd_row.get("search_alg")) if combo_cmd_row else None
        )
        cmd_search = cmd_search_raw or _default_search_alg_for_backend(cmd_backend, tr_search)
        cmd_dataset_schema = (
            str(combo_cmd_row.get("dataset_schema", tr_dataset_schema)) if combo_cmd_row else str(tr_dataset_schema)
        )
        cmd_dataset_table = (
            str(combo_cmd_row.get("dataset_table", tr_dataset_table)) if combo_cmd_row else str(tr_dataset_table)
        )
        cmd_dataset_input_method = (
            str(combo_cmd_row.get("dataset_input_method", tr_dataset_input_method))
            if combo_cmd_row
            else str(tr_dataset_input_method)
        )
        cmd_dataframe_backend = (
            str(combo_cmd_row.get("dataframe_backend", tr_dataframe_backend))
            if combo_cmd_row
            else str(tr_dataframe_backend)
        )
        cmd_dataset_path = (
            str(combo_cmd_row.get("dataset_path", tr_dataset_path)) if combo_cmd_row else str(tr_dataset_path)
        )
        cmd_dataset_sql = (
            str(combo_cmd_row.get("dataset_sql", tr_dataset_sql)) if combo_cmd_row else str(tr_dataset_sql)
        )
        cmd_group_mode = str(combo_cmd_row.get("group_mode", group_mode)) if combo_cmd_row else str(group_mode)
        cmd_loto = _to_csv_list(combo_cmd_row.get("loto")) if combo_cmd_row else list(tr_loto)
        cmd_uid = _to_csv_list(combo_cmd_row.get("unique_id")) if combo_cmd_row else list(tr_uid)
        if str(cmd_group_mode) == "loto_ts_type":
            cmd_uid = []
        cmd_ts_type = _to_csv_list(combo_cmd_row.get("ts_type")) if combo_cmd_row else list(tr_ts_type)
        cmd_freq = str(combo_cmd_row.get("freq", tr_freq)) if combo_cmd_row else str(tr_freq)
        cmd_local_scaler_raw = (
            combo_cmd_row.get("local_scaler_type", tr_local_scaler_type) if combo_cmd_row else tr_local_scaler_type
        )
        cmd_local_scaler_type = None if str(cmd_local_scaler_raw) == "(none)" else str(cmd_local_scaler_raw)

        cmd_h_from_axis = _parse_horizon_axis_value(combo_cmd_row.get("horizon") if combo_cmd_row else None)

        cmd_uid_count = len(cmd_uid) if cmd_uid else 0
        if (
            cmd_h_from_axis is None
            and cmd_uid_count <= 0
            and engine is not None
            and cmd_dataset_table
            and str(cmd_dataset_input_method) == "db_table"
        ):
            try:
                where_parts = [f"{_safe_ident(settings.id_col)} IS NOT NULL"]
                if str(tr_dataset_where).strip():
                    where_parts.append(f"({str(tr_dataset_where).strip()})")

                def _quote_list(vals: list[str]) -> str:
                    return ", ".join(["'" + str(v).replace("'", "''") + "'" for v in vals])

                if cmd_loto:
                    where_parts.append(f"{_safe_ident('loto')} IN ({_quote_list(cmd_loto)})")
                if cmd_uid:
                    where_parts.append(f"{_safe_ident(settings.id_col)} IN ({_quote_list(cmd_uid)})")
                if cmd_ts_type:
                    where_parts.append(f"{_safe_ident('ts_type')} IN ({_quote_list(cmd_ts_type)})")
                q_uid = (
                    f"SELECT COUNT(DISTINCT {_safe_ident(settings.id_col)}) AS uid_cnt "
                    f"FROM {_safe_ident(cmd_dataset_schema)}.{_safe_ident(cmd_dataset_table)} "
                    f"WHERE {' AND '.join(where_parts)}"
                )
                uid_df = _query_df(engine, q_uid)
                cmd_uid_count = int(uid_df.iloc[0]["uid_cnt"]) if not uid_df.empty else 0
            except Exception:
                cmd_uid_count = 0
        if cmd_h_from_axis is not None:
            cmd_effective_h = int(cmd_h_from_axis)
        else:
            cmd_effective_h = int(cmd_uid_count) if auto_h_by_uid and cmd_uid_count > 0 else int(manual_h)
        if cmd_effective_h <= 0:
            cmd_effective_h = 1
        cmd_effective_h_resolved, cmd_h_adjust_note = _resolve_model_horizon(cmd_model, int(cmd_effective_h))
        cmd_effective_h = int(cmd_effective_h_resolved or 1)

        tr_errors: list[str] = []
        if tr_param_builder_errors:
            tr_errors.extend(list(tr_param_builder_errors))
        base_params_obj: dict[str, Any] = {}
        try:
            base_params_obj = json.loads(tr_params_raw) if str(tr_params_raw).strip() else {}
            if not isinstance(base_params_obj, dict):
                raise ValueError("base params-json must be object")
        except Exception as e:
            tr_errors.append(f"base params-json: {e}")
            base_params_obj = {}

        tr_nf_fit = _json_text(tr_nf_fit_raw, "dict", "nf-fit-kwargs-json", tr_errors)
        tr_nf_predict = _json_text(tr_nf_predict_raw, "dict", "nf-predict-kwargs-json", tr_errors)
        tr_nf_cv = _json_text(tr_nf_cv_raw, "dict", "nf-cross-validation-kwargs-json", tr_errors)
        tr_nf_save = _json_text(tr_nf_save_raw, "dict", "nf-save-kwargs-json", tr_errors)
        tr_nf_load = _json_text(tr_nf_load_raw, "dict", "nf-load-kwargs-json", tr_errors)
        tr_nf_ins = _json_text(tr_nf_ins_raw, "dict", "nf-predict-insample-kwargs-json", tr_errors)
        tr_nf_fit_obj = json.loads(tr_nf_fit) if tr_nf_fit is not None else {}
        tr_nf_predict_obj = json.loads(tr_nf_predict) if tr_nf_predict is not None else {}
        tr_nf_cv_obj = json.loads(tr_nf_cv) if tr_nf_cv is not None else {}
        tr_nf_save_obj = json.loads(tr_nf_save) if tr_nf_save is not None else {}
        tr_nf_load_obj = json.loads(tr_nf_load) if tr_nf_load is not None else {}
        tr_nf_ins_obj = json.loads(tr_nf_ins) if tr_nf_ins is not None else {}

        def _fit_kwargs_from_combo_row(row_obj: dict[str, Any] | None) -> dict[str, Any]:
            merged = dict(tr_nf_fit_obj)
            if not isinstance(row_obj, dict):
                return merged
            for axis_key, fit_key in fit_axis_to_fit_key.items():
                if axis_key not in row_obj:
                    continue
                merged[fit_key] = _decode_fit_axis_value(axis_key, row_obj.get(axis_key))
            return merged

        cmd_nf_fit_obj = _fit_kwargs_from_combo_row(combo_cmd_row if combo_cmd_row else None)
        cmd_nf_fit = json.dumps(cmd_nf_fit_obj, ensure_ascii=False) if tr_nf_fit is not None else None
        predict_h_raw = tr_nf_predict_obj.get("h")
        cv_h_raw = tr_nf_cv_obj.get("h")

        def _as_int_or_none(v: Any) -> int | None:
            try:
                if v is None or str(v).strip() == "":
                    return None
                return int(v)
            except Exception:
                return None

        predict_h_int = _as_int_or_none(predict_h_raw)
        cv_h_int = _as_int_or_none(cv_h_raw)
        predict_h_mismatch = predict_h_int is not None and int(predict_h_int) != int(cmd_effective_h)
        cv_h_mismatch = cv_h_int is not None and int(cv_h_int) != int(cmd_effective_h)
        if bool(st.session_state.get("nf_lab_train_runtime_sync_h", True)):
            tr_nf_predict_obj["h"] = int(cmd_effective_h)
            tr_nf_cv_obj["h"] = int(cmd_effective_h)
        tr_nf_predict = json.dumps(tr_nf_predict_obj, ensure_ascii=False) if tr_nf_predict is not None else None
        tr_nf_cv = json.dumps(tr_nf_cv_obj, ensure_ascii=False) if tr_nf_cv is not None else None
        tr_nf_fit = cmd_nf_fit
        tr_nf_save = json.dumps(tr_nf_save_obj, ensure_ascii=False) if tr_nf_save is not None else None
        tr_nf_load = json.dumps(tr_nf_load_obj, ensure_ascii=False) if tr_nf_load is not None else None
        tr_nf_ins = json.dumps(tr_nf_ins_obj, ensure_ascii=False) if tr_nf_ins is not None else None
        tr_exog_auto = bool(st.session_state.get("nf_lab_train_exog_auto", True))
        tr_futr = None if tr_exog_auto else _json_text(tr_futr_raw, "list", "futr-exog-list-json", tr_errors)
        tr_hist = None if tr_exog_auto else _json_text(tr_hist_raw, "list", "hist-exog-list-json", tr_errors)
        tr_stat = None if tr_exog_auto else _json_text(tr_stat_raw, "list", "stat-exog-list-json", tr_errors)

        def _json_list_from_text(raw_v: Any) -> list[str]:
            if raw_v is None:
                return []
            if isinstance(raw_v, list):
                return [str(x).strip() for x in raw_v if str(x).strip()]
            sv = str(raw_v).strip()
            if not sv:
                return []
            try:
                loaded = json.loads(sv)
                if isinstance(loaded, list):
                    return [str(x).strip() for x in loaded if str(x).strip()]
            except Exception:
                pass
            return []

        manual_futr_list = _json_list_from_text(tr_futr)
        manual_hist_list = _json_list_from_text(tr_hist)
        manual_stat_list = _json_list_from_text(tr_stat)
        cmd_detected_exog = (
            _prefixed_exog_cols_for_table(str(cmd_dataset_schema), str(cmd_dataset_table))
            if str(cmd_dataset_input_method) == "db_table"
            else {"futr_exog": [], "hist_exog": [], "stat_exog": []}
        )
        cmd_futr_list = _decode_exog_axis_value(combo_cmd_row.get("futr_exog")) if combo_cmd_row else []
        cmd_hist_list = _decode_exog_axis_value(combo_cmd_row.get("hist_exog")) if combo_cmd_row else []
        cmd_stat_list = _decode_exog_axis_value(combo_cmd_row.get("stat_exog")) if combo_cmd_row else []
        if not cmd_futr_list:
            cmd_futr_list = list(cmd_detected_exog.get("futr_exog", [])) if tr_exog_auto else list(manual_futr_list)
        if not cmd_hist_list:
            cmd_hist_list = list(cmd_detected_exog.get("hist_exog", [])) if tr_exog_auto else list(manual_hist_list)
        if not cmd_stat_list:
            cmd_stat_list = list(cmd_detected_exog.get("stat_exog", [])) if tr_exog_auto else list(manual_stat_list)
        cmd_exog_adjustments: list[str] = []
        cmd_support = (
            dict(get_model_exog_support(str(cmd_model)))
            if get_model_exog_support is not None
            else {"futr": False, "hist": False, "stat": False}
        )
        if cmd_futr_list and not bool(cmd_support.get("futr", False)):
            cmd_exog_adjustments.append(f"futr_exog({len(cmd_futr_list)})")
            cmd_futr_list = []
        if cmd_hist_list and not bool(cmd_support.get("hist", False)):
            cmd_exog_adjustments.append(f"hist_exog({len(cmd_hist_list)})")
            cmd_hist_list = []
        if cmd_stat_list and not bool(cmd_support.get("stat", False)):
            cmd_exog_adjustments.append(f"stat_exog({len(cmd_stat_list)})")
            cmd_stat_list = []
        cmd_futr = json.dumps(cmd_futr_list, ensure_ascii=False)
        cmd_hist = json.dumps(cmd_hist_list, ensure_ascii=False)
        cmd_stat = json.dumps(cmd_stat_list, ensure_ascii=False)

        def _prune_empty_params(payload: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in payload.items():
                if isinstance(v, str) and v.strip() == "":
                    continue
                out[str(k)] = v
            return out

        params_payload = _prune_empty_params(dict(base_params_obj))
        params_payload.update(tr_param_builder_obj)
        params_payload.update(
            {
                "backend": str(cmd_backend),
                "num_samples": int(cmd_num_samples),
                "loss_name": str(cmd_loss),
                "valid_loss_name": str(cmd_loss),
                "freq": str(cmd_freq),
                "local_scaler_type": cmd_local_scaler_type,
                "local_static_scaler_type": cmd_local_scaler_type,
                "dataset_schema": cmd_dataset_schema,
                "dataset_table": cmd_dataset_table,
                "dataset_input_method": str(cmd_dataset_input_method),
                "dataset_where": tr_dataset_where.strip() or None,
                "dataset_path": str(cmd_dataset_path).strip() or None,
                "dataset_sql": str(cmd_dataset_sql).strip() or None,
                "dataframe_backend": str(cmd_dataframe_backend),
                "group_by_mode": cmd_group_mode,
                "target_loto": ",".join([str(x) for x in cmd_loto]) if cmd_loto else "",
                "target_unique_id": ",".join([str(x) for x in cmd_uid]) if cmd_uid else "",
                "target_ts_type": ",".join([str(x) for x in cmd_ts_type]) if cmd_ts_type else "",
                "h_mode": "fixed" if (cmd_h_from_axis is not None or not auto_h_by_uid) else "unique_id_count",
                "auto_exog_from_table": bool(tr_exog_auto),
                "futr_exog_list": list(cmd_futr_list),
                "hist_exog_list": list(cmd_hist_list),
                "stat_exog_list": list(cmd_stat_list),
            }
        )
        if str(cmd_search).strip():
            params_payload["search_alg_name"] = str(cmd_search).strip()
        else:
            params_payload.pop("search_alg_name", None)
        params_payload = _prune_empty_params(params_payload)
        for _blocked in ["h", "loss", "valid_loss", "search_alg"]:
            params_payload.pop(_blocked, None)
        tr_params = json.dumps(params_payload, ensure_ascii=False)

        preflight_errors: list[str] = []
        preflight_warnings: list[str] = []
        if str(cmd_dataset_input_method) == "db_table":
            if not str(cmd_dataset_table).strip():
                preflight_errors.append("dataset_input_method=db_table では dataset table が必須です。")
        elif str(cmd_dataset_input_method) == "db_sql":
            if not str(cmd_dataset_sql).strip():
                preflight_errors.append("dataset_input_method=db_sql では dataset SQL が必須です。")
        elif str(cmd_dataset_input_method) in {"csv", "parquet", "json"} and not str(cmd_dataset_path).strip():
            preflight_errors.append(f"dataset_input_method={cmd_dataset_input_method} では dataset path が必須です。")
        if not _is_supported_backend_for_input_method(str(cmd_dataset_input_method), str(cmd_dataframe_backend)):
            allowed_cmd = _supported_backends_for_input_method(str(cmd_dataset_input_method))
            preflight_errors.append(
                f"unsupported dataframe backend: {cmd_dataframe_backend} "
                f"(allowed for {cmd_dataset_input_method}: {', '.join(allowed_cmd)})"
            )
        uid_group_error_cmd = _group_mode_unique_id_validation_error(cmd_group_mode, cmd_uid)
        if uid_group_error_cmd:
            preflight_errors.append(uid_group_error_cmd)
        invalid_cmd_choice = _validate_train_combo_choice(
            cmd_model, str(params_payload.get("backend", "")), cmd_valid_loss, cmd_search
        )
        if invalid_cmd_choice:
            preflight_errors.append(invalid_cmd_choice)
        prereq_err = _validate_model_runtime_prerequisites(cmd_model, int(cmd_effective_h))
        if prereq_err:
            preflight_errors.append(prereq_err)
        for k in ["h", "loss", "valid_loss", "search_alg"]:
            if k in params_payload:
                preflight_errors.append(f"params-json に禁止キー `{k}` が残っています。")
        if int(tr_cpus) == 0 and int(tr_gpus) == 0:
            preflight_warnings.append("cpus/gpus が 0 です。学習が進まない可能性があります。")
        if auto_h_by_uid and str(cmd_dataset_input_method) != "db_table":
            preflight_warnings.append(
                "dataset_input_method が db_table 以外のため、h自動算出は manual h を使用します。"
            )
        if str(cmd_group_mode) == "loto_ts_type":
            preflight_warnings.append("学習単位 候補=loto_ts_type のため target_unique_id は自動で None 扱いです。")
        if int(cmd_num_samples) <= 1 and not bool(tr_run_cv):
            preflight_warnings.append("num_samples=1 かつ CV無効 のため、短時間で終了する最小構成実行になります。")
        if cmd_exog_adjustments:
            preflight_warnings.append("モデル未対応のため exog を自動除外: " + ", ".join(cmd_exog_adjustments))
        if str(cmd_h_adjust_note or "").strip():
            preflight_warnings.append(str(cmd_h_adjust_note))
        if (predict_h_mismatch or cv_h_mismatch) and bool(st.session_state.get("nf_lab_train_runtime_sync_h", True)):
            preflight_warnings.append(f"runtime kwargs の h を effective h={int(cmd_effective_h)} に同期しました。")
        elif predict_h_mismatch or cv_h_mismatch:
            preflight_warnings.append(
                "runtime kwargs の h と effective h が不一致です。predict/cv結果の解釈に注意してください。"
            )

        fit_defaults = {
            "df": "auto",
            "static_df": None,
            "val_size": 0,
            "use_init_models": False,
            "verbose": False,
            "id_col": "unique_id",
            "time_col": "ds",
            "target_col": "y",
            "distributed_config": None,
            "prediction_intervals": None,
        }
        predict_defaults: dict[str, Any] = {
            "df": "auto",
            "static_df": None,
            "futr_df": "auto",
            "verbose": False,
            "engine": None,
            "level": None,
            "quantiles": None,
            "h": None,
            "data_kwargs": {},
        }
        cv_defaults: dict[str, Any] = {
            "df": "auto",
            "static_df": None,
            "n_windows": 1,
            "step_size": 1,
            "val_size": 0,
            "test_size": None,
            "use_init_models": False,
            "verbose": False,
            "refit": False,
            "id_col": "unique_id",
            "time_col": "ds",
            "target_col": "y",
            "prediction_intervals": None,
            "level": None,
            "quantiles": None,
            "h": None,
            "data_kwargs": {},
        }
        save_defaults = {"path": "<artifact_path>", "model_index": None, "save_dataset": True, "overwrite": False}
        load_defaults = {"path": "<artifact_path>", "verbose": False, "kwargs": {}}
        insample_defaults = {"step_size": 1, "level": None, "quantiles": None}
        runtime_summary_df = pd.concat(
            [
                _nf_signature_rows("fit", fit_defaults, cmd_nf_fit_obj),
                _nf_signature_rows(
                    "predict",
                    predict_defaults,
                    tr_nf_predict_obj,
                    forced=(
                        {"h": int(cmd_effective_h)}
                        if bool(st.session_state.get("nf_lab_train_runtime_sync_h", True))
                        else None
                    ),
                ),
                _nf_signature_rows(
                    "cross_validation",
                    cv_defaults,
                    tr_nf_cv_obj,
                    forced=(
                        {"h": int(cmd_effective_h)}
                        if bool(st.session_state.get("nf_lab_train_runtime_sync_h", True))
                        else None
                    ),
                ),
                _nf_signature_rows("save", save_defaults, tr_nf_save_obj),
                _nf_signature_rows("load", load_defaults, tr_nf_load_obj),
                _nf_signature_rows("predict_insample", insample_defaults, tr_nf_ins_obj),
            ],
            axis=0,
            ignore_index=True,
        )
        with st.expander("NeuralForecast API引数サマリ（現在値）", expanded=False):
            _show_df(runtime_summary_df, hide_index=True)
            st.caption(
                "source=default/override/forced。`forced` はこの画面の `effective h` と整合させるために自動調整された値です。"
            )

        tr_parts = [
            "python",
            "-m",
            "loto_forecast.cli",
            "train",
            "--model",
            cmd_model,
            "--h",
            str(int(cmd_effective_h)),
            "--cpus",
            str(int(tr_cpus)),
            "--gpus",
            str(int(tr_gpus)),
            _bool_opt("refit-with-val", tr_refit),
            _bool_opt("verbose", tr_verbose),
            _bool_opt("strict-exog", tr_strict),
            _bool_opt("run-cross-validation", tr_run_cv),
        ]
        if cmd_search.strip():
            tr_parts.extend(["--search-alg-name", cmd_search.strip()])
        for flg, val in [
            ("--params-json", tr_params),
            ("--nf-fit-kwargs-json", tr_nf_fit),
            ("--nf-predict-kwargs-json", tr_nf_predict),
            ("--nf-cross-validation-kwargs-json", tr_nf_cv),
            ("--nf-save-kwargs-json", tr_nf_save),
            ("--nf-load-kwargs-json", tr_nf_load),
            ("--nf-predict-insample-kwargs-json", tr_nf_ins),
            ("--futr-exog-list-json", cmd_futr),
            ("--hist-exog-list-json", cmd_hist),
            ("--stat-exog-list-json", cmd_stat),
        ]:
            if val is not None:
                tr_parts.extend([flg, val])
        cmd_train = " ".join(shlex.quote(x) for x in tr_parts)
        combo_skip_reason_counts: dict[str, int] = {}
        if "reason" in pd.DataFrame(skipped_rows if "skipped_rows" in locals() else []).columns:
            skipped_reason_df = pd.DataFrame(skipped_rows)
            combo_skip_reason_counts = (
                skipped_reason_df["reason"].astype(str).value_counts().to_dict() if not skipped_reason_df.empty else {}
            )
        wizard_state = build_nf_wizard_state(
            mode=str(train_ui_mode),
            action="学習(train)",
            model=str(cmd_model),
            dataset_input_method=str(cmd_dataset_input_method),
            dataset_table=str(cmd_dataset_table),
            dataset_path=str(cmd_dataset_path),
            dataset_sql=str(cmd_dataset_sql),
            unique_ids=list(cmd_uid),
            ts_types=list(cmd_ts_type),
            horizon=int(cmd_effective_h),
            backend=str(cmd_backend),
            loss=str(cmd_loss),
            search_alg=str(cmd_search),
            combo_total=(int(combo_df.shape[0]) if isinstance(combo_df, pd.DataFrame) else None),
            combo_skip_reasons=combo_skip_reason_counts,
            command_preview=cmd_train,
        )
        st.markdown("### Step Wizard")
        wizard_step_df = pd.DataFrame(
            [{"step": step.title, "status": step.status, "summary": step.summary} for step in wizard_state.steps]
        )
        _show_df(wizard_step_df, hide_index=True)
        if wizard_state.required_missing:
            st.warning("必須不足: " + " / ".join(wizard_state.required_missing))
        for issue in wizard_state.issues:
            st.warning(f"原因: {issue.reason}\n影響: {issue.impact}\n対処: {issue.fix}")
        if wizard_state.can_run:
            st.success("実行可能です。次は実行前チェックを確認して Run train を押してください。")
        else:
            st.info("まだ実行前の修正が必要です。次にやること: " + " / ".join(wizard_state.next_actions))
        with st.expander("実行前チェック", expanded=True):
            if preflight_errors:
                st.error("\n".join([f"- {m}" for m in preflight_errors]))
            else:
                st.success("エラーなし: 実行可能です。")
            if preflight_warnings:
                st.warning("\n".join([f"- {m}" for m in preflight_warnings]))
        _render_command_preview(
            cmd_train,
            copy_key="nf_lab_copy_train_cmd",
            copy_label="Copy train command",
            cwd=lab_cwd,
            show_arg_table=True,
        )
        if tr_errors:
            st.error("\n".join(tr_errors))
        run_train_disabled = bool(tr_errors) or bool(preflight_errors)
        if st.button("Run train", key="nf_lab_run_train", disabled=run_train_disabled):
            if not lab_cwd.exists() or not lab_cwd.is_dir():
                st.error("lab cwd が有効なディレクトリではありません。")
            else:
                st.session_state["nf_lab_train_result"] = _run_shell_command_live(
                    cmd_train,
                    cwd=lab_cwd,
                    timeout_sec=lab_timeout_sec,
                    title="NF train",
                )
        _render_command_result_block("nf_lab_train_result", "train 実行結果", "nf_lab_train")

        st.markdown("**有効候補の全件実行（順次）**")
        st.caption(
            "組合せ表を作成しただけでは単一コマンドのみ実行されます。"
            " 下記で複数行を選んで実行すると、全体進捗バー付きで順次実行します。"
        )
        st.info(
            "72件など大量実行は `固定/網羅メタ反映` の `反映後に全件自動実行` を使うと、失敗時も継続し DB へ結果記録できます。"
        )
        if combo_df.empty:
            st.info("固定/網羅メタ反映 の有効組合せがないため、一括実行できません。")
        else:
            combo_run_all = st.toggle(
                "全件実行モード: 有効候補をすべて対象にする",
                value=True,
                key="nf_lab_train_combo_run_all_mode",
            )
            if combo_run_all:
                st.session_state["nf_lab_train_combo_run_start"] = 1
                st.session_state["nf_lab_train_combo_run_count"] = int(combo_df.shape[0])
            else:
                st.session_state["nf_lab_train_combo_run_start"] = max(
                    1,
                    min(int(st.session_state.get("nf_lab_train_combo_run_start", 1) or 1), int(combo_df.shape[0])),
                )
                st.session_state["nf_lab_train_combo_run_count"] = max(
                    1,
                    min(
                        int(st.session_state.get("nf_lab_train_combo_run_count", int(combo_df.shape[0])) or 1),
                        int(combo_df.shape[0]),
                    ),
                )
            run_start = int(
                st.number_input(
                    "実行開始行 (1-based)",
                    min_value=1,
                    max_value=int(combo_df.shape[0]),
                    value=1,
                    step=1,
                    key="nf_lab_train_combo_run_start",
                    disabled=combo_run_all,
                )
            )
            run_count = int(
                st.number_input(
                    "実行行数",
                    min_value=1,
                    max_value=int(combo_df.shape[0]),
                    value=int(combo_df.shape[0]),
                    step=1,
                    key="nf_lab_train_combo_run_count",
                    disabled=combo_run_all,
                )
            )
            if int(run_count) < int(combo_df.shape[0]):
                st.warning(
                    f"現在は部分実行です: {run_count}/{int(combo_df.shape[0])} 件。"
                    " 全件実行したい場合は `全件実行モード: 有効候補をすべて対象にする` をONにしてください。"
                )
            run_continue_on_error = st.toggle(
                "失敗しても継続", value=True, key="nf_lab_train_combo_continue_on_error"
            )
            post_c1, post_c2 = st.columns(2)
            run_predict_after_train = post_c1.toggle(
                "一括実行: train後にpredict(未知予測)を実行",
                value=False,
                key="nf_lab_train_combo_run_predict_after_train",
            )
            run_evaluate_after_train = post_c2.toggle(
                "一括実行: train後にevaluate(予測ステップ分割)を実行",
                value=False,
                key="nf_lab_train_combo_run_evaluate_after_train",
            )
            pe1, pe2 = st.columns([1.3, 1.7])
            run_eval_step_size = int(
                pe1.number_input(
                    "evaluate step分割サイズ",
                    min_value=1,
                    max_value=3650,
                    value=1,
                    step=1,
                    key="nf_lab_train_combo_run_eval_step_size",
                    disabled=(not run_evaluate_after_train),
                )
            )
            run_apply_where_post = pe2.toggle(
                "predict/evaluate でも dataset_where を適用（OFF=全件）",
                value=False,
                key="nf_lab_train_combo_run_post_apply_where",
                disabled=(not run_predict_after_train and not run_evaluate_after_train),
            )

            combo_uid_count_cache: dict[str, int] = {}

            def _sql_in(values: list[str]) -> str:
                return ", ".join(["'" + str(v).replace("'", "''") + "'" for v in values if str(v).strip()])

            def _uid_count_for_train_combo(
                schema_v: str, table_v: str, loto_vals_v: list[str], uid_vals_v: list[str], ts_type_vals_v: list[str]
            ) -> int:
                cache_k = "|".join(
                    [
                        schema_v,
                        table_v,
                        ",".join(loto_vals_v),
                        ",".join(uid_vals_v),
                        ",".join(ts_type_vals_v),
                        str(tr_dataset_where).strip(),
                    ]
                )
                if cache_k in combo_uid_count_cache:
                    return int(combo_uid_count_cache[cache_k])
                if engine is None:
                    return 1
                where_parts = [f"{_safe_ident(settings.id_col)} IS NOT NULL"]
                if str(tr_dataset_where).strip():
                    where_parts.append(f"({str(tr_dataset_where).strip()})")
                if loto_vals_v:
                    where_parts.append(f"{_safe_ident('loto')} IN ({_sql_in(loto_vals_v)})")
                if uid_vals_v:
                    where_parts.append(f"{_safe_ident(settings.id_col)} IN ({_sql_in(uid_vals_v)})")
                if ts_type_vals_v:
                    where_parts.append(f"{_safe_ident('ts_type')} IN ({_sql_in(ts_type_vals_v)})")
                q = (
                    f"SELECT COUNT(DISTINCT {_safe_ident(settings.id_col)}) AS uid_cnt "
                    f"FROM {_safe_ident(schema_v)}.{_safe_ident(table_v)} "
                    f"WHERE {' AND '.join(where_parts)}"
                )
                try:
                    out_df = _query_df(engine, q)
                    cnt = int(out_df.iloc[0]["uid_cnt"]) if not out_df.empty else 1
                except Exception:
                    cnt = 1
                cnt = max(1, int(cnt))
                combo_uid_count_cache[cache_k] = cnt
                return cnt

            def _safe_row_get(row_obj: dict[str, Any], key: str, default: Any) -> Any:
                if key not in row_obj:
                    return default
                v = row_obj.get(key)
                if v is None:
                    return default
                if isinstance(v, float) and np.isnan(v):
                    return default
                if str(v).strip().lower() == "nan":
                    return default
                return v

            def _append_dataset_source_args(
                args: list[str],
                *,
                dataset_input_method_v: str,
                dataframe_backend_v: str,
                schema_v: str,
                table_v: str,
                dataset_where_v: str,
                dataset_path_v: str,
                dataset_sql_v: str,
                apply_where: bool,
            ) -> None:
                args.extend(["--dataset-input-method", str(dataset_input_method_v)])
                args.extend(["--dataframe-backend", str(dataframe_backend_v)])
                if str(dataset_input_method_v) == "db_table":
                    args.extend(["--dataset-schema", str(schema_v), "--dataset-table", str(table_v)])
                    if apply_where and str(dataset_where_v).strip():
                        args.extend(["--dataset-where", str(dataset_where_v).strip()])
                elif str(dataset_input_method_v) == "db_sql":
                    if str(dataset_sql_v).strip():
                        args.extend(["--dataset-sql", str(dataset_sql_v).strip()])
                else:
                    if str(dataset_path_v).strip():
                        args.extend(["--dataset-path", str(dataset_path_v).strip()])

            def _compose_train_command_for_combo_row(row_obj: dict[str, Any]) -> dict[str, Any]:
                model_v = str(_safe_row_get(row_obj, "model", cmd_model))
                backend_row_v = _normalize_optional_train_core_value(_safe_row_get(row_obj, "backend", None))
                backend_v = backend_row_v or str(cmd_backend)
                num_samples_v = int(_safe_row_get(row_obj, "num_samples", cmd_num_samples))
                loss_row_v = _normalize_optional_train_core_value(_safe_row_get(row_obj, "loss", None))
                loss_v = str(loss_row_v or cmd_loss)
                valid_loss_row_v = _normalize_optional_train_core_value(_safe_row_get(row_obj, "valid_loss", None))
                valid_loss_v = str(loss_v) if valid_loss_row_v is not None else str(loss_v)
                search_row_v = _normalize_optional_train_core_value(_safe_row_get(row_obj, "search_alg", None))
                search_v = str(search_row_v or _default_search_alg_for_backend(backend_v, cmd_search)).strip()
                local_scaler_v_raw = _safe_row_get(row_obj, "local_scaler_type", tr_local_scaler_type)
                local_scaler_v = None if str(local_scaler_v_raw) == "(none)" else str(local_scaler_v_raw)
                schema_v = str(_safe_row_get(row_obj, "dataset_schema", cmd_dataset_schema))
                table_v = str(_safe_row_get(row_obj, "dataset_table", cmd_dataset_table))
                dataset_input_method_v = str(_safe_row_get(row_obj, "dataset_input_method", cmd_dataset_input_method))
                dataframe_backend_v = str(_safe_row_get(row_obj, "dataframe_backend", cmd_dataframe_backend))
                dataset_path_v = str(_safe_row_get(row_obj, "dataset_path", cmd_dataset_path)).strip()
                dataset_sql_v = str(_safe_row_get(row_obj, "dataset_sql", cmd_dataset_sql)).strip()
                group_mode_v = str(_safe_row_get(row_obj, "group_mode", cmd_group_mode))
                loto_v = _to_csv_list(_safe_row_get(row_obj, "loto", ",".join(cmd_loto)))
                uid_v = _to_csv_list(_safe_row_get(row_obj, "unique_id", ",".join(cmd_uid)))
                uid_group_error_local = _group_mode_unique_id_validation_error(group_mode_v, uid_v)
                if group_mode_v == "loto_ts_type":
                    uid_v = []
                ts_type_v = _to_csv_list(_safe_row_get(row_obj, "ts_type", ",".join(cmd_ts_type)))

                h_from_axis = _parse_horizon_axis_value(_safe_row_get(row_obj, "horizon", None))
                if h_from_axis is not None:
                    h_v = int(h_from_axis)
                elif auto_h_by_uid:
                    if dataset_input_method_v != "db_table":
                        h_v = int(manual_h)
                    elif group_mode_v == "loto_unique_id_ts_type":
                        h_v = 1 if uid_v else _uid_count_for_train_combo(schema_v, table_v, loto_v, uid_v, ts_type_v)
                    else:
                        h_v = _uid_count_for_train_combo(schema_v, table_v, loto_v, uid_v, ts_type_v)
                else:
                    h_v = int(manual_h)
                h_v = max(1, int(h_v))
                h_v_resolved, _h_adjust_note = _resolve_model_horizon(model_v, int(h_v))
                h_v = int(h_v_resolved or 1)

                params_local = _prune_empty_params(dict(base_params_obj))
                params_local.update(tr_param_builder_obj)
                nf_fit_local_obj = _normalize_fit_kwargs_for_horizon(_fit_kwargs_from_combo_row(row_obj), int(h_v))
                nf_fit_local_text = json.dumps(nf_fit_local_obj, ensure_ascii=False) if tr_nf_fit is not None else None
                detected_exog_local = (
                    _prefixed_exog_cols_for_table(schema_v, table_v)
                    if dataset_input_method_v == "db_table"
                    else {"futr_exog": [], "hist_exog": [], "stat_exog": []}
                )
                futr_local_list = _decode_exog_axis_value(_safe_row_get(row_obj, "futr_exog", []))
                hist_local_list = _decode_exog_axis_value(_safe_row_get(row_obj, "hist_exog", []))
                stat_local_list = _decode_exog_axis_value(_safe_row_get(row_obj, "stat_exog", []))
                if not futr_local_list:
                    futr_local_list = (
                        list(detected_exog_local.get("futr_exog", [])) if tr_exog_auto else list(manual_futr_list)
                    )
                if not hist_local_list:
                    hist_local_list = (
                        list(detected_exog_local.get("hist_exog", [])) if tr_exog_auto else list(manual_hist_list)
                    )
                if not stat_local_list:
                    stat_local_list = (
                        list(detected_exog_local.get("stat_exog", [])) if tr_exog_auto else list(manual_stat_list)
                    )
                support_local = (
                    dict(get_model_exog_support(model_v))
                    if get_model_exog_support is not None
                    else {"futr": False, "hist": False, "stat": False}
                )
                if futr_local_list and not bool(support_local.get("futr", False)):
                    futr_local_list = []
                if hist_local_list and not bool(support_local.get("hist", False)):
                    hist_local_list = []
                if stat_local_list and not bool(support_local.get("stat", False)):
                    stat_local_list = []
                futr_local_text = json.dumps(futr_local_list, ensure_ascii=False)
                hist_local_text = json.dumps(hist_local_list, ensure_ascii=False)
                stat_local_text = json.dumps(stat_local_list, ensure_ascii=False)
                params_local.update(
                    {
                        "backend": backend_v,
                        "num_samples": int(num_samples_v),
                        "loss_name": loss_v,
                        "valid_loss_name": loss_v,
                        "freq": str(_safe_row_get(row_obj, "freq", cmd_freq)),
                        "local_scaler_type": local_scaler_v,
                        "local_static_scaler_type": local_scaler_v,
                        "dataset_schema": schema_v,
                        "dataset_table": table_v,
                        "dataset_input_method": dataset_input_method_v,
                        "dataset_where": str(tr_dataset_where).strip() or None,
                        "dataset_path": dataset_path_v or None,
                        "dataset_sql": dataset_sql_v or None,
                        "dataframe_backend": dataframe_backend_v,
                        "group_by_mode": group_mode_v,
                        "target_loto": ",".join(loto_v) if loto_v else "",
                        "target_unique_id": ",".join(uid_v) if uid_v else "",
                        "target_ts_type": ",".join(ts_type_v) if ts_type_v else "",
                        "h_mode": "fixed" if (h_from_axis is not None or not auto_h_by_uid) else "unique_id_count",
                        "auto_exog_from_table": bool(tr_exog_auto),
                        "futr_exog_list": list(futr_local_list),
                        "hist_exog_list": list(hist_local_list),
                        "stat_exog_list": list(stat_local_list),
                    }
                )
                if search_v:
                    params_local["search_alg_name"] = search_v
                else:
                    params_local.pop("search_alg_name", None)
                for _blocked in ["h", "loss", "valid_loss", "search_alg"]:
                    params_local.pop(_blocked, None)
                params_local = _prune_empty_params(params_local)

                errors_local: list[str] = []
                if uid_group_error_local:
                    errors_local.append(uid_group_error_local)
                if dataset_input_method_v == "db_table" and not table_v.strip():
                    errors_local.append("dataset_input_method=db_table では dataset table が必須")
                if dataset_input_method_v == "db_sql" and not dataset_sql_v:
                    errors_local.append("dataset_input_method=db_sql では dataset SQL が必須")
                if dataset_input_method_v in {"csv", "parquet", "json"} and not dataset_path_v:
                    errors_local.append(f"dataset_input_method={dataset_input_method_v} では dataset path が必須")
                if not _is_supported_backend_for_input_method(dataset_input_method_v, dataframe_backend_v):
                    allowed_local = _supported_backends_for_input_method(dataset_input_method_v)
                    errors_local.append(
                        f"unsupported dataframe backend: {dataframe_backend_v} "
                        f"(allowed for {dataset_input_method_v}: {', '.join(allowed_local)})"
                    )
                invalid_local_choice = _validate_train_combo_choice(model_v, backend_v, valid_loss_v, search_v)
                if invalid_local_choice:
                    errors_local.append(invalid_local_choice)
                prereq_local = _validate_model_runtime_prerequisites(model_v, h_v)
                if prereq_local:
                    errors_local.append(prereq_local)

                parts_local = [
                    "python",
                    "-m",
                    "loto_forecast.cli",
                    "train",
                    "--model",
                    model_v,
                    "--h",
                    str(int(h_v)),
                    "--cpus",
                    str(int(tr_cpus)),
                    "--gpus",
                    str(int(tr_gpus)),
                    _bool_opt("refit-with-val", tr_refit),
                    _bool_opt("verbose", tr_verbose),
                    _bool_opt("strict-exog", tr_strict),
                    _bool_opt("run-cross-validation", tr_run_cv),
                ]
                if search_v:
                    parts_local.extend(["--search-alg-name", search_v])
                run_id_local = _make_param_based_run_id(model_v, h_v, params_local)
                parts_local.extend(["--run-id", run_id_local])
                params_local_text = json.dumps(params_local, ensure_ascii=False)
                for flg, val in [
                    ("--params-json", params_local_text),
                    ("--nf-fit-kwargs-json", nf_fit_local_text),
                    ("--nf-predict-kwargs-json", tr_nf_predict),
                    ("--nf-cross-validation-kwargs-json", tr_nf_cv),
                    ("--nf-save-kwargs-json", tr_nf_save),
                    ("--nf-load-kwargs-json", tr_nf_load),
                    ("--nf-predict-insample-kwargs-json", tr_nf_ins),
                    ("--futr-exog-list-json", futr_local_text),
                    ("--hist-exog-list-json", hist_local_text),
                    ("--stat-exog-list-json", stat_local_text),
                ]:
                    if val is not None:
                        parts_local.extend([flg, val])
                cmd_local = " ".join(shlex.quote(x) for x in parts_local)
                combo_signature = _build_train_combo_signature(model_v, h_v, params_local)
                command_items: list[dict[str, str]] = [{"phase": "train", "command": cmd_local}]
                if run_predict_after_train and not errors_local:
                    predict_parts_local = [
                        "python",
                        "-m",
                        "loto_forecast.cli",
                        "predict",
                        "--run-id",
                        str(run_id_local),
                        "--h",
                        str(int(h_v)),
                    ]
                    _append_dataset_source_args(
                        predict_parts_local,
                        dataset_input_method_v=dataset_input_method_v,
                        dataframe_backend_v=dataframe_backend_v,
                        schema_v=schema_v,
                        table_v=table_v,
                        dataset_where_v=str(tr_dataset_where),
                        dataset_path_v=dataset_path_v,
                        dataset_sql_v=dataset_sql_v,
                        apply_where=bool(run_apply_where_post),
                    )
                    command_items.append(
                        {
                            "phase": "predict",
                            "command": " ".join(shlex.quote(x) for x in predict_parts_local),
                        }
                    )
                if run_evaluate_after_train and not errors_local:
                    evaluate_parts_local = [
                        "python",
                        "-m",
                        "loto_forecast.cli",
                        "evaluate",
                        "--run-id",
                        str(run_id_local),
                        "--step-eval-size",
                        str(int(run_eval_step_size)),
                    ]
                    _append_dataset_source_args(
                        evaluate_parts_local,
                        dataset_input_method_v=dataset_input_method_v,
                        dataframe_backend_v=dataframe_backend_v,
                        schema_v=schema_v,
                        table_v=table_v,
                        dataset_where_v=str(tr_dataset_where),
                        dataset_path_v=dataset_path_v,
                        dataset_sql_v=dataset_sql_v,
                        apply_where=bool(run_apply_where_post),
                    )
                    command_items.append(
                        {
                            "phase": "evaluate",
                            "command": " ".join(shlex.quote(x) for x in evaluate_parts_local),
                        }
                    )
                return {
                    "command": cmd_local,
                    "commands": command_items,
                    "errors": errors_local,
                    "h": int(h_v),
                    "model": model_v,
                    "backend": backend_v,
                    "loss": loss_v,
                    "valid_loss": valid_loss_v,
                    "search_alg": search_v,
                    "dataset_schema": schema_v,
                    "dataset_table": table_v,
                    "dataset_input_method": dataset_input_method_v,
                    "dataframe_backend": dataframe_backend_v,
                    "run_id": run_id_local,
                    "combo_signature": combo_signature,
                    "params_json_obj": params_local,
                }

            end_idx = min(int(combo_df.shape[0]), run_start + run_count - 1)
            run_slice_df = combo_df.iloc[run_start - 1 : end_idx]
            run_plan: list[dict[str, Any]] = []
            for local_idx, (_, row_obj) in enumerate(run_slice_df.iterrows(), start=run_start):
                row_dict = row_obj.to_dict()
                built = _compose_train_command_for_combo_row(row_dict)
                run_plan.append(
                    {
                        "row_no": int(local_idx),
                        "model": built["model"],
                        "backend": built["backend"],
                        "loss": built["loss"],
                        "valid_loss": built["valid_loss"],
                        "search_alg": built["search_alg"],
                        "h": built["h"],
                        "dataset": f"{built['dataset_schema']}.{built['dataset_table']}",
                        "dataset_input": built.get("dataset_input_method"),
                        "df_backend": built.get("dataframe_backend"),
                        "run_id": built["run_id"],
                        "combo_signature": built["combo_signature"],
                        "params_json_obj": built["params_json_obj"],
                        "commands": list(built.get("commands", [])),
                        "phase_count": int(len(list(built.get("commands", []))) or 1),
                        "error": "; ".join(built["errors"]),
                        "command": built["command"],
                    }
                )
            completed_combo_index = _load_completed_combo_signatures(engine)
            for rec in run_plan:
                rec["already_executed"] = _is_combo_signature_completed(
                    str(rec.get("combo_signature", "")), completed_combo_index
                )
            run_plan_df = pd.DataFrame(run_plan)
            if not run_plan_df.empty:
                st.caption(f"実行予定: rows {run_start}..{end_idx}")
                display_cols = [
                    c
                    for c in [
                        "row_no",
                        "model",
                        "backend",
                        "loss",
                        "valid_loss",
                        "search_alg",
                        "h",
                        "dataset_input",
                        "df_backend",
                        "dataset",
                        "run_id",
                        "phase_count",
                        "already_executed",
                        "error",
                    ]
                    if c in run_plan_df.columns
                ]
                _show_df(run_plan_df[display_cols].head(500), hide_index=True)
                keep_open_cmds = st.toggle(
                    "実行コマンド一覧を開いたままにする",
                    value=True,
                    key="nf_lab_train_combo_cmds_expand",
                )
                with st.expander("実行コマンド一覧", expanded=bool(keep_open_cmds)):
                    cmd_rows = [
                        rec
                        for rec in run_plan
                        if (not str(rec.get("error", "")).strip()) and (not bool(rec.get("already_executed", False)))
                    ]
                    if not cmd_rows:
                        st.info("生成対象コマンドはありません（全件実行済み、または不整合）。")
                    else:
                        for rec in cmd_rows[:200]:
                            st.caption(f"row={int(rec.get('row_no', 0))} run_id={str(rec.get('run_id', ''))}")
                            row_cmds = rec.get("commands", [])
                            if isinstance(row_cmds, list) and row_cmds:
                                for cmd_item in row_cmds:
                                    st.caption(f"phase={str((cmd_item or {}).get('phase', 'train'))}")
                                    st.code(str((cmd_item or {}).get("command", "")), language="bash")
                            else:
                                st.code(str(rec.get("command", "")), language="bash")

            run_valid_plan = [rec for rec in run_plan if not str(rec.get("error", "")).strip()]
            run_exec_plan_rows = [rec for rec in run_valid_plan if not bool(rec.get("already_executed", False))]
            run_exec_plan: list[dict[str, Any]] = []
            for rec in run_exec_plan_rows:
                cmd_items = rec.get("commands", [])
                if not isinstance(cmd_items, list) or not cmd_items:
                    cmd_items = [{"phase": "train", "command": str(rec.get("command", ""))}]
                total_phases = int(len(cmd_items))
                for phase_idx, cmd_item in enumerate(cmd_items, start=1):
                    run_exec_plan.append(
                        {
                            **dict(rec),
                            "phase": str((cmd_item or {}).get("phase", "train")),
                            "phase_idx": int(phase_idx),
                            "phase_total": int(total_phases),
                            "command": str((cmd_item or {}).get("command", "")),
                        }
                    )
            run_commands = [str(rec.get("command", "")) for rec in run_exec_plan if str(rec.get("command", "")).strip()]
            invalid_n = int(sum(1 for rec in run_plan if str(rec.get("error", "")).strip()))
            already_done_n = int(sum(1 for rec in run_valid_plan if bool(rec.get("already_executed", False))))
            excluded_plan_rows = [
                {
                    "step": None,
                    "row_no": rec.get("row_no"),
                    "phase": "plan",
                    "phase_idx": 0,
                    "phase_total": 0,
                    "ok": False,
                    "returncode": None,
                    "elapsed_sec": 0.0,
                    "status": "excluded",
                    "model": rec.get("model"),
                    "run_id": rec.get("run_id"),
                    "artifact_exists": None,
                    "meta_exists": None,
                    "step_metrics_count": None,
                    "saved_to_db": False,
                    "reason": str(rec.get("error", "")).strip(),
                }
                for rec in run_plan
                if str(rec.get("error", "")).strip()
            ]
            skipped_plan_rows = [
                {
                    "step": None,
                    "row_no": rec.get("row_no"),
                    "phase": "plan",
                    "phase_idx": 0,
                    "phase_total": int(rec.get("phase_count", 1) or 1),
                    "ok": True,
                    "returncode": 0,
                    "elapsed_sec": 0.0,
                    "status": "skipped",
                    "model": rec.get("model"),
                    "run_id": rec.get("run_id"),
                    "artifact_exists": None,
                    "meta_exists": None,
                    "step_metrics_count": None,
                    "saved_to_db": False,
                    "reason": "既実行の組合せのためスキップしました。",
                }
                for rec in run_valid_plan
                if bool(rec.get("already_executed", False))
            ]
            if invalid_n > 0:
                st.warning(f"除外対象の設定組合せ: {invalid_n}件")
            if already_done_n > 0:
                st.info(f"既実行のためスキップ: {already_done_n}件")
            if run_valid_plan and already_done_n == len(run_valid_plan):
                st.success("選択範囲は全件実行済みです。コマンド/メタ生成をスキップします。")
            if run_commands:
                total_phase_n = int(len(run_commands))
                row_phase_n = int(len(run_exec_plan_rows))
                if run_predict_after_train or run_evaluate_after_train:
                    st.caption(
                        f"実行コマンド総数: {total_phase_n} "
                        f"(rows={row_phase_n}, predict={bool(run_predict_after_train)}, evaluate={bool(run_evaluate_after_train)})"
                    )
                joined_cmds = "\n".join(run_commands)
                _render_copy_button(
                    joined_cmds, key="nf_lab_train_combo_copy_all_cmds", label="実行コマンドを一括コピー", cwd=lab_cwd
                )
                out_dir = (PROJECT_ROOT / "artifacts" / "generated_commands").resolve()
                c_export, c_split_count, c_export_split = st.columns([1.0, 1.1, 1.2])
                split_default = min(4, max(1, len(run_commands)))
                export_single = c_export.button("bashファイルを生成", key="nf_lab_btn_train_combo_export_bash")
                split_count = int(
                    c_split_count.selectbox(
                        "分割数",
                        options=list(range(1, 51)),
                        index=max(0, split_default - 1),
                        key="nf_lab_train_combo_bash_split_count",
                    )
                )
                export_split = c_export_split.button(
                    "bashファイルを分割生成", key="nf_lab_btn_train_combo_export_bash_split"
                )
                if export_single:
                    script_path = write_bash_script(
                        run_commands,
                        out_dir=out_dir,
                        cwd=lab_cwd,
                        stop_on_error=(not bool(run_continue_on_error)),
                        file_stem="nf_combo_batch",
                    )
                    st.session_state["nf_lab_train_combo_batch_script_path"] = str(script_path)
                    st.session_state["nf_lab_train_combo_batch_split_script_paths"] = []
                    st.session_state["nf_lab_train_combo_batch_split_launch_command"] = ""
                    st.session_state["nf_lab_train_combo_batch_split_launcher_script_path"] = ""
                    st.success(f"bashファイルを作成しました: {script_path}")
                if export_split:
                    split_paths = write_split_bash_scripts(
                        run_commands,
                        split_count=split_count,
                        out_dir=out_dir,
                        cwd=lab_cwd,
                        stop_on_error=(not bool(run_continue_on_error)),
                        file_stem="nf_combo_batch",
                    )
                    st.session_state["nf_lab_train_combo_batch_split_script_paths"] = [str(p) for p in split_paths]
                    launch_cmd = build_split_tab_launch_command(split_paths)
                    st.session_state["nf_lab_train_combo_batch_split_launch_command"] = str(launch_cmd)
                    launcher_path = write_split_tab_launcher_script(
                        split_paths,
                        out_dir=out_dir,
                        file_stem="nf_combo_batch_launch_tabs",
                    )
                    st.session_state["nf_lab_train_combo_batch_split_launcher_script_path"] = str(launcher_path or "")
                    if split_paths:
                        st.success(f"分割bashを作成しました: {len(split_paths)}ファイル")
                    else:
                        st.warning("分割対象コマンドがありません。")
                saved_script_path = str(st.session_state.get("nf_lab_train_combo_batch_script_path", "")).strip()
                if saved_script_path:
                    st.code(saved_script_path, language="text")
                    _render_copy_button(
                        saved_script_path, key="nf_lab_train_combo_copy_bash_path", label="bashパスをコピー"
                    )
                saved_split_script_paths = st.session_state.get("nf_lab_train_combo_batch_split_script_paths", [])
                split_launch_cmd = str(
                    st.session_state.get("nf_lab_train_combo_batch_split_launch_command", "")
                ).strip()
                if split_launch_cmd:
                    st.code(split_launch_cmd, language="bash")
                    _render_copy_button(
                        split_launch_cmd,
                        key="nf_lab_train_combo_copy_split_bash_launch_cmd",
                        label="分割bashをタブ並列実行する1コマンドをコピー",
                        cwd=lab_cwd,
                    )
                if isinstance(saved_split_script_paths, list) and saved_split_script_paths:
                    split_paths_text = "\n".join([str(p) for p in saved_split_script_paths if str(p).strip()])
                    with st.expander("分割bashパス一覧", expanded=False):
                        if split_paths_text:
                            st.code(split_paths_text, language="text")
                            _render_copy_button(
                                split_paths_text,
                                key="nf_lab_train_combo_copy_split_bash_paths",
                                label="分割bashパスをコピー",
                            )
                split_launcher_path = str(
                    st.session_state.get("nf_lab_train_combo_batch_split_launcher_script_path", "")
                ).strip()
                if split_launcher_path:
                    st.code(split_launcher_path, language="text")
                    _render_copy_button(
                        split_launcher_path,
                        key="nf_lab_train_combo_copy_split_bash_launcher_path",
                        label="タブ並列実行launcherのbashパスをコピー",
                    )
            if st.button(
                "有効候補をすべて実行", key="nf_lab_run_train_combo_batch", disabled=(len(run_commands) == 0)
            ):
                if not lab_cwd.exists() or not lab_cwd.is_dir():
                    st.error("lab cwd が有効なディレクトリではありません。")
                else:
                    st.session_state["nf_lab_train_combo_batch_exec_plan"] = run_exec_plan
                    st.session_state["nf_lab_train_combo_batch_non_exec_rows"] = excluded_plan_rows + skipped_plan_rows
                    st.session_state["nf_lab_train_combo_batch_result"] = _run_command_sequence_live(
                        run_commands,
                        cwd=lab_cwd,
                        timeout_sec_per_command=lab_timeout_sec,
                        stop_on_error=(not bool(run_continue_on_error)),
                    )

            if "nf_lab_train_combo_batch_result" in st.session_state:
                batch_res = st.session_state["nf_lab_train_combo_batch_result"]
                exec_plan = st.session_state.get("nf_lab_train_combo_batch_exec_plan", [])
                non_exec_rows = st.session_state.get("nf_lab_train_combo_batch_non_exec_rows", [])
                if not isinstance(exec_plan, list):
                    exec_plan = []
                if not isinstance(non_exec_rows, list):
                    non_exec_rows = []
                b_rows: list[dict[str, Any]] = []
                model_ok_count = 0
                model_ng_count = 0
                phase_counts: Counter[str] = Counter()
                for i, r in enumerate(batch_res):
                    rec = exec_plan[i] if i < len(exec_plan) and isinstance(exec_plan[i], dict) else {}
                    phase_v = str(rec.get("phase", "train") or "train").strip() or "train"
                    phase_counts[phase_v] += 1
                    out_t = str(r.get("stdout", ""))
                    err_t = str(r.get("stderr", ""))
                    status_v = "success" if bool(r.get("ok", False)) else "failed"
                    artifact_v: Any = None
                    meta_exists_v: Any = None
                    run_id_v = str(
                        rec.get("run_id")
                        or _extract_last_run_id("\n".join([out_t, err_t]))
                        or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i + 1:04d}"
                    ).strip()
                    batch_model_name: str | None = str(rec.get("model") or "").strip() or None
                    error_text = ""
                    eval_step_metrics_n: int | None = None
                    saved_to_db = False

                    if phase_v == "train":
                        summ = _extract_train_result_summary(out_t, err_t) or {}
                        status_v = str(summ.get("status") or status_v).strip().lower() or status_v
                        artifact_v = summ.get("artifact_exists")
                        if artifact_v is None:
                            artifact_v = bool(status_v == "success" and bool(r.get("ok", False)))
                        meta_exists_v = summ.get("meta_exists")
                        if status_v == "success" and artifact_v is True:
                            model_ok_count += 1
                        elif status_v == "failed":
                            model_ng_count += 1
                        run_id_v = str(
                            summ.get("run_id") or _extract_last_run_id("\n".join([out_t, err_t])) or run_id_v
                        ).strip()
                        batch_model_name = (
                            str(summ.get("model_name") or batch_model_name or rec.get("model") or "").strip() or None
                        )
                        error_text = str(summ.get("error") or "").strip()
                        if not error_text and status_v != "success":
                            err_tail = _safe_tail(err_t, 30).strip() or _safe_tail(out_t, 30).strip()
                            error_text = err_tail[:8000] if err_tail else "batch command failed"
                    else:
                        parsed_tail = _try_parse_json_tail(str(out_t))
                        if phase_v == "evaluate" and isinstance(parsed_tail, dict):
                            step_metrics = parsed_tail.get("step_metrics")
                            if isinstance(step_metrics, list):
                                eval_step_metrics_n = int(len(step_metrics))
                        if status_v != "success":
                            err_tail = _safe_tail(err_t, 30).strip() or _safe_tail(out_t, 30).strip()
                            error_text = err_tail[:8000] if err_tail else f"{phase_v} command failed"

                    started_raw = str(r.get("started_at") or "").strip()
                    ended_raw = str(r.get("ended_at") or "").strip()
                    try:
                        started_at = datetime.fromisoformat(started_raw) if started_raw else datetime.now(timezone.utc)
                    except Exception:
                        started_at = datetime.now(timezone.utc)
                    try:
                        ended_at = datetime.fromisoformat(ended_raw) if ended_raw else datetime.now(timezone.utc)
                    except Exception:
                        ended_at = datetime.now(timezone.utc)

                    if phase_v == "train":
                        summ = _extract_train_result_summary(out_t, err_t) or {}
                        params_for_db: dict[str, Any] = (
                            dict(rec["params_json_obj"]) if isinstance(rec.get("params_json_obj"), dict) else {}
                        )
                        diagnostics = {
                            "source": "operations_dashboard_combo_batch",
                            "row_no": rec.get("row_no"),
                            "combo_signature": rec.get("combo_signature"),
                            "returncode": int(r.get("returncode", -1)),
                            "ok": bool(r.get("ok", False)),
                            "phase": phase_v,
                            "command": rec.get("command"),
                        }
                        model_save_json = {
                            "artifact_exists": bool(artifact_v) if artifact_v is not None else None,
                            "meta_exists": meta_exists_v,
                            "result_status": status_v,
                        }
                        try:
                            _upsert_combo_run_result_log(
                                engine,
                                run_id=run_id_v,
                                status=status_v,
                                model_name=str(batch_model_name or rec.get("model") or "unknown"),
                                horizon=int(rec.get("h", 1) or 1),
                                params_json=dict(params_for_db),
                                diagnostics_json=diagnostics,
                                model_save_json=model_save_json,
                                artifact_path=summ.get("artifact_path"),
                                log_path=summ.get("log_path"),
                                error_message=error_text,
                                started_at=started_at,
                                ended_at=ended_at,
                            )
                            saved_to_db = True
                        except Exception:
                            saved_to_db = False
                    b_rows.append(
                        {
                            "step": i + 1,
                            "row_no": rec.get("row_no"),
                            "phase": phase_v,
                            "phase_idx": int(rec.get("phase_idx", 1) or 1),
                            "phase_total": int(rec.get("phase_total", 1) or 1),
                            "ok": bool(r.get("ok", False)),
                            "returncode": int(r.get("returncode", -1)),
                            "elapsed_sec": float(r.get("elapsed_sec", 0.0) or 0.0),
                            "status": status_v or None,
                            "model": batch_model_name or rec.get("model"),
                            "run_id": run_id_v,
                            "artifact_exists": artifact_v,
                            "meta_exists": meta_exists_v,
                            "step_metrics_count": eval_step_metrics_n,
                            "saved_to_db": bool(saved_to_db),
                            "reason": error_text or None,
                        }
                    )
                for row in non_exec_rows:
                    if isinstance(row, dict):
                        b_rows.append(dict(row))
                bdf = pd.DataFrame(b_rows)
                if not bdf.empty:
                    status_counts = Counter(str(item).strip().lower() for item in bdf.get("status", pd.Series(dtype=str)).fillna("unknown"))
                    s1, s2, s3, s4, s5, s6 = st.columns(6)
                    s1.metric("success", int(status_counts.get("success", 0)))
                    s2.metric("failed", int(status_counts.get("failed", 0)))
                    s3.metric("skipped", int(status_counts.get("skipped", 0)))
                    s4.metric("excluded", int(status_counts.get("excluded", 0)))
                    s5.metric("model generated", int(model_ok_count))
                    s6.metric(
                        "phase内訳",
                        ", ".join([f"{k}:{int(v)}" for k, v in sorted(phase_counts.items())])
                        if phase_counts
                        else "train:0",
                    )
                    if int(status_counts.get("failed", 0)) == 0:
                        st.success("一括実行判定: 成功")
                    else:
                        st.error("一括実行判定: 失敗あり（failed > 0）")
                    _show_df(bdf, hide_index=True)

        st.markdown("---")
        st.markdown("**選択したパラメータの組み合わせ（下部表示）**")
        st.caption("現在の選択状態から自動で組み合わせ表を再計算します。")
        auto_combo_enabled = st.toggle(
            "自動で探索表を生成",
            value=True,
            key="nf_lab_bottom_combo_auto",
        )
        auto_combo_build = st.button(
            "無効組合せを自動除外して有効候補のみ作成",
            key="nf_lab_bottom_combo_build_valid_only",
        )
        auto_combo_use_allowed = st.toggle(
            "詳細モードでは allowed/type から候補を自動展開",
            value=True,
            key="nf_lab_bottom_combo_use_allowed",
            disabled=not auto_combo_enabled,
        )
        auto_combo_max_axis_vals = int(
            st.number_input(
                "1パラメータあたり自動候補の最大数",
                min_value=1,
                max_value=30,
                value=6,
                step=1,
                key="nf_lab_bottom_combo_max_axis_vals",
                disabled=not auto_combo_enabled,
            )
        )
        combo_bottom_limit = int(
            st.number_input(
                "下部表示の上限行数",
                min_value=10,
                max_value=50000,
                value=1000,
                step=10,
                key="nf_lab_bottom_combo_limit",
            )
        )
        combo_axes_bottom: dict[str, list[Any]] = {}

        def _cell(v: Any) -> Any:
            if isinstance(v, (dict, list, tuple, set)):
                return _stable_json_dumps(v)
            return v

        if auto_combo_enabled or auto_combo_build:
            if isinstance(axis_plan, dict) and axis_plan:
                for k, spec in axis_plan.items():
                    vals = spec.get("values", []) if isinstance(spec, dict) else []
                    vals = list(vals) if isinstance(vals, list) else [vals]
                    vals = [_cell(x) for x in vals if not (isinstance(x, str) and x.strip() == "")]
                    if vals:
                        combo_axes_bottom[str(k)] = vals

            if str(train_ui_mode) == "詳細" and selected_params:
                for raw_param in selected_params:
                    param_name = _parameter_name(raw_param)
                    if param_name in combo_axes_bottom:
                        continue
                    expand_key = f"nf_lab_combo_expand_{_slug(tr_model)}_{_slug(param_name)}"
                    cand_key = f"nf_lab_combo_candidates_{_slug(tr_model)}_{_slug(param_name)}"
                    if bool(st.session_state.get(expand_key, False)):
                        try:
                            parsed = json.loads(str(st.session_state.get(cand_key, "[]")).strip() or "[]")
                            if isinstance(parsed, list) and parsed:
                                combo_axes_bottom[param_name] = [_cell(x) for x in parsed[:auto_combo_max_axis_vals]]
                                continue
                        except Exception:
                            pass

                    cur_v = tr_param_builder_obj.get(param_name, default_param_values.get(param_name))
                    if auto_combo_use_allowed:
                        spec = dict(reserved_param_specs.get(param_name, {}))
                        allowed_vals = spec.get("allowed") if isinstance(spec.get("allowed"), list) else []
                        type_hint = str(spec.get("type", "") or "").strip().lower()
                        if allowed_vals:
                            combo_axes_bottom[param_name] = [_cell(x) for x in allowed_vals[:auto_combo_max_axis_vals]]
                            continue
                        if type_hint == "bool" or isinstance(cur_v, bool):
                            combo_axes_bottom[param_name] = [False, True]
                            continue
                    combo_axes_bottom[param_name] = [_cell(cur_v)]

            if not combo_axes_bottom:
                combo_axes_bottom = {
                    "model": [str(tr_model)],
                    "backend": [str(tr_backend)],
                    "num_samples": [int(tr_num_samples)],
                    "loss": [str(tr_loss)],
                    "valid_loss": [str(tr_valid_loss)],
                    "search_alg": [str(tr_search)],
                }

            def _normalize_bottom_combo_row(row: dict[str, Any], context: ComboContext) -> dict[str, Any]:
                normalized = dict(row)
                normalized["model"] = str(normalized.get("model") or tr_model)
                normalized["backend"] = _decode_optional_train_core_choice("backend", normalized.get("backend"))
                normalized["loss"] = _decode_optional_train_core_choice("loss", normalized.get("loss"))
                valid_loss_bottom_raw = _decode_optional_train_core_choice("valid_loss", normalized.get("valid_loss"))
                normalized["search_alg"] = _decode_optional_train_core_choice("search_alg", normalized.get("search_alg"))
                normalized["valid_loss"] = None if valid_loss_bottom_raw is None else str(normalized.get("loss") or tr_loss)
                group_mode_v = str(normalized.get("group_mode") or group_mode)
                normalized["group_mode"] = group_mode_v
                uid_values_v = _csv_nonempty_list(normalized.get("unique_id"))
                if group_mode_v == "loto_ts_type":
                    normalized["unique_id"] = None
                elif uid_values_v:
                    normalized["unique_id"] = ",".join(uid_values_v)
                schema_v = str(normalized.get("dataset_schema") or tr_dataset_schema)
                table_v = str(normalized.get("dataset_table") or context.dataset_table or tr_dataset_table)
                dataset_input_method_v = str(normalized.get("dataset_input_method") or tr_dataset_input_method)
                normalized["dataset_input_method"] = dataset_input_method_v
                normalized["dataframe_backend"] = str(normalized.get("dataframe_backend") or tr_dataframe_backend)
                normalized["dataset_table"] = table_v
                if context.dataset_path:
                    normalized["dataset_path"] = context.dataset_path
                if context.dataset_sql:
                    normalized["dataset_sql"] = context.dataset_sql
                detected_exog_v = (
                    _prefixed_exog_cols_for_table(schema_v, table_v)
                    if dataset_input_method_v == "db_table"
                    else {"futr_exog": [], "hist_exog": [], "stat_exog": []}
                )
                futr_exog_v = _decode_exog_axis_value(normalized.get("futr_exog")) or list(
                    detected_exog_v.get("futr_exog", [])
                )
                hist_exog_v = _decode_exog_axis_value(normalized.get("hist_exog")) or list(
                    detected_exog_v.get("hist_exog", [])
                )
                stat_exog_v = _decode_exog_axis_value(normalized.get("stat_exog")) or list(
                    detected_exog_v.get("stat_exog", [])
                )
                support_v = (
                    dict(get_model_exog_support(str(normalized.get("model", tr_model))))
                    if get_model_exog_support is not None
                    else {"futr": False, "hist": False, "stat": False}
                )
                preview_exog_adjustments: list[str] = []
                if futr_exog_v and not bool(support_v.get("futr", False)):
                    preview_exog_adjustments.append(f"drop futr_exog({len(futr_exog_v)})")
                    futr_exog_v = []
                if hist_exog_v and not bool(support_v.get("hist", False)):
                    preview_exog_adjustments.append(f"drop hist_exog({len(hist_exog_v)})")
                    hist_exog_v = []
                if stat_exog_v and not bool(support_v.get("stat", False)):
                    preview_exog_adjustments.append(f"drop stat_exog({len(stat_exog_v)})")
                    stat_exog_v = []
                normalized["futr_exog"] = futr_exog_v
                normalized["hist_exog"] = hist_exog_v
                normalized["stat_exog"] = stat_exog_v
                normalized["supports_futr_exog(F)"] = bool(support_v.get("futr", False))
                normalized["supports_hist_exog(H)"] = bool(support_v.get("hist", False))
                normalized["supports_stat_exog(S)"] = bool(support_v.get("stat", False))
                if preview_exog_adjustments:
                    normalized["exog_adjustment"] = "; ".join(preview_exog_adjustments)
                return normalized

            bottom_eval = evaluate_train_combinations(
                combo_axes_bottom,
                context=ComboContext(
                    dataset_table=str(tr_dataset_table),
                    dataset_path=str(tr_dataset_path or "").strip(),
                    dataset_sql=str(tr_dataset_sql or "").strip(),
                    default_values={
                        "model": str(tr_model),
                        "backend": str(tr_backend),
                        "valid_loss": str(tr_valid_loss),
                        "search_alg": str(tr_search),
                    },
                ),
                row_normalizer=_normalize_bottom_combo_row,
                runtime_validator=_validate_model_runtime_prerequisites,
            )
            rows_bottom = list(bottom_eval.valid_combinations[:combo_bottom_limit])
            st.session_state["nf_lab_bottom_combo_total"] = int(len(bottom_eval.valid_combinations))
            st.session_state["nf_lab_bottom_combo_total_theoretical"] = int(bottom_eval.theoretical_count)
            st.session_state["nf_lab_bottom_combo_skipped"] = int(len(bottom_eval.excluded_combinations))
            st.session_state["nf_lab_bottom_combo_skip_reasons"] = dict(bottom_eval.reason_summary)
            st.session_state["nf_lab_bottom_combo_reason_rows"] = _build_combo_reason_rows(bottom_eval)
            st.session_state["nf_lab_bottom_combo_fix_suggestions"] = list(bottom_eval.fix_suggestions)
            st.session_state["nf_lab_bottom_combo_df"] = pd.DataFrame(rows_bottom)
        elif st.button("選択したパラメータの組み合わせを表示", key="nf_lab_bottom_combo_show_manual"):
            # Manual mode fallback.
            st.session_state["nf_lab_bottom_combo_total"] = 1
            st.session_state["nf_lab_bottom_combo_total_theoretical"] = 1
            st.session_state["nf_lab_bottom_combo_skipped"] = 0
            st.session_state["nf_lab_bottom_combo_skip_reasons"] = {}
            st.session_state["nf_lab_bottom_combo_reason_rows"] = []
            st.session_state["nf_lab_bottom_combo_fix_suggestions"] = []
            st.session_state["nf_lab_bottom_combo_df"] = pd.DataFrame(
                [
                    {
                        "model": str(tr_model),
                        "backend": str(tr_backend),
                        "num_samples": int(tr_num_samples),
                        "loss": str(tr_loss),
                        "valid_loss": str(tr_valid_loss),
                        "search_alg": str(tr_search),
                        "dataset_input_method": str(tr_dataset_input_method),
                        "dataframe_backend": str(tr_dataframe_backend),
                        "futr_exog": (
                            list(
                                _prefixed_exog_cols_for_table(str(tr_dataset_schema), str(tr_dataset_table)).get(
                                    "futr_exog", []
                                )
                            )
                            if str(tr_dataset_input_method) == "db_table"
                            else []
                        ),
                        "hist_exog": (
                            list(
                                _prefixed_exog_cols_for_table(str(tr_dataset_schema), str(tr_dataset_table)).get(
                                    "hist_exog", []
                                )
                            )
                            if str(tr_dataset_input_method) == "db_table"
                            else []
                        ),
                        "stat_exog": (
                            list(
                                _prefixed_exog_cols_for_table(str(tr_dataset_schema), str(tr_dataset_table)).get(
                                    "stat_exog", []
                                )
                            )
                            if str(tr_dataset_input_method) == "db_table"
                            else []
                        ),
                        "supports_futr_exog(F)": bool(
                            dict(get_model_exog_support(str(tr_model))).get("futr", False)
                            if get_model_exog_support is not None
                            else False
                        ),
                        "supports_hist_exog(H)": bool(
                            dict(get_model_exog_support(str(tr_model))).get("hist", False)
                            if get_model_exog_support is not None
                            else False
                        ),
                        "supports_stat_exog(S)": bool(
                            dict(get_model_exog_support(str(tr_model))).get("stat", False)
                            if get_model_exog_support is not None
                            else False
                        ),
                    }
                ]
            )

        bottom_df = st.session_state.get("nf_lab_bottom_combo_df")
        bottom_total = int(st.session_state.get("nf_lab_bottom_combo_total", 0) or 0)
        bottom_total_theoretical = int(st.session_state.get("nf_lab_bottom_combo_total_theoretical", bottom_total) or 0)
        bottom_skipped = int(st.session_state.get("nf_lab_bottom_combo_skipped", 0) or 0)
        bottom_skip_reasons = st.session_state.get("nf_lab_bottom_combo_skip_reasons", {})
        bottom_reason_rows = st.session_state.get("nf_lab_bottom_combo_reason_rows", [])
        bottom_fix_suggestions = st.session_state.get("nf_lab_bottom_combo_fix_suggestions", [])
        if isinstance(bottom_df, pd.DataFrame) and not bottom_df.empty:
            c_b1, c_b2, c_b3, c_b4 = st.columns(4)
            c_b1.metric("理論組合せ数", int(bottom_total_theoretical))
            c_b2.metric("自動除外後の有効件数", int(bottom_total))
            c_b3.metric("除外件数", int(bottom_skipped))
            c_b4.metric("表示行数", int(bottom_df.shape[0]))
            if isinstance(bottom_skip_reasons, dict) and bottom_skip_reasons:
                with st.expander("除外理由（件数）", expanded=False):
                    _show_df(pd.DataFrame(bottom_reason_rows).sort_values("count", ascending=False), hide_index=True)
            if isinstance(bottom_fix_suggestions, list) and bottom_fix_suggestions:
                st.info("推奨修正: " + " / ".join([str(item) for item in bottom_fix_suggestions]))
            _show_df(bottom_df, hide_index=True)
            st.session_state.pop("nf_lab_bottom_combo_download", None)
            st.download_button(
                "Download bottom_selected_param_combinations.csv",
                data=bottom_df.to_csv(index=False),
                file_name="bottom_selected_param_combinations.csv",
                mime="text/csv",
                key="nf_lab_bottom_combo_download",
            )
        elif "nf_lab_bottom_combo_df" in st.session_state:
            if isinstance(bottom_reason_rows, list) and bottom_reason_rows:
                class _BottomEvalProxy:
                    reason_rows = [type("Row", (), row)() for row in bottom_reason_rows]
                    fix_suggestions = bottom_fix_suggestions

                _render_zero_combo_diagnostics(_BottomEvalProxy(), label="下部の組合せ表示")
            else:
                st.info(
                    f"組合せ表は空です。理論={bottom_total_theoretical}, 実行可能={bottom_total}, 除外={bottom_skipped}。"
                    " 候補設定や backend/search_alg の整合性を見直してください。"
                )
                if isinstance(bottom_fix_suggestions, list) and bottom_fix_suggestions:
                    st.info("有効件数を増やすには: " + " / ".join([str(item) for item in bottom_fix_suggestions]))

    if lab_section == "再学習(retrain)":
        st.caption("既存runを基準に retrain を実行します。")
        rr1, rr2, rr3 = st.columns([2, 1, 1])
        rt_base = rr1.selectbox("base-run-id", [""] + run_id_options, index=0, key="nf_lab_retrain_base")
        rt_h = int(rr2.number_input("h (0=omit)", min_value=0, max_value=3650, value=0, step=1, key="nf_lab_retrain_h"))
        rt_params_raw = rr3.text_input("params-json", value="{}", key="nf_lab_retrain_params")

        rt_seed_ready = False
        if rt_base:
            local_meta = (PROJECT_ROOT / "artifacts" / str(rt_base) / "meta.json").resolve()
            if local_meta.exists():
                st.success(f"retrain seed: local meta available ({local_meta})")
                rt_seed_ready = True
            else:
                db_ready = False
                if engine is not None:
                    try:
                        m_cfg = re.match(r"^cfg(?P<cfg>\d+)_", str(rt_base))
                        cfg_id_hint = int(m_cfg.group("cfg")) if m_cfg else None
                        chk_df = _query_df(
                            engine,
                            """
                            SELECT
                              EXISTS(SELECT 1 FROM model.nf_automodel WHERE run_id = :run_id) AS in_model,
                              EXISTS(SELECT 1 FROM meta.model_run WHERE run_id = :run_id) AS in_run,
                              EXISTS(
                                SELECT 1
                                FROM meta.nf_automodel
                                WHERE (:cfg_id_hint IS NOT NULL AND config_id = :cfg_id_hint)
                              ) AS in_meta_cfg
                            """,
                            {"run_id": str(rt_base), "cfg_id_hint": cfg_id_hint},
                        )
                        if not chk_df.empty:
                            in_model = bool(chk_df.iloc[0].get("in_model", False))
                            in_run = bool(chk_df.iloc[0].get("in_run", False))
                            in_meta_cfg = bool(chk_df.iloc[0].get("in_meta_cfg", False))
                            db_ready = bool(in_model or in_run or in_meta_cfg)
                    except Exception:
                        db_ready = False
                if db_ready:
                    st.info(
                        "retrain seed: local metaなし。DBフォールバックで実行します（model.nf_automodel / meta.model_run）。"
                    )
                    rt_seed_ready = True
                else:
                    st.error("retrain seedが見つかりません。local artifacts と DB の両方で未検出です。")

        rt_errors: list[str] = []
        rt_params = _json_text(rt_params_raw, "dict", "params-json", rt_errors)
        rt_parts = ["python", "-m", "loto_forecast.cli", "retrain", "--base-run-id", str(rt_base or "")]
        if rt_h > 0:
            rt_parts.extend(["--h", str(int(rt_h))])
        if rt_params is not None:
            rt_parts.extend(["--params-json", rt_params])
        cmd_retrain = " ".join(shlex.quote(x) for x in rt_parts)
        _render_command_preview(
            cmd_retrain,
            copy_key="nf_lab_copy_retrain_cmd",
            copy_label="Copy retrain command",
            cwd=lab_cwd,
            show_arg_table=True,
        )
        if not rt_base:
            st.warning("base-run-id を選択してください。")
        if rt_errors:
            st.error("\n".join(rt_errors))
        if st.button(
            "Run retrain", key="nf_lab_run_retrain", disabled=(not rt_base or (not rt_seed_ready) or bool(rt_errors))
        ):
            if not lab_cwd.exists() or not lab_cwd.is_dir():
                st.error("lab cwd が有効なディレクトリではありません。")
            else:
                st.session_state["nf_lab_retrain_result"] = _run_shell_command_live(
                    cmd_retrain,
                    cwd=lab_cwd,
                    timeout_sec=lab_timeout_sec,
                    title="NF retrain",
                )
        _render_command_result_block("nf_lab_retrain_result", "retrain 実行結果", "nf_lab_retrain")

    if lab_section == "予測/評価":
        st.caption("predict / evaluate / explain を独立タブで実行します。")
        st.info("推奨順: 1) predict 2) evaluate 3) explain。各タブに現在設定サマリを表示します。")
        pe_sections = ["予測(predict)", "評価(evaluate)", "説明(explain)"]
        pe_section = st.selectbox(
            "予測/評価 サブメニュー",
            pe_sections,
            index=0,
            key="nf_lab_predict_eval_sub_select",
        )
        predict_defaults = {
            "df": "auto",
            "static_df": None,
            "futr_df": "auto",
            "verbose": False,
            "engine": None,
            "level": None,
            "quantiles": None,
            "h": None,
            "data_kwargs": {},
        }

        if pe_section == "予測(predict)":
            _render_tab_playbook(
                purpose="保存済みrunのモデルを使って予測を生成します。",
                required_inputs=["run-id", "predict h(必要時のみ)"],
                outputs=["forecast.parquet", "stdout/json tail"],
                steps=["run-id選択", "h確認", "Run predict"],
            )
            pe1, pe2 = st.columns([2, 1])
            pe_run = pe1.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_predict_run")
            pe_h = int(
                pe2.number_input(
                    "predict h (0=omit)", min_value=0, max_value=3650, value=0, step=1, key="nf_lab_predict_h"
                )
            )
            st.caption("全件データを読み込んで実行し、未来の未知予測値を出力できます。")
            psrc1, psrc2 = st.columns(2)
            pe_dataset_input_method = psrc1.selectbox(
                "predict dataset input method",
                DATASET_INPUT_METHOD_OPTIONS,
                index=0,
                key="nf_lab_predict_dataset_input_method",
            )
            pe_backend_candidates = _supported_backends_for_input_method(str(pe_dataset_input_method))
            pe_backend_default = str(
                st.session_state.get("nf_lab_predict_dataframe_backend", DATAFRAME_BACKEND_OPTIONS[0])
            )
            if pe_backend_default not in pe_backend_candidates:
                pe_backend_default = (
                    str(pe_backend_candidates[0]) if pe_backend_candidates else DATAFRAME_BACKEND_OPTIONS[0]
                )
            pe_dataframe_backend = psrc2.selectbox(
                "predict dataframe backend",
                pe_backend_candidates if pe_backend_candidates else DATAFRAME_BACKEND_OPTIONS,
                index=(
                    (pe_backend_candidates.index(pe_backend_default))
                    if (pe_backend_candidates and pe_backend_default in pe_backend_candidates)
                    else 0
                ),
                key="nf_lab_predict_dataframe_backend",
            )
            pe_dataset_schema_options = (
                sorted(list({s for s, _ in tables if s in {"dataset", "exog"}})) if tables else [settings.db_schema]
            )
            if not pe_dataset_schema_options:
                pe_dataset_schema_options = [settings.db_schema]
            pe_dataset_schema = st.selectbox(
                "predict dataset schema",
                pe_dataset_schema_options,
                index=(
                    pe_dataset_schema_options.index(settings.db_schema)
                    if settings.db_schema in pe_dataset_schema_options
                    else 0
                ),
                key="nf_lab_predict_dataset_schema",
            )
            pe_dataset_table_options = sorted([t for s, t in tables if s == pe_dataset_schema]) if tables else []
            pe_default_table = (
                settings.db_table
                if pe_dataset_schema == settings.db_schema
                else (pe_dataset_table_options[0] if pe_dataset_table_options else settings.db_table)
            )
            if pe_dataset_table_options:
                pe_dataset_table = st.selectbox(
                    "predict dataset table",
                    pe_dataset_table_options,
                    index=(
                        pe_dataset_table_options.index(pe_default_table)
                        if pe_default_table in pe_dataset_table_options
                        else 0
                    ),
                    key="nf_lab_predict_dataset_table",
                )
            else:
                pe_dataset_table = st.text_input(
                    "predict dataset table", value=str(pe_default_table), key="nf_lab_predict_dataset_table_text"
                )
            pe_dataset_where = st.text_input(
                "predict dataset where SQL (optional)", value="", key="nf_lab_predict_dataset_where"
            )
            pe_dataset_path = st.text_input(
                "predict dataset path (csv/parquet/json)", value="", key="nf_lab_predict_dataset_path"
            )
            pe_dataset_sql = st.text_area(
                "predict dataset SQL (db_sql)", value="", height=80, key="nf_lab_predict_dataset_sql"
            )
            with st.expander("predict: DataFrame候補と読み込み対応表", expanded=False):
                _show_df(_dataset_loader_support_df(), hide_index=True)
            pe_meta = (
                _safe_read_json_file(run_id_to_dir[pe_run] / "meta.json") if pe_run and pe_run in run_id_to_dir else {}
            )
            pe_runtime = (
                pe_meta.get("nf_runtime_kwargs", {}).get("nf_predict_kwargs", {})
                if isinstance(pe_meta.get("nf_runtime_kwargs"), dict)
                else {}
            )
            pe_runtime_h = pe_runtime.get("h") if isinstance(pe_runtime, dict) else None
            pe_meta_h = pe_meta.get("h")
            pe_effective_h = int(pe_h) if pe_h > 0 else int(pe_runtime_h or pe_meta_h or settings.default_horizon)
            pe_h_source = (
                "CLI --h"
                if pe_h > 0
                else (
                    "meta nf_predict_kwargs.h"
                    if pe_runtime_h is not None
                    else ("meta h" if pe_meta_h is not None else "settings.default_horizon")
                )
            )
            pe_summary_df = pd.DataFrame(
                [
                    {"arg": "run_id", "value": str(pe_run or ""), "source": "UI"},
                    {"arg": "h", "value": int(pe_effective_h), "source": pe_h_source},
                    {"arg": "meta.h", "value": pe_meta_h, "source": "meta.json"},
                    {"arg": "meta.nf_predict_kwargs.h", "value": pe_runtime_h, "source": "meta.json"},
                    {"arg": "dataset_input_method", "value": str(pe_dataset_input_method), "source": "UI"},
                    {"arg": "dataframe_backend", "value": str(pe_dataframe_backend), "source": "UI"},
                    {"arg": "dataset_schema", "value": str(pe_dataset_schema), "source": "UI"},
                    {"arg": "dataset_table", "value": str(pe_dataset_table), "source": "UI"},
                ]
            )
            with st.expander("predict 設定サマリ", expanded=True):
                _show_df(pe_summary_df, hide_index=True)
                _show_df(
                    _nf_signature_rows(
                        "predict",
                        predict_defaults,
                        pe_runtime if isinstance(pe_runtime, dict) else {},
                        forced={"h": int(pe_effective_h)},
                    ),
                    hide_index=True,
                )
            cmd_predict_parts = ["python", "-m", "loto_forecast.cli", "predict", "--run-id", str(pe_run or "")]
            if pe_h > 0:
                cmd_predict_parts.extend(["--h", str(int(pe_h))])
            cmd_predict_parts.extend(["--dataset-input-method", str(pe_dataset_input_method)])
            cmd_predict_parts.extend(["--dataframe-backend", str(pe_dataframe_backend)])
            if str(pe_dataset_input_method) == "db_table":
                cmd_predict_parts.extend(
                    ["--dataset-schema", str(pe_dataset_schema), "--dataset-table", str(pe_dataset_table)]
                )
                if str(pe_dataset_where).strip():
                    cmd_predict_parts.extend(["--dataset-where", str(pe_dataset_where).strip()])
            elif str(pe_dataset_input_method) == "db_sql":
                if str(pe_dataset_sql).strip():
                    cmd_predict_parts.extend(["--dataset-sql", str(pe_dataset_sql).strip()])
            else:
                if str(pe_dataset_path).strip():
                    cmd_predict_parts.extend(["--dataset-path", str(pe_dataset_path).strip()])
            cmd_predict = " ".join(shlex.quote(x) for x in cmd_predict_parts)
            _render_command_preview(
                cmd_predict,
                copy_key="nf_lab_copy_predict_cmd",
                copy_label="Copy predict command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            if not pe_run:
                st.warning("run-id を選択してください。")
            pe_predict_invalid = (
                (str(pe_dataset_input_method) == "db_table" and not str(pe_dataset_table).strip())
                or (str(pe_dataset_input_method) == "db_sql" and not str(pe_dataset_sql).strip())
                or (str(pe_dataset_input_method) in {"csv", "parquet", "json"} and not str(pe_dataset_path).strip())
            )
            if not _is_supported_backend_for_input_method(str(pe_dataset_input_method), str(pe_dataframe_backend)):
                allowed_predict = _supported_backends_for_input_method(str(pe_dataset_input_method))
                pe_predict_invalid = True
                st.warning(
                    "predict の dataframe backend が未対応です: "
                    f"{pe_dataframe_backend} (allowed: {', '.join(allowed_predict)})"
                )
            if pe_predict_invalid:
                st.warning("predict のデータセット入力条件が不足しています。")
            if st.button("Run predict", key="nf_lab_run_predict", disabled=(not pe_run or pe_predict_invalid)):
                st.session_state["nf_lab_predict_result"] = _run_shell_command_live(
                    cmd_predict, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF predict"
                )
            _render_command_result_block("nf_lab_predict_result", "predict 実行結果", "nf_lab_predict")

        if pe_section == "評価(evaluate)":
            _render_tab_playbook(
                purpose="同一runで予測精度を再計算し、評価JSONを更新します。",
                required_inputs=["run-id"],
                outputs=["evaluation.json", "metrics/diagnostics"],
                steps=["run-id選択", "評価対象h確認", "Run evaluate"],
            )
            ev_run = st.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_evaluate_run")
            esrc1, esrc2 = st.columns(2)
            ev_dataset_input_method = esrc1.selectbox(
                "evaluate dataset input method",
                DATASET_INPUT_METHOD_OPTIONS,
                index=0,
                key="nf_lab_evaluate_dataset_input_method",
            )
            ev_backend_candidates = _supported_backends_for_input_method(str(ev_dataset_input_method))
            ev_backend_default = str(
                st.session_state.get("nf_lab_evaluate_dataframe_backend", DATAFRAME_BACKEND_OPTIONS[0])
            )
            if ev_backend_default not in ev_backend_candidates:
                ev_backend_default = (
                    str(ev_backend_candidates[0]) if ev_backend_candidates else DATAFRAME_BACKEND_OPTIONS[0]
                )
            ev_dataframe_backend = esrc2.selectbox(
                "evaluate dataframe backend",
                ev_backend_candidates if ev_backend_candidates else DATAFRAME_BACKEND_OPTIONS,
                index=(
                    (ev_backend_candidates.index(ev_backend_default))
                    if (ev_backend_candidates and ev_backend_default in ev_backend_candidates)
                    else 0
                ),
                key="nf_lab_evaluate_dataframe_backend",
            )
            ev_step_eval_size = int(
                st.number_input(
                    "予測ステップ分割サイズ (1=ステップ単位)",
                    min_value=1,
                    max_value=3650,
                    value=1,
                    step=1,
                    key="nf_lab_evaluate_step_eval_size",
                )
            )
            ev_dataset_schema_options = (
                sorted(list({s for s, _ in tables if s in {"dataset", "exog"}})) if tables else [settings.db_schema]
            )
            if not ev_dataset_schema_options:
                ev_dataset_schema_options = [settings.db_schema]
            ev_dataset_schema = st.selectbox(
                "evaluate dataset schema",
                ev_dataset_schema_options,
                index=(
                    ev_dataset_schema_options.index(settings.db_schema)
                    if settings.db_schema in ev_dataset_schema_options
                    else 0
                ),
                key="nf_lab_evaluate_dataset_schema",
            )
            ev_dataset_table_options = sorted([t for s, t in tables if s == ev_dataset_schema]) if tables else []
            ev_default_table = (
                settings.db_table
                if ev_dataset_schema == settings.db_schema
                else (ev_dataset_table_options[0] if ev_dataset_table_options else settings.db_table)
            )
            if ev_dataset_table_options:
                ev_dataset_table = st.selectbox(
                    "evaluate dataset table",
                    ev_dataset_table_options,
                    index=(
                        ev_dataset_table_options.index(ev_default_table)
                        if ev_default_table in ev_dataset_table_options
                        else 0
                    ),
                    key="nf_lab_evaluate_dataset_table",
                )
            else:
                ev_dataset_table = st.text_input(
                    "evaluate dataset table", value=str(ev_default_table), key="nf_lab_evaluate_dataset_table_text"
                )
            ev_dataset_where = st.text_input(
                "evaluate dataset where SQL (optional)", value="", key="nf_lab_evaluate_dataset_where"
            )
            ev_dataset_path = st.text_input(
                "evaluate dataset path (csv/parquet/json)", value="", key="nf_lab_evaluate_dataset_path"
            )
            ev_dataset_sql = st.text_area(
                "evaluate dataset SQL (db_sql)", value="", height=80, key="nf_lab_evaluate_dataset_sql"
            )
            with st.expander("evaluate: DataFrame候補と読み込み対応表", expanded=False):
                _show_df(_dataset_loader_support_df(), hide_index=True)
            ev_meta = (
                _safe_read_json_file(run_id_to_dir[ev_run] / "meta.json") if ev_run and ev_run in run_id_to_dir else {}
            )
            ev_summary_df = pd.DataFrame(
                [
                    {"arg": "run_id", "value": str(ev_run or ""), "source": "UI"},
                    {"arg": "h(評価で使用)", "value": ev_meta.get("h"), "source": "meta.json"},
                    {"arg": "step_eval_size", "value": int(ev_step_eval_size), "source": "UI"},
                    {"arg": "dataset_input_method", "value": str(ev_dataset_input_method), "source": "UI"},
                    {"arg": "dataframe_backend", "value": str(ev_dataframe_backend), "source": "UI"},
                    {
                        "arg": "nf_predict_kwargs.h(評価時は上書きされる)",
                        "value": (
                            ev_meta.get("nf_runtime_kwargs", {}).get("nf_predict_kwargs", {}).get("h")
                            if isinstance(ev_meta.get("nf_runtime_kwargs"), dict)
                            else None
                        ),
                        "source": "meta.json",
                    },
                ]
            )
            with st.expander("evaluate 設定サマリ", expanded=True):
                _show_df(ev_summary_df, hide_index=True)
            cmd_evaluate_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "evaluate",
                "--run-id",
                str(ev_run or ""),
                "--step-eval-size",
                str(int(ev_step_eval_size)),
                "--dataset-input-method",
                str(ev_dataset_input_method),
                "--dataframe-backend",
                str(ev_dataframe_backend),
            ]
            if str(ev_dataset_input_method) == "db_table":
                cmd_evaluate_parts.extend(
                    ["--dataset-schema", str(ev_dataset_schema), "--dataset-table", str(ev_dataset_table)]
                )
                if str(ev_dataset_where).strip():
                    cmd_evaluate_parts.extend(["--dataset-where", str(ev_dataset_where).strip()])
            elif str(ev_dataset_input_method) == "db_sql":
                if str(ev_dataset_sql).strip():
                    cmd_evaluate_parts.extend(["--dataset-sql", str(ev_dataset_sql).strip()])
            else:
                if str(ev_dataset_path).strip():
                    cmd_evaluate_parts.extend(["--dataset-path", str(ev_dataset_path).strip()])
            cmd_evaluate = " ".join(shlex.quote(x) for x in cmd_evaluate_parts)
            _render_command_preview(
                cmd_evaluate,
                copy_key="nf_lab_copy_evaluate_cmd",
                copy_label="Copy evaluate command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            if not ev_run:
                st.warning("run-id を選択してください。")
            ev_evaluate_invalid = (
                (str(ev_dataset_input_method) == "db_table" and not str(ev_dataset_table).strip())
                or (str(ev_dataset_input_method) == "db_sql" and not str(ev_dataset_sql).strip())
                or (str(ev_dataset_input_method) in {"csv", "parquet", "json"} and not str(ev_dataset_path).strip())
            )
            if not _is_supported_backend_for_input_method(str(ev_dataset_input_method), str(ev_dataframe_backend)):
                allowed_eval = _supported_backends_for_input_method(str(ev_dataset_input_method))
                ev_evaluate_invalid = True
                st.warning(
                    "evaluate の dataframe backend が未対応です: "
                    f"{ev_dataframe_backend} (allowed: {', '.join(allowed_eval)})"
                )
            if ev_evaluate_invalid:
                st.warning("evaluate のデータセット入力条件が不足しています。")
            if st.button("Run evaluate", key="nf_lab_run_evaluate", disabled=(not ev_run or ev_evaluate_invalid)):
                st.session_state["nf_lab_evaluate_result"] = _run_shell_command_live(
                    cmd_evaluate, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF evaluate"
                )
            _render_command_result_block("nf_lab_evaluate_result", "evaluate 実行結果", "nf_lab_evaluate")
            if "nf_lab_evaluate_result" in st.session_state:
                ev_last = dict(st.session_state.get("nf_lab_evaluate_result", {}) or {})
                ev_obj = _try_parse_json_tail(str(ev_last.get("stdout", "")))
                if isinstance(ev_obj, dict):
                    eval_step_rows: Any = ev_obj.get("step_metrics")
                    if isinstance(eval_step_rows, list) and eval_step_rows:
                        st.markdown("**予測ステップ分割評価 (step_metrics)**")
                        _show_df(pd.DataFrame(eval_step_rows).head(500), hide_index=True)

        if pe_section == "説明(explain)":
            _render_tab_playbook(
                purpose="寄与特徴量や因果候補の説明情報を算出します。",
                required_inputs=["run-id", "method", "maxlag/top-k"],
                outputs=["explain stdout", "feature importance / screening"],
                steps=["run-id選択", "method設定", "Run explain"],
            )
            ex1, ex2, ex3 = st.columns([2, 1, 1])
            ex_run = ex1.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_explain_run")
            pe_method = ex2.selectbox(
                "explain method", ["permutation", "neuralforecast", "granger"], index=0, key="nf_lab_explain_method"
            )
            pe_topk = ex3.number_input(
                "explain top-k", min_value=1, max_value=200, value=20, step=1, key="nf_lab_explain_topk"
            )
            pe_maxlag = st.number_input(
                "granger maxlag", min_value=1, max_value=64, value=8, step=1, key="nf_lab_explain_maxlag"
            )
            ex_summary_df = pd.DataFrame(
                [
                    {"arg": "run_id", "value": str(ex_run or ""), "source": "UI"},
                    {"arg": "method", "value": str(pe_method), "source": "UI"},
                    {"arg": "maxlag", "value": int(pe_maxlag), "source": "UI"},
                    {"arg": "top_k", "value": int(pe_topk), "source": "UI"},
                ]
            )
            with st.expander("explain 設定サマリ", expanded=True):
                _show_df(ex_summary_df, hide_index=True)
            cmd_explain = " ".join(
                shlex.quote(x)
                for x in [
                    "python",
                    "-m",
                    "loto_forecast.cli",
                    "explain",
                    "--run-id",
                    str(ex_run or ""),
                    "--method",
                    str(pe_method),
                    "--maxlag",
                    str(int(pe_maxlag)),
                    "--top-k",
                    str(int(pe_topk)),
                ]
            )
            _render_command_preview(
                cmd_explain,
                copy_key="nf_lab_copy_explain_cmd",
                copy_label="Copy explain command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            if not ex_run:
                st.warning("run-id を選択してください。")
            if st.button("Run explain", key="nf_lab_run_explain", disabled=not ex_run):
                st.session_state["nf_lab_explain_result"] = _run_shell_command_live(
                    cmd_explain, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF explain"
                )
            _render_command_result_block("nf_lab_explain_result", "explain 実行結果", "nf_lab_explain")

    if lab_section == "CV/Insample":
        st.caption("cross_validation は train経由で実行、predict_insample は load-check経由で検証します。")
        st.info("推奨順: 1) cross_validation で汎化確認 2) predict_insample で保存モデル再現性確認")
        cv_sections = ["cross_validation", "predict_insample"]
        cv_section = st.selectbox(
            "CV/Insample サブメニュー",
            cv_sections,
            index=0,
            key="nf_lab_cv_ins_sub_select",
        )
        if cv_section == "cross_validation":
            _render_tab_playbook(
                purpose="学習時系列の窓分割検証を実施し、汎化性能を確認します。",
                required_inputs=["model", "h", "nf-cross-validation-kwargs-json"],
                outputs=["cross_validation.parquet", "stdout/json tail"],
                steps=["model/h設定", "kwargs確認", "Run cross_validation"],
            )
            cv1, cv2 = st.columns(2)
            cv_model = cv1.text_input("cv model", value="AutoNHITS", key="nf_lab_cv_model")
            cv_h = int(
                cv2.number_input(
                    "cv h", min_value=1, max_value=3650, value=settings.default_horizon, step=1, key="nf_lab_cv_h"
                )
            )
            cv_kwargs_raw = st.text_area(
                "nf-cross-validation-kwargs-json",
                value='{"n_windows": 3, "step_size": 1, "refit": false}',
                height=100,
                key="nf_lab_cv_kwargs",
            )
            cv_errors: list[str] = []
            cv_kwargs = _json_text(cv_kwargs_raw, "dict", "nf-cross-validation-kwargs-json", cv_errors)
            cv_kwargs_obj = json.loads(cv_kwargs) if cv_kwargs is not None else {}
            cv_defaults = {
                "df": "auto",
                "static_df": None,
                "n_windows": 1,
                "step_size": 1,
                "val_size": 0,
                "test_size": None,
                "use_init_models": False,
                "verbose": False,
                "refit": False,
                "id_col": "unique_id",
                "time_col": "ds",
                "target_col": "y",
                "prediction_intervals": None,
                "level": None,
                "quantiles": None,
                "h": None,
                "data_kwargs": {},
            }
            with st.expander("cross_validation 設定サマリ", expanded=True):
                _show_df(
                    _nf_signature_rows("cross_validation", cv_defaults, cv_kwargs_obj, forced={"h": int(cv_h)}),
                    hide_index=True,
                )
            cv_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "train",
                "--model",
                cv_model.strip() or "AutoNHITS",
                "--h",
                str(int(cv_h)),
                "--run-cross-validation",
            ]
            if cv_kwargs is not None:
                cv_parts.extend(["--nf-cross-validation-kwargs-json", cv_kwargs])
            cmd_cv = " ".join(shlex.quote(x) for x in cv_parts)
            _render_command_preview(
                cmd_cv,
                copy_key="nf_lab_copy_cv_cmd",
                copy_label="Copy cross_validation command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            if cv_errors:
                st.error("\n".join(cv_errors))
            if st.button("Run cross_validation", key="nf_lab_run_cv", disabled=bool(cv_errors)):
                st.session_state["nf_lab_cv_result"] = _run_shell_command_live(
                    cmd_cv, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF cross_validation"
                )
            _render_command_result_block("nf_lab_cv_result", "cross_validation 実行結果", "nf_lab_cv")

        if cv_section == "predict_insample":
            _render_tab_playbook(
                purpose="保存済みモデルの読み込み後に insample 予測を実行して再現性を確認します。",
                required_inputs=["run-id", "source-path", "save-path", "predict-insample-kwargs-json"],
                outputs=["load疎通結果", "predict_insample結果"],
                steps=["run-id/path確認", "kwargs/step設定", "Run predict_insample check"],
            )
            ins_run = st.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_ins_run")
            default_source = str(run_id_to_dir.get(ins_run, PROJECT_ROOT / "artifacts" / (ins_run or "")))
            ins_source = st.text_input("source-path", value=default_source, key="nf_lab_ins_source")
            ins_save = st.text_input(
                "save-path",
                value=str(PROJECT_ROOT / "artifacts" / "saved_models" / (ins_run or "")),
                key="nf_lab_ins_save",
            )
            ins_kwargs_raw = st.text_area(
                "predict-insample-kwargs-json", value='{"step_size": 1}', height=80, key="nf_lab_ins_kwargs"
            )
            ins_step = st.number_input(
                "insample-step-size", min_value=1, max_value=365, value=1, step=1, key="nf_lab_ins_step"
            )
            ins_errors: list[str] = []
            ins_kwargs = _json_text(ins_kwargs_raw, "dict", "predict-insample-kwargs-json", ins_errors)
            ins_kwargs_obj = json.loads(ins_kwargs) if ins_kwargs is not None else {}
            with st.expander("predict_insample 設定サマリ", expanded=True):
                _show_df(
                    _nf_signature_rows(
                        "predict_insample",
                        {"step_size": 1, "level": None, "quantiles": None},
                        ins_kwargs_obj,
                        forced={"step_size": int(ins_step)},
                    ),
                    hide_index=True,
                )
            ins_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "model-save-load-analyze",
                "--run-id",
                str(ins_run or ""),
                "--source-path",
                str(ins_source),
                "--save-path",
                str(ins_save),
                "--no-run-save",
                "--run-load",
                "--no-run-analyze",
                "--load-check-predict",
                "--insample-step-size",
                str(int(ins_step)),
            ]
            if ins_kwargs is not None:
                ins_parts.extend(["--predict-insample-kwargs-json", ins_kwargs])
            cmd_ins = " ".join(shlex.quote(x) for x in ins_parts)
            _render_command_preview(
                cmd_ins,
                copy_key="nf_lab_copy_ins_cmd",
                copy_label="Copy predict_insample command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            ins_preflight_df, ins_hints = _model_ops_preflight_checks(
                run_id=str(ins_run),
                source_path=str(ins_source),
                save_path=str(ins_save),
                run_save=False,
                run_load=True,
            )
            with st.expander("predict_insample 実行前チェック", expanded=False):
                _show_df(ins_preflight_df, hide_index=True)
                if ins_hints:
                    st.warning("\n".join([f"- {h}" for h in ins_hints]))
            if not ins_run:
                st.warning("run-id を選択してください。")
            if ins_errors:
                st.error("\n".join(ins_errors))
            if st.button(
                "Run predict_insample check", key="nf_lab_run_insample", disabled=(not ins_run or bool(ins_errors))
            ):
                st.session_state["nf_lab_ins_result"] = _run_shell_command_live(
                    cmd_ins, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF predict_insample check"
                )
            _render_command_result_block("nf_lab_ins_result", "predict_insample 実行結果", "nf_lab_ins")

    if lab_section == "保存/ロード":
        st.caption("保存(save) / ロード(load) / 保存+ロード+分析 を独立実行できます。")
        st.info("推奨順: 1) 保存(save) 2) ロード(load) 3) 保存+ロード+分析(一括検証)")
        sl_sections = ["保存(save)", "ロード(load)", "保存+ロード+分析"]
        sl_section = st.selectbox(
            "保存/ロード サブメニュー",
            sl_sections,
            index=0,
            key="nf_lab_save_load_sub_select",
        )

        if sl_section == "保存(save)":
            _render_tab_playbook(
                purpose="学習成果物を保存ディレクトリへエクスポートします。",
                required_inputs=["run-id", "source-path", "save-path", "save-kwargs-json"],
                outputs=["saved_models配下の保存物", "save実行ログ"],
                steps=["run-id/path確認", "save設定確認", "Run save"],
            )
            sv_run = st.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_save_run")
            sv_source = st.text_input(
                "source-path",
                value=str(run_id_to_dir.get(sv_run, PROJECT_ROOT / "artifacts" / (sv_run or ""))),
                key="nf_lab_save_source",
            )
            sv_save = st.text_input(
                "save-path",
                value=str(PROJECT_ROOT / "artifacts" / "saved_models" / (sv_run or "")),
                key="nf_lab_save_path",
            )
            sv1, sv2 = st.columns(2)
            sv_dataset = sv1.toggle("save_dataset", value=False, key="nf_lab_save_dataset")
            sv_overwrite = sv2.toggle("save_overwrite", value=True, key="nf_lab_save_overwrite")
            sv_kwargs_raw = st.text_area("save-kwargs-json", value="{}", height=80, key="nf_lab_save_kwargs")
            sv_errors: list[str] = []
            sv_kwargs = _json_text(sv_kwargs_raw, "dict", "save-kwargs-json", sv_errors)
            sv_kwargs_obj = json.loads(sv_kwargs) if sv_kwargs is not None else {}
            with st.expander("save 設定サマリ", expanded=True):
                _show_df(
                    _nf_signature_rows(
                        "save",
                        {"path": "<save-path>", "model_index": None, "save_dataset": True, "overwrite": False},
                        sv_kwargs_obj,
                        forced={
                            "path": str(sv_save),
                            "save_dataset": bool(sv_dataset),
                            "overwrite": bool(sv_overwrite),
                        },
                    ),
                    hide_index=True,
                )
            sv_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "model-save-load-analyze",
                "--run-id",
                str(sv_run or ""),
                "--source-path",
                str(sv_source),
                "--save-path",
                str(sv_save),
                "--run-save",
                "--no-run-load",
                "--no-run-analyze",
                _bool_opt("save-dataset", sv_dataset),
                _bool_opt("save-overwrite", sv_overwrite),
                "--no-load-check-predict",
            ]
            if sv_kwargs is not None:
                sv_parts.extend(["--save-kwargs-json", sv_kwargs])
            cmd_save = " ".join(shlex.quote(x) for x in sv_parts)
            _render_command_preview(
                cmd_save,
                copy_key="nf_lab_copy_save_cmd",
                copy_label="Copy save command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            save_preflight_df, save_hints = _model_ops_preflight_checks(
                run_id=str(sv_run),
                source_path=str(sv_source),
                save_path=str(sv_save),
                run_save=True,
                run_load=False,
            )
            with st.expander("save 実行前チェック", expanded=False):
                _show_df(save_preflight_df, hide_index=True)
                if save_hints:
                    st.warning("\n".join([f"- {h}" for h in save_hints]))
            if not sv_run:
                st.warning("run-id を選択してください。")
            if sv_errors:
                st.error("\n".join(sv_errors))
            if st.button("Run save", key="nf_lab_run_save", disabled=(not sv_run or bool(sv_errors))):
                st.session_state["nf_lab_save_result"] = _run_shell_command_live(
                    cmd_save, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF save"
                )
            _render_command_result_block("nf_lab_save_result", "save 実行結果", "nf_lab_save")

        if sl_section == "ロード(load)":
            _render_tab_playbook(
                purpose="保存済みモデルをロードして復元可否を確認します。",
                required_inputs=["run-id", "source-path", "save-path", "load-kwargs-json"],
                outputs=["load結果", "必要時predict_insample疎通結果"],
                steps=["run-id/path確認", "load設定確認", "Run load"],
            )
            ld_run = st.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_load_run")
            ld_source = st.text_input(
                "source-path",
                value=str(run_id_to_dir.get(ld_run, PROJECT_ROOT / "artifacts" / (ld_run or ""))),
                key="nf_lab_load_source",
            )
            ld_save = st.text_input(
                "save-path",
                value=str(PROJECT_ROOT / "artifacts" / "saved_models" / (ld_run or "")),
                key="nf_lab_load_path",
            )
            ld1, ld2 = st.columns(2)
            ld_check = ld1.toggle("load_check_predict", value=True, key="nf_lab_load_check")
            ld_step = int(
                ld2.number_input(
                    "insample-step-size", min_value=1, max_value=365, value=1, step=1, key="nf_lab_load_step"
                )
            )
            ld_load_kwargs_raw = st.text_area("load-kwargs-json", value="{}", height=80, key="nf_lab_load_kwargs")
            ld_ins_kwargs_raw = st.text_area(
                "predict-insample-kwargs-json", value='{"step_size": 1}', height=80, key="nf_lab_load_ins_kwargs"
            )
            ld_errors: list[str] = []
            ld_load_kwargs = _json_text(ld_load_kwargs_raw, "dict", "load-kwargs-json", ld_errors)
            ld_ins_kwargs = _json_text(ld_ins_kwargs_raw, "dict", "predict-insample-kwargs-json", ld_errors)
            ld_load_obj = json.loads(ld_load_kwargs) if ld_load_kwargs is not None else {}
            ld_ins_obj = json.loads(ld_ins_kwargs) if ld_ins_kwargs is not None else {}
            with st.expander("load 設定サマリ", expanded=True):
                load_sig_df = _nf_signature_rows(
                    "load",
                    {"path": "<save-path>", "verbose": False, "kwargs": {}},
                    ld_load_obj,
                    forced={"path": str(ld_save)},
                )
                ins_sig_df = _nf_signature_rows(
                    "predict_insample",
                    {"step_size": 1, "level": None, "quantiles": None},
                    ld_ins_obj,
                    forced={"step_size": int(ld_step)},
                )
                _show_df(pd.concat([load_sig_df, ins_sig_df], ignore_index=True), hide_index=True)
            ld_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "model-save-load-analyze",
                "--run-id",
                str(ld_run or ""),
                "--source-path",
                str(ld_source),
                "--save-path",
                str(ld_save),
                "--no-run-save",
                "--run-load",
                "--no-run-analyze",
                _bool_opt("load-check-predict", ld_check),
                "--insample-step-size",
                str(int(ld_step)),
            ]
            if ld_load_kwargs is not None:
                ld_parts.extend(["--load-kwargs-json", ld_load_kwargs])
            if ld_ins_kwargs is not None:
                ld_parts.extend(["--predict-insample-kwargs-json", ld_ins_kwargs])
            cmd_load = " ".join(shlex.quote(x) for x in ld_parts)
            _render_command_preview(
                cmd_load,
                copy_key="nf_lab_copy_load_cmd",
                copy_label="Copy load command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            load_preflight_df, load_hints = _model_ops_preflight_checks(
                run_id=str(ld_run),
                source_path=str(ld_source),
                save_path=str(ld_save),
                run_save=False,
                run_load=True,
            )
            with st.expander("load 実行前チェック", expanded=False):
                _show_df(load_preflight_df, hide_index=True)
                if load_hints:
                    st.warning("\n".join([f"- {h}" for h in load_hints]))
            if not ld_run:
                st.warning("run-id を選択してください。")
            if ld_errors:
                st.error("\n".join(ld_errors))
            if st.button("Run load", key="nf_lab_run_load", disabled=(not ld_run or bool(ld_errors))):
                st.session_state["nf_lab_load_result"] = _run_shell_command_live(
                    cmd_load, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF load"
                )
            _render_command_result_block("nf_lab_load_result", "load 実行結果", "nf_lab_load")

        if sl_section == "保存+ロード+分析":
            st.caption("従来の save/load/analyze 一括実行です。")
            _render_tab_playbook(
                purpose="save/load/analyze を一括実行し、保存と再利用をまとめて検証します。",
                required_inputs=["run-id", "run_save/run_load/run_analyze", "save/load kwargs"],
                outputs=["保存結果", "ロード結果", "分析結果"],
                steps=["実行フラグ調整", "kwargs確認", "Run save/load/analyze"],
            )
            sl_run = st.selectbox("run-id", [""] + run_id_options, index=0, key="nf_lab_sl_run")
            sl_source = st.text_input(
                "source-path",
                value=str(run_id_to_dir.get(sl_run, PROJECT_ROOT / "artifacts" / (sl_run or ""))),
                key="nf_lab_sl_source",
            )
            sl_save = st.text_input(
                "save-path",
                value=str(PROJECT_ROOT / "artifacts" / "saved_models" / (sl_run or "")),
                key="nf_lab_sl_save",
            )
            s1, s2, s3, s4, s5 = st.columns(5)
            sl_run_save = s1.toggle("run_save", value=True, key="nf_lab_sl_run_save")
            sl_run_load = s2.toggle("run_load", value=True, key="nf_lab_sl_run_load")
            sl_run_analyze = s3.toggle("run_analyze", value=True, key="nf_lab_sl_run_analyze")
            sl_save_dataset = s4.toggle("save_dataset", value=False, key="nf_lab_sl_save_dataset")
            sl_overwrite = s5.toggle("save_overwrite", value=True, key="nf_lab_sl_overwrite")
            s6, s7 = st.columns(2)
            sl_load_check = s6.toggle("load_check_predict", value=False, key="nf_lab_sl_load_check")
            sl_step = int(
                s7.number_input("insample-step-size", min_value=1, max_value=365, value=1, step=1, key="nf_lab_sl_step")
            )
            sl_save_kwargs_raw = st.text_area("save-kwargs-json", value="{}", height=80, key="nf_lab_sl_save_kwargs")
            sl_load_kwargs_raw = st.text_area("load-kwargs-json", value="{}", height=80, key="nf_lab_sl_load_kwargs")
            sl_ins_kwargs_raw = st.text_area(
                "predict-insample-kwargs-json", value='{"step_size": 1}', height=80, key="nf_lab_sl_ins_kwargs"
            )
            sl_errors: list[str] = []
            sl_save_kwargs = _json_text(sl_save_kwargs_raw, "dict", "save-kwargs-json", sl_errors)
            sl_load_kwargs = _json_text(sl_load_kwargs_raw, "dict", "load-kwargs-json", sl_errors)
            sl_ins_kwargs = _json_text(sl_ins_kwargs_raw, "dict", "predict-insample-kwargs-json", sl_errors)
            sl_save_obj = json.loads(sl_save_kwargs) if sl_save_kwargs is not None else {}
            sl_load_obj = json.loads(sl_load_kwargs) if sl_load_kwargs is not None else {}
            sl_ins_obj = json.loads(sl_ins_kwargs) if sl_ins_kwargs is not None else {}
            with st.expander("save/load/analyze 設定サマリ", expanded=True):
                sig_save_df = _nf_signature_rows(
                    "save",
                    {"path": "<save-path>", "model_index": None, "save_dataset": True, "overwrite": False},
                    sl_save_obj,
                    forced={
                        "path": str(sl_save),
                        "save_dataset": bool(sl_save_dataset),
                        "overwrite": bool(sl_overwrite),
                    },
                )
                sig_load_df = _nf_signature_rows(
                    "load",
                    {"path": "<save-path>", "verbose": False, "kwargs": {}},
                    sl_load_obj,
                    forced={"path": str(sl_save)},
                )
                sig_ins_df = _nf_signature_rows(
                    "predict_insample",
                    {"step_size": 1, "level": None, "quantiles": None},
                    sl_ins_obj,
                    forced={"step_size": int(sl_step)},
                )
                _show_df(pd.concat([sig_save_df, sig_load_df, sig_ins_df], ignore_index=True), hide_index=True)
            sl_parts = [
                "python",
                "-m",
                "loto_forecast.cli",
                "model-save-load-analyze",
                "--run-id",
                str(sl_run or ""),
                "--source-path",
                str(sl_source),
                "--save-path",
                str(sl_save),
                _bool_opt("run-save", sl_run_save),
                _bool_opt("run-load", sl_run_load),
                _bool_opt("run-analyze", sl_run_analyze),
                _bool_opt("save-dataset", sl_save_dataset),
                _bool_opt("save-overwrite", sl_overwrite),
                _bool_opt("load-check-predict", sl_load_check),
                "--insample-step-size",
                str(int(sl_step)),
            ]
            for flg, val in [
                ("--save-kwargs-json", sl_save_kwargs),
                ("--load-kwargs-json", sl_load_kwargs),
                ("--predict-insample-kwargs-json", sl_ins_kwargs),
            ]:
                if val is not None:
                    sl_parts.extend([flg, val])
            cmd_sl = " ".join(shlex.quote(x) for x in sl_parts)
            _render_command_preview(
                cmd_sl,
                copy_key="nf_lab_copy_sl_cmd",
                copy_label="Copy save/load/analyze command",
                cwd=lab_cwd,
                show_arg_table=True,
            )
            sl_preflight_df, sl_hints = _model_ops_preflight_checks(
                run_id=str(sl_run),
                source_path=str(sl_source),
                save_path=str(sl_save),
                run_save=bool(sl_run_save),
                run_load=bool(sl_run_load),
            )
            with st.expander("save/load/analyze 実行前チェック", expanded=False):
                _show_df(sl_preflight_df, hide_index=True)
                if sl_hints:
                    st.warning("\n".join([f"- {h}" for h in sl_hints]))
            if not sl_run:
                st.warning("run-id を選択してください。")
            if sl_errors:
                st.error("\n".join(sl_errors))
            if st.button("Run save/load/analyze", key="nf_lab_run_sl", disabled=(not sl_run or bool(sl_errors))):
                st.session_state["nf_lab_sl_result"] = _run_shell_command_live(
                    cmd_sl, cwd=lab_cwd, timeout_sec=lab_timeout_sec, title="NF save-load-analyze"
                )
            _render_command_result_block("nf_lab_sl_result", "save/load/analyze 実行結果", "nf_lab_sl")

    if lab_section == "効果検証・因果":
        st.caption("model.nf_automodel の metrics/params を展開し、評価寄与とproxy因果(ATE)を可視化します。")
        if engine is None or ("model", "nf_automodel") not in tables:
            st.info("DB接続と model.nf_automodel が必要です。")
        elif model_df.empty:
            st.info("分析対象の model.nf_automodel データがありません。")
        else:
            flat_rows: list[dict[str, Any]] = []
            for row in model_df.head(max(200, row_limit * 2)).to_dict(orient="records"):
                flat_rec: dict[str, Any] = {
                    "run_id": str(row.get("run_id", "")),
                    "config_id": row.get("config_id"),
                    "model_name": str(row.get("model_name", "")),
                    "status": str(row.get("status", "")),
                }
                metrics_obj = _parse_json_like(row.get("metrics_json"))
                if isinstance(metrics_obj, dict):
                    for k, v in metrics_obj.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            flat_rec[f"metric__{k}"] = float(v)
                params_obj = _parse_json_like(row.get("params_json"))
                if isinstance(params_obj, dict):
                    for k, v in params_obj.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            flat_rec[f"param__{k}"] = float(v)
                flat_rows.append(flat_rec)
            flat_df = pd.DataFrame(flat_rows)
            numeric_cols = [c for c in flat_df.columns if pd.api.types.is_numeric_dtype(flat_df[c])]
            metric_candidates = [c for c in numeric_cols if c.startswith("metric__")]
            if not metric_candidates:
                st.info("数値メトリクス列が不足しているため効果/因果分析を実行できません。")
            else:
                pref = ["metric__mae", "metric__rmse", "metric__mape"]
                default_metric = next((x for x in pref if x in metric_candidates), metric_candidates[0])
                metric_col = st.selectbox(
                    "分析対象メトリクス",
                    metric_candidates,
                    index=metric_candidates.index(default_metric),
                    key="nf_lab_effect_metric",
                )
                base = flat_df.dropna(subset=[metric_col]).copy()
                base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")
                base = base.dropna(subset=[metric_col])
                fail_n = (
                    int((base["status"].astype(str).str.lower() == "failed").sum()) if "status" in base.columns else 0
                )
                total_n = int(base.shape[0])
                ci = _bootstrap_mean_ci(
                    pd.to_numeric(base[metric_col], errors="coerce"), alpha=0.05, n_iter=1200, seed=42
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("rows", int(base.shape[0]))
                c2.metric("failed rate", f"{(fail_n / total_n * 100.0):.1f}%" if total_n > 0 else "n/a")
                c3.metric(
                    "target mean",
                    f"{float(pd.to_numeric(base[metric_col], errors='coerce').mean()):.4g}" if total_n > 0 else "n/a",
                )
                ci_txt = (
                    f"{ci.get('ci_low'):.4g} .. {ci.get('ci_high'):.4g}"
                    if ci.get("ci_low") is not None and ci.get("ci_high") is not None
                    else "n/a"
                )
                c4.metric("bootstrap 95% CI", ci_txt)
                st.markdown("**target description**")
                _show_df(
                    pd.to_numeric(base[metric_col], errors="coerce").describe().to_frame("value").reset_index(),
                    hide_index=True,
                )

                treat_candidates = [c for c in numeric_cols if c != metric_col][:80]
                corr_rows: list[dict[str, Any]] = []
                for c in treat_candidates:
                    s = base[[c, metric_col]].dropna()
                    if len(s) < 12:
                        continue
                    corr_rows.append(
                        {
                            "feature": c,
                            "pearson": float(s[c].corr(s[metric_col])),
                            "spearman": float(s[c].corr(s[metric_col], method="spearman")),
                            "n": int(len(s)),
                        }
                    )
                corr_df = (
                    pd.DataFrame(corr_rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
                    if corr_rows
                    else pd.DataFrame()
                )
                st.markdown("**寄与候補（相関ベース）**")
                if corr_df.empty:
                    st.info("寄与候補を算出できる数値特徴量が不足しています。")
                else:
                    _show_df(corr_df.head(30), hide_index=True)
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            corr_df.head(20), x="feature", y="spearman", title=f"{metric_col} とのSpearman相関"
                        )
                        fig.update_layout(height=330)
                        st.plotly_chart(fig, width="stretch")
                if corr_df.empty:
                    st.info("proxy因果のtreatment候補がありません。")
                else:
                    top_treats = corr_df.head(20)["feature"].tolist()
                    treat_col = st.selectbox("proxy因果 treatment", top_treats, index=0, key="nf_lab_causal_treat")
                    ate = _causal_proxy_ate(base, target_col=metric_col, treatment_col=str(treat_col))
                    st.markdown("**proxy ATE（調整回帰）**")
                    st.json(ate)

    if lab_section == "Run-ID統合分析":
        render_runid_integrated_panel(
            project_root=PROJECT_ROOT,
            run_id_options=run_id_options,
            run_id_to_dir=run_id_to_dir,
            model_df=model_df,
            engine=engine,
            tables=tables,
            row_limit=row_limit,
            settings=settings,
            query_df=_query_df,
            show_df=_show_df,
            parse_json_like=_parse_json_like,
            safe_read_json_file=_safe_read_json_file,
            has_model_artifacts=_has_model_artifacts,
            causal_proxy_ate=_causal_proxy_ate,
            stable_json_dumps=_stable_json_dumps,
            plotly_available=PLOTLY_AVAILABLE,
            px=px if PLOTLY_AVAILABLE else None,
        )

    if lab_section == "リソース解析":
        render_nf_resource_analytics_panel(
            engine=engine,
            tables=tables,
            row_limit=int(row_limit),
            query_df=_query_df,
            show_df=_show_df,
            plotly_available=PLOTLY_AVAILABLE,
            px=px if PLOTLY_AVAILABLE else None,
        )

    if lab_section == "DB管理/ER":
        st.caption("DB・スキーマ・テーブルのCRAD、テーブル確認、ER図をこのラボ内で実行できます。")
        if engine is None:
            st.info("DB未接続のため表示できません。")
        else:
            render_db_admin_panel(
                engine=engine,
                database=database,
                row_limit=int(row_limit),
                sample_limit=int(sample_limit),
                show_df=_show_df,
                query_df=_query_df,
                table_columns=_table_columns,
                sample_table=_sample_table,
                exact_count=_exact_count,
                clear_query_cache=_clear_query_cache,
            )
    _save_nf_lab_ui_state(
        engine=engine,
        host=host,
        port=int(port),
        user=user,
        database=database,
    )
    return


def _render_overview(engine: Engine, tables: set[tuple[str, str]]) -> None:
    st.header("概要")
    catalog = _table_catalog(engine)
    if catalog.empty:
        st.info("dataset/exog/resources にテーブルが見つかりません。")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("datasetテーブル", int((catalog["table_schema"] == "dataset").sum()))
    c2.metric("exogテーブル", int((catalog["table_schema"] == "exog").sum()))
    c3.metric("resourcesテーブル", int((catalog["table_schema"] == "resources").sum()))
    c4.metric("meta/modelテーブル", int((catalog["table_schema"].isin(["meta", "model"])).sum()))
    c5.metric("全テーブル", int(len(catalog)))
    _show_df(catalog, hide_index=True)

    if ("resources", "run") in tables:
        run_stats = _query_df(
            engine,
            """
            SELECT
              COUNT(*)::bigint AS total_runs,
              COUNT(*) FILTER (WHERE status='success')::bigint AS success_runs,
              COUNT(*) FILTER (WHERE status='failed')::bigint AS failed_runs
            FROM resources.run
            """,
        )
        if not run_stats.empty:
            s1, s2, s3 = st.columns(3)
            s1.metric("実行数", int(run_stats.iloc[0]["total_runs"]))
            s2.metric("成功", int(run_stats.iloc[0]["success_runs"]))
            s3.metric("失敗", int(run_stats.iloc[0]["failed_runs"]))

        latest = _query_df(
            engine,
            """
            SELECT
              run_id,
              started_at,
              ended_at,
              status,
              COALESCE(tags->>'execution_os', tags->'runtime_env'->>'execution_os', 'unknown') AS execution_os,
              app_name,
              command,
              rows_target,
              rows_written,
              rows_failed
            FROM resources.run
            ORDER BY started_at DESC
            LIMIT 30
            """,
        )
        st.caption("Latest resources.run")
        _show_df(latest, hide_index=True)

    st.markdown("**プロジェクトツリー(depth=3)**")
    st.code(_tree_lines(PROJECT_ROOT, max_depth=3, max_entries=500))


def _render_runs(engine: Engine, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("実行履歴 / Span / メトリクス")
    if ("resources", "run") not in tables:
        st.info("resources.run が存在しません。")
        return

    runs = _query_df(
        engine,
        """
        SELECT
          run_id, started_at, ended_at, status, env, profile, app_name, command,
          rows_target, rows_written, rows_failed, error_summary, tags,
          COALESCE(tags->>'execution_os', tags->'runtime_env'->>'execution_os', 'unknown') AS execution_os
        FROM resources.run
        ORDER BY started_at DESC
        LIMIT :limit
        """,
        {"limit": int(row_limit)},
    )
    _show_df(runs.drop(columns=["tags"], errors="ignore"), hide_index=True)
    if runs.empty:
        return

    run_id = st.selectbox("run_id", runs["run_id"].astype(str).tolist(), index=0)
    selected = runs[runs["run_id"].astype(str) == run_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("status", str(selected.get("status", "")))
    c2.metric("rows_written", str(selected.get("rows_written", "")))
    c3.metric("rows_failed", str(selected.get("rows_failed", "")))
    c4.metric("execution_os", str(selected.get("execution_os", "unknown")))
    st.code(str(selected.get("command", "")))
    if selected.get("tags") is not None:
        try:
            tags = selected.get("tags")
            st.json(json.loads(tags) if isinstance(tags, str) else tags)
        except Exception:
            st.text(str(selected.get("tags")))

    if ("resources", "stage_span") in tables:
        spans = _query_df(
            engine,
            """
            SELECT
              stage_name, started_at, ended_at, duration_ms, rows_in, rows_out,
              db_time_ms, db_rows, gpu_util_avg, gpu_mem_used_mb_avg, exception_type, exception_msg
            FROM resources.stage_span
            WHERE run_id = :run_id
            ORDER BY started_at
            """,
            {"run_id": run_id},
        )
        st.markdown("**resources.stage_span**")
        _show_df(spans, hide_index=True)
        if not spans.empty:
            bar = spans.groupby("stage_name", as_index=False)["duration_ms"].sum().set_index("stage_name")
            st.bar_chart(bar, height=260)

    if ("resources", "resource_metric") in tables:
        metrics = _query_df(
            engine,
            """
            SELECT sampled_at, metric_key, metric_value, unit
            FROM resources.resource_metric
            WHERE run_id = :run_id
            ORDER BY sampled_at
            LIMIT 5000
            """,
            {"run_id": run_id},
        )
        st.markdown("**resources.resource_metric**")
        _show_df(metrics, hide_index=True)
        if not metrics.empty:
            pivot = metrics.pivot_table(
                index="sampled_at",
                columns="metric_key",
                values="metric_value",
                aggfunc="mean",
            )
            st.line_chart(pivot, height=320)


def _render_resources_analytics(engine: Engine, tables: set[tuple[str, str]]) -> None:
    st.subheader("リソース分析")
    st.caption("`resources.run` / `resources.stage_span` / `resources.resource_metric` を横断分析します。")
    if ("resources", "run") not in tables:
        st.info("resources.run が存在しません。")
        return

    run_df = _query_df(
        engine,
        """
        SELECT
          run_id::text AS run_id,
          started_at,
          ended_at,
          status,
          app_name,
          command,
          rows_target,
          rows_written,
          rows_failed,
          COALESCE(tags->>'execution_os', tags->'runtime_env'->>'execution_os', 'unknown') AS execution_os,
          EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))::double precision AS duration_sec
        FROM resources.run
        ORDER BY started_at DESC
        LIMIT 5000
        """,
    )
    if run_df.empty:
        st.info("resources.run にデータがありません。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("実行数", int(len(run_df)))
    c2.metric("成功", int((run_df["status"].astype(str) == "success").sum()))
    c3.metric("失敗", int((run_df["status"].astype(str) == "failed").sum()))
    duration_sec = pd.to_numeric(run_df["duration_sec"], errors="coerce")
    avg_duration = float(duration_sec.dropna().mean()) if duration_sec.notna().any() else 0.0
    c4.metric("平均処理時間(s)", f"{avg_duration:.2f}")

    run_df["started_day"] = pd.to_datetime(run_df["started_at"], errors="coerce").dt.date
    status_counts = run_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
    st.markdown("**status件数**")
    _show_df(status_counts, hide_index=True)
    st.bar_chart(status_counts.set_index("status")[["count"]], height=260)

    os_counts = (
        run_df["execution_os"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis("execution_os")
        .reset_index(name="count")
    )
    st.markdown("**実行OS別件数 (wsl / native_linux / windows)**")
    _show_df(os_counts, hide_index=True)
    st.bar_chart(os_counts.set_index("execution_os")[["count"]], height=240)

    day_counts = run_df.groupby("started_day", as_index=False).size().rename(columns={"size": "runs"})
    day_counts = day_counts.sort_values("started_day")
    st.markdown("**日次実行数**")
    if not day_counts.empty:
        day_plot = day_counts.set_index("started_day")[["runs"]]
        st.line_chart(day_plot, height=260)

    st.markdown("**処理量 / 品質**")
    qdf = run_df.copy()
    qdf["status"] = _normalize_status_series(qdf.get("status", pd.Series(dtype=object)), default="unknown")
    qdf["rows_written"] = pd.to_numeric(qdf["rows_written"], errors="coerce").fillna(0.0)
    qdf["rows_failed"] = pd.to_numeric(qdf["rows_failed"], errors="coerce").fillna(0.0)
    den = qdf["rows_written"] + qdf["rows_failed"]
    qdf["fail_rate"] = np.where(den > 0, qdf["rows_failed"] / den, 0.0)
    qdf["fail_rate"] = pd.to_numeric(qdf["fail_rate"], errors="coerce").fillna(0.0)
    _show_df(
        qdf[["run_id", "status", "duration_sec", "rows_written", "rows_failed", "fail_rate"]].head(300), hide_index=True
    )
    if PLOTLY_AVAILABLE:
        try:
            fig = _build_categorical_scatter_figure(
                qdf,
                x="duration_sec",
                y="rows_written",
                color="status",
                hover_fields=["run_id", "rows_failed", "fail_rate"],
                color_map=STATUS_COLOR_MAP,
                color_order=_present_category_order(qdf["status"], ["success", "failed", "running", "pending", "unknown"]),
                title="rows_written vs duration_sec",
            )
            st.plotly_chart(fig, width="stretch")
        except Exception:
            pass

    if ("resources", "stage_span") in tables:
        stage_df = _query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              stage_name,
              duration_ms,
              rows_in,
              rows_out,
              db_time_ms,
              db_rows,
              gpu_util_avg,
              gpu_mem_used_mb_avg,
              exception_type
            FROM resources.stage_span
            ORDER BY started_at DESC
            LIMIT 50000
            """,
        )
        st.markdown("**Stage Span Summary**")
        if not stage_df.empty:
            stage_agg = (
                stage_df.groupby("stage_name", as_index=False)
                .agg(
                    count=("stage_name", "count"),
                    avg_duration_ms=("duration_ms", "mean"),
                    p95_duration_ms=("duration_ms", lambda x: float(pd.Series(x).quantile(0.95))),
                    exception_count=("exception_type", lambda x: int(pd.Series(x).notna().sum())),
                )
                .sort_values("avg_duration_ms", ascending=False)
            )
            _show_df(stage_agg, hide_index=True)
            st.bar_chart(stage_agg.set_index("stage_name")[["avg_duration_ms"]], height=280)

            if PLOTLY_AVAILABLE:
                try:
                    heat = stage_df.copy()
                    heat["duration_ms"] = pd.to_numeric(heat["duration_ms"], errors="coerce")
                    pivot = (
                        heat.groupby(["run_id", "stage_name"], as_index=False)["duration_ms"]
                        .mean()
                        .pivot(index="run_id", columns="stage_name", values="duration_ms")
                    )
                    pivot = pivot.apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0.0)
                    if not pivot.empty and pivot.shape[0] <= 200:
                        fig_hm = px.imshow(pivot, aspect="auto", title="Run x Stage (avg duration_ms)")
                        st.plotly_chart(fig_hm, width="stretch")
                except Exception:
                    pass

    if ("resources", "resource_metric") in tables:
        metric_df = _query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              sampled_at,
              metric_key,
              metric_value,
              unit
            FROM resources.resource_metric
            ORDER BY sampled_at DESC
            LIMIT 100000
            """,
        )
        st.markdown("**Resource Metric Summary**")
        if not metric_df.empty:
            metric_agg = (
                metric_df.groupby("metric_key", as_index=False)
                .agg(
                    count=("metric_key", "count"),
                    avg_value=("metric_value", "mean"),
                    min_value=("metric_value", "min"),
                    max_value=("metric_value", "max"),
                )
                .sort_values("count", ascending=False)
            )
            _show_df(metric_agg, hide_index=True)
            st.bar_chart(metric_agg.head(20).set_index("metric_key")[["count"]], height=260)

            keys = metric_agg["metric_key"].astype(str).tolist()
            selected_key = st.selectbox("metric key", keys, index=0, key="res_metric_key")
            key_df = metric_df[metric_df["metric_key"].astype(str) == selected_key].copy()
            key_df["sampled_at"] = pd.to_datetime(key_df["sampled_at"], errors="coerce")
            key_df = key_df.sort_values("sampled_at")
            _show_df(key_df[["sampled_at", "run_id", "metric_value", "unit"]].tail(500), hide_index=True)
            if not key_df.empty:
                st.line_chart(key_df.set_index("sampled_at")[["metric_value"]], height=280)


def _render_exog_tables(engine: Engine, tables: set[tuple[str, str]], sample_limit: int) -> None:
    st.subheader("Exogテーブル")
    exog_tables = sorted([name for schema, name in tables if schema == "exog"])
    if not exog_tables:
        st.info("exog スキーマにテーブルがありません。")
        return
    selected = st.selectbox("exog table", exog_tables, index=0)
    cols_df = _table_columns(engine, "exog", selected)
    _show_df(cols_df, hide_index=True)

    c_names = cols_df["column_name"].astype(str) if not cols_df.empty else pd.Series(dtype=str)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("hist_*", int(c_names.str.startswith("hist_").sum()))
    c2.metric("stat_*", int(c_names.str.startswith("stat_").sum()))
    c3.metric("feat_*", int(c_names.str.startswith("feat_").sum()))
    c4.metric("all columns", int(len(cols_df)))

    if st.button("Count rows (exact)", key=f"cnt_exog_{selected}"):
        try:
            st.metric("row_count", _exact_count(engine, "exog", selected))
        except Exception as e:
            st.error(str(e))
    _show_df(_sample_table(engine, "exog", selected, sample_limit), hide_index=True)


def _render_dataset_model_grid(engine: Engine, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("Model / Grid / Meta / Eventログ")
    if ("dataset", "model_run") in tables:
        runs = _query_df(
            engine,
            """
            SELECT run_id, started_at, ended_at, status, model_name, library_name, adapter_name, grid_id, task_id, error_message
            FROM meta.model_run
            ORDER BY started_at DESC
            LIMIT :limit
            """,
            {"limit": int(row_limit)},
        )
        st.markdown("**meta.model_run**")
        _show_df(runs, hide_index=True)

    if ("dataset", "grid_search_definition") in tables:
        defs = _query_df(
            engine,
            """
            SELECT grid_id, library_name, adapter_name, model_name, horizon, created_at
            FROM meta.grid_search_definition
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"limit": int(row_limit)},
        )
        st.markdown("**meta.grid_search_definition**")
        _show_df(defs, hide_index=True)

    if ("dataset", "grid_search_task") in tables:
        tasks = _query_df(
            engine,
            """
            SELECT grid_id, status, COUNT(*)::bigint AS cnt
            FROM meta.grid_search_task
            GROUP BY grid_id, status
            ORDER BY grid_id, status
            """,
        )
        st.markdown("**meta.grid_search_task (status summary)**")
        _show_df(tasks, hide_index=True)

    if ("dataset", "execution_event_log") in tables:
        events = _query_df(
            engine,
            """
            SELECT event_ts, level, event_type, run_id, task_id, message
            FROM log.execution_event_log
            ORDER BY event_ts DESC
            LIMIT :limit
            """,
            {"limit": int(row_limit)},
        )
        st.markdown("**log.execution_event_log**")
        _show_df(events, hide_index=True)

    if ("meta", "nf_automodel") in tables:
        meta_df = _query_df(
            engine,
            """
            SELECT
              config_id, active, priority, config_name,
              base_schema, base_table, hist_schema, hist_table, exog_schema,
              output_schema, output_table,
              unified_filter_json,
              unified_group_cols_json, unified_group_validate_strict,
              model_name, horizon, auto_cls_model, auto_h,
              auto_backend, auto_num_samples, auto_loss, auto_valid_loss, max_tasks, recursive_depth,
              run_predict, run_evaluate, run_explain, run_save, run_load, run_analyze,
              save_path,
              last_status, last_run_id, last_run_at, updated_at
            FROM meta.nf_automodel
            ORDER BY priority, config_id
            LIMIT :limit
            """,
            {"limit": int(row_limit)},
        )
        st.markdown("**meta.nf_automodel**")
        _show_df(meta_df, hide_index=True)

    if ("model", "nf_automodel") in tables:
        model_df = _query_df(
            engine,
            """
            SELECT
              result_id, config_id, run_id, status, model_name, horizon,
              dataset_rows, feature_cols, artifact_path, model_store_path, error_message,
              started_at, ended_at, created_at
            FROM model.nf_automodel
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"limit": int(row_limit)},
        )
        st.markdown("**model.nf_automodel**")
        _show_df(model_df, hide_index=True)

    if ("dataset", "loto_y_ts_unified") in tables:
        st.markdown("**dataset.loto_y_ts_unified (sample)**")
        try:
            sample = _query_df(
                engine,
                """
                SELECT *
                FROM dataset.loto_y_ts_unified
                ORDER BY ds DESC NULLS LAST
                LIMIT :limit
                """,
                {"limit": int(min(50, row_limit))},
            )
            _show_df(sample, hide_index=True)
        except Exception as e:
            st.warning(f"dataset.loto_y_ts_unified sample failed: {e}")


def _flatten_recursive_tree_rows(tree: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any], path: str) -> None:
        if not isinstance(node, dict):
            return
        cond = node.get("condition") if isinstance(node.get("condition"), dict) else {}
        rows.append(
            {
                "path": path,
                "depth": int(node.get("depth", 0) or 0),
                "rows": int(node.get("rows", 0) or 0),
                "target_mean": node.get("target_mean"),
                "target_median": node.get("target_median"),
                "target_std": node.get("target_std"),
                "split_parameter": node.get("split_parameter"),
                "condition": json.dumps(cond, ensure_ascii=False),
            }
        )
        children = node.get("children")
        if not isinstance(children, list):
            return
        for idx, child in enumerate(children, start=1):
            _walk(child if isinstance(child, dict) else {}, f"{path}.{idx}")

    _walk(tree if isinstance(tree, dict) else {}, "root")
    return rows


def _bootstrap_mean_ci(
    series: pd.Series,
    alpha: float = 0.05,
    n_iter: int = 1200,
    seed: int = 42,
) -> dict[str, float | int | None]:
    y = pd.to_numeric(series, errors="coerce").dropna()
    n = int(y.shape[0])
    if n <= 1:
        mean_val = float(y.mean()) if n > 0 else None
        return {"n": n, "mean": mean_val, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    values = y.to_numpy(dtype=float)
    means = np.empty(max(100, int(n_iter)), dtype=float)
    for i in range(means.shape[0]):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(np.mean(sample))
    lo_q = float(np.quantile(means, alpha / 2.0))
    hi_q = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {"n": n, "mean": float(np.mean(values)), "ci_low": lo_q, "ci_high": hi_q}


def _iqr_outlier_mask(series: pd.Series, whisker: float = 1.5) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or float(iqr) <= 0.0:
        return pd.Series([False] * len(y), index=y.index)
    lower = float(q1 - whisker * iqr)
    upper = float(q3 + whisker * iqr)
    return (y < lower) | (y > upper)


def _extract_jsonl(path: Path, max_rows: int = 5000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(rows) >= max_rows:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


_parse_json_like = dashboard_helpers.parse_json_like
_flatten_json_like_value = dashboard_helpers.flatten_json_like_value
_flatten_json_columns = dashboard_helpers.flatten_json_columns
_expand_semistructured_columns = dashboard_helpers.expand_semistructured_columns


_bayesian_success_posterior = dashboard_helpers.bayesian_success_posterior
_impact_contribution_rates = dashboard_helpers.impact_contribution_rates
_r2_for_features = dashboard_helpers.r2_for_features
_approx_shapley_contrib = dashboard_helpers.approx_shapley_contrib
_graph_centrality_from_impact = dashboard_helpers.graph_centrality_from_impact
_causal_proxy_ate = dashboard_helpers.causal_proxy_ate


def _render_meta_deep_analysis(engine: Engine, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("メタ深層分析")
    st.caption("`model.nf_automodel` を対象に、統計検定・多角可視化・再帰分析・出力を実行します。")
    tab_labels = [
        "概要サマリ",
        "分布・時系列",
        "群比較・検定",
        "相関・欠損",
        "パラメータ影響",
        "設定反映監査",
        "寄与・因果拡張",
        "再帰ツリー",
        "フラットデータ",
        "エクスポート",
    ]
    with st.expander("画面ガイド / クイック移動", expanded=False):
        guide_tabs = st.tabs(["タブ説明", "読み方", "クイックリンク"])
        with guide_tabs[0]:
            guide_rows = [
                {
                    "tab": "概要サマリ",
                    "目的": "全体状態・CI・記述統計を確認",
                    "主出力": "failed rate / bootstrap CI / target description",
                },
                {"tab": "分布・時系列", "目的": "分布歪み・経時変化を確認", "主出力": "hist / trend / status別箱ひげ"},
                {
                    "tab": "群比較・検定",
                    "目的": "群間差と有意性の確認",
                    "主出力": "group summary / stat tests / Bayesian",
                },
                {
                    "tab": "相関・欠損",
                    "目的": "欠損偏りと共変動を確認",
                    "主出力": "missing ratio / correlation heatmap",
                },
                {"tab": "パラメータ影響", "目的": "精度へ効く設定を特定", "主出力": "effect ranking / Shapley proxy"},
                {"tab": "設定反映監査", "目的": "設定値の反映漏れ検知", "主出力": "mismatch rate / mismatch details"},
                {"tab": "寄与・因果拡張", "目的": "寄与因子とproxy因果を確認", "主出力": "Permutation / causal hints"},
                {"tab": "再帰ツリー", "目的": "分割ロジックを追跡", "主出力": "recursive tree + flattened rows"},
                {"tab": "フラットデータ", "目的": "明細を直接確認", "主出力": "flattened table"},
                {"tab": "エクスポート", "目的": "再現可能な成果物を保存", "主出力": "json/csv download"},
            ]
            _show_df(pd.DataFrame(guide_rows), hide_index=True)
        with guide_tabs[1]:
            st.markdown(
                "\n".join(
                    [
                        "1. `概要サマリ` で失敗率とCIを確認",
                        "2. `分布・時系列` と `群比較・検定` で異常条件を切り分け",
                        "3. `パラメータ影響` と `設定反映監査` で改善対象を確定",
                        "4. `寄与・因果拡張` で施策候補を優先付け",
                        "5. `エクスポート` で結果を保存し再現性を担保",
                    ]
                )
            )
        with guide_tabs[2]:
            st.caption(
                "Streamlitの制約でリンククリック時にタブ自体は自動選択されないため、上段タブ名をクリックして移動してください。"
            )
            st.markdown(" | ".join([f"[{t}](#meta-deep-{_slug(t)})" for t in tab_labels]))

    if ("model", "nf_automodel") not in tables:
        st.info("model.nf_automodel が存在しないため解析できません。")
        return

    from loto_forecast.analysis.meta_automodel_report import build_meta_automodel_report

    c1, c2, c3, c4 = st.columns(4)
    config_id_raw = c1.text_input("config id（任意）", value="", key="meta_deep_config_id")
    run_id_raw = c2.text_input("run id（任意）", value="", key="meta_deep_run_id")
    status_opt = c3.selectbox("status", ["all", "success", "failed"], index=0, key="meta_deep_status")
    limit_val = c4.number_input(
        "取得上限行",
        min_value=10,
        max_value=200000,
        value=int(max(100, min(5000, row_limit * 20))),
        step=10,
        key="meta_deep_limit",
    )

    preset = st.selectbox("分析プリセット", ["標準", "詳細"], index=0, key="meta_deep_preset")
    default_depth = 4 if preset == "詳細" else 3
    default_topk = 40 if preset == "詳細" else 20
    a1, a2, a3, a4, a5 = st.columns(5)
    target_metric = a1.text_input("対象指標", value="mae", key="meta_deep_target_metric")
    higher_is_better = a2.toggle("値が高いほど良い", value=False, key="meta_deep_hib")
    recursive_depth = a3.slider(
        "再帰深度", min_value=1, max_value=8, value=default_depth, key="meta_deep_recursive_depth"
    )
    min_group_size = a4.slider(
        "最小グループサイズ", min_value=2, max_value=100, value=5, key="meta_deep_min_group_size"
    )
    alpha = a5.slider("有意水準 alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.001, key="meta_deep_alpha")
    top_k = st.slider(
        "パラメータ影響 上位件数", min_value=5, max_value=100, value=default_topk, step=1, key="meta_deep_top_k"
    )

    has_meta_cfg = ("meta", "nf_automodel") in tables
    col_prefix = "m." if has_meta_cfg else ""

    where_parts = ["1=1"]
    params: dict[str, Any] = {"limit": int(limit_val)}
    if config_id_raw.strip():
        try:
            params["config_id"] = int(config_id_raw.strip())
            where_parts.append(f"{col_prefix}config_id = :config_id")
        except Exception:
            st.warning("config id は整数で入力してください。")
    if run_id_raw.strip():
        params["run_id"] = run_id_raw.strip()
        where_parts.append(f"{col_prefix}run_id = :run_id")
    if status_opt != "all":
        params["status"] = status_opt
        where_parts.append(f"{col_prefix}status = :status")

    if has_meta_cfg:
        sql = f"""
            SELECT
              m.*,
              cfg.model_name AS cfg_model_name,
              cfg.horizon AS cfg_horizon,
              cfg.auto_num_samples AS cfg_auto_num_samples,
              cfg.auto_backend AS cfg_auto_backend,
              cfg.auto_loss AS cfg_auto_loss,
              cfg.auto_valid_loss AS cfg_auto_valid_loss,
              cfg.auto_search_alg AS cfg_auto_search_alg,
              cfg.auto_cpus AS cfg_auto_cpus,
              cfg.auto_gpus AS cfg_auto_gpus,
              cfg.auto_refit_with_val AS cfg_auto_refit_with_val,
              cfg.auto_verbose AS cfg_auto_verbose
            FROM model.nf_automodel m
            LEFT JOIN meta.nf_automodel cfg
              ON cfg.config_id = m.config_id
            WHERE {" AND ".join(where_parts)}
            ORDER BY m.created_at DESC
            LIMIT :limit
        """
    else:
        sql = f"""
            SELECT *
            FROM model.nf_automodel
            WHERE {" AND ".join(where_parts)}
            ORDER BY created_at DESC
            LIMIT :limit
        """

    try:
        raw_df = _query_df(engine, sql, params=params)
    except Exception as e:
        st.error(f"query failed: {e}")
        return

    if raw_df.empty:
        st.warning("対象データが0件です。filter/limit を見直してください。")
        return

    _log_dashboard_event(
        "meta_deep_analysis_start",
        {"rows": int(raw_df.shape[0]), "config_id": config_id_raw.strip(), "run_id": run_id_raw.strip()},
    )

    report = build_meta_automodel_report(
        raw_df=raw_df,
        target_metric=target_metric,
        higher_is_better=higher_is_better,
        recursive_depth=int(recursive_depth),
        min_group_size=int(min_group_size),
        alpha=float(alpha),
        top_k=int(top_k),
    )

    overview = dict(report.get("overview", {}) or {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("rows", int(overview.get("rows", 0) or 0))
    m2.metric("runs", int(overview.get("run_count", 0) or 0))
    m3.metric("configs", int(overview.get("config_count", 0) or 0))
    status_counts = overview.get("status_counts", {}) if isinstance(overview.get("status_counts", {}), dict) else {}
    fail_n = int(status_counts.get("failed", 0) or 0)
    total_n = int(sum(int(v or 0) for v in status_counts.values()))
    m4.metric("failed rate", f"{(fail_n / total_n * 100.0):.1f}%" if total_n > 0 else "n/a")

    st.markdown("**インサイト**")
    insights = report.get("insights", []) if isinstance(report.get("insights", []), list) else []
    if insights:
        for idx, item in enumerate(insights, start=1):
            st.write(f"{idx}. {item}")
    else:
        st.write("insightなし")

    target_col = report.get("target_metric")
    metric_summary_df = report.get("metric_summary", pd.DataFrame())
    impact_df = report.get("parameter_impact", pd.DataFrame())
    reflection_df = report.get("parameter_reflection", pd.DataFrame())
    reflection_detail_df = report.get("parameter_reflection_detail", pd.DataFrame())
    contribution_df = report.get("feature_contribution", pd.DataFrame())
    causal_hints_df = report.get("causal_hints", pd.DataFrame())
    flat_df = report.get("flattened_results", pd.DataFrame())
    if target_col and target_col in flat_df.columns:
        flat_df = flat_df.copy()
        flat_df[target_col] = pd.to_numeric(flat_df[target_col], errors="coerce")

    y = (
        pd.to_numeric(flat_df[target_col], errors="coerce").dropna()
        if target_col and target_col in flat_df.columns
        else pd.Series(dtype=float)
    )
    outlier_mask = (
        _iqr_outlier_mask(flat_df[target_col])
        if target_col and target_col in flat_df.columns
        else pd.Series(dtype=bool)
    )
    bootstrap_ci = _bootstrap_mean_ci(y, alpha=float(alpha), n_iter=2000 if preset == "詳細" else 1200, seed=42)

    st.markdown("**統計検定**")
    st.json(report.get("stat_tests", {}))

    st.caption("クイック移動: " + " | ".join([f"`{t}`" for t in tab_labels]))
    sub = st.tabs(tab_labels)

    with sub[0]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[0])}'></div>", unsafe_allow_html=True)
        st.caption("対象指標の基本統計・失敗率・CIを最初に確認するタブです。")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("対象指標", str(target_col) if target_col else "n/a")
        c2.metric("有効サンプル", int(y.shape[0]))
        c3.metric("外れ値数(IQR)", int(outlier_mask.sum()) if not outlier_mask.empty else 0)
        ci_text = (
            f"{bootstrap_ci.get('ci_low'):.4g} .. {bootstrap_ci.get('ci_high'):.4g}"
            if bootstrap_ci.get("ci_low") is not None and bootstrap_ci.get("ci_high") is not None
            else "n/a"
        )
        c4.metric("平均95%CI(Bootstrap)", ci_text)

        if target_col and target_col in flat_df.columns:
            target_desc = (
                pd.to_numeric(flat_df[target_col], errors="coerce")
                .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                .to_frame("value")
            )
            st.markdown("**対象指標の記述統計**")
            _show_df(target_desc.reset_index(), hide_index=True)

        st.markdown("**metrics.* 要約**")
        _show_df(metric_summary_df, hide_index=True)
        if not metric_summary_df.empty:
            chart_df = metric_summary_df.copy()
            chart_df = chart_df.sort_values("mean", ascending=False).head(20).set_index("metric")
            st.bar_chart(chart_df[["mean"]], height=300)

    with sub[1]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[1])}'></div>", unsafe_allow_html=True)
        st.caption("分布形状と時間方向の変化を同時に確認し、ドリフトや外れ値を早期検知します。")
        if not target_col or target_col not in flat_df.columns:
            st.info("target metric が見つからないため可視化できません。")
        else:
            plot_df = flat_df.copy()
            plot_df[target_col] = pd.to_numeric(plot_df[target_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[target_col])
            if "created_at" in plot_df.columns:
                plot_df["created_at"] = pd.to_datetime(plot_df["created_at"], errors="coerce")
            if plot_df.empty:
                st.info("target metric の有効データがありません。")
            else:
                c_hist, c_line = st.columns(2)
                with c_hist:
                    if PLOTLY_AVAILABLE:
                        if "status" in plot_df.columns:
                            plot_df["status"] = _normalize_status_series(plot_df["status"], default="unknown")
                            fig_hist = _build_categorical_histogram_figure(
                                plot_df,
                                x=target_col,
                                color="status",
                                color_map=STATUS_COLOR_MAP,
                                color_order=_present_category_order(
                                    plot_df["status"], ["success", "failed", "running", "pending", "unknown"]
                                ),
                                title=f"分布: {target_col}",
                            )
                        else:
                            fig_hist = px.histogram(plot_df, x=target_col)
                        fig_hist.update_layout(height=320, title=f"分布: {target_col}")
                        st.plotly_chart(fig_hist, width="stretch")
                    else:
                        st.bar_chart(plot_df[target_col].value_counts().head(30), height=320)
                with c_line:
                    if "created_at" in plot_df.columns:
                        trend = plot_df[["created_at", target_col]].dropna().sort_values("created_at")
                        if not trend.empty:
                            st.line_chart(trend.set_index("created_at")[[target_col]], height=320)
                    else:
                        st.info("created_at 列がないため時系列表示をスキップ")
                if PLOTLY_AVAILABLE and "status" in plot_df.columns:
                    fig_box = px.box(plot_df, x="status", y=target_col, points="outliers")
                    fig_box.update_layout(height=340, title=f"status 別比較: {target_col}")
                    st.plotly_chart(fig_box, width="stretch")
                if not outlier_mask.empty:
                    flagged = plot_df.loc[outlier_mask.reindex(plot_df.index, fill_value=False)]
                    if not flagged.empty:
                        st.markdown("**外れ値サンプル(IQR)**")
                        show_cols = [
                            c
                            for c in ["run_id", "config_id", "status", "model_name", target_col]
                            if c in flagged.columns
                        ]
                        _show_df(flagged[show_cols].head(100), hide_index=True)

    with sub[2]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[2])}'></div>", unsafe_allow_html=True)
        st.caption("status/model 単位の群比較と検定結果で、差分の統計的妥当性を確認します。")
        st.markdown("**status別の統計要約**")
        if target_col and target_col in flat_df.columns and "status" in flat_df.columns:
            grp = (
                flat_df[["status", target_col]]
                .dropna()
                .groupby("status", as_index=False)
                .agg(
                    count=(target_col, "count"),
                    mean=(target_col, "mean"),
                    median=(target_col, "median"),
                    std=(target_col, "std"),
                    min=(target_col, "min"),
                    max=(target_col, "max"),
                )
            )
            _show_df(grp, hide_index=True)
        else:
            st.info("status または target 指標が不足しているため群比較を省略しました。")

        if target_col and target_col in flat_df.columns and "model_name" in flat_df.columns:
            model_grp = (
                flat_df[["model_name", target_col]]
                .dropna()
                .groupby("model_name", as_index=False)
                .agg(
                    count=(target_col, "count"),
                    mean=(target_col, "mean"),
                    median=(target_col, "median"),
                    std=(target_col, "std"),
                )
                .sort_values("mean", ascending=bool(higher_is_better))
            )
            st.markdown("**model_name別 指標比較**")
            _show_df(model_grp.head(30), hide_index=True)
            if PLOTLY_AVAILABLE and not model_grp.empty:
                fig_m = px.bar(model_grp.head(20), x="model_name", y="mean", title=f"model_name別 平均 {target_col}")
                fig_m.update_layout(height=320)
                st.plotly_chart(fig_m, width="stretch")

        st.markdown("**検定結果(JSON)**")
        st.json(report.get("stat_tests", {}))

        bayes_df = _bayesian_success_posterior(flat_df, group_col="model_name")
        if not bayes_df.empty:
            st.markdown("**ベイズ推定: model別 成功確率 事後分布(Beta)**")
            _show_df(bayes_df, hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_b = px.bar(
                    bayes_df.head(20),
                    x="model_name",
                    y="posterior_mean",
                    error_y=(bayes_df.head(20)["posterior_ci_high_95"] - bayes_df.head(20)["posterior_mean"]),
                    error_y_minus=(bayes_df.head(20)["posterior_mean"] - bayes_df.head(20)["posterior_ci_low_95"]),
                    title="model別 事後成功確率",
                )
                fig_b.update_layout(height=320)
                st.plotly_chart(fig_b, width="stretch")

    with sub[3]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[3])}'></div>", unsafe_allow_html=True)
        st.caption("欠損の偏りと数値列間の相関を点検し、学習データ品質の問題を見つけます。")
        st.markdown("**欠損率 上位列**")
        miss_ratio = flat_df.isna().mean().sort_values(ascending=False)
        miss_df = miss_ratio.reset_index()
        miss_df.columns = ["column", "missing_ratio"]
        _show_df(miss_df.head(40), hide_index=True)

        numeric_df = flat_df.select_dtypes(include=[np.number]).copy()
        if target_col and target_col in flat_df.columns:
            candidate_cols = [c for c in numeric_df.columns if c.startswith("metrics.") or c.startswith("params.")]
            candidate_cols = [c for c in candidate_cols if numeric_df[c].notna().sum() >= 8]
            if target_col not in candidate_cols:
                candidate_cols = [target_col, *candidate_cols]
            candidate_cols = candidate_cols[:15]
            dev = str(st.session_state.get("ui_compute_device", "auto"))
            gpu_info = _gpu_runtime_info()
            use_gpu = bool(dev == "gpu" or (dev == "auto" and gpu_info.get("cuda_available") and CUPY_AVAILABLE))
            corr_df = (
                _corr_matrix_fast(numeric_df[candidate_cols], use_gpu=use_gpu) if candidate_cols else pd.DataFrame()
            )
            if not corr_df.empty:
                st.markdown("**相関行列 (numeric)**")
                st.caption(f"計算デバイス: {'GPU' if use_gpu else 'CPU'}")
                _show_df(corr_df.reset_index(), hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig_corr = px.imshow(corr_df, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                    fig_corr.update_layout(height=520, title="相関ヒートマップ")
                    st.plotly_chart(fig_corr, width="stretch")
            else:
                st.info("相関計算可能な数値列が不足しています。")
        else:
            st.info("target 指標が未解決のため相関分析を省略しました。")

    with sub[4]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[4])}'></div>", unsafe_allow_html=True)
        st.caption("パラメータ別の影響量・寄与率・Shapley近似をまとめて確認します。")
        st.markdown("**パラメータ影響ランキング**")
        _show_df(impact_df, hide_index=True)
        if not impact_df.empty:
            top = impact_df.head(20).copy()
            top = top.sort_values("effect_abs", ascending=True)
            if PLOTLY_AVAILABLE:
                top["is_significant"] = top["is_significant"].fillna(False).map(
                    lambda value: "significant" if bool(value) else "not_significant"
                )
                fig_imp = _build_categorical_bar_figure(
                    top,
                    x="effect_abs",
                    y="parameter",
                    orientation="h",
                    color="is_significant",
                    color_map={"significant": "#0f766e", "not_significant": "#94a3b8"},
                    color_order=["not_significant", "significant"],
                    title="Top Parameter Effects",
                    height=480,
                )
                st.plotly_chart(fig_imp, width="stretch")
            else:
                st.bar_chart(top.set_index("parameter")[["effect_abs"]], height=420)

            contrib_df = _impact_contribution_rates(impact_df)
            if not contrib_df.empty:
                st.markdown("**寄与率（effect_abs正規化）**")
                _show_df(contrib_df.head(30), hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig_c = px.pie(
                        contrib_df.head(12),
                        names="parameter",
                        values="contribution_rate",
                        title="寄与率トップ12",
                    )
                    fig_c.update_layout(height=380)
                    st.plotly_chart(fig_c, width="stretch")

            cand_cols = [c for c in flat_df.columns if c.startswith("params.")]
            shapley_df = _approx_shapley_contrib(
                flat_df,
                target_col=target_col if isinstance(target_col, str) else "",
                candidate_cols=cand_cols,
                n_perm=100 if preset == "詳細" else 60,
                seed=42,
            )
            if not shapley_df.empty:
                st.markdown("**ゲーム理論近似: Shapley-R2寄与**")
                _show_df(shapley_df, hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig_s = px.bar(
                        shapley_df,
                        x="feature",
                        y="shapley_rate",
                        title="Shapley寄与率",
                    )
                    fig_s.update_layout(height=320)
                    st.plotly_chart(fig_s, width="stretch")

            graph_df = _graph_centrality_from_impact(impact_df)
            if not graph_df.empty:
                st.markdown("**グラフ理論指標（影響ネットワーク近似）**")
                _show_df(graph_df.head(30), hide_index=True)

            if target_col and isinstance(target_col, str):
                treat_col = str(impact_df.iloc[0]["parameter"]) if "parameter" in impact_df.columns else ""
                if treat_col:
                    ate = _causal_proxy_ate(flat_df, target_col=target_col, treatment_col=treat_col)
                    st.markdown("**因果推論(代理): 調整ATE推定**")
                    st.json(ate)

    with sub[5]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[5])}'></div>", unsafe_allow_html=True)
        st.caption("config設定値と実際のparamsを突合し、反映漏れや上書きを監査します。")
        st.markdown("**設定値反映監査（config -> params）**")
        if isinstance(reflection_df, pd.DataFrame) and not reflection_df.empty:
            _show_df(reflection_df, hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_ref = px.bar(
                    reflection_df.head(20),
                    x="parameter",
                    y="mismatch_rate",
                    title="設定反映 mismatch rate",
                )
                fig_ref.update_layout(height=320, yaxis_tickformat=".1%")
                st.plotly_chart(fig_ref, width="stretch")
        else:
            st.info("設定反映監査データがありません。meta設定列との結合がない可能性があります。")

        if isinstance(reflection_detail_df, pd.DataFrame) and not reflection_detail_df.empty:
            st.markdown("**不一致詳細（上位200件）**")
            _show_df(reflection_detail_df.head(200), hide_index=True)

    with sub[6]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[6])}'></div>", unsafe_allow_html=True)
        st.caption("寄与モデルとproxy因果推定を同時に確認し、改善アクション候補を抽出します。")
        st.markdown("**精度寄与（Permutation Importance）**")
        if isinstance(contribution_df, pd.DataFrame) and not contribution_df.empty:
            _show_df(contribution_df, hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_fc = px.bar(
                    contribution_df.head(20),
                    x="feature",
                    y="importance_mean",
                    title="評価指標への寄与要因",
                )
                fig_fc.update_layout(height=360)
                st.plotly_chart(fig_fc, width="stretch")
        else:
            st.info("寄与分析に必要なデータ量または特徴量が不足しています。")

        st.markdown("**因果候補（調整回帰による代理推定）**")
        if isinstance(causal_hints_df, pd.DataFrame) and not causal_hints_df.empty:
            _show_df(causal_hints_df, hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_ch = px.bar(
                    causal_hints_df.head(20),
                    x="treatment",
                    y="std_coef",
                    color="significant_0.05",
                    title="標準化効果量（proxy）",
                )
                fig_ch.update_layout(height=360)
                st.plotly_chart(fig_ch, width="stretch")
        else:
            st.info("因果候補推定はサンプル不足または説明変数不足のため算出されませんでした。")

    with sub[7]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[7])}'></div>", unsafe_allow_html=True)
        st.caption("再帰分割の根拠をツリーで追跡し、分析結果の経路を確認します。")
        tree = report.get("recursive_tree", {})
        rows = _flatten_recursive_tree_rows(tree if isinstance(tree, dict) else {})
        tree_df = pd.DataFrame(rows)
        _show_df(tree_df, hide_index=True)
        with st.expander("raw recursive tree json", expanded=False):
            st.json(tree)

    with sub[8]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[8])}'></div>", unsafe_allow_html=True)
        st.caption("統合済みの明細データを直接確認して、個別runの裏取りを行います。")
        sample_cols = [
            c
            for c in [
                "config_id",
                "run_id",
                "status",
                "model_name",
                "horizon",
                "dataset_rows",
                "feature_cols",
                "created_at",
                target_col if isinstance(target_col, str) else None,
            ]
            if c and c in flat_df.columns
        ]
        if sample_cols:
            _show_df(flat_df[sample_cols].head(500), hide_index=True)
        else:
            _show_df(flat_df.head(200), hide_index=True)

    with sub[9]:
        st.markdown(f"<div id='meta-deep-{_slug(tab_labels[9])}'></div>", unsafe_allow_html=True)
        st.caption("解析結果をJSON/CSVとして保存し、監査・再現・共有に利用します。")
        export_payload = {
            "overview": overview,
            "target_metric": target_col,
            "insights": insights,
            "stat_tests": report.get("stat_tests", {}),
            "bootstrap_target_ci": bootstrap_ci,
            "top_parameter_impact": impact_df.head(50).to_dict(orient="records")
            if isinstance(impact_df, pd.DataFrame)
            else [],
            "parameter_reflection": reflection_df.head(100).to_dict(orient="records")
            if isinstance(reflection_df, pd.DataFrame)
            else [],
            "feature_contribution": contribution_df.head(100).to_dict(orient="records")
            if isinstance(contribution_df, pd.DataFrame)
            else [],
            "causal_hints": causal_hints_df.head(100).to_dict(orient="records")
            if isinstance(causal_hints_df, pd.DataFrame)
            else [],
            "recursive_tree": report.get("recursive_tree", {}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        export_json = json.dumps(export_payload, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "Download report.json",
            data=export_json,
            file_name=f"meta_deep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="meta_deep_download_json",
        )
        if isinstance(metric_summary_df, pd.DataFrame) and not metric_summary_df.empty:
            st.download_button(
                "Download metric_summary.csv",
                data=metric_summary_df.to_csv(index=False),
                file_name="metric_summary.csv",
                mime="text/csv",
                key="meta_deep_download_metric_csv",
            )
        if isinstance(impact_df, pd.DataFrame) and not impact_df.empty:
            st.download_button(
                "Download parameter_impact.csv",
                data=impact_df.to_csv(index=False),
                file_name="parameter_impact.csv",
                mime="text/csv",
                key="meta_deep_download_impact_csv",
            )
        if isinstance(reflection_df, pd.DataFrame) and not reflection_df.empty:
            st.download_button(
                "Download parameter_reflection.csv",
                data=reflection_df.to_csv(index=False),
                file_name="parameter_reflection.csv",
                mime="text/csv",
                key="meta_deep_download_reflection_csv",
            )
        if isinstance(contribution_df, pd.DataFrame) and not contribution_df.empty:
            st.download_button(
                "Download feature_contribution.csv",
                data=contribution_df.to_csv(index=False),
                file_name="feature_contribution.csv",
                mime="text/csv",
                key="meta_deep_download_contrib_csv",
            )
        if isinstance(causal_hints_df, pd.DataFrame) and not causal_hints_df.empty:
            st.download_button(
                "Download causal_hints.csv",
                data=causal_hints_df.to_csv(index=False),
                file_name="causal_hints.csv",
                mime="text/csv",
                key="meta_deep_download_causal_csv",
            )
        if isinstance(flat_df, pd.DataFrame) and not flat_df.empty:
            st.download_button(
                "Download flattened_results.csv",
                data=flat_df.head(20000).to_csv(index=False),
                file_name="flattened_results.csv",
                mime="text/csv",
                key="meta_deep_download_flat_csv",
            )

    _log_dashboard_event(
        "meta_deep_analysis_end",
        {
            "rows": int(raw_df.shape[0]),
            "target_metric": str(target_col),
            "impact_rows": int(impact_df.shape[0]) if isinstance(impact_df, pd.DataFrame) else 0,
            "reflection_rows": int(reflection_df.shape[0]) if isinstance(reflection_df, pd.DataFrame) else 0,
            "contribution_rows": int(contribution_df.shape[0]) if isinstance(contribution_df, pd.DataFrame) else 0,
            "causal_rows": int(causal_hints_df.shape[0]) if isinstance(causal_hints_df, pd.DataFrame) else 0,
        },
    )


def _render_model_resource_relationships(engine: Engine, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("モデル・外生変数・リソース関連分析")
    st.caption("model/meta/resources を統合し、相関・寄与・検定・関係性を分析します。")
    rel_tab_labels = ["概要", "監査", "寄与", "因果", "ゲーム理論", "Export"]
    with st.expander("画面ガイド / クイック移動", expanded=False):
        guide_tabs = st.tabs(["タブ説明", "読み方", "クイックリンク"])
        with guide_tabs[0]:
            guide_rows = [
                {
                    "tab": "概要",
                    "目的": "全体状況・失敗率・CI・相関を把握",
                    "主出力": "failed rate / bootstrap CI / target description / correlations",
                },
                {"tab": "監査", "目的": "設定反映の不一致を検知", "主出力": "mismatch rate / mismatch details"},
                {
                    "tab": "寄与",
                    "目的": "評価指標へ寄与した要因を確認",
                    "主出力": "Permutation contribution / effect_abs",
                },
                {"tab": "因果", "目的": "proxy因果候補を抽出", "主出力": "causal hints / ATE proxy"},
                {"tab": "ゲーム理論", "目的": "説明の多面的比較", "主出力": "contribution rate / Shapley / centrality"},
                {"tab": "Export", "目的": "再利用可能な成果物を出力", "主出力": "json/csv download"},
            ]
            _show_df(pd.DataFrame(guide_rows), hide_index=True)
        with guide_tabs[1]:
            st.markdown(
                "\n".join(
                    [
                        "1. `概要` で failed rate, bootstrap CI, target description を確認",
                        "2. `監査` で config -> params 不一致の原因を特定",
                        "3. `寄与` と `因果` で改善施策の候補を抽出",
                        "4. `ゲーム理論` で説明の整合性を横断確認",
                        "5. `Export` で監査・再現用成果物を保存",
                    ]
                )
            )
        with guide_tabs[2]:
            st.caption(
                "Streamlitの制約でリンククリック時にタブ自体は自動選択されないため、上段タブ名をクリックして移動してください。"
            )
            st.markdown(" | ".join([f"[{t}](#model-rel-{_slug(t)})" for t in rel_tab_labels]))
    if ("model", "nf_automodel") not in tables:
        st.info("model.nf_automodel が存在しません。")
        return

    max_rows = int(max(200, min(100000, row_limit * 50)))
    use_parallel = bool(st.session_state.get("ui_enable_parallel_query", True))
    workers = int(st.session_state.get("ui_parallel_workers", 4))
    query_specs: dict[str, tuple[str, dict[str, Any] | None]] = {
        "model": (
            """
            SELECT *
            FROM model.nf_automodel
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"limit": max_rows},
        )
    }
    if ("meta", "nf_automodel") in tables:
        query_specs["meta"] = (
            """
            SELECT config_id, config_name, model_name, horizon, auto_backend, auto_num_samples,
                   auto_loss, auto_valid_loss, auto_search_alg, auto_cpus, auto_gpus,
                   auto_refit_with_val, auto_verbose, unified_filter_json, updated_at
            FROM meta.nf_automodel
            """,
            None,
        )
    if ("resources", "run") in tables:
        query_specs["run"] = (
            """
            SELECT run_id::text AS run_id, status AS run_status, started_at, ended_at,
                   rows_target, rows_written, rows_failed,
                   EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))::double precision AS run_duration_sec
            FROM resources.run
            ORDER BY started_at DESC
            LIMIT :limit
            """,
            {"limit": max_rows},
        )
    if ("resources", "stage_span") in tables:
        query_specs["stage"] = (
            """
            SELECT run_id::text AS run_id,
                   AVG(duration_ms)::double precision AS stage_avg_ms,
                   SUM(duration_ms)::double precision AS stage_total_ms,
                   AVG(gpu_util_avg)::double precision AS gpu_util_avg,
                   AVG(gpu_mem_used_mb_avg)::double precision AS gpu_mem_avg_mb
            FROM resources.stage_span
            GROUP BY run_id
            """,
            None,
        )

    if use_parallel:
        frames = _parallel_query_frames(engine, query_specs=query_specs, max_workers=workers)
    else:
        frames = {k: _query_df(engine, spec[0], params=spec[1]) for k, spec in query_specs.items()}

    model_df = frames.get("model", pd.DataFrame())
    if model_df.empty:
        st.info("model.nf_automodel にデータがありません。")
        return

    flat = _flatten_json_columns(
        model_df,
        {
            "metrics_json": "metrics",
            "params_json": "params",
            "exog_json": "exog",
            "diagnostics_json": "diag",
            "model_analyze_json": "analyze",
        },
    )

    if "run_id" in flat.columns:
        flat["run_id"] = flat["run_id"].astype(str)
    if "config_id" in flat.columns:
        flat["config_id"] = pd.to_numeric(flat["config_id"], errors="coerce")

    meta_df = frames.get("meta", pd.DataFrame())
    if not meta_df.empty and "config_id" in meta_df.columns:
        meta_df = meta_df.copy()
        meta_df["config_id"] = pd.to_numeric(meta_df["config_id"], errors="coerce")
        flat = flat.merge(
            meta_df.add_prefix("meta_"),
            left_on="config_id",
            right_on="meta_config_id",
            how="left",
        )
        cfg_map = {
            "meta_model_name": "cfg_model_name",
            "meta_horizon": "cfg_horizon",
            "meta_auto_num_samples": "cfg_auto_num_samples",
            "meta_auto_backend": "cfg_auto_backend",
            "meta_auto_loss": "cfg_auto_loss",
            "meta_auto_valid_loss": "cfg_auto_valid_loss",
            "meta_auto_search_alg": "cfg_auto_search_alg",
            "meta_auto_cpus": "cfg_auto_cpus",
            "meta_auto_gpus": "cfg_auto_gpus",
            "meta_auto_refit_with_val": "cfg_auto_refit_with_val",
            "meta_auto_verbose": "cfg_auto_verbose",
        }
        for src_col, dst_col in cfg_map.items():
            if src_col in flat.columns and dst_col not in flat.columns:
                flat[dst_col] = flat[src_col]

    run_df = frames.get("run", pd.DataFrame())
    if not run_df.empty and "run_id" in run_df.columns:
        run_df = run_df.copy()
        run_df["run_id"] = run_df["run_id"].astype(str)
        flat = flat.merge(run_df, on="run_id", how="left")

    stage_df = frames.get("stage", pd.DataFrame())
    if not stage_df.empty and "run_id" in stage_df.columns:
        stage_df = stage_df.copy()
        stage_df["run_id"] = stage_df["run_id"].astype(str)
        flat = flat.merge(stage_df, on="run_id", how="left")

    metric_candidates = [c for c in flat.columns if c.startswith("metrics.")]
    default_metric = (
        "metrics.mae" if "metrics.mae" in metric_candidates else (metric_candidates[0] if metric_candidates else "")
    )
    metric_col = st.selectbox(
        "分析対象メトリクス",
        [""] + metric_candidates,
        index=(metric_candidates.index(default_metric) + 1 if default_metric else 0),
    )
    top_n = st.slider("表示上位件数", min_value=5, max_value=80, value=20, step=1, key="model_rel_top_n")
    _show_df(flat.head(300), hide_index=True)

    if not metric_col:
        st.info("metrics.* 列がないため詳細分析をスキップします。")
        return
    flat[metric_col] = pd.to_numeric(flat[metric_col], errors="coerce")
    base = flat.dropna(subset=[metric_col]).copy()
    if base.empty:
        st.info("対象メトリクスに有効データがありません。")
        return

    # Exogenous and resource relationships
    exog_like_cols = [
        c
        for c in base.columns
        if c.startswith("exog.")
        or "exog" in c
        or c in ["run_duration_sec", "rows_written", "rows_failed", "stage_total_ms", "gpu_util_avg", "gpu_mem_avg_mb"]
    ]
    exog_like_cols = [c for c in exog_like_cols if c in base.columns]
    numeric_rel = base[exog_like_cols + [metric_col]].copy()
    for c in numeric_rel.columns:
        numeric_rel[c] = pd.to_numeric(numeric_rel[c], errors="coerce")
    corr_rows: list[dict[str, Any]] = []
    for c in exog_like_cols:
        s = numeric_rel[[c, metric_col]].dropna()
        if s.shape[0] < 8:
            continue
        pear = float(s[c].corr(s[metric_col]))
        spear = float(s[c].corr(s[metric_col], method="spearman"))
        corr_rows.append({"feature": c, "pearson": pear, "spearman": spear, "n": int(s.shape[0])})
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_spearman"] = corr_df["spearman"].abs()
        corr_df = corr_df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    # Model property tests and Bayesian summary
    bayes_df = pd.DataFrame()
    chi_square_summary: dict[str, Any] = {}
    if "model_name" in base.columns:
        bayes_df = _bayesian_success_posterior(base, group_col="model_name")
        if SCIPY_AVAILABLE and "status" in base.columns:
            ct = pd.crosstab(base["model_name"].astype(str), base["status"].astype(str))
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, dof, _ = spstats.chi2_contingency(ct.values)
                chi_square_summary = {
                    "chi2": float(chi2),
                    "pvalue": float(p),
                    "dof": int(dof),
                    "significant_0.05": bool(float(p) < 0.05),
                }

    # Advanced audit/contribution/causal analyses (same depth as meta deep analysis)
    reflection_df = pd.DataFrame()
    reflection_detail_df = pd.DataFrame()
    contribution_df = pd.DataFrame()
    causal_hints_df = pd.DataFrame()
    try:
        from loto_forecast.analysis import meta_automodel_report as mar

        reflection_df, reflection_detail_df = mar._build_parameter_reflection(base)  # noqa: SLF001
        contribution_df = mar._build_feature_contribution(  # noqa: SLF001
            base,
            target_metric=metric_col,
            max_features=max(20, int(top_n)),
            random_state=42,
        )
        causal_candidates = [
            c
            for c in base.columns
            if (
                c.startswith("params.")
                or c.startswith("exog.")
                or c
                in [
                    "run_duration_sec",
                    "rows_written",
                    "rows_failed",
                    "stage_total_ms",
                    "gpu_util_avg",
                    "gpu_mem_avg_mb",
                ]
            )
            and pd.to_numeric(base[c], errors="coerce").notna().sum() >= 30
        ]
        causal_hints_df = mar._build_causal_hints(  # noqa: SLF001
            base,
            target_metric=metric_col,
            candidate_cols=causal_candidates[:40],
            max_rows=max(10, int(top_n)),
        )
    except Exception as e:
        st.warning(f"advanced relationship analysis failed: {e}")

    # Contribution / game / graph / causal proxy
    impact_like = []
    for c in [x for x in base.columns if x.startswith("params.")][:30]:
        s = base[[c, metric_col]].dropna()
        if s.shape[0] < 10:
            continue
        if pd.api.types.is_numeric_dtype(s[c]):
            try:
                bins = pd.qcut(s[c], q=4, duplicates="drop")
                gnum = s.groupby(bins)[metric_col].mean()
                if gnum.shape[0] < 2:
                    continue
                spread = float(gnum.max() - gnum.min())
            except Exception:
                continue
        else:
            g = s.groupby(c)[metric_col].mean()
            if g.shape[0] < 2:
                continue
            spread = float(g.max() - g.min())
        impact_like.append({"parameter": c, "effect_abs": abs(spread), "is_significant": bool(abs(spread) > 0)})
    impact_like_df = (
        pd.DataFrame(impact_like).sort_values("effect_abs", ascending=False).reset_index(drop=True)
        if impact_like
        else pd.DataFrame()
    )
    contrib_game_df = _impact_contribution_rates(impact_like_df) if not impact_like_df.empty else pd.DataFrame()
    shapley_df = (
        _approx_shapley_contrib(
            base, target_col=metric_col, candidate_cols=impact_like_df["parameter"].tolist()[:8], n_perm=80, seed=42
        )
        if not impact_like_df.empty
        else pd.DataFrame()
    )
    graph_df = _graph_centrality_from_impact(impact_like_df) if not impact_like_df.empty else pd.DataFrame()
    ate_proxy: dict[str, Any] = (
        _causal_proxy_ate(base, target_col=metric_col, treatment_col=str(impact_like_df.iloc[0]["parameter"]))
        if not impact_like_df.empty
        else {}
    )
    target_values = pd.to_numeric(base[metric_col], errors="coerce").dropna()
    outlier_mask = _iqr_outlier_mask(base[metric_col]) if metric_col in base.columns else pd.Series(dtype=bool)
    bootstrap_ci = _bootstrap_mean_ci(target_values, alpha=0.05, n_iter=1200, seed=42)
    target_desc = (
        pd.to_numeric(base[metric_col], errors="coerce")
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_frame("value")
        .reset_index()
    )
    status_counts = (
        flat["status"].astype(str).value_counts().to_dict() if "status" in flat.columns and not flat.empty else {}
    )
    fail_n = int(status_counts.get("failed", 0) or 0)
    total_n = int(sum(int(v or 0) for v in status_counts.values()))

    st.markdown("**インサイト**")
    insights: list[str] = []
    if not reflection_df.empty:
        worst = reflection_df.iloc[0]
        insights.append(
            f"設定反映ミスマッチ最大: {worst.get('parameter')} ({float(worst.get('mismatch_rate', 0.0)) * 100.0:.1f}%)"
        )
    if not contribution_df.empty:
        top_contrib = contribution_df.iloc[0]
        insights.append(
            f"精度寄与トップ: {top_contrib.get('feature')} ({float(top_contrib.get('contribution_rate', 0.0)) * 100.0:.1f}%)"
        )
    if not causal_hints_df.empty:
        top_causal = causal_hints_df.iloc[0]
        insights.append(
            f"因果候補トップ: {top_causal.get('treatment')} (std_coef={float(top_causal.get('std_coef', 0.0)):.3f}, p={float(top_causal.get('pvalue', 1.0)):.4g})"
        )
    if insights:
        for i, item in enumerate(insights, start=1):
            st.write(f"{i}. {item}")
    else:
        st.write("insightなし")

    st.caption("クイック移動: " + " | ".join([f"`{t}`" for t in rel_tab_labels]))
    tabs = st.tabs(rel_tab_labels)

    with tabs[0]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[0])}'></div>", unsafe_allow_html=True)
        st.caption("概要では、meta_deep_analysisと同等粒度の基礎監査指標を表示します。")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("rows", int(flat.shape[0]))
        m2.metric("runs", int(flat["run_id"].nunique()) if "run_id" in flat.columns and not flat.empty else 0)
        m3.metric("configs", int(flat["config_id"].nunique()) if "config_id" in flat.columns and not flat.empty else 0)
        m4.metric("failed rate", f"{(fail_n / total_n * 100.0):.1f}%" if total_n > 0 else "n/a")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("対象指標", metric_col)
        c2.metric("有効サンプル", int(target_values.shape[0]))
        c3.metric("外れ値数(IQR)", int(outlier_mask.sum()) if not outlier_mask.empty else 0)
        ci_text = (
            f"{bootstrap_ci.get('ci_low'):.4g} .. {bootstrap_ci.get('ci_high'):.4g}"
            if bootstrap_ci.get("ci_low") is not None and bootstrap_ci.get("ci_high") is not None
            else "n/a"
        )
        c4.metric("平均95%CI(Bootstrap)", ci_text)

        st.markdown("**対象指標の記述統計**")
        _show_df(target_desc, hide_index=True)

        st.markdown("**外生変数・リソース相関**")
        if not corr_df.empty:
            _show_df(corr_df.head(top_n), hide_index=True)
            if PLOTLY_AVAILABLE:
                fig = px.bar(corr_df.head(top_n), x="feature", y="spearman", title=f"{metric_col} とのSpearman相関")
                fig.update_layout(height=340)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("相関を算出できる外生/リソース列がありません。")

        if not bayes_df.empty:
            st.markdown("**モデル別 ベイズ成功確率**")
            _show_df(bayes_df.head(top_n), hide_index=True)
        if chi_square_summary:
            st.markdown("**統計検定: model_name と status の独立性(Chi-square)**")
            st.json(chi_square_summary)

    with tabs[1]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[1])}'></div>", unsafe_allow_html=True)
        st.caption("設定反映監査: config値と実行時paramsの不一致を検知します。")
        st.markdown("**設定反映監査（config -> params）**")
        if not reflection_df.empty:
            _show_df(reflection_df.head(max(20, top_n)), hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_ref = px.bar(
                    reflection_df.head(max(20, top_n)),
                    x="parameter",
                    y="mismatch_rate",
                    title="設定反映 mismatch rate",
                )
                fig_ref.update_layout(height=320, yaxis_tickformat=".1%")
                st.plotly_chart(fig_ref, width="stretch")
        else:
            st.info("設定反映監査の比較可能列が不足しています。")
        if not reflection_detail_df.empty:
            st.markdown("**不一致詳細（上位200件）**")
            _show_df(reflection_detail_df.head(200), hide_index=True)

    with tabs[2]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[2])}'></div>", unsafe_allow_html=True)
        st.caption("寄与モデル: 評価指標に効いた要素をPermutation Importanceで可視化します。")
        st.markdown("**寄与モデル（Permutation Importance）**")
        if not contribution_df.empty:
            _show_df(contribution_df.head(max(20, top_n)), hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_fc = px.bar(
                    contribution_df.head(max(20, top_n)),
                    x="feature",
                    y="importance_mean",
                    title=f"{metric_col} への寄与要因",
                )
                fig_fc.update_layout(height=360)
                st.plotly_chart(fig_fc, width="stretch")
        else:
            st.info("寄与モデルはサンプル不足または特徴量不足で算出できませんでした。")
        if not impact_like_df.empty:
            st.markdown("**補助指標: パラメータ別 effect_abs**")
            _show_df(impact_like_df.head(max(20, top_n)), hide_index=True)

    with tabs[3]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[3])}'></div>", unsafe_allow_html=True)
        st.caption("proxy因果: 調整回帰ベースの効果量と有意性を確認します。")
        st.markdown("**proxy因果（調整回帰）**")
        if not causal_hints_df.empty:
            _show_df(causal_hints_df.head(max(20, top_n)), hide_index=True)
            if PLOTLY_AVAILABLE:
                fig_ch = px.bar(
                    causal_hints_df.head(max(20, top_n)),
                    x="treatment",
                    y="std_coef",
                    color="significant_0.05",
                    title="標準化効果量（proxy）",
                )
                fig_ch.update_layout(height=360)
                st.plotly_chart(fig_ch, width="stretch")
        else:
            st.info("proxy因果は有効な候補変数が不足しているため算出できませんでした。")
        if ate_proxy:
            st.markdown("**proxy ATE（単一treatment）**")
            st.json(ate_proxy)

    with tabs[4]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[4])}'></div>", unsafe_allow_html=True)
        st.caption("ゲーム理論/グラフ理論: 寄与率・Shapley・中心性で説明を多面的に評価します。")
        st.markdown("**ゲーム理論・グラフ理論 近似分析**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("寄与率")
            _show_df(contrib_game_df.head(10), hide_index=True)
        with c2:
            st.markdown("Shapley-R2")
            _show_df(shapley_df.head(10), hide_index=True)
        with c3:
            st.markdown("中心性")
            _show_df(graph_df.head(10), hide_index=True)

    with tabs[5]:
        st.markdown(f"<div id='model-rel-{_slug(rel_tab_labels[5])}'></div>", unsafe_allow_html=True)
        st.caption("解析結果をJSON/CSVでエクスポートし、再現・共有を容易にします。")
        export_payload = {
            "metric_col": metric_col,
            "correlations": corr_df.head(200).to_dict(orient="records") if not corr_df.empty else [],
            "bayesian_success": bayes_df.head(200).to_dict(orient="records") if not bayes_df.empty else [],
            "chi_square": chi_square_summary,
            "reflection": reflection_df.head(200).to_dict(orient="records") if not reflection_df.empty else [],
            "feature_contribution": contribution_df.head(200).to_dict(orient="records")
            if not contribution_df.empty
            else [],
            "causal_hints": causal_hints_df.head(200).to_dict(orient="records") if not causal_hints_df.empty else [],
            "game_contribution": contrib_game_df.head(200).to_dict(orient="records")
            if not contrib_game_df.empty
            else [],
            "shapley": shapley_df.head(200).to_dict(orient="records") if not shapley_df.empty else [],
            "graph_centrality": graph_df.head(200).to_dict(orient="records") if not graph_df.empty else [],
            "ate_proxy": ate_proxy,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        st.download_button(
            "Download model_rel_advanced_report.json",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2, default=str),
            file_name=f"model_rel_advanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="model_rel_adv_json",
        )
        if not reflection_df.empty:
            st.download_button(
                "Download model_rel_parameter_reflection.csv",
                data=reflection_df.to_csv(index=False),
                file_name="model_rel_parameter_reflection.csv",
                mime="text/csv",
                key="model_rel_reflection_csv",
            )
        if not contribution_df.empty:
            st.download_button(
                "Download model_rel_feature_contribution.csv",
                data=contribution_df.to_csv(index=False),
                file_name="model_rel_feature_contribution.csv",
                mime="text/csv",
                key="model_rel_contribution_csv",
            )
        if not causal_hints_df.empty:
            st.download_button(
                "Download model_rel_causal_hints.csv",
                data=causal_hints_df.to_csv(index=False),
                file_name="model_rel_causal_hints.csv",
                mime="text/csv",
                key="model_rel_causal_csv",
            )
        if not contrib_game_df.empty:
            st.download_button(
                "Download model_rel_game_contribution.csv",
                data=contrib_game_df.to_csv(index=False),
                file_name="model_rel_game_contribution.csv",
                mime="text/csv",
                key="model_rel_game_contrib_csv",
            )
        if not shapley_df.empty:
            st.download_button(
                "Download model_rel_shapley.csv",
                data=shapley_df.to_csv(index=False),
                file_name="model_rel_shapley.csv",
                mime="text/csv",
                key="model_rel_shapley_csv",
            )
        if not graph_df.empty:
            st.download_button(
                "Download model_rel_graph_centrality.csv",
                data=graph_df.to_csv(index=False),
                file_name="model_rel_graph_centrality.csv",
                mime="text/csv",
                key="model_rel_graph_csv",
            )


@st.cache_data(show_spinner=False)
def _load_forecast_parquet_cached(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)


_safe_read_json_file = dashboard_helpers.safe_read_json_file
_artifact_file_stats = dashboard_helpers.artifact_file_stats
_has_model_artifacts = dashboard_helpers.has_model_artifacts
_has_analysis_bundle = dashboard_helpers.has_analysis_bundle


def _discover_run_directories(require_model_artifacts: bool = False) -> list[Path]:
    art = PROJECT_ROOT / "artifacts"
    if not art.exists() or not art.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(art.iterdir(), key=lambda x: x.name, reverse=True):
        if not p.is_dir():
            continue
        if require_model_artifacts and not _has_model_artifacts(p):
            continue
        if (p / "meta.json").exists() or (p / "forecast.parquet").exists():
            out.append(p)
    return out


def _render_feature_verification_lab(engine: Engine | None, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("機能動作確認ラボ")
    st.caption("NeuralForecast/AutoModel の設定反映、runtime kwargs、外生変数対応、保存/読込を段階的に検証します。")

    from loto_forecast.models.neuralforecast_model import (
        AUTO_MODEL_NAMES,
        load_model,
        model_exog_support_table,
        validate_runtime_kwargs,
    )
    from loto_forecast.models.registry import get_adapter

    run_df = pd.DataFrame()
    if engine is not None and ("model", "nf_automodel") in tables:
        try:
            run_df = _query_df(
                engine,
                """
                SELECT run_id, config_id, status, model_name, created_at, artifact_path, model_store_path, params_json, exog_json
                FROM model.nf_automodel
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"limit": int(max(100, row_limit))},
            )
        except Exception as e:
            st.warning(f"model.nf_automodel 読み込み失敗: {e}")

    run_dirs = _discover_run_directories()
    run_dir_map = {p.name: p for p in run_dirs}
    run_id_options = list(
        dict.fromkeys(
            [str(x) for x in run_df.get("run_id", pd.Series(dtype=str)).dropna().astype(str).tolist()]
            + list(run_dir_map.keys())
        )
    )

    st.caption("クイック移動: `ガイド` | `API・設定監査` | `Autoformer専用チェック` | `保存物・スモーク`")
    c0, c1 = st.columns([2, 1])
    selected_run_id = c0.selectbox("run_id", [""] + run_id_options, index=0, key="verify_run_id")
    run_source = c1.selectbox(
        "run source", ["artifacts", "model_store_path", "artifact_path"], index=0, key="verify_run_source"
    )

    selected_run_dir: Path | None = None
    selected_row = pd.Series(dtype=object)
    if selected_run_id:
        if not run_df.empty and "run_id" in run_df.columns:
            matched = run_df[run_df["run_id"].astype(str) == str(selected_run_id)]
            if not matched.empty:
                selected_row = matched.iloc[0]
        if run_source == "artifacts":
            selected_run_dir = run_dir_map.get(selected_run_id)
        elif run_source == "model_store_path":
            raw = str(selected_row.get("model_store_path", "") or "").strip()
            selected_run_dir = Path(raw).expanduser() if raw else None
        else:
            raw = str(selected_row.get("artifact_path", "") or "").strip()
            selected_run_dir = Path(raw).expanduser() if raw else None
        if selected_run_dir is not None and not selected_run_dir.is_absolute():
            selected_run_dir = (PROJECT_ROOT / selected_run_dir).resolve()

    selected_meta = _safe_read_json_file(selected_run_dir / "meta.json") if selected_run_dir else {}
    selected_eval = _safe_read_json_file(selected_run_dir / "evaluation.json") if selected_run_dir else {}

    lab_tabs = st.tabs(["ガイド", "API・設定監査", "Autoformer専用チェック", "保存物・スモーク"])

    with lab_tabs[0]:
        gtab = st.tabs(["全体像", "チェック項目", "補助情報"])
        with gtab[0]:
            st.markdown(
                "\n".join(
                    [
                        "1. `API・設定監査`: NeuralForecast/AutoModel の仕様と設定値を検証",
                        "2. `Autoformer専用チェック`: futr/hist/stat exog の実入力整合を監査",
                        "3. `保存物・スモーク`: run保存物監査 + load/predict_insample 動作確認",
                    ]
                )
            )
        with gtab[1]:
            checklist_rows = [
                {"領域": "API", "確認内容": "fit/predict/cv/save/load シグネチャ一致", "判定": "シグネチャ表"},
                {"領域": "設定", "確認内容": "runtime kwargs の許可キー/禁止キー", "判定": "validation report"},
                {"領域": "Autoformer", "確認内容": "futr/hist/stat exog 実入力整合", "判定": "role別監査表"},
                {"領域": "保存物", "確認内容": "meta/evaluation/forecast/checkpoint 存在", "判定": "artifact監査"},
            ]
            _show_df(pd.DataFrame(checklist_rows), hide_index=True)
        with gtab[2]:
            st.caption("選択runがある場合は `meta.json` の設定を自動読込し、検証入力の初期値に反映します。")
            st.json(
                {
                    "selected_run_id": selected_run_id or None,
                    "selected_run_dir": str(selected_run_dir) if selected_run_dir else None,
                    "meta_loaded": bool(selected_meta),
                    "evaluation_loaded": bool(selected_eval),
                }
            )

    with lab_tabs[1]:
        st.markdown("<div id='verify-api-settings'></div>", unsafe_allow_html=True)
        st.caption("モデル対応表・APIシグネチャ・runtime kwargs・AutoModel params を段階的に検証します。")
        support_df = pd.DataFrame(model_exog_support_table())
        st.markdown("**モデル別 外生変数サポート表 (F/H/S)**")
        _show_df(support_df, hide_index=True)
        if not support_df.empty and PLOTLY_AVAILABLE:
            support_counts = pd.DataFrame(
                [
                    {"role": "futr", "supported_models": int(support_df["supports_futr_exog"].sum())},
                    {"role": "hist", "supported_models": int(support_df["supports_hist_exog"].sum())},
                    {"role": "stat", "supported_models": int(support_df["supports_stat_exog"].sum())},
                ]
            )
            fig_sup = px.bar(support_counts, x="role", y="supported_models", title="外生変数ロール別 サポートモデル数")
            fig_sup.update_layout(height=320)
            st.plotly_chart(fig_sup, width="stretch")

        api_tabs = st.tabs(["APIシグネチャ", "runtime kwargs 検証", "AutoModel パラメータ検証"])
        with api_tabs[0]:
            sig_rows: list[dict[str, Any]] = []
            try:
                from neuralforecast import NeuralForecast

                sig_rows.append(
                    {"symbol": "NeuralForecast.fit", "signature": str(inspect.signature(NeuralForecast.fit))}
                )
                sig_rows.append(
                    {"symbol": "NeuralForecast.predict", "signature": str(inspect.signature(NeuralForecast.predict))}
                )
                sig_rows.append(
                    {
                        "symbol": "NeuralForecast.cross_validation",
                        "signature": str(inspect.signature(NeuralForecast.cross_validation)),
                    }
                )
                sig_rows.append(
                    {"symbol": "NeuralForecast.save", "signature": str(inspect.signature(NeuralForecast.save))}
                )
                sig_rows.append(
                    {"symbol": "NeuralForecast.load", "signature": str(inspect.signature(NeuralForecast.load))}
                )
            except Exception as e:
                sig_rows.append({"symbol": "NeuralForecast", "signature": f"import failed: {e}"})
            try:
                from neuralforecast.models import Autoformer

                sig_rows.append(
                    {"symbol": "Autoformer.__init__", "signature": str(inspect.signature(Autoformer.__init__))}
                )
            except Exception as e:
                sig_rows.append({"symbol": "Autoformer", "signature": f"import failed: {e}"})
            try:
                from neuralforecast.common._base_auto import BaseAuto

                sig_rows.append({"symbol": "BaseAuto.__init__", "signature": str(inspect.signature(BaseAuto.__init__))})
            except Exception as e:
                sig_rows.append({"symbol": "BaseAuto", "signature": f"import failed: {e}"})
            _show_df(pd.DataFrame(sig_rows), hide_index=True)

        with api_tabs[1]:
            default_runtime_obj: dict[str, Any] = {
                "nf_fit_kwargs": {"val_size": 0, "verbose": False},
                "nf_predict_kwargs": {"h": settings.default_horizon},
                "nf_cross_validation_kwargs": {},
                "nf_save_kwargs": {"save_dataset": False, "overwrite": True},
                "nf_load_kwargs": {"verbose": False},
                "nf_predict_insample_kwargs": {"step_size": 1},
            }
            if selected_run_dir is not None:
                if isinstance(selected_meta.get("nf_runtime_kwargs_raw"), dict) and selected_meta.get(
                    "nf_runtime_kwargs_raw"
                ):
                    default_runtime_obj = dict(selected_meta["nf_runtime_kwargs_raw"])
                elif isinstance(selected_meta.get("nf_runtime_kwargs"), dict) and selected_meta.get(
                    "nf_runtime_kwargs"
                ):
                    default_runtime_obj = dict(selected_meta["nf_runtime_kwargs"])
            runtime_raw = st.text_area(
                "runtime kwargs JSON",
                value=json.dumps(default_runtime_obj, ensure_ascii=False, indent=2),
                height=220,
                key="verify_runtime_json",
            )
            runtime_report: dict[str, Any] = {
                "ok": False,
                "errors": ["not validated yet"],
                "warnings": [],
                "normalized": {},
            }
            if st.button("runtime kwargs を検証", key="verify_runtime_btn"):
                try:
                    parsed_runtime = json.loads(runtime_raw) if runtime_raw.strip() else {}
                    runtime_report = validate_runtime_kwargs(parsed_runtime if isinstance(parsed_runtime, dict) else {})
                    _publish_notification(
                        kind=(
                            NotificationEventKind.OPERATION_SUCCESS
                            if bool(runtime_report.get("ok", False))
                            else NotificationEventKind.OPERATION_FAILURE
                        ),
                        severity=(
                            NotificationSeverity.SUCCESS
                            if bool(runtime_report.get("ok", False))
                            else NotificationSeverity.WARNING
                        ),
                        title="runtime kwargs 検証を実行しました",
                        message="runtime kwargs の検証結果を更新しました。",
                        action="verify_runtime_kwargs",
                        status="success" if bool(runtime_report.get("ok", False)) else "warning",
                        command_summary="validate runtime kwargs",
                        error_summary="\n".join([str(x) for x in runtime_report.get("errors", [])[:3]]),
                    )
                except Exception as e:
                    runtime_report = {"ok": False, "errors": [str(e)], "warnings": [], "normalized": {}}
                    _publish_notification(
                        kind=NotificationEventKind.EXCEPTION,
                        severity=NotificationSeverity.FAILURE,
                        title="runtime kwargs 検証で例外が発生しました",
                        message="入力 JSON か validator 実装を確認してください。",
                        action="verify_runtime_kwargs",
                        status="failed",
                        command_summary="validate runtime kwargs",
                        error_summary=str(e),
                    )
                st.session_state["verify_runtime_report"] = runtime_report
            if "verify_runtime_report" in st.session_state:
                runtime_report = dict(st.session_state["verify_runtime_report"])
            st.json(runtime_report)
            if PLOTLY_AVAILABLE:
                rsum = pd.DataFrame(
                    [
                        {"type": "errors", "count": int(len(runtime_report.get("errors", [])))},
                        {"type": "warnings", "count": int(len(runtime_report.get("warnings", [])))},
                    ]
                )
                fig_r = px.bar(rsum, x="type", y="count", title="runtime kwargs 検証サマリ")
                fig_r.update_layout(height=260)
                st.plotly_chart(fig_r, width="stretch")

        with api_tabs[2]:
            adapter = get_adapter("neuralforecast_auto")
            model_name = st.selectbox("model", sorted(list(AUTO_MODEL_NAMES)), index=0, key="verify_model_name")
            params_raw = st.text_area(
                "model_params JSON",
                value=json.dumps(
                    {
                        "backend": "optuna",
                        "num_samples": 10,
                        "search_alg_name": "BasicVariantGenerator",
                        "cpus": 1,
                        "gpus": 0,
                        "refit_with_val": False,
                        "verbose": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                height=200,
                key="verify_model_params_json",
            )
            if st.button("model params を検証", key="verify_model_btn"):
                try:
                    parsed_params = json.loads(params_raw) if params_raw.strip() else {}
                    if not isinstance(parsed_params, dict):
                        raise ValueError("model_params must be JSON object")
                    st.session_state["verify_model_report"] = adapter.validate(
                        model_name=model_name, model_params=parsed_params
                    )
                    report = st.session_state["verify_model_report"]
                    _publish_notification(
                        kind=(
                            NotificationEventKind.OPERATION_SUCCESS
                            if bool(report.get("ok", False))
                            else NotificationEventKind.OPERATION_FAILURE
                        ),
                        severity=NotificationSeverity.SUCCESS if bool(report.get("ok", False)) else NotificationSeverity.WARNING,
                        title="model params 検証を実行しました",
                        message="モデルパラメータ検証結果を更新しました。",
                        action="verify_model_params",
                        status="success" if bool(report.get("ok", False)) else "warning",
                        command_summary=f"validate model params {model_name}",
                        error_summary="\n".join([str(x) for x in report.get("errors", [])[:3]]),
                    )
                except Exception as e:
                    st.session_state["verify_model_report"] = {"ok": False, "errors": [str(e)], "warnings": []}
                    _publish_notification(
                        kind=NotificationEventKind.EXCEPTION,
                        severity=NotificationSeverity.FAILURE,
                        title="model params 検証で例外が発生しました",
                        message="入力 JSON またはモデル選択を確認してください。",
                        action="verify_model_params",
                        status="failed",
                        command_summary=f"validate model params {model_name}",
                        error_summary=str(e),
                    )
            if "verify_model_report" in st.session_state:
                st.json(st.session_state["verify_model_report"])

    with lab_tabs[2]:
        st.markdown("<div id='verify-autoformer-audit'></div>", unsafe_allow_html=True)
        st.caption("Autoformer専用: futr/hist/stat exog の設定値・実使用値・実入力列の整合を監査します。")

        def _to_list(v: Any) -> list[str]:
            if v is None:
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str):
                raw = v.strip()
                if not raw:
                    return []
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, list):
                        return [str(x).strip() for x in obj if str(x).strip()]
                except Exception:
                    pass
                return [x.strip() for x in raw.split(",") if x.strip()]
            return []

        autoformer_sig: inspect.Signature | None = None
        autoformer_error = ""
        try:
            from neuralforecast.models import Autoformer

            autoformer_sig = inspect.signature(Autoformer.__init__)
        except Exception as e:
            autoformer_error = str(e)

        role_map = {
            "futr_exog": "futr_exog_list",
            "hist_exog": "hist_exog_list",
            "stat_exog": "stat_exog_list",
        }
        af_supported: dict[str, bool] = {}
        if autoformer_sig is not None:
            af_params = set(autoformer_sig.parameters.keys())
            af_supported = {role: (param in af_params) for role, param in role_map.items()}
        else:
            af_supported = {role: False for role in role_map}

        row_params = _parse_json_like(selected_row.get("params_json")) if not selected_row.empty else None
        row_params = row_params if isinstance(row_params, dict) else {}
        row_exog = _parse_json_like(selected_row.get("exog_json")) if not selected_row.empty else None
        row_exog = row_exog if isinstance(row_exog, dict) else {}
        meta_exog = selected_meta.get("exog", {}) if isinstance(selected_meta.get("exog", {}), dict) else {}
        model_params_meta = (
            selected_meta.get("model_params", {}) if isinstance(selected_meta.get("model_params", {}), dict) else {}
        )

        source_cols: list[str] = []
        source_label = "none"
        if engine is not None and tables:
            candidates = [(s, t) for s, t in sorted(tables) if s in {"dataset", "exog"}]
            if candidates:
                default_pair = ("dataset", "loto_y_ts_unified")
                default_idx = candidates.index(default_pair) if default_pair in candidates else 0
                source_label = st.selectbox(
                    "実入力列の監査ソース",
                    [f"{s}.{t}" for s, t in candidates],
                    index=default_idx,
                    key="verify_autoformer_source_table",
                )
                schema_name, table_name = source_label.split(".", 1)
                try:
                    cols_df = _table_columns(engine, schema_name, table_name)
                    if not cols_df.empty and "column_name" in cols_df.columns:
                        source_cols = sorted(cols_df["column_name"].astype(str).dropna().unique().tolist())
                except Exception as e:
                    st.warning(f"監査ソース列取得に失敗: {e}")

        source_set = set(source_cols)
        audit_rows: list[dict[str, Any]] = []
        used_sets: dict[str, set[str]] = {}
        for role, param_name in role_map.items():
            cfg = list(
                dict.fromkeys(
                    _to_list(model_params_meta.get(param_name))
                    + _to_list(row_params.get(param_name))
                    + _to_list(meta_exog.get(role))
                    + _to_list(row_exog.get(role))
                )
            )
            used = list(dict.fromkeys(_to_list(meta_exog.get(role)) + _to_list(row_exog.get(role))))
            used_sets[role] = set(used)
            missing_used = [c for c in used if source_set and c not in source_set]
            mismatch = sorted(list(set(cfg) ^ set(used)))
            status = "ok"
            if missing_used:
                status = "error"
            elif mismatch:
                status = "warn"
            elif used and not af_supported.get(role, False):
                status = "error"
            audit_rows.append(
                {
                    "role": role,
                    "autoformer_param": param_name,
                    "supported_by_signature": bool(af_supported.get(role, False)),
                    "configured_count": int(len(cfg)),
                    "used_count": int(len(used)),
                    "missing_in_source_count": int(len(missing_used)),
                    "config_used_mismatch_count": int(len(mismatch)),
                    "status": status,
                    "configured_cols": ", ".join(cfg[:20]),
                    "used_cols": ", ".join(used[:20]),
                    "missing_in_source": ", ".join(missing_used[:20]),
                }
            )
        audit_df = pd.DataFrame(audit_rows)
        st.markdown("**Autoformer exog 整合監査**")
        _show_df(audit_df, hide_index=True)

        if PLOTLY_AVAILABLE and not audit_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                bar_df = audit_df[["role", "configured_count", "used_count", "missing_in_source_count"]].melt(
                    id_vars=["role"],
                    var_name="metric",
                    value_name="count",
                )
                fig_a1 = _build_categorical_bar_figure(
                    bar_df,
                    x="role",
                    y="count",
                    color="metric",
                    color_map={
                        "configured_count": "#0f766e",
                        "used_count": "#2563eb",
                        "missing_in_source_count": "#b91c1c",
                    },
                    color_order=["configured_count", "used_count", "missing_in_source_count"],
                    barmode="group",
                    title="role別 exog 数量監査",
                    height=320,
                )
                st.plotly_chart(fig_a1, width="stretch")
            with c2:
                st_df = (
                    _normalize_status_series(audit_df["status"], default="unknown")
                    .value_counts()
                    .rename_axis("status")
                    .reset_index(name="count")
                )
                fig_a2 = px.bar(st_df, x="status", y="count", title="監査ステータス件数")
                fig_a2.update_layout(height=320)
                st.plotly_chart(fig_a2, width="stretch")

        overlap_rows: list[dict[str, Any]] = []
        roles = list(role_map.keys())
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                a = roles[i]
                b = roles[j]
                ov = sorted(list(used_sets.get(a, set()) & used_sets.get(b, set())))
                if ov:
                    overlap_rows.append(
                        {"role_a": a, "role_b": b, "overlap_count": int(len(ov)), "overlap_cols": ", ".join(ov[:20])}
                    )
        if overlap_rows:
            st.markdown("**role間の重複列（注意）**")
            _show_df(pd.DataFrame(overlap_rows), hide_index=True)
        else:
            st.info("futr/hist/stat の重複列は検出されませんでした。")

        st.json(
            {
                "run_id": selected_run_id or None,
                "model_name": selected_meta.get("model_name", selected_row.get("model_name")),
                "source_table": source_label,
                "source_col_count": int(len(source_cols)),
                "autoformer_signature_loaded": bool(autoformer_sig is not None),
                "autoformer_signature_error": autoformer_error or None,
            }
        )

    with lab_tabs[3]:
        st.markdown("<div id='verify-artifacts-smoke'></div>", unsafe_allow_html=True)
        st.caption("選択runの保存物監査と load_model / predict_insample の動作確認を行います。")
        if selected_run_dir is None:
            st.info("run_id を選択すると保存物監査と読込テストを実行できます。")
            return
        selected_run_dir = selected_run_dir.expanduser().resolve()
        file_stats = _artifact_file_stats(selected_run_dir)

        smoke_tabs = st.tabs(["保存物監査", "load/predict_insample スモーク"])
        with smoke_tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("exists", "yes" if file_stats.get("exists") else "no")
            c2.metric("file_count", int(file_stats.get("file_count", 0)))
            c3.metric("total_size", _format_bytes(file_stats.get("total_bytes", 0)))
            c4.metric("run_dir", selected_run_dir.name)
            checks = [
                {"check": "meta.json", "ok": bool((selected_run_dir / "meta.json").exists())},
                {"check": "forecast.parquet", "ok": bool((selected_run_dir / "forecast.parquet").exists())},
                {"check": "evaluation.json", "ok": bool((selected_run_dir / "evaluation.json").exists())},
                {"check": "configuration.pkl", "ok": bool((selected_run_dir / "configuration.pkl").exists())},
                {"check": "alias_to_model.pkl", "ok": bool((selected_run_dir / "alias_to_model.pkl").exists())},
                {"check": "checkpoint exists", "ok": bool(_has_model_artifacts(selected_run_dir))},
                {"check": "nf_runtime_kwargs in meta", "ok": isinstance(selected_meta.get("nf_runtime_kwargs"), dict)},
                {
                    "check": "model_exog_support in meta",
                    "ok": isinstance(selected_meta.get("model_exog_support"), dict),
                },
                {"check": "metrics in evaluation", "ok": isinstance(selected_eval.get("metrics"), dict)},
            ]
            _show_df(pd.DataFrame(checks), hide_index=True)
            if PLOTLY_AVAILABLE:
                ext_df = pd.DataFrame(
                    [
                        {"ext": k, "count": int(v)}
                        for k, v in sorted(
                            (file_stats.get("ext_counts", {}) or {}).items(), key=lambda x: (-int(x[1]), str(x[0]))
                        )
                    ]
                )
                if not ext_df.empty:
                    fig_ext = px.bar(ext_df.head(20), x="ext", y="count", title="artifact 拡張子分布")
                    fig_ext.update_layout(height=300)
                    st.plotly_chart(fig_ext, width="stretch")
            if selected_meta:
                with st.expander("meta.json (抜粋)", expanded=False):
                    st.json(
                        {
                            "run_id": selected_meta.get("run_id"),
                            "model_name": selected_meta.get("model_name"),
                            "h": selected_meta.get("h"),
                            "backend": selected_meta.get("backend"),
                            "num_samples": selected_meta.get("num_samples"),
                            "strict_exog": selected_meta.get("strict_exog"),
                            "exog": selected_meta.get("exog", {}),
                            "nf_runtime_kwargs": selected_meta.get("nf_runtime_kwargs", {}),
                            "cross_validation": selected_meta.get("cross_validation", {}),
                        }
                    )
            show_file_n = st.slider(
                "表示するファイル件数", min_value=20, max_value=500, value=120, step=10, key="verify_file_n"
            )
            _show_df(pd.DataFrame(file_stats.get("files", [])).head(int(show_file_n)), hide_index=True)

        with smoke_tabs[1]:
            c5, c6 = st.columns(2)
            do_predict_insample = c5.toggle("predict_insample も実行", value=False, key="verify_load_insample")
            insample_step = c6.number_input(
                "insample step_size", min_value=1, max_value=365, value=1, step=1, key="verify_insample_step"
            )
            load_kwargs_raw = st.text_input(
                "load kwargs JSON (optional)",
                value=json.dumps(
                    selected_meta.get("nf_runtime_kwargs", {}).get("nf_load_kwargs", {}), ensure_ascii=False
                ),
                key="verify_load_kwargs_raw",
            )
            if st.button("load_model スモークテスト実行", key="verify_load_btn"):
                t0 = time.perf_counter()
                load_report: dict[str, Any] = {"ok": False, "run_dir": str(selected_run_dir)}
                try:
                    load_kwargs = json.loads(load_kwargs_raw) if str(load_kwargs_raw).strip() else {}
                    if not isinstance(load_kwargs, dict):
                        raise ValueError("load kwargs must be JSON object")
                    nf = load_model(selected_run_dir, load_kwargs=load_kwargs)
                    model_names = [type(m).__name__ for m in getattr(nf, "models", [])]
                    load_report.update({"ok": True, "model_count": int(len(model_names)), "model_names": model_names})
                    if do_predict_insample:
                        ins_kwargs = {"step_size": int(insample_step)}
                        ins_kwargs.update(
                            dict(selected_meta.get("nf_runtime_kwargs", {}).get("nf_predict_insample_kwargs", {}))
                        )
                        ins = nf.predict_insample(**ins_kwargs)
                        load_report["predict_insample"] = {
                            "ok": True,
                            "rows": int(len(ins)),
                            "columns": [str(c) for c in ins.columns],
                            "kwargs": ins_kwargs,
                        }
                    _publish_notification(
                        kind=NotificationEventKind.OPERATION_SUCCESS,
                        severity=NotificationSeverity.SUCCESS,
                        title="load_model スモークテストが完了しました",
                        message="モデル読込の検証が完了しました。必要なら predict_insample 結果も確認してください。",
                        action="verify_load_model",
                        status="success",
                        command_summary=f"load_model {selected_run_dir}",
                        metadata={"predict_insample": bool(do_predict_insample)},
                    )
                except Exception as e:
                    load_report["error"] = str(e)
                    _publish_notification(
                        kind=NotificationEventKind.OPERATION_FAILURE,
                        severity=NotificationSeverity.FAILURE,
                        title="load_model スモークテストに失敗しました",
                        message="保存物、load kwargs、モデル互換性を確認してください。",
                        action="verify_load_model",
                        status="failed",
                        command_summary=f"load_model {selected_run_dir}",
                        error_summary=str(e),
                    )
                report["elapsed_sec"] = float(time.perf_counter() - t0)
                st.session_state["verify_load_report"] = load_report
            if "verify_load_report" in st.session_state:
                st.json(st.session_state["verify_load_report"])


def _render_model_analysis_lab(engine: Engine | None, tables: set[tuple[str, str]], row_limit: int) -> None:
    st.subheader("モデル解析ラボ")
    st.caption("保存済み forecast/evaluation/model を用いて、誤差構造・地平別劣化・残差検定・区間被覆率を分析します。")
    top_tabs = st.tabs(["ガイド/ナビ", "解析ワークスペース"])

    with top_tabs[0]:
        gtabs = st.tabs(["全体像", "タブ説明", "クイックリンク"])
        with gtabs[0]:
            st.markdown(
                "\n".join(
                    [
                        "1. run_id は **実モデル保存物(ckpt/pkl)が存在するもののみ** 表示します。",
                        "2. forecast/evaluation/model artifact を統合し、誤差構造・地平別劣化・残差検定・区間被覆率を解析します。",
                        "3. 実測テーブルが接続可能なら自動突合し、無い場合は forecast 単体分析にフォールバックします。",
                    ]
                )
            )
        with gtabs[1]:
            rows = [
                {"tab": "概要", "目的": "品質サマリと記述統計の確認", "主出力": "MAE/RMSE/MAPE, describe"},
                {"tab": "誤差", "目的": "誤差分布と時系列挙動の確認", "主出力": "actual/pred line, residual hist"},
                {"tab": "地平別", "目的": "horizon増加時の劣化確認", "主出力": "horizon-wise mae/rmse"},
                {"tab": "残差検定", "目的": "自己相関/定常性の点検", "主出力": "ADF, Ljung-Box"},
                {"tab": "予測区間", "目的": "区間被覆率の妥当性確認", "主出力": "coverage, avg width"},
                {
                    "tab": "保存物",
                    "目的": "artifact整合性・同一config比較",
                    "主出力": "artifact list, Bayesian summary",
                },
            ]
            _show_df(pd.DataFrame(rows), hide_index=True)
        with gtabs[2]:
            labels = ["概要", "誤差", "地平別", "残差検定", "予測区間", "保存物"]
            st.caption(
                "Streamlitの制約でリンククリック時にタブ自体は自動選択されないため、上段タブ名をクリックして移動してください。"
            )
            st.markdown(" | ".join([f"[{t}](#model-analysis-{_slug(t)})" for t in labels]))

    with top_tabs[1]:
        run_df = pd.DataFrame()
        if engine is not None and ("model", "nf_automodel") in tables:
            try:
                run_df = _query_df(
                    engine,
                    """
                    SELECT run_id, config_id, model_name, status, created_at, artifact_path, model_store_path, metrics_json, params_json
                    FROM model.nf_automodel
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """,
                    {"limit": int(max(300, row_limit))},
                )
            except Exception as e:
                st.warning(f"model.nf_automodel 読み込み失敗: {e}")

        run_dir_map: dict[str, Path] = {}
        for p in _discover_run_directories(require_model_artifacts=True):
            if _has_analysis_bundle(p):
                run_dir_map[p.name] = p

        if not run_df.empty:
            for _, row in run_df.iterrows():
                rid = str(row.get("run_id", "") or "").strip()
                if not rid:
                    continue
                if rid in run_dir_map:
                    continue
                for col in ("artifact_path", "model_store_path"):
                    raw = str(row.get(col, "") or "").strip()
                    if not raw:
                        continue
                    cand = Path(raw).expanduser()
                    cand = cand if cand.is_absolute() else (PROJECT_ROOT / cand).resolve()
                    if _has_analysis_bundle(cand):
                        run_dir_map[rid] = cand
                        break

        ordered_from_db = (
            [str(x) for x in run_df.get("run_id", pd.Series(dtype=str)).dropna().astype(str).tolist()]
            if not run_df.empty
            else []
        )
        run_ids = [rid for rid in ordered_from_db if rid in run_dir_map]
        for rid in sorted(run_dir_map.keys(), reverse=True):
            if rid not in run_ids:
                run_ids.append(rid)

        if not run_ids:
            st.warning("実モデル保存物を持つ run_id が見つかりません。")
            return

        selected_run_id = st.selectbox("解析対象 run_id", run_ids, index=0, key="analysis_run_id")
        run_dir = run_dir_map.get(selected_run_id)
        if run_dir is None:
            st.warning("run directory を解決できませんでした。")
            return
        run_dir = run_dir.expanduser().resolve()

        selected_row = pd.Series(dtype=object)
        if not run_df.empty and "run_id" in run_df.columns:
            matched = run_df[run_df["run_id"].astype(str) == str(selected_run_id)]
            if not matched.empty:
                selected_row = matched.iloc[0]

        meta_json = _safe_read_json_file(run_dir / "meta.json")
        eval_json = _safe_read_json_file(run_dir / "evaluation.json")
        forecast_df = _load_forecast_parquet_cached(str(run_dir / "forecast.parquet"))
        if not forecast_df.empty and settings.time_col in forecast_df.columns:
            forecast_df[settings.time_col] = pd.to_datetime(forecast_df[settings.time_col], errors="coerce")

        stats = _artifact_file_stats(run_dir)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("artifact files", int(stats.get("file_count", 0)))
        c2.metric("artifact size", _format_bytes(stats.get("total_bytes", 0)))
        c3.metric("forecast rows", int(len(forecast_df)))
        c4.metric("status", str(selected_row.get("status", "n/a")))

        overview = {
            "run_id": selected_run_id,
            "run_dir": str(run_dir),
            "model_name": meta_json.get("model_name", selected_row.get("model_name")),
            "h": meta_json.get("h"),
            "backend": meta_json.get("backend"),
            "num_samples": meta_json.get("num_samples"),
            "saved_metrics": eval_json.get("metrics", {}),
        }
        st.json(overview)

        if forecast_df.empty:
            st.warning("forecast.parquet が存在しないため、モデル解析を続行できません。")
            return

        time_col = settings.time_col if settings.time_col in forecast_df.columns else "ds"
        id_col = settings.id_col if settings.id_col in forecast_df.columns else "unique_id"
        numeric_cols = [c for c in forecast_df.columns if pd.api.types.is_numeric_dtype(forecast_df[c])]
        pred_candidates = [c for c in numeric_cols if c not in {settings.target_col, "y"}]
        if not pred_candidates:
            st.warning("予測列を特定できません。")
            return
        default_pred = (
            meta_json.get("model_name") if isinstance(meta_json.get("model_name"), str) else pred_candidates[0]
        )
        pred_col = st.selectbox(
            "予測列",
            pred_candidates,
            index=pred_candidates.index(default_pred) if default_pred in pred_candidates else 0,
            key="analysis_pred_col",
        )

        work = forecast_df.copy()
        if id_col in work.columns:
            uid_options = sorted(work[id_col].dropna().astype(str).unique().tolist())
            uid_sel = st.selectbox("unique_id", ["__ALL__"] + uid_options, index=0, key="analysis_uid")
            if uid_sel != "__ALL__":
                work = work[work[id_col].astype(str) == str(uid_sel)].copy()
        else:
            uid_sel = "__ALL__"
        work = work.dropna(subset=[time_col]).sort_values([id_col, time_col] if id_col in work.columns else [time_col])
        if work.empty:
            st.warning("選択条件で予測データが0件です。")
            return

        st.markdown("**予測データ(抜粋)**")
        preview_cols = [c for c in [id_col, time_col, pred_col, settings.target_col, "y"] if c in work.columns]
        _show_df(work[preview_cols].head(200), hide_index=True)

        actual_df = pd.DataFrame()
        actual_schema_table_options: list[tuple[str, str]] = []
        if engine is not None:
            if ("dataset", "loto_y_ts_unified") in tables:
                actual_schema_table_options.append(("dataset", "loto_y_ts_unified"))
            if (settings.db_schema, settings.db_table) in tables and (
                settings.db_schema,
                settings.db_table,
            ) not in actual_schema_table_options:
                actual_schema_table_options.append((settings.db_schema, settings.db_table))

        if actual_schema_table_options and engine is not None:
            default_idx = 0
            if (settings.db_schema, settings.db_table) in actual_schema_table_options:
                default_idx = actual_schema_table_options.index((settings.db_schema, settings.db_table))
            src_label = st.selectbox(
                "実測ソース",
                [f"{s}.{t}" for s, t in actual_schema_table_options],
                index=default_idx,
                key="analysis_actual_source",
            )
            src_schema, src_table = src_label.split(".", 1)
            actual_id_col = st.text_input("実測 id_col", value=settings.id_col, key="analysis_actual_id")
            actual_time_col = st.text_input("実測 time_col", value=settings.time_col, key="analysis_actual_time")
            actual_target_col = st.text_input(
                "実測 target_col", value=settings.target_col, key="analysis_actual_target"
            )
            ds_min = pd.to_datetime(work[time_col].min(), errors="coerce")
            ds_max = pd.to_datetime(work[time_col].max(), errors="coerce")
            where_parts = [f"{_safe_ident(actual_time_col)} >= :ds_min", f"{_safe_ident(actual_time_col)} <= :ds_max"]
            params: dict[str, Any] = {"ds_min": ds_min, "ds_max": ds_max}
            if uid_sel != "__ALL__":
                where_parts.append(f"{_safe_ident(actual_id_col)} = :uid")
                params["uid"] = str(uid_sel)
            sql = f"""
            SELECT
              {_safe_ident(actual_id_col)} AS id_col,
              {_safe_ident(actual_time_col)} AS ds_col,
              {_safe_ident(actual_target_col)} AS y_col
            FROM {_safe_ident(src_schema)}.{_safe_ident(src_table)}
            WHERE {" AND ".join(where_parts)}
            ORDER BY {_safe_ident(actual_time_col)}
            """
            try:
                actual_df = _query_df(engine, sql, params=params)
                if not actual_df.empty:
                    actual_df["ds_col"] = pd.to_datetime(actual_df["ds_col"], errors="coerce")
                    actual_df["y_col"] = pd.to_numeric(actual_df["y_col"], errors="coerce")
            except Exception as e:
                st.warning(f"実測データ取得失敗: {e}")
                actual_df = pd.DataFrame()
        else:
            st.info("DB接続または実測テーブルが無いため、誤差解析は forecast 単体情報のみ表示します。")

        merged = pd.DataFrame()
        if not actual_df.empty:
            pred_merge = work.copy()
            pred_merge = pred_merge.rename(columns={id_col: "id_col", time_col: "ds_col", pred_col: "y_pred"})
            keep_cols = [c for c in ["id_col", "ds_col", "y_pred"] if c in pred_merge.columns]
            interval_cols = [
                c for c in pred_merge.columns if c.startswith(f"{pred_col}-lo-") or c.startswith(f"{pred_col}-hi-")
            ]
            keep_cols.extend(interval_cols)
            pred_merge = pred_merge[keep_cols]
            merged = actual_df.merge(pred_merge, on=["id_col", "ds_col"], how="inner")
            merged = merged.dropna(subset=["y_col", "y_pred"]).sort_values(["id_col", "ds_col"])

        tab_labels = ["概要", "誤差", "地平別", "残差検定", "予測区間", "保存物"]
        st.caption("クイック移動: " + " | ".join([f"`{t}`" for t in tab_labels]))
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[0])}'></div>", unsafe_allow_html=True)
            st.caption("まず品質指標と記述統計を確認し、解析対象として妥当かを判断します。")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("forecast rows", int(len(work)))
            c2.metric("actual rows", int(len(actual_df)))
            c3.metric("overlap rows", int(len(merged)))
            c4.metric("unique_id", int(work[id_col].nunique()) if id_col in work.columns else 1)
            if not merged.empty:
                y = pd.to_numeric(merged["y_col"], errors="coerce")
                yp = pd.to_numeric(merged["y_pred"], errors="coerce")
                mae = float(np.mean(np.abs(y - yp)))
                rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
                mape = float(np.nanmean(np.abs((y - yp) / y.replace(0, np.nan))) * 100.0)
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{mae:.6g}")
                m2.metric("RMSE", f"{rmse:.6g}")
                m3.metric("MAPE(%)", f"{mape:.4g}")
                target_desc = (
                    pd.to_numeric(merged["y_col"], errors="coerce")
                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                    .to_frame("actual")
                )
                pred_desc = (
                    pd.to_numeric(merged["y_pred"], errors="coerce")
                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
                    .to_frame("pred")
                )
                desc = target_desc.join(pred_desc, how="outer").reset_index(names="stat")
                st.markdown("**記述統計**")
                _show_df(desc, hide_index=True)
            elif eval_json:
                st.markdown("**evaluation.json**")
                st.json(eval_json)

        with tabs[1]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[1])}'></div>", unsafe_allow_html=True)
            st.caption("actual vs pred と残差分布を同時に確認し、系統誤差を検出します。")
            if merged.empty:
                st.info("誤差解析は overlap rows が必要です。")
            else:
                plot_df = merged.copy()
                plot_df["residual"] = plot_df["y_col"] - plot_df["y_pred"]
                _show_df(plot_df.head(200), hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_df["ds_col"], y=plot_df["y_col"], mode="lines", name="actual"))
                    fig.add_trace(go.Scatter(x=plot_df["ds_col"], y=plot_df["y_pred"], mode="lines", name="pred"))
                    fig.update_layout(height=360, title=f"actual vs pred ({selected_run_id})")
                    st.plotly_chart(fig, width="stretch")
                    fig_r = px.histogram(plot_df, x="residual", nbins=50, title="residual distribution")
                    fig_r.update_layout(height=320)
                    st.plotly_chart(fig_r, width="stretch")
                else:
                    st.line_chart(plot_df.set_index("ds_col")[["y_col", "y_pred"]], height=360)

        with tabs[2]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[2])}'></div>", unsafe_allow_html=True)
            st.caption("予測地平(horizon)の増加に伴う精度劣化を定量化します。")
            if merged.empty:
                st.info("地平別解析は overlap rows が必要です。")
            else:
                horizon_df = merged.copy()
                horizon_df["horizon"] = horizon_df.groupby("id_col").cumcount() + 1
                grp = (
                    horizon_df.groupby("horizon", as_index=False)
                    .agg(
                        n=("y_col", "count"),
                        mae=("y_col", lambda s: float(np.mean(np.abs(s - horizon_df.loc[s.index, "y_pred"])))),
                        rmse=("y_col", lambda s: float(np.sqrt(np.mean((s - horizon_df.loc[s.index, "y_pred"]) ** 2)))),
                    )
                    .sort_values("horizon")
                )
                _show_df(grp.head(200), hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig_h = px.line(grp, x="horizon", y=["mae", "rmse"], markers=True, title="horizon-wise error")
                    fig_h.update_layout(height=360)
                    st.plotly_chart(fig_h, width="stretch")
                else:
                    st.line_chart(grp.set_index("horizon")[["mae", "rmse"]], height=320)

        with tabs[3]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[3])}'></div>", unsafe_allow_html=True)
            st.caption("残差の定常性と自己相関を検定し、モデル仮定の破綻を検知します。")
            if merged.empty:
                st.info("残差検定は overlap rows が必要です。")
            else:
                from loto_forecast.analysis.diagnostics import adf_test, ljung_box

                resid = pd.to_numeric(merged["y_col"], errors="coerce") - pd.to_numeric(
                    merged["y_pred"], errors="coerce"
                )
                adf = adf_test(pd.Series(resid))
                lb_lag = st.slider("Ljung-Box lag", min_value=5, max_value=80, value=20, step=1, key="analysis_lb_lag")
                lb = ljung_box(pd.Series(resid), lags=int(lb_lag))
                st.json({"adf": adf, "ljung_box": lb})

        with tabs[4]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[4])}'></div>", unsafe_allow_html=True)
            st.caption("予測区間の実測被覆率と区間幅を確認し、不確実性の妥当性を監査します。")
            if merged.empty:
                st.info("区間評価は overlap rows が必要です。")
            else:
                interval_pairs: list[tuple[str, str, str]] = []
                for c in merged.columns:
                    if c.startswith(f"{pred_col}-lo-"):
                        level = c.replace(f"{pred_col}-lo-", "")
                        hi = f"{pred_col}-hi-{level}"
                        if hi in merged.columns:
                            interval_pairs.append((level, c, hi))
                if not interval_pairs:
                    st.info("予測区間列が見つかりませんでした。")
                else:
                    coverage_rows: list[dict[str, Any]] = []
                    for level, lo_col, hi_col in interval_pairs:
                        y = pd.to_numeric(merged["y_col"], errors="coerce")
                        lo = pd.to_numeric(merged[lo_col], errors="coerce")
                        hi = pd.to_numeric(merged[hi_col], errors="coerce")
                        ok = y.notna() & lo.notna() & hi.notna()
                        n = int(ok.sum())
                        if n <= 0:
                            continue
                        cov = float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean())
                        width = float((hi[ok] - lo[ok]).mean())
                        coverage_rows.append(
                            {"level": str(level), "n": n, "empirical_coverage": cov, "avg_width": width}
                        )
                    cov_df = (
                        pd.DataFrame(coverage_rows).sort_values("level") if coverage_rows else pd.DataFrame()
                    )
                    _show_df(cov_df, hide_index=True)
                    if PLOTLY_AVAILABLE and not cov_df.empty:
                        fig_cov = px.bar(cov_df, x="level", y="empirical_coverage", title="interval coverage")
                        fig_cov.update_layout(height=320, yaxis_range=[0, 1])
                        st.plotly_chart(fig_cov, width="stretch")

        with tabs[5]:
            st.markdown(f"<div id='model-analysis-{_slug(tab_labels[5])}'></div>", unsafe_allow_html=True)
            st.caption("artifact一覧と同一config比較で、run再現性と劣化傾向を監査します。")
            st.markdown("**artifact files**")
            _show_df(pd.DataFrame(stats.get("files", [])).head(300), hide_index=True)
            if not run_df.empty and "config_id" in run_df.columns and pd.notna(selected_row.get("config_id")):
                cfg_id = int(selected_row.get("config_id"))
                cfg_runs = run_df[run_df["config_id"].fillna(-1).astype(int) == cfg_id].copy()
                if not cfg_runs.empty:
                    st.markdown(f"**同一config_id={cfg_id} の runs**")
                    cfg_flat = _flatten_json_columns(cfg_runs, {"metrics_json": "metrics", "params_json": "params"})
                    _show_df(cfg_flat.head(max(100, int(row_limit))), hide_index=True)
                    bayes = _bayesian_success_posterior(cfg_runs, group_col="model_name")
                    if not bayes.empty:
                        st.markdown("**model別 ベイズ成功確率**")
                        _show_df(bayes, hide_index=True)


def _render_actual_vs_forecast(engine: Engine | None, tables: set[tuple[str, str]]) -> None:
    st.subheader("実測値 vs 予測値")
    st.caption("artifact の `forecast.parquet` と実測テーブルを突合して作図します。")

    forecast_files = sorted(PROJECT_ROOT.glob("artifacts/cfg*/forecast.parquet"))
    if not forecast_files:
        st.info("forecast.parquet が見つかりません。")
        return
    run_ids = [p.parent.name for p in forecast_files]
    run_id = st.selectbox("run_id", run_ids, index=len(run_ids) - 1 if run_ids else 0)
    p = next((x for x in forecast_files if x.parent.name == run_id), forecast_files[-1])
    fdf = _load_forecast_parquet_cached(str(p))
    if fdf.empty:
        st.warning("予測データの読み込みに失敗しました。")
        return
    if "ds" in fdf.columns:
        fdf["ds"] = pd.to_datetime(fdf["ds"], errors="coerce")
    id_col = (
        "unique_id"
        if "unique_id" in fdf.columns
        else (
            fdf.select_dtypes(include=["object"]).columns[0]
            if len(fdf.select_dtypes(include=["object"]).columns) > 0
            else None
        )
    )
    pred_cols = [c for c in fdf.columns if c not in {"ds", "unique_id"} and pd.api.types.is_numeric_dtype(fdf[c])]
    if not pred_cols:
        st.warning("予測列が見つかりません。")
        return
    pred_col = st.selectbox("予測列", pred_cols, index=0)
    if id_col and id_col in fdf.columns:
        uid = st.selectbox("unique_id", sorted(fdf[id_col].dropna().astype(str).unique().tolist()), index=0)
        pred = fdf[fdf[id_col].astype(str) == str(uid)].copy()
    else:
        uid = None
        pred = fdf.copy()
    pred = pred.dropna(subset=["ds"])
    pred = pred.sort_values("ds")
    pred = pred.rename(columns={pred_col: "y_pred"})

    actual = pd.DataFrame()
    if engine is not None and ("dataset", "loto_y_ts_unified") in tables and uid is not None:
        ds_min = pred["ds"].min()
        ds_max = pred["ds"].max()
        hist_days = st.slider("実測表示の過去日数", min_value=14, max_value=720, value=180, step=7, key="af_hist_days")
        ds_from = (ds_min - pd.Timedelta(days=int(hist_days))) if pd.notna(ds_min) else None
        params: dict[str, Any] = {"uid": str(uid)}
        where = ["unique_id = :uid"]
        if ds_from is not None:
            params["ds_from"] = ds_from
            where.append("ds >= :ds_from")
        if pd.notna(ds_max):
            params["ds_to"] = ds_max
            where.append("ds <= :ds_to")
        sql = f"""
        SELECT ds, y, unique_id
        FROM dataset.loto_y_ts_unified
        WHERE {" AND ".join(where)}
        ORDER BY ds
        """
        try:
            actual = _query_df(engine, sql, params=params)
            if not actual.empty:
                actual["ds"] = pd.to_datetime(actual["ds"], errors="coerce")
                actual["y"] = pd.to_numeric(actual["y"], errors="coerce")
        except Exception:
            actual = pd.DataFrame()

    st.markdown("**予測データ**")
    _show_df(pred.head(200), hide_index=True)
    if actual.empty:
        st.info("実測データを取得できなかったため、予測のみ表示します。")
        plot_df = pred[["ds", "y_pred"]].dropna()
        if not plot_df.empty:
            st.line_chart(plot_df.set_index("ds")[["y_pred"]], height=340)
        return

    merged = actual.merge(pred[["ds", "y_pred"]], on="ds", how="outer").sort_values("ds")
    st.markdown("**実測+予測 結合データ**")
    _show_df(merged.tail(400), hide_index=True)
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], mode="lines+markers", name="actual"))
        fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y_pred"], mode="lines+markers", name="pred"))
        fig.update_layout(height=420, title=f"実測値 vs 予測値 ({run_id})")
        st.plotly_chart(fig, width="stretch")
    else:
        st.line_chart(merged.set_index("ds")[["y", "y_pred"]], height=380)
    ov = merged.dropna(subset=["y", "y_pred"]).copy()
    if not ov.empty:
        mae = float(np.mean(np.abs(ov["y"] - ov["y_pred"])))
        rmse = float(np.sqrt(np.mean((ov["y"] - ov["y_pred"]) ** 2)))
        mape = float(np.mean(np.abs((ov["y"] - ov["y_pred"]) / ov["y"].replace(0, np.nan))) * 100.0)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.5g}")
        c2.metric("RMSE", f"{rmse:.5g}")
        c3.metric("MAPE(%)", f"{mape:.3f}")


def _render_async_xai_api_runner() -> None:
    st.markdown("### Async XAI API 連携")
    st.caption("`/loops/submit -> /tasks/{id} -> /evaluations/{id}/contract|drift` を1画面で実行・監視・可視化します。")

    c1, c2, c3 = st.columns([2, 1, 1])
    base_url = c1.text_input(
        "API base URL",
        value=str(st.session_state.get("runner_async_api_base_url", "http://127.0.0.1:8000")),
        key="runner_async_api_base_url",
    ).strip()
    timeout_sec = float(
        c2.number_input(
            "api timeout(sec)",
            min_value=1.0,
            max_value=120.0,
            value=float(st.session_state.get("runner_async_api_timeout_sec", 10.0)),
            step=1.0,
            key="runner_async_api_timeout_sec",
        )
    )
    health_auto = c3.toggle("auto health", value=False, key="runner_async_health_auto")

    health_key = "runner_async_health"
    if st.button("Health check", key="runner_async_health_btn") or bool(health_auto):
        health_res = _api_get_json(base_url, "/health", timeout_sec=timeout_sec)
        st.session_state[health_key] = health_res
        _publish_notification(
            kind=(
                NotificationEventKind.OPERATION_SUCCESS
                if bool(health_res.get("ok", False))
                else NotificationEventKind.OPERATION_FAILURE
            ),
            severity=NotificationSeverity.SUCCESS if bool(health_res.get("ok", False)) else NotificationSeverity.WARNING,
            title="Async XAI health check を実行しました",
            message="API health の取得結果を更新しました。",
            action="async_xai_health",
            status="success" if bool(health_res.get("ok", False)) else "warning",
            command_summary=f"GET {base_url}/health",
            error_summary=str(health_res.get("error") or ""),
        )
    health_state = st.session_state.get(health_key)
    if isinstance(health_state, dict):
        st.json(
            {
                "ok": bool(health_state.get("ok", False)),
                "status": health_state.get("status"),
                "elapsed_ms": round(float(health_state.get("elapsed_ms", 0.0) or 0.0), 1),
                "data": health_state.get("data"),
                "error": health_state.get("error"),
            }
        )

    st.markdown("**1) 再帰ループ投入 (`POST /loops/submit`)**")
    s1, s2, s3, s4 = st.columns(4)
    kind = s1.selectbox("kind", ["train", "analyze", "eval"], index=0, key="runner_async_kind")
    recursive_depth = int(
        s2.number_input(
            "recursive depth", min_value=1, max_value=64, value=3, step=1, key="runner_async_recursive_depth"
        )
    )
    num_cpus = float(
        s3.number_input("num cpus", min_value=0.1, max_value=64.0, value=2.0, step=0.1, key="runner_async_num_cpus")
    )
    num_gpus = float(
        s4.number_input("num gpus", min_value=0.0, max_value=8.0, value=0.0, step=0.1, key="runner_async_num_gpus")
    )
    callable_path = st.text_input(
        "callable (package.module:function)",
        value="loto_forecast.pipeline_hooks:demo_train_and_predict",
        key="runner_async_callable",
    )

    p1, p2, p3 = st.columns(3)
    strategy = p1.selectbox("strategy", ["seed_increment"], index=0, key="runner_async_strategy")
    seed_key = p2.text_input("seed key", value="seed", key="runner_async_seed_key")
    seed_start = int(
        p3.number_input("seed start", min_value=0, max_value=1000000, value=1, step=1, key="runner_async_seed_start")
    )
    seed_step = int(
        st.number_input("seed step", min_value=1, max_value=1000000, value=1, step=1, key="runner_async_seed_step")
    )

    params_json = st.text_area(
        "params json",
        value=json.dumps(
            {
                "dataset_id": "demo_dataset",
                "n": 240,
                "resource_interval_s": 1.0,
                "interval_coverage": 0.9,
                "attribution_top_k": 5,
                "drift_bins": 10,
            },
            ensure_ascii=False,
            indent=2,
        ),
        height=170,
        key="runner_async_params_json",
    )
    if st.button("Submit recursive loop", key="runner_async_submit_loop", type="primary"):
        params_obj, parse_err = _parse_json_dict_input(params_json, default={})
        if parse_err:
            st.error(parse_err)
            _publish_notification(
                kind=NotificationEventKind.OPERATION_FAILURE,
                severity=NotificationSeverity.FAILURE,
                title="再帰ループ投入に失敗しました",
                message="params json を修正してから再実行してください。",
                action="async_xai_submit",
                status="failed",
                command_summary=f"POST {base_url}/loops/submit",
                error_summary=parse_err,
            )
        else:
            payload = {
                "kind": str(kind),
                "callable": str(callable_path),
                "params": params_obj or {},
                "recursive_depth": int(recursive_depth),
                "strategy": str(strategy),
                "seed_key": str(seed_key),
                "seed_start": int(seed_start),
                "seed_step": int(seed_step),
                "num_cpus": float(num_cpus),
                "num_gpus": float(num_gpus),
            }
            res = _api_post_json(base_url, "/loops/submit", payload=payload, timeout_sec=timeout_sec)
            st.session_state["runner_async_last_submit"] = res
            if bool(res.get("ok", False)) and isinstance(res.get("data"), dict):
                st.session_state["runner_async_last_loop"] = dict(res.get("data") or {})
                _publish_notification(
                    kind=NotificationEventKind.OPERATION_SUCCESS,
                    severity=NotificationSeverity.SUCCESS,
                    title="再帰ループ投入が完了しました",
                    message="loop_id を取得しました。次は tasks と evaluation の追跡に進めます。",
                    action="async_xai_submit",
                    status="success",
                    command_summary=f"POST {base_url}/loops/submit",
                    metadata={"loop_id": (res.get("data") or {}).get("loop_id")},
                )
                _log_dashboard_event(
                    "runner_async_loop_submit_ok",
                    {
                        "base_url": base_url,
                        "loop_id": (res.get("data") or {}).get("loop_id"),
                        "recursive_depth": (res.get("data") or {}).get("recursive_depth"),
                    },
                )
            else:
                _publish_notification(
                    kind=NotificationEventKind.OPERATION_FAILURE,
                    severity=NotificationSeverity.FAILURE,
                    title="再帰ループ投入に失敗しました",
                    message="API 応答が失敗でした。status と error を確認してください。",
                    action="async_xai_submit",
                    status="failed",
                    command_summary=f"POST {base_url}/loops/submit",
                    error_summary=str(res.get("error") or ""),
                )
                _log_dashboard_event(
                    "runner_async_loop_submit_failed",
                    {"base_url": base_url, "status": res.get("status"), "error": str(res.get("error"))},
                    level="ERROR",
                )
    if "runner_async_last_submit" in st.session_state:
        st.markdown("**last submit response**")
        st.json(st.session_state["runner_async_last_submit"])

    last_loop = st.session_state.get("runner_async_last_loop", {})
    if isinstance(last_loop, dict) and last_loop:
        st.markdown("**last loop summary**")
        st.json(last_loop)

    st.markdown("**2) タスク監視 (`GET /tasks`, `GET /tasks/{id}`)**")
    l1, l2 = st.columns([1, 1])
    tasks_limit = int(
        l1.number_input(
            "tasks list limit", min_value=1, max_value=500, value=80, step=1, key="runner_async_tasks_limit"
        )
    )
    if l2.button("Refresh tasks list", key="runner_async_refresh_tasks"):
        res = _api_get_json(base_url, "/tasks", timeout_sec=timeout_sec, params={"limit": tasks_limit})
        st.session_state["runner_async_tasks_res"] = res
        rows = (res.get("data") or {}).get("rows") if isinstance(res.get("data"), dict) else []
        st.session_state["runner_async_tasks_df"] = pd.DataFrame(rows if isinstance(rows, list) else [])

    tasks_df = st.session_state.get("runner_async_tasks_df", pd.DataFrame())
    if isinstance(tasks_df, pd.DataFrame) and not tasks_df.empty:
        _show_df(tasks_df.head(300), hide_index=True)

    task_ids: list[str] = []
    if isinstance(last_loop, dict):
        task_ids.extend([str(x) for x in list(last_loop.get("task_ids", [])) if str(x)])
    if isinstance(tasks_df, pd.DataFrame) and "id" in tasks_df.columns:
        task_ids.extend([str(x) for x in tasks_df["id"].astype(str).tolist() if str(x)])
    seen: set[str] = set()
    unique_task_ids: list[str] = []
    for task_id in task_ids:
        if task_id in seen:
            continue
        seen.add(task_id)
        unique_task_ids.append(task_id)
    task_ids = unique_task_ids

    selected_task_id = st.selectbox(
        "selected task id",
        task_ids if task_ids else [""],
        index=0,
        key="runner_async_selected_task_id",
    )
    c_eval1, c_eval2, c_eval3 = st.columns(3)
    manual_eval_id = c_eval1.text_input(
        "manual evaluation id(optional)", value="", key="runner_async_manual_eval_id"
    ).strip()
    pred_limit = int(
        c_eval2.number_input(
            "predictions limit", min_value=100, max_value=200000, value=10000, step=100, key="runner_async_pred_limit"
        )
    )
    res_limit = int(
        c_eval3.number_input(
            "resource limit", min_value=100, max_value=50000, value=5000, step=100, key="runner_async_res_limit"
        )
    )

    if st.button("Fetch task + evaluation + contract + drift", key="runner_async_fetch_detail"):
        detail: dict[str, Any] = {"base_url": base_url, "task_id": selected_task_id}
        if selected_task_id:
            task_res = _api_get_json(base_url, f"/tasks/{selected_task_id}", timeout_sec=timeout_sec)
            detail["task_res"] = task_res
            task_obj = task_res.get("data") if isinstance(task_res.get("data"), dict) else {}
            eval_id_val: int | None = None
            if manual_eval_id:
                try:
                    eval_id_val = int(manual_eval_id)
                except Exception:
                    eval_id_val = None
            if eval_id_val is None and isinstance(task_obj, dict):
                try:
                    task_result = task_obj.get("result")
                    if isinstance(task_result, dict) and task_result.get("evaluation_id") is not None:
                        eval_id_val = int(task_result["evaluation_id"])
                    else:
                        eval_id_val = None
                except Exception:
                    eval_id_val = None
            detail["evaluation_id"] = eval_id_val
            if eval_id_val is not None:
                detail["evaluation_res"] = _api_get_json(
                    base_url, f"/evaluations/{eval_id_val}", timeout_sec=timeout_sec
                )
                detail["contract_res"] = _api_get_json(
                    base_url, f"/evaluations/{eval_id_val}/contract", timeout_sec=timeout_sec
                )
                detail["drift_res"] = _api_get_json(
                    base_url, f"/evaluations/{eval_id_val}/drift", timeout_sec=timeout_sec
                )
                detail["predictions_res"] = _api_get_json(
                    base_url,
                    f"/evaluations/{eval_id_val}/predictions",
                    timeout_sec=timeout_sec,
                    params={"limit": int(pred_limit)},
                )
            detail["resources_res"] = _api_get_json(
                base_url,
                f"/tasks/{selected_task_id}/resources",
                timeout_sec=timeout_sec,
                params={"limit": int(res_limit)},
            )
            _log_dashboard_event(
                "runner_async_fetch_detail",
                {"task_id": selected_task_id, "evaluation_id": detail.get("evaluation_id")},
            )
        st.session_state["runner_async_detail"] = detail
        detail_task_res = detail.get("task_res") if isinstance(detail.get("task_res"), dict) else {}
        detail_ok = bool(selected_task_id) and bool((detail_task_res or {}).get("ok", False))
        _publish_notification(
            kind=NotificationEventKind.OPERATION_SUCCESS if detail_ok else NotificationEventKind.OPERATION_FAILURE,
            severity=NotificationSeverity.SUCCESS if detail_ok else NotificationSeverity.WARNING,
            title="Async XAI 詳細取得を実行しました",
            message="task / evaluation / contract / drift の最新結果を更新しました。",
            action="async_xai_fetch",
            status="success" if detail_ok else "warning",
            command_summary=f"GET {base_url}/tasks/{selected_task_id}",
            error_summary=str((detail_task_res or {}).get("error") or ""),
        )

    detail_state = st.session_state.get("runner_async_detail")
    if not isinstance(detail_state, dict) or not detail_state:
        return
    detail = detail_state

    st.markdown("**3) 可視化・分析結果**")
    task_obj = (
        (detail.get("task_res") or {}).get("data")
        if isinstance((detail.get("task_res") or {}).get("data"), dict)
        else {}
    )
    if isinstance(task_obj, dict) and task_obj:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("task status", str(task_obj.get("status") or "-"))
        t2.metric("task kind", str(task_obj.get("kind") or "-"))
        t3.metric("task id", str(task_obj.get("id") or "-")[:10] + "...")
        t4.metric("evaluation id", str(detail.get("evaluation_id") or "-"))

    eval_obj = (
        (detail.get("evaluation_res") or {}).get("data")
        if isinstance((detail.get("evaluation_res") or {}).get("data"), dict)
        else {}
    )
    contract_obj = (
        (detail.get("contract_res") or {}).get("data")
        if isinstance((detail.get("contract_res") or {}).get("data"), dict)
        else {}
    )
    drift_obj = (
        (detail.get("drift_res") or {}).get("data")
        if isinstance((detail.get("drift_res") or {}).get("data"), dict)
        else {}
    )
    pred_obj = (
        (detail.get("predictions_res") or {}).get("data")
        if isinstance((detail.get("predictions_res") or {}).get("data"), dict)
        else {}
    )
    res_obj = (
        (detail.get("resources_res") or {}).get("data")
        if isinstance((detail.get("resources_res") or {}).get("data"), dict)
        else {}
    )

    metrics_dict = (eval_obj.get("metrics") or {}) if isinstance(eval_obj, dict) else {}
    if isinstance(metrics_dict, dict) and metrics_dict:
        st.markdown("**evaluation metrics**")
        _show_df(
            pd.DataFrame([{"metric": str(k), "value": v} for k, v in metrics_dict.items()]),
            hide_index=True,
        )

    if isinstance(contract_obj, dict) and contract_obj:
        contract_data = contract_obj.get("contract")
        contract: dict[str, Any] = contract_data if isinstance(contract_data, dict) else {}
        st.markdown("**explainability contract**")
        st.json(
            {
                "point_forecast": contract.get("point_forecast"),
                "prediction_interval": contract.get("prediction_interval"),
                "residual_diagnostics": contract.get("residual_diagnostics"),
            }
        )
        top_features = (
            ((contract.get("attribution") or {}).get("top_features"))
            if isinstance(contract.get("attribution"), dict)
            else []
        )
        if isinstance(top_features, list) and top_features:
            st.markdown("寄与上位特徴量")
            _show_df(pd.DataFrame(top_features), hide_index=True)
        scenarios = (
            ((contract.get("what_if") or {}).get("scenarios")) if isinstance(contract.get("what_if"), dict) else []
        )
        if isinstance(scenarios, list) and scenarios:
            st.markdown("What-ifシナリオ")
            _show_df(pd.DataFrame(scenarios), hide_index=True)

    if isinstance(drift_obj, dict) and drift_obj:
        drift = drift_obj.get("drift") if isinstance(drift_obj.get("drift"), dict) else {}
        st.markdown("**drift**")
        st.json(drift)
        drift_rows: list[dict[str, Any]] = []
        if isinstance(drift, dict):
            for key in ["y_true", "y_pred", "residual"]:
                obj = drift.get(key)
                if isinstance(obj, dict):
                    drift_rows.append(
                        {
                            "signal": key,
                            "psi": obj.get("psi"),
                            "ks_stat": obj.get("ks_stat"),
                            "ks_pvalue": obj.get("ks_pvalue"),
                            "mean_shift": obj.get("mean_shift"),
                            "reference_n": obj.get("reference_n"),
                            "current_n": obj.get("current_n"),
                        }
                    )
        if drift_rows:
            _show_df(pd.DataFrame(drift_rows), hide_index=True)

    pred_rows = pred_obj.get("rows") if isinstance(pred_obj, dict) else []
    if isinstance(pred_rows, list) and pred_rows:
        pdf = pd.DataFrame(pred_rows)
        for c in ["y_true", "y_pred"]:
            if c in pdf.columns:
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
        if "t" in pdf.columns:
            pdf["t_dt"] = pd.to_datetime(pdf["t"], errors="coerce")
        st.markdown("**actual vs predicted (API)**")
        if PLOTLY_AVAILABLE and "y_true" in pdf.columns and "y_pred" in pdf.columns:
            if "t_dt" in pdf.columns and pdf["t_dt"].notna().any():
                x = pdf["t_dt"]
            else:
                x = np.arange(len(pdf))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=pdf["y_true"], mode="lines", name="actual"))
            fig.add_trace(go.Scatter(x=x, y=pdf["y_pred"], mode="lines", name="pred"))
            fig.update_layout(height=360, title="Actual vs Predicted (API)")
            st.plotly_chart(fig, width="stretch")
        elif "y_true" in pdf.columns and "y_pred" in pdf.columns:
            tmp = pdf[["y_true", "y_pred"]].copy()
            st.line_chart(tmp, height=320)
        _show_df(pdf.tail(200), hide_index=True)

    res_rows = res_obj.get("rows") if isinstance(res_obj, dict) else []
    if isinstance(res_rows, list) and res_rows:
        rdf = pd.DataFrame(res_rows)
        if "ts" in rdf.columns:
            rdf["ts"] = pd.to_datetime(rdf["ts"], errors="coerce")
        st.markdown("**resource timeline (task)**")
        if "ts" in rdf.columns and rdf["ts"].notna().any():
            rdf2 = rdf.sort_values("ts")
            metric_cols = [c for c in ["cpu_percent", "rss_mb", "gpu_util", "gpu_mem_mb"] if c in rdf2.columns]
            if metric_cols:
                st.line_chart(rdf2.set_index("ts")[metric_cols], height=320)
        _show_df(rdf.tail(300), hide_index=True)

    bundle = {
        "task": task_obj,
        "evaluation": eval_obj,
        "contract": contract_obj,
        "drift": drift_obj,
    }
    st.download_button(
        "Download async_xai_bundle.json",
        data=json.dumps(bundle, ensure_ascii=False, indent=2, default=str),
        file_name=f"async_xai_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="runner_async_download_bundle",
    )


def _render_operation_runner(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    engine: Engine | None = None,
) -> None:
    st.subheader("実行Runner（進捗表示付き）")
    st.caption(
        "`meta-automodel-create` / `run-table-pyspark` / `meta-automodel-run` / "
        "`model-save-load-analyze` / フル一括bash実行に加え、"
        "`Async XAI API(loop/contract/drift)` を進捗付きで運用します。"
    )
    with st.expander("Runner 引数ガイド / 補助情報", expanded=False):
        r_tabs = st.tabs(["共通設定", "A-F セクション要点", "運用時の注意"])
        with r_tabs[0]:
            rows = [
                {
                    "name": "runner cwd",
                    "meaning": "コマンド実行カレントディレクトリ",
                    "effect": "相対パスの解決先が変わる",
                },
                {"name": "timeout", "meaning": "1コマンドの最大実行秒数", "effect": "超過時は強制終了"},
                {"name": "stop_on_error", "meaning": "シーケンス失敗時停止", "effect": "ONで早期停止、OFFで継続実行"},
                {"name": "optimized", "meaning": "高速プリセット", "effect": "unified/build周りに高速向け既定値を適用"},
            ]
            _show_df(pd.DataFrame(rows), hide_index=True)
        with r_tabs[1]:
            rows = [
                {"section": "A meta-automodel-create", "purpose": "設定登録", "key_args": "model_name, h, auto_*"},
                {
                    "section": "B run-table-pyspark",
                    "purpose": "データ変換",
                    "key_args": "source_sql, transform_sql, execution_backend",
                },
                {
                    "section": "C meta-automodel-run",
                    "purpose": "設定実行",
                    "key_args": "config_id/limit, stop_on_error",
                },
                {"section": "D Fast Sequence", "purpose": "一括実行", "key_args": "include_* toggles"},
                {
                    "section": "E model-save-load-analyze",
                    "purpose": "保存再利用検証",
                    "key_args": "run_save/run_load/run_analyze",
                },
                {"section": "F Full Local Pipeline", "purpose": "ローカル総合実行", "key_args": "MODEL_OPS_* env"},
            ]
            _show_df(pd.DataFrame(rows), hide_index=True)
        with r_tabs[2]:
            st.markdown(
                "\n".join(
                    [
                        "- `source_sql` は必ず対象を絞ってから実行してください。",
                        "- `output_if_exists=replace` は上書きするため、対象テーブルを再確認してください。",
                        "- `auto_num_samples` を増やすと探索精度は上がりますが時間も増えます。",
                        "- `save_dataset=true` は保存容量が大きくなります。通常は false 推奨です。",
                    ]
                )
            )

    cwd = Path(st.text_input("runner cwd", value=str(PROJECT_ROOT), key="runner_cwd")).expanduser()
    timeout_sec = st.slider("runner: コマンドごとのタイムアウト(秒)", min_value=60, max_value=7200, value=1200, step=30)
    stop_on_error = st.checkbox("シーケンスで失敗時に停止", value=True)
    optimized = st.checkbox("高速プリセットを使用", value=True)

    def _q(val: Any) -> str:
        return shlex.quote(str(val))

    with st.expander("A. meta-automodel-create", expanded=True):
        cfg_name = st.text_input("config name", value="local_nf_run_01", key="runner_cfg_name")
        model_name = st.text_input("model name", value="AutoNHITS", key="runner_model_name")
        horizon = st.number_input("horizon", min_value=1, max_value=3650, value=28, step=1, key="runner_h")
        auto_cls_model = st.text_input("auto cls model(optional)", value="AutoNHITS", key="runner_auto_cls_model")
        auto_h = st.number_input(
            "auto h (0: use horizon)",
            min_value=0,
            max_value=3650,
            value=28,
            step=1,
            key="runner_auto_h",
        )
        auto_backend = st.selectbox("auto backend", ["optuna", "ray"], index=0, key="runner_auto_backend")
        auto_loss = st.text_input("auto loss", value="MAE", key="runner_auto_loss")
        auto_valid_loss = st.text_input("auto valid loss", value="MAE", key="runner_auto_valid_loss")
        search_options = (
            ["RandomSampler", "TPESampler", "CmaEsSampler", "NSGAIISampler"]
            if auto_backend == "optuna"
            else ["BasicVariantGenerator", "OptunaSearch", "HyperOptSearch", "BayesOptSearch"]
        )
        if (
            "runner_auto_search_alg" in st.session_state
            and str(st.session_state.get("runner_auto_search_alg")) not in search_options
        ):
            st.session_state["runner_auto_search_alg"] = search_options[0]
        current_search = str(st.session_state.get("runner_auto_search_alg", search_options[0]))
        if current_search not in search_options:
            current_search = search_options[0]
        auto_search_alg = st.selectbox(
            "auto search alg",
            search_options,
            index=search_options.index(current_search),
            key="runner_auto_search_alg",
        )
        auto_num_samples = st.number_input(
            "auto num samples",
            min_value=1,
            max_value=10000,
            value=10,
            step=1,
            key="runner_auto_num_samples",
        )
        auto_cpus = st.number_input(
            "auto cpus (0: library default)", min_value=0, max_value=256, value=0, step=1, key="runner_auto_cpus"
        )
        auto_gpus = st.number_input(
            "auto gpus (0: library default)", min_value=0, max_value=64, value=0, step=1, key="runner_auto_gpus"
        )
        base_schema = st.text_input("base schema", value="dataset", key="runner_base_schema")
        base_table = st.text_input("base table", value="loto_y_ts_unified", key="runner_base_table")
        hist_schema = st.text_input("hist schema", value="dataset", key="runner_hist_schema")
        hist_table = st.text_input("hist table", value="loto_hist_feat", key="runner_hist_table")
        exog_schema = st.text_input("exog schema", value="exog", key="runner_exog_schema")
        output_schema = st.text_input("output schema", value="dataset", key="runner_output_schema")
        output_table = st.text_input("output table", value="loto_y_ts_unified", key="runner_output_table")
        unified_filter_json = st.text_area(
            "unified filter json",
            value='{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}',
            height=90,
            key="runner_unified_filter_json",
        )
        unified_group_cols_json = st.text_area(
            "unified group cols json",
            value='["loto","unique_id","ts_type"]',
            height=70,
            key="runner_unified_group_cols_json",
        )
        unified_group_validate_strict = st.toggle(
            "strict validate unified group (loto/unique_id/ts_type + ds)",
            value=False,
            key="runner_unified_group_validate_strict",
        )
        model_params_json = st.text_area(
            "model params json",
            value='{"backend":"optuna","num_samples":20}',
            height=90,
            key="runner_model_params_json",
        )
        auto_config_json = st.text_area(
            "auto config json(BaseAuto config)",
            value='{"backend":"optuna","num_samples":10}',
            height=90,
            key="runner_auto_config_json",
        )
        param_space_json = st.text_area(
            "param space json",
            value='{"num_samples":[10,20],"seed":[1,2]}',
            height=90,
            key="runner_param_space_json",
        )
        param_mode_json = st.text_area(
            "param mode json (fixed/vary flags)",
            value='{"learning_rate":{"mode":"vary","values":[0.001,0.0005]},"batch_size":{"mode":"fixed","value":32}}',
            height=110,
            key="runner_param_mode_json",
        )
        st.caption("mode=vary は param_space に展開、mode=fixed は固定値として採用、enabled=false は対象外にします。")
        recursive_depth = st.number_input(
            "recursive depth", min_value=1, max_value=100, value=2, step=1, key="runner_recursive"
        )
        max_tasks = st.number_input(
            "max tasks (0: all combinations)", min_value=0, max_value=1000000, value=0, step=1, key="runner_max_tasks"
        )
        run_predict = st.toggle("run predict", value=True, key="runner_run_predict")
        run_evaluate = st.toggle("run evaluate", value=True, key="runner_run_eval")
        run_explain = st.toggle("run explain", value=True, key="runner_run_explain")
        run_save = st.toggle("run save", value=True, key="runner_run_save")
        run_load = st.toggle("run load", value=True, key="runner_run_load")
        run_analyze = st.toggle("run analyze", value=True, key="runner_run_analyze")
        save_dataset = st.toggle("save dataset", value=False, key="runner_save_dataset")
        save_overwrite = st.toggle("save overwrite", value=True, key="runner_save_overwrite")
        load_check_predict = st.toggle("load check predict_insample", value=False, key="runner_load_check_predict")
        meta_create_ensure_db = st.toggle(
            "ensure db-init before create", value=True, key="runner_meta_create_ensure_db"
        )
        save_path = st.text_input(
            "save path template",
            value=str(PROJECT_ROOT / "artifacts" / "saved_models" / "{run_id}"),
            key="runner_save_path",
        )
        auto_callbacks_json = st.text_area(
            "auto callbacks json(array)",
            value="[]",
            height=70,
            key="runner_auto_callbacks_json",
        )

        cmd_meta_create_parts = [
            "python -m loto_forecast.cli meta-automodel-create",
            f"--config-name {_q(cfg_name)}",
            f"--base-schema {_q(base_schema)} --base-table {_q(base_table)}",
            f"--hist-schema {_q(hist_schema)} --hist-table {_q(hist_table)}",
            f"--exog-schema {_q(exog_schema)}",
            f"--output-schema {_q(output_schema)} --output-table {_q(output_table)}",
            f"--unified-filter-json {_q(unified_filter_json)}",
            f"--unified-group-cols-json {_q(unified_group_cols_json)}",
            f"--model-name {_q(model_name)} --h {int(horizon)}",
            f"--auto-cls-model {_q(auto_cls_model)}",
            f"--auto-loss {_q(auto_loss)}",
            f"--auto-valid-loss {_q(auto_valid_loss)}",
            f"--auto-search-alg {_q(auto_search_alg)}",
            f"--auto-num-samples {int(auto_num_samples)}",
            f"--auto-backend {_q(auto_backend)}",
            f"--auto-config-json {_q(auto_config_json)}",
            f"--auto-callbacks-json {_q(auto_callbacks_json)}",
            f"--model-params-json {_q(model_params_json)}",
            f"--param-space-json {_q(param_space_json)}",
            f"--param-mode-json {_q(param_mode_json)}",
            f"--recursive-depth {int(recursive_depth)}",
            ("--run-predict" if run_predict else "--no-run-predict"),
            ("--run-evaluate" if run_evaluate else "--no-run-evaluate"),
            ("--run-explain" if run_explain else "--no-run-explain"),
            ("--run-save" if run_save else "--no-run-save"),
            ("--run-load" if run_load else "--no-run-load"),
            ("--run-analyze" if run_analyze else "--no-run-analyze"),
            ("--save-dataset" if save_dataset else "--no-save-dataset"),
            ("--save-overwrite" if save_overwrite else "--no-save-overwrite"),
            ("--load-check-predict" if load_check_predict else "--no-load-check-predict"),
            ("--ensure-db-init" if meta_create_ensure_db else "--no-ensure-db-init"),
            (
                "--unified-group-validate-strict"
                if unified_group_validate_strict
                else "--no-unified-group-validate-strict"
            ),
            f"--save-path {_q(save_path)}",
        ]
        if int(auto_h) > 0:
            cmd_meta_create_parts.append(f"--auto-h {int(auto_h)}")
        if int(auto_cpus) > 0:
            cmd_meta_create_parts.append(f"--auto-cpus {int(auto_cpus)}")
        if int(auto_gpus) > 0:
            cmd_meta_create_parts.append(f"--auto-gpus {int(auto_gpus)}")
        if int(max_tasks) > 0:
            cmd_meta_create_parts.append(f"--max-tasks {int(max_tasks)}")
        cmd_meta_create = " ".join(cmd_meta_create_parts)
        _render_command_preview(
            cmd_meta_create,
            copy_key="runner_copy_meta_create",
            copy_label="Copy meta-create command",
            cwd=cwd,
            show_arg_table=True,
        )
        if st.button("Run meta-automodel-create", key="runner_run_meta_create"):
            if not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_meta_create"] = _run_shell_command_live(
                    cmd_meta_create,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    title="meta-automodel-create",
                )

    with st.expander("Status: Meta/Model ライブ状態", expanded=True):
        if engine is None:
            st.caption(
                "現在のダッシュボードはDB未接続です。ここでは入力済み接続情報で都度ステータス問い合わせを行います。"
            )
        status_cfg_name = st.text_input("status config name", value=str(cfg_name), key="runner_status_cfg_name")
        c_status_1, c_status_2 = st.columns([1, 1])
        refresh_status = c_status_1.button("Refresh status", key="runner_status_refresh")
        auto_refresh_status = c_status_2.toggle("auto refresh on rerun", value=True, key="runner_status_auto_refresh")

        status_key = "runner_status_snapshot"
        last_cfg_key = "runner_status_snapshot_cfg"
        should_refresh = (
            bool(refresh_status)
            or bool(auto_refresh_status)
            or (status_key not in st.session_state)
            or (str(st.session_state.get(last_cfg_key, "")) != str(status_cfg_name))
        )
        if should_refresh:
            st.session_state[status_key] = _fetch_runner_live_status(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=database,
                config_name=str(status_cfg_name),
                row_limit=30,
            )
            st.session_state[last_cfg_key] = str(status_cfg_name)

        snapshot = st.session_state.get(status_key, {"ok": False, "error": "no snapshot"})
        if not bool(snapshot.get("ok", False)):
            st.warning(f"status query failed: {snapshot.get('error', 'unknown error')}")
        else:
            meta_obj = dict(snapshot.get("meta") or {})
            latest_model = dict(snapshot.get("latest_model") or {})
            success_count = int(snapshot.get("success_count", 0) or 0)
            failed_count = int(snapshot.get("failed_count", 0) or 0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("meta last status", str(meta_obj.get("last_status") or "-"))
            m2.metric("meta last run id", str(meta_obj.get("last_run_id") or "-"))
            m3.metric("model runs", f"success={success_count} failed={failed_count}")
            m4.metric("latest model status", str(latest_model.get("status") or "-"))
            st.json(
                {
                    "config_id": meta_obj.get("config_id"),
                    "config_name": meta_obj.get("config_name"),
                    "meta_last_run_at": meta_obj.get("last_run_at"),
                    "meta_updated_at": meta_obj.get("updated_at"),
                    "latest_result_id": latest_model.get("result_id"),
                    "latest_run_id": latest_model.get("run_id"),
                    "latest_model_name": latest_model.get("model_name"),
                    "latest_started_at": latest_model.get("started_at"),
                    "latest_ended_at": latest_model.get("ended_at"),
                    "latest_error": latest_model.get("error_message"),
                }
            )
            model_rows = snapshot.get("model_rows")
            if isinstance(model_rows, pd.DataFrame) and not model_rows.empty:
                st.markdown("**recent model.nf_automodel rows (for selected config)**")
                _show_df(model_rows.head(30), hide_index=True)

    with st.expander("B. run-table-pyspark", expanded=True):
        src_schema = st.text_input("source schema", value="dataset", key="runner_src_schema")
        src_table = st.text_input("source table", value="loto_y_ts_unified", key="runner_src_table")
        apply_key_filter_sql = st.toggle(
            "apply key filter in source sql(pushdown)",
            value=True,
            key="runner_apply_key_filter_sql",
        )
        filter_loto = st.text_input("filter loto", value="bingo5", key="runner_filter_loto")
        filter_unique_id = st.text_input("filter unique_id", value="N1", key="runner_filter_unique_id")
        filter_ts_type = st.text_input("filter ts_type", value="raw", key="runner_filter_ts_type")
        source_sql = st.text_area(
            "source sql (executed on PostgreSQL)",
            value=f'SELECT * FROM "{src_schema}"."{src_table}" WHERE y IS NOT NULL',
            height=120,
            key="runner_source_sql",
        )
        apply_transform_sql = st.toggle(
            "apply transform sql after source sql",
            value=False,
            key="runner_apply_transform_sql",
        )
        transform_sql = st.text_area(
            "transform sql (Spark SQL, use {{source}})",
            value="SELECT * FROM {{source}}",
            height=120,
            key="runner_transform_sql",
        )
        source_sql_effective = (
            f'SELECT * FROM "{src_schema}"."{src_table}" WHERE y IS NOT NULL '
            f"AND loto = '{filter_loto}' "
            f"AND unique_id = '{filter_unique_id}' "
            f"AND ts_type = '{filter_ts_type}'"
            if apply_key_filter_sql
            else source_sql
        )
        tgt_schema = st.text_input("target schema", value="dataset", key="runner_tgt_schema")
        tgt_table = st.text_input("target table", value="loto_y_ts_unified_spark", key="runner_tgt_table")
        output_if_exists = st.selectbox(
            "output if exists", ["replace", "append", "fail"], index=0, key="runner_out_mode"
        )
        output_parquet = st.text_input(
            "output parquet path",
            value=str(PROJECT_ROOT / "artifacts" / "datasets" / "loto_y_ts_unified_spark.parquet"),
            key="runner_out_parquet",
        )
        repartition = st.number_input(
            "repartition", min_value=0, max_value=4096, value=0, step=1, key="runner_repartition"
        )
        spark_master = st.text_input("spark master(optional)", value="", key="runner_spark_master")
        execution_backend = st.selectbox(
            "execution backend",
            ["auto", "polars", "pandas", "dask", "spark"],
            index=0,
            key="runner_execution_backend",
        )
        dask_npartitions = st.number_input(
            "dask npartitions",
            min_value=0,
            max_value=4096,
            value=0,
            step=1,
            key="runner_dask_npartitions",
        )
        prefer_pandas = st.toggle("prefer pandas engine (fast local)", value=False, key="runner_prefer_pandas")
        skip_row_count = st.toggle("skip spark row count", value=True, key="runner_skip_row_count")
        spark_ui_enabled = st.toggle("spark ui enabled", value=False, key="runner_spark_ui_enabled")
        spark_shuffle_partitions = st.number_input(
            "spark shuffle partitions",
            min_value=1,
            max_value=2000,
            value=16,
            step=1,
            key="runner_spark_shuffle_partitions",
        )
        spark_reader_fetchsize = st.number_input(
            "spark jdbc fetchsize",
            min_value=0,
            max_value=1000000,
            value=10000,
            step=1000,
            key="runner_spark_reader_fetchsize",
        )
        postgres_write_mode = st.selectbox(
            "postgres write mode", ["copy", "to_sql"], index=0, key="runner_postgres_write_mode"
        )
        postgres_copy_chunk_rows = st.number_input(
            "postgres copy chunk rows",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000,
            key="runner_postgres_copy_chunk_rows",
        )
        postgres_lock_timeout_ms = st.number_input(
            "postgres lock timeout ms",
            min_value=0,
            max_value=600000,
            value=10000,
            step=1000,
            key="runner_postgres_lock_timeout_ms",
        )

        cmd_spark_parts = [
            "python -m loto_forecast.cli run-table-pyspark",
            _safe_db_cli_flags(host=host, port=int(port), user=user, database=database),
            f"--source-schema {_q(src_schema)} --source-table {_q(src_table)}",
            f"--source-sql {_q(source_sql_effective)}",
            f"--target-schema {_q(tgt_schema)} --target-table {_q(tgt_table)}",
            f"--output-if-exists {_q(output_if_exists)}",
            f"--output-parquet-path {_q(output_parquet)}",
            f"--execution-backend {_q(execution_backend)}",
            ("--prefer-pandas" if prefer_pandas else "--no-prefer-pandas"),
            ("--skip-row-count" if skip_row_count else "--no-skip-row-count"),
            ("--spark-ui-enabled" if spark_ui_enabled else "--no-spark-ui-enabled"),
            f"--spark-shuffle-partitions {int(spark_shuffle_partitions)}",
            f"--postgres-write-mode {_q(postgres_write_mode)}",
            f"--postgres-copy-chunk-rows {int(postgres_copy_chunk_rows)}",
            f"--postgres-lock-timeout-ms {int(postgres_lock_timeout_ms)}",
        ]
        if int(dask_npartitions) > 0:
            cmd_spark_parts.append(f"--dask-npartitions {int(dask_npartitions)}")
        if int(spark_reader_fetchsize) > 0:
            cmd_spark_parts.append(f"--spark-reader-fetchsize {int(spark_reader_fetchsize)}")
        if apply_transform_sql:
            cmd_spark_parts.append(f"--transform-sql {_q(transform_sql)}")
        if int(repartition) > 0:
            cmd_spark_parts.append(f"--repartition {int(repartition)}")
        if spark_master.strip():
            cmd_spark_parts.append(f"--spark-master {_q(spark_master.strip())}")
        cmd_spark = " ".join(cmd_spark_parts)
        _render_command_preview(
            cmd_spark,
            copy_key="runner_copy_pyspark",
            copy_label="Copy pyspark command",
            cwd=cwd,
            show_arg_table=True,
        )
        if st.button("Run run-table-pyspark", key="runner_run_pyspark"):
            if not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_pyspark"] = _run_shell_command_live(
                    cmd_spark,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    title="run-table-pyspark",
                )

    with st.expander("C. meta-automodel-run", expanded=True):
        meta_run_config_id = st.text_input("config id(optional)", value="", key="runner_meta_run_config_id")
        meta_run_limit = st.number_input(
            "meta run limit", min_value=1, max_value=100000, value=1, step=1, key="runner_meta_run_limit"
        )
        meta_run_stop_on_error = st.toggle("meta run stop on error", value=True, key="runner_meta_run_stop_on_error")
        meta_run_ensure_db = st.toggle("ensure db-init before run", value=True, key="runner_meta_run_ensure_db")
        meta_run_skip_success = st.toggle(
            "meta run skip existing success", value=True, key="runner_meta_run_skip_success"
        )

        cmd_meta_run_parts = ["python -m loto_forecast.cli meta-automodel-run"]
        meta_run_cfg_id_int: int | None = None
        if meta_run_config_id.strip():
            try:
                meta_run_cfg_id_int = int(meta_run_config_id)
            except Exception:
                st.warning("config id は整数で指定してください。--limit を使用します。")
                meta_run_cfg_id_int = None
        if meta_run_cfg_id_int is not None:
            cmd_meta_run_parts.append(f"--config-id {meta_run_cfg_id_int}")
        else:
            cmd_meta_run_parts.append(f"--limit {int(meta_run_limit)}")
        if meta_run_stop_on_error:
            cmd_meta_run_parts.append("--stop-on-error")
        cmd_meta_run_parts.append("--ensure-db-init" if meta_run_ensure_db else "--no-ensure-db-init")
        cmd_meta_run_parts.append("--skip-existing-success" if meta_run_skip_success else "--no-skip-existing-success")
        cmd_meta_run = " ".join(cmd_meta_run_parts)
        _render_command_preview(
            cmd_meta_run,
            copy_key="runner_copy_meta_run",
            copy_label="Copy meta-run command",
            cwd=cwd,
            show_arg_table=True,
        )
        if st.button("Run meta-automodel-run", key="runner_run_meta_run"):
            if not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_meta_run"] = _run_shell_command_live(
                    cmd_meta_run,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    title="meta-automodel-run",
                )

    with st.expander("D. Fast Sequence (db-init -> unified -> meta-run -> pyspark)", expanded=True):
        seq_limit = st.number_input(
            "meta run limit (sequence)", min_value=1, max_value=100000, value=100, step=1, key="runner_seq_limit"
        )
        include_db_init = st.toggle("include db-init", value=True, key="runner_seq_db_init")
        include_meta_create = st.toggle("include meta-automodel-create", value=False, key="runner_seq_meta_create")
        include_pyspark = st.toggle("include run-table-pyspark", value=True, key="runner_seq_pyspark")

        cmd_unified = (
            "python -m loto_forecast.cli build-unified-dataset "
            f"{_safe_db_cli_flags(host=host, port=int(port), user=user, database=database)} "
            f"--base-schema {_q(base_schema)} --base-table {_q(base_table)} "
            f"--hist-schema {_q(hist_schema)} --hist-table {_q(hist_table)} "
            f"--exog-schema {_q(exog_schema)} "
            f"--output-schema {_q(output_schema)} --output-table {_q(output_table)} "
            + (
                "--fast-mode --postgres-write-mode copy --postgres-copy-chunk-rows 50000 --show-progress "
                if optimized
                else "--show-progress "
            )
        ).strip()
        seq_meta_run = (
            f"python -m loto_forecast.cli meta-automodel-run --limit {int(seq_limit)} --skip-existing-success"
        )
        seq_cmds: list[str] = []
        if include_db_init:
            seq_cmds.append("python -m loto_forecast.cli db-init")
        if include_meta_create:
            seq_cmds.append(cmd_meta_create)
        seq_cmds.append(cmd_unified)
        seq_cmds.append(seq_meta_run)
        if include_pyspark:
            seq_cmds.append(cmd_spark)

        st.markdown("**Sequence Commands**")
        st.code("\n".join([f"{i + 1}. {c}" for i, c in enumerate(seq_cmds)]), language="bash")
        _render_copy_button("\n".join(seq_cmds), key="runner_copy_sequence", label="Copy sequence commands", cwd=cwd)

        if st.button("Run Fast Sequence", key="runner_run_sequence", type="primary"):
            if not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_sequence"] = _run_command_sequence_live(
                    seq_cmds,
                    cwd=cwd,
                    timeout_sec_per_command=timeout_sec,
                    stop_on_error=stop_on_error,
                )

    with st.expander("E. model-save-load-analyze", expanded=False):
        run_id_mode = st.selectbox(
            "run id source",
            ["auto latest by config", "manual run id"],
            index=0,
            key="runner_model_ops_run_id_mode",
        )
        run_id_for_save = st.text_input("run id(manual)", value="", key="runner_model_ops_run_id")
        config_name_for_save = st.text_input(
            "config name(auto mode)", value=str(cfg_name), key="runner_model_ops_cfg_name"
        )
        config_id_for_save = st.text_input("config id(optional)", value="", key="runner_model_ops_cfg_id")
        source_path_for_save = st.text_input("source path(optional)", value="", key="runner_model_ops_source_path")
        save_path_for_save = st.text_input(
            "save path(template supports {run_id})",
            value=str(PROJECT_ROOT / "artifacts" / "saved_models" / "{run_id}"),
            key="runner_model_ops_save_path",
        )
        run_save_only = st.toggle("ops: run save", value=True, key="runner_model_ops_run_save")
        run_load_only = st.toggle("ops: run load", value=True, key="runner_model_ops_run_load")
        run_analyze_only = st.toggle("ops: run analyze", value=True, key="runner_model_ops_run_analyze")
        save_dataset_only = st.toggle("ops: save dataset", value=False, key="runner_model_ops_save_dataset")
        save_overwrite_only = st.toggle("ops: save overwrite", value=True, key="runner_model_ops_save_overwrite")
        load_check_only = st.toggle("ops: load check predict_insample", value=False, key="runner_model_ops_load_check")
        insample_step_size = st.number_input(
            "insample step size",
            min_value=1,
            max_value=365,
            value=1,
            step=1,
            key="runner_model_ops_insample_step_size",
        )
        model_ops_env: list[str] = []
        if run_id_mode == "manual run id" and run_id_for_save.strip():
            model_ops_env.append(f"RUN_ID={_q(run_id_for_save.strip())}")
        elif run_id_mode == "manual run id":
            st.info("manual mode では run id を指定してください。")
        if config_name_for_save.strip():
            model_ops_env.append(f"CONFIG_NAME={_q(config_name_for_save.strip())}")
        cfg_id_int: int | None = None
        if config_id_for_save.strip():
            try:
                cfg_id_int = int(config_id_for_save)
            except Exception:
                st.warning("config id(optional) は整数で指定してください。")
        if cfg_id_int is not None:
            model_ops_env.append(f"CONFIG_ID={cfg_id_int}")
        model_ops_env.extend(
            [
                f"SOURCE_PATH={_q(source_path_for_save.strip())}" if source_path_for_save.strip() else "",
                f"SAVE_PATH={_q(save_path_for_save)}",
                f"RUN_SAVE={'true' if run_save_only else 'false'}",
                f"RUN_LOAD={'true' if run_load_only else 'false'}",
                f"RUN_ANALYZE={'true' if run_analyze_only else 'false'}",
                f"SAVE_DATASET={'true' if save_dataset_only else 'false'}",
                f"SAVE_OVERWRITE={'true' if save_overwrite_only else 'false'}",
                f"LOAD_CHECK_PREDICT={'true' if load_check_only else 'false'}",
                f"INSAMPLE_STEP_SIZE={int(insample_step_size)}",
            ]
        )
        model_ops_env = [x for x in model_ops_env if x]
        cmd_model_ops = " ".join([*model_ops_env, "bash scripts/run_model_save_load_analyze.sh"]).strip()
        _render_command_preview(
            cmd_model_ops,
            copy_key="runner_copy_model_ops",
            copy_label="Copy model-save-load-analyze bash command",
            cwd=cwd,
            show_arg_table=True,
        )

        if st.button("Run model-save-load-analyze", key="runner_run_model_ops"):
            if run_id_mode == "manual run id" and not run_id_for_save.strip():
                st.error("manual mode では run id を指定してください。")
            elif not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_model_ops"] = _run_shell_command_live(
                    cmd_model_ops,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    title="model-save-load-analyze",
                )

    with st.expander("F. Full Local Pipeline (train + save/load + analyze + eval + predict)", expanded=True):
        full_script = st.text_input(
            "full pipeline script path",
            value="scripts/run_local_nf_full_pipeline.sh",
            key="runner_full_script_path",
        )
        full_run_check_grouping = st.toggle(
            "full: run unified grouping check", value=True, key="runner_full_check_grouping"
        )
        full_run_meta = st.toggle("full: run meta-automodel-run", value=True, key="runner_full_run_meta")
        full_run_model_ops = st.toggle(
            "full: run model save/load/analyze after meta-run", value=True, key="runner_full_run_model_ops"
        )
        full_meta_stop_on_error = st.toggle(
            "full: meta stop on error", value=True, key="runner_full_meta_stop_on_error"
        )
        full_model_ops_run_id_mode = st.selectbox(
            "full: model ops run id source",
            ["auto latest by config", "manual run id"],
            index=0,
            key="runner_full_model_ops_run_id_mode",
        )
        full_model_ops_run_id = st.text_input(
            "full: model ops run id(manual)", value="", key="runner_full_model_ops_run_id"
        )
        full_model_ops_save_path = st.text_input(
            "full: model ops save path(template supports {run_id})",
            value=str(PROJECT_ROOT / "artifacts" / "saved_models" / "{run_id}"),
            key="runner_full_model_ops_save_path",
        )
        full_model_ops_save = st.toggle("full: model ops run save", value=True, key="runner_full_model_ops_run_save")
        full_model_ops_load = st.toggle("full: model ops run load", value=True, key="runner_full_model_ops_run_load")
        full_model_ops_analyze = st.toggle(
            "full: model ops run analyze", value=True, key="runner_full_model_ops_run_analyze"
        )
        full_model_ops_save_dataset = st.toggle(
            "full: model ops save dataset", value=False, key="runner_full_model_ops_save_dataset"
        )
        full_model_ops_load_check = st.toggle(
            "full: model ops load check predict_insample", value=False, key="runner_full_model_ops_load_check"
        )
        full_model_ops_insample_step = st.number_input(
            "full: model ops insample step size",
            min_value=1,
            max_value=365,
            value=1,
            step=1,
            key="runner_full_model_ops_insample_step",
        )

        full_env = [
            f"CONFIG_NAME={_q(cfg_name)}",
            f"TARGET_LOTO={_q(filter_loto)}",
            f"TARGET_UNIQUE_ID={_q(filter_unique_id)}",
            f"TARGET_TS_TYPE={_q(filter_ts_type)}",
            f"RUN_CHECK_GROUPING={'true' if full_run_check_grouping else 'false'}",
            f"RUN_META_AUTOMODEL_RUN={'true' if full_run_meta else 'false'}",
            f"META_STOP_ON_ERROR={'true' if full_meta_stop_on_error else 'false'}",
            f"RUN_MODEL_OPS_AFTER={'true' if full_run_model_ops else 'false'}",
            f"MODEL_OPS_CONFIG_NAME={_q(cfg_name)}",
            f"MODEL_OPS_SAVE_PATH={_q(full_model_ops_save_path)}",
            f"MODEL_OPS_RUN_SAVE={'true' if full_model_ops_save else 'false'}",
            f"MODEL_OPS_RUN_LOAD={'true' if full_model_ops_load else 'false'}",
            f"MODEL_OPS_RUN_ANALYZE={'true' if full_model_ops_analyze else 'false'}",
            f"MODEL_OPS_SAVE_DATASET={'true' if full_model_ops_save_dataset else 'false'}",
            "MODEL_OPS_SAVE_OVERWRITE=true",
            f"MODEL_OPS_LOAD_CHECK_PREDICT={'true' if full_model_ops_load_check else 'false'}",
            f"MODEL_OPS_INSAMPLE_STEP_SIZE={int(full_model_ops_insample_step)}",
            f"AUTO_BACKEND={_q(auto_backend)}",
            f"EXECUTION_BACKEND={_q(execution_backend)}",
            f"PREFER_PANDAS={'true' if prefer_pandas else 'false'}",
            f"POSTGRES_WRITE_MODE={_q(postgres_write_mode)}",
            f"POSTGRES_COPY_CHUNK_ROWS={int(postgres_copy_chunk_rows)}",
            f"POSTGRES_LOCK_TIMEOUT_MS={int(postgres_lock_timeout_ms)}",
        ]
        if full_model_ops_run_id_mode == "manual run id" and full_model_ops_run_id.strip():
            full_env.append(f"MODEL_OPS_RUN_ID={_q(full_model_ops_run_id.strip())}")
        cmd_full_pipeline = " ".join([*full_env, f"bash {_q(full_script)}"]).strip()
        st.code(cmd_full_pipeline, language="bash")
        _render_copy_button(
            cmd_full_pipeline, key="runner_copy_full_pipeline", label="Copy full pipeline bash command", cwd=cwd
        )
        if st.button("Run Full Local Pipeline", key="runner_run_full_pipeline", type="primary"):
            if full_model_ops_run_id_mode == "manual run id" and not full_model_ops_run_id.strip():
                st.error("manual mode では full: model ops run id を指定してください。")
            elif not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                st.session_state["runner_last_full_pipeline"] = _run_shell_command_live(
                    cmd_full_pipeline,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    title="local-nf-full-pipeline",
                )

    with st.expander("G. Parallel Command Runner", expanded=False):
        st.caption("複数コマンドを並列実行します（互いに競合しない処理のみ推奨）。")
        parallel_cmds_text = st.text_area(
            "parallel commands (1行1コマンド)",
            value="\n".join(
                [
                    "python -m loto_forecast.cli meta-automodel-run --config-id 2 --stop-on-error",
                    "DB_PASSWORD=${DB_PASSWORD} python -m loto_forecast.cli check-unified-grouping --host 127.0.0.1 --port 5432 --user loto --database loto --schema dataset --table loto_y_ts_unified_spark --group-cols loto,unique_id,ts_type --time-col ds",
                ]
            ),
            height=140,
            key="runner_parallel_cmds",
        )
        parallel_workers = st.slider(
            "parallel workers", min_value=1, max_value=8, value=2, step=1, key="runner_parallel_workers"
        )
        if st.button("Run Parallel Commands", key="runner_run_parallel_commands"):
            if not cwd.exists() or not cwd.is_dir():
                st.error("runner cwd が有効ではありません。")
            else:
                cmds = [line.strip() for line in parallel_cmds_text.splitlines() if line.strip()]
                st.session_state["runner_last_parallel"] = _run_commands_parallel(
                    cmds,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    max_workers=int(parallel_workers),
                )

    with st.expander("H. Async XAI API (loop/contract/drift)", expanded=True):
        _render_async_xai_api_runner()

    if "runner_last_meta_create" in st.session_state:
        meta_last = dict(st.session_state["runner_last_meta_create"])
        parsed_meta = _try_parse_json_tail(str(meta_last.get("stdout", "")))
        st.markdown("**Last meta-automodel-create result**")
        st.json(
            {
                "ok": meta_last.get("ok"),
                "returncode": meta_last.get("returncode"),
                "elapsed_sec": meta_last.get("elapsed_sec"),
                "started_at": meta_last.get("started_at"),
                "ended_at": meta_last.get("ended_at"),
                "config_id": parsed_meta.get("config_id") if isinstance(parsed_meta, dict) else None,
                "action": parsed_meta.get("action") if isinstance(parsed_meta, dict) else None,
            }
        )
    if "runner_last_pyspark" in st.session_state:
        pyspark_last = dict(st.session_state["runner_last_pyspark"])
        parsed_spark = _try_parse_json_tail(str(pyspark_last.get("stdout", "")))
        st.markdown("**Last run-table-pyspark result**")
        st.json(
            {
                "ok": pyspark_last.get("ok"),
                "returncode": pyspark_last.get("returncode"),
                "elapsed_sec": pyspark_last.get("elapsed_sec"),
                "started_at": pyspark_last.get("started_at"),
                "ended_at": pyspark_last.get("ended_at"),
                "spark_available": parsed_spark.get("spark_available") if isinstance(parsed_spark, dict) else None,
                "rows": parsed_spark.get("rows") if isinstance(parsed_spark, dict) else None,
                "fallback_engine": parsed_spark.get("fallback_engine") if isinstance(parsed_spark, dict) else None,
            }
        )
    if "runner_last_meta_run" in st.session_state:
        meta_run_last = dict(st.session_state["runner_last_meta_run"])
        parsed_meta_run = _try_parse_json_tail(str(meta_run_last.get("stdout", "")))
        st.markdown("**Last meta-automodel-run result**")
        st.json(
            {
                "ok": meta_run_last.get("ok"),
                "returncode": meta_run_last.get("returncode"),
                "elapsed_sec": meta_run_last.get("elapsed_sec"),
                "started_at": meta_run_last.get("started_at"),
                "ended_at": meta_run_last.get("ended_at"),
                "executed": parsed_meta_run.get("executed") if isinstance(parsed_meta_run, dict) else None,
                "success": parsed_meta_run.get("success") if isinstance(parsed_meta_run, dict) else None,
                "failed": parsed_meta_run.get("failed") if isinstance(parsed_meta_run, dict) else None,
                "skipped": parsed_meta_run.get("skipped") if isinstance(parsed_meta_run, dict) else None,
                "skip_existing_success": parsed_meta_run.get("skip_existing_success")
                if isinstance(parsed_meta_run, dict)
                else None,
                "stopped_on_error": parsed_meta_run.get("stopped_on_error")
                if isinstance(parsed_meta_run, dict)
                else None,
            }
        )
    if "runner_last_sequence" in st.session_state:
        seq_df = pd.DataFrame(
            [
                {
                    "step": i + 1,
                    "ok": bool(r.get("ok", False)),
                    "returncode": int(r.get("returncode", -1)),
                    "command": str(r.get("command", "")),
                    "elapsed_sec": float(r.get("elapsed_sec", 0.0) or 0.0),
                    "started_at": str(r.get("started_at", "")),
                    "ended_at": str(r.get("ended_at", "")),
                }
                for i, r in enumerate(st.session_state["runner_last_sequence"])
            ]
        )
        st.markdown("**Last sequence summary**")
        _show_df(seq_df, hide_index=True)
    if "runner_last_model_ops" in st.session_state:
        model_ops_last = dict(st.session_state["runner_last_model_ops"])
        parsed_model_ops = _try_parse_json_tail(str(model_ops_last.get("stdout", "")))
        st.markdown("**Last model-save-load-analyze result**")
        st.json(
            {
                "ok": model_ops_last.get("ok"),
                "returncode": model_ops_last.get("returncode"),
                "elapsed_sec": model_ops_last.get("elapsed_sec"),
                "started_at": model_ops_last.get("started_at"),
                "ended_at": model_ops_last.get("ended_at"),
                "save_ok": parsed_model_ops.get("save", {}).get("ok") if isinstance(parsed_model_ops, dict) else None,
                "load_ok": parsed_model_ops.get("load", {}).get("ok") if isinstance(parsed_model_ops, dict) else None,
                "analyze_file_count": parsed_model_ops.get("analyze", {}).get("file_count")
                if isinstance(parsed_model_ops, dict)
                else None,
            }
        )
    if "runner_last_full_pipeline" in st.session_state:
        full_last = dict(st.session_state["runner_last_full_pipeline"])
        parsed_full = _try_parse_json_tail(str(full_last.get("stdout", "")))
        full_text = "\n".join([str(full_last.get("stdout", "")), str(full_last.get("stderr", ""))])
        resolved_run_id = _extract_last_run_id(full_text)
        st.markdown("**Last full local pipeline result**")
        st.json(
            {
                "ok": full_last.get("ok"),
                "returncode": full_last.get("returncode"),
                "elapsed_sec": full_last.get("elapsed_sec"),
                "started_at": full_last.get("started_at"),
                "ended_at": full_last.get("ended_at"),
                "resolved_run_id": resolved_run_id,
                "save_ok": parsed_full.get("save", {}).get("ok") if isinstance(parsed_full, dict) else None,
                "load_ok": parsed_full.get("load", {}).get("ok") if isinstance(parsed_full, dict) else None,
                "analyze_file_count": parsed_full.get("analyze", {}).get("file_count")
                if isinstance(parsed_full, dict)
                else None,
            }
        )
    if "runner_last_parallel" in st.session_state:
        parallel_rows: list[dict[str, Any]] = []
        for r in st.session_state["runner_last_parallel"]:
            parallel_rows.append(
                {
                    "ok": bool(r.get("ok", False)),
                    "returncode": int(r.get("returncode", -1)),
                    "command": str(r.get("command", "")),
                    "started_at": str(r.get("started_at", "")),
                    "ended_at": str(r.get("ended_at", "")),
                    "stderr_tail": str(r.get("stderr", ""))[-200:],
                }
            )
        st.markdown("**Last parallel command summary**")
        _show_df(pd.DataFrame(parallel_rows), hide_index=True)


def _render_table_inspector(engine: Engine, tables: set[tuple[str, str]], sample_limit: int) -> None:
    st.subheader("テーブル検査")
    by_schema: dict[str, list[str]] = defaultdict(list)
    for schema, table in sorted(tables):
        by_schema[schema].append(table)
    if not by_schema:
        st.info("対象テーブルがありません。")
        return

    schema = st.selectbox("schema", sorted(by_schema.keys()), index=0)
    table = st.selectbox("table", by_schema[schema], index=0)
    _show_df(_table_columns(engine, schema, table), hide_index=True)

    is_nf_model_table = schema == "model" and table == "nf_automodel"
    expand_key_base = f"tbl_expand_{_slug(schema)}_{_slug(table)}"
    expand_semistructured = st.toggle(
        "JSON/List/Dict カラムを展開",
        value=is_nf_model_table,
        key=f"{expand_key_base}_enabled",
        help="半構造カラムを展開して、model.nf_automodel の SELECT * を見やすく表示します。",
    )
    expand_max_depth = 3
    expand_list_items = 4
    expand_max_cols = 120
    if expand_semistructured:
        c1, c2, c3 = st.columns(3)
        expand_max_depth = c1.slider("展開深さ", min_value=1, max_value=6, value=3, key=f"{expand_key_base}_depth")
        expand_list_items = c2.slider(
            "配列の展開要素数", min_value=1, max_value=12, value=4, key=f"{expand_key_base}_list_items"
        )
        expand_max_cols = c3.slider(
            "元カラムごとの追加列上限",
            min_value=20,
            max_value=400,
            value=120,
            step=10,
            key=f"{expand_key_base}_max_cols",
        )

    if st.button("Count rows (exact)", key=f"cnt_any_{schema}_{table}"):
        try:
            row_count = _exact_count(engine, schema, table)
            st.metric("row_count", row_count)
            _publish_notification(
                kind=NotificationEventKind.OPERATION_SUCCESS,
                severity=NotificationSeverity.SUCCESS,
                title="行数カウントが完了しました",
                message="正確な行数を取得しました。必要なら続けて SELECT 実行やサンプル確認に進めます。",
                action="count_rows",
                status="success",
                command_summary=f"count rows {schema}.{table}",
                metadata={"row_count": int(row_count), "schema": schema, "table": table},
            )
        except Exception as e:
            st.error(str(e))
            _publish_notification(
                kind=NotificationEventKind.OPERATION_FAILURE,
                severity=NotificationSeverity.FAILURE,
                title="行数カウントに失敗しました",
                message="対象テーブルの行数取得に失敗しました。接続状態か権限を確認してください。",
                action="count_rows",
                status="failed",
                command_summary=f"count rows {schema}.{table}",
                error_summary=str(e),
                metadata={"schema": schema, "table": table},
            )
    sample_df = _sample_table(engine, schema, table, sample_limit)
    if expand_semistructured:
        sample_df = _expand_semistructured_columns(
            sample_df,
            max_depth=int(expand_max_depth),
            max_list_items=int(expand_list_items),
            max_new_cols_per_source=int(expand_max_cols),
        )
    _show_df(sample_df, hide_index=True)

    st.markdown("**Ad-hoc SELECT SQL (read-only)**")
    default_sql = f"SELECT * FROM {_safe_ident(schema)}.{_safe_ident(table)} LIMIT 50"
    raw_sql = st.text_area("sql", value=default_sql, height=120)
    if st.button("Run SQL"):
        if not raw_sql.strip().lower().startswith("select"):
            st.error("SELECT のみ実行できます。")
            _publish_notification(
                kind=NotificationEventKind.OPERATION_FAILURE,
                severity=NotificationSeverity.WARNING,
                title="SQL 実行を拒否しました",
                message="SELECT 以外は実行できません。読み取り専用 SQL に修正してください。",
                action="run_sql",
                status="failed",
                command_summary=raw_sql[:300],
                error_summary="SELECT only",
                metadata={"schema": schema, "table": table},
            )
        else:
            try:
                result_df = _query_df(engine, raw_sql)
                if expand_semistructured:
                    result_df = _expand_semistructured_columns(
                        result_df,
                        max_depth=int(expand_max_depth),
                        max_list_items=int(expand_list_items),
                        max_new_cols_per_source=int(expand_max_cols),
                    )
                _show_df(result_df, hide_index=True)
                _publish_notification(
                    kind=NotificationEventKind.OPERATION_SUCCESS,
                    severity=NotificationSeverity.SUCCESS,
                    title="SQL 実行が完了しました",
                    message="SELECT 結果を表示しました。必要なら絞り込み条件を調整して再実行してください。",
                    action="run_sql",
                    status="success",
                    command_summary=raw_sql[:300],
                    metadata={
                        "schema": schema,
                        "table": table,
                        "row_count": int(result_df.shape[0]),
                        "column_count": int(result_df.shape[1]),
                    },
                )
            except Exception as e:
                st.error(str(e))
                _publish_notification(
                    kind=NotificationEventKind.OPERATION_FAILURE,
                    severity=NotificationSeverity.FAILURE,
                    title="SQL 実行に失敗しました",
                    message="SQL 実行に失敗しました。構文、接続状態、対象テーブルを確認してください。",
                    action="run_sql",
                    status="failed",
                    command_summary=raw_sql[:300],
                    error_summary=str(e),
                    metadata={"schema": schema, "table": table},
                )


def _render_schema_export(
    engine: Engine,
    *,
    host: str,
    port: int,
    user: str,
    database: str,
) -> None:
    st.subheader("スキーマスナップショット出力")
    st.caption("dataset/exog/resources/meta/model のテーブル・カラム情報をまとめてエクスポートします。")

    snapshot = _snapshot_schema(engine, database)
    flat = pd.DataFrame(_snapshot_flat_rows(snapshot))
    if flat.empty:
        st.info("スナップショット対象データがありません。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("schemas", int(len(snapshot.get("schemas", []))))
    c2.metric("tables", int(flat[["schema", "table"]].drop_duplicates().shape[0]))
    c3.metric("columns", int(len(flat[flat["column_name"].astype(str) != ""])))
    _show_df(flat, hide_index=True)

    fmt = st.selectbox("export format", ["json", "csv", "yaml", "md", "html"], index=0)
    content, mime, ext = _snapshot_to_format(snapshot, fmt)
    _render_export_controls(content, mime, ext, filename_base="schema_snapshot", key=f"schema_{fmt}")

    st.markdown("---")
    st.markdown("### バックアップ / リストア用ファイル出力")
    st.caption("選択したスキーマ・テーブルに対する `manifest.json` / `backup.sh` / `restore.sh` を生成します。")

    user_tables = _all_user_tables(engine)
    if user_tables.empty:
        st.info("バックアップ対象のユーザーテーブルがありません。")
        return

    schema_options = sorted(user_tables["table_schema"].astype(str).unique().tolist())
    default_schemas = [s for s in SCHEMAS if s in schema_options] or schema_options
    selected_schemas = st.multiselect(
        "backup schemas",
        schema_options,
        default=default_schemas,
        key="schema_backup_selected_schemas",
    )
    if not selected_schemas:
        st.info("スキーマを1つ以上選択してください。")
        return

    filtered = user_tables[user_tables["table_schema"].astype(str).isin(selected_schemas)].copy()
    table_labels = [f"{r.table_schema}.{r.table_name}" for r in filtered.itertuples(index=False)]
    selected_table_labels = st.multiselect(
        "backup tables (未選択なら選択スキーマ配下を全件)",
        table_labels,
        default=[],
        key="schema_backup_selected_tables",
    )

    if selected_table_labels:
        selected_tables = [(x.split(".", 1)[0], x.split(".", 1)[1]) for x in selected_table_labels if "." in x]
        dump_tables = list(selected_tables)
    else:
        selected_tables = [(str(r.table_schema), str(r.table_name)) for r in filtered.itertuples(index=False)]
        dump_tables = []
    if not selected_tables:
        st.info("対象テーブルがありません。")
        return

    selected_df = pd.DataFrame([{"schema": s, "table": t} for s, t in sorted({(s, t) for s, t in selected_tables})])
    st.markdown("**選択対象テーブル**")
    _show_df(selected_df, hide_index=True)

    bundle = _build_backup_restore_bundle(
        engine,
        host=host,
        port=int(port),
        user=user,
        database=database,
        selected_schemas=selected_schemas,
        selected_tables=dump_tables,
    )

    st.markdown("**backup manifest (json)**")
    _render_export_controls(
        bundle["manifest_json"],
        "application/json",
        "json",
        filename_base="db_backup_manifest",
        key="backup_manifest_json",
    )
    st.markdown("**backup script (bash)**")
    _render_export_controls(
        bundle["backup_script"],
        "text/x-shellscript",
        "sh",
        filename_base="db_backup",
        key="backup_script_sh",
    )
    st.markdown("**restore script (bash)**")
    _render_export_controls(
        bundle["restore_script"],
        "text/x-shellscript",
        "sh",
        filename_base="db_restore",
        key="restore_script_sh",
    )


def _render_directory_compiler() -> None:
    st.subheader("ディレクトリ統合コンパイラ")
    st.caption("指定ディレクトリ内の json/csv/yaml/md/html/mmd を集約し、表示・エクスポート・読み上げできます。")

    default_root = str(PROJECT_ROOT)
    root_input = st.text_input("directory path", value=default_root)
    max_files = st.slider("max files", min_value=50, max_value=5000, value=1200, step=50)
    root = Path(root_input).expanduser()

    if not root.exists() or not root.is_dir():
        st.error("有効なディレクトリを指定してください。")
        return

    files = _scan_supported_files(root, max_files=max_files)
    st.write(f"detected files: {len(files)}")
    if not files:
        st.info("対象ファイルがありません。")
        return

    rows = []
    for p in files:
        rows.append(
            {
                "path": str(p.relative_to(root)),
                "suffix": p.suffix.lower(),
                "size_bytes": int(p.stat().st_size),
                "size_human": _format_bytes(p.stat().st_size),
            }
        )
    files_df = pd.DataFrame(rows)
    _show_df(files_df, hide_index=True)

    selected_rel = st.selectbox("preview file", files_df["path"].tolist(), index=0)
    selected_path = root / selected_rel
    summary = _summarize_supported_file(selected_path)
    st.json({"meta": summary.get("meta", {}), "size": summary.get("size_human", "")})
    preview = str(summary.get("preview", ""))
    if selected_path.suffix.lower() == ".csv":
        try:
            _show_df(pd.read_csv(selected_path, nrows=200), hide_index=True)
        except Exception:
            st.code(preview)
    else:
        _render_rich_content(preview[:50000], ext=selected_path.suffix, key=f"dir_preview_{selected_rel}")

    _render_read_aloud(preview, key="file_preview")

    if st.button("Compile Directory Payload"):
        try:
            compiled_payload = _compile_directory_payload(root, files)
            st.session_state["compiled_bundle"] = compiled_payload
            _publish_notification(
                kind=NotificationEventKind.OPERATION_SUCCESS,
                severity=NotificationSeverity.SUCCESS,
                title="ディレクトリ統合が完了しました",
                message="集約結果を生成しました。必要なら export format を切り替えて保存できます。",
                action="directory_compile",
                status="success",
                command_summary=f"compile directory {root}",
                metadata={"root": str(root), "file_count": int(len(files))},
            )
        except Exception as e:
            st.error(str(e))
            _publish_notification(
                kind=NotificationEventKind.OPERATION_FAILURE,
                severity=NotificationSeverity.FAILURE,
                title="ディレクトリ統合に失敗しました",
                message="対象ファイルの読み込みか集約処理に失敗しました。",
                action="directory_compile",
                status="failed",
                command_summary=f"compile directory {root}",
                error_summary=str(e),
                metadata={"root": str(root)},
            )

    compiled_bundle = st.session_state.get("compiled_bundle")
    if not compiled_bundle:
        return
    bundle_data: dict[str, Any] = compiled_bundle

    st.markdown("**Compiled Summary**")
    st.json(
        {
            "root": bundle_data.get("root"),
            "generated_at": bundle_data.get("generated_at"),
            "file_count": bundle_data.get("file_count"),
            "suffix_counts": bundle_data.get("suffix_counts"),
        }
    )

    export_fmt = st.selectbox("compiled export format", ["json", "csv", "yaml", "md", "html"], index=0)
    content, mime, ext = _compiled_to_format(bundle_data, export_fmt)
    _render_export_controls(content, mime, ext, filename_base="compiled_directory", key=f"compiled_{export_fmt}")


def _render_markdown_compiler() -> None:
    st.subheader("Markdown資料コンパイラ")
    st.caption("各種 md 資料を収集・結合し、リッチ表示/エクスポートします。")

    default_roots = [
        str(PROJECT_ROOT / "docs"),
        str(PROJECT_ROOT),
        str(EXTERNAL_TARGETS["trend"]),
        str(EXTERNAL_TARGETS["timesfm"]),
    ]
    roots_raw = st.text_area(
        "root directories (one per line)",
        value="\n".join(default_roots),
        height=140,
    )
    roots = [Path(line.strip()).expanduser() for line in roots_raw.splitlines() if line.strip()]
    max_files = st.slider("max markdown files", min_value=20, max_value=5000, value=1200, step=20)

    files = _scan_markdown_files(roots, max_files=max_files)
    st.write(f"detected markdown files: {len(files)}")
    if not files:
        st.info("markdown files not found")
        return

    rows = [
        {"path": str(p), "size_human": _format_bytes(p.stat().st_size), "size_bytes": int(p.stat().st_size)}
        for p in files
    ]
    files_df = pd.DataFrame(rows)
    _show_df(files_df, hide_index=True)

    if st.button("Compile Markdown Bundle"):
        try:
            st.session_state["compiled_markdown_bundle"] = _compile_markdown_bundle(roots, files)
            _publish_notification(
                kind=NotificationEventKind.OPERATION_SUCCESS,
                severity=NotificationSeverity.SUCCESS,
                title="Markdown 統合が完了しました",
                message="資料バンドルを生成しました。リッチ表示かエクスポートで内容を確認できます。",
                action="markdown_compile",
                status="success",
                command_summary="compile markdown bundle",
                metadata={"root_count": int(len(roots)), "file_count": int(len(files))},
            )
        except Exception as e:
            st.error(str(e))
            _publish_notification(
                kind=NotificationEventKind.OPERATION_FAILURE,
                severity=NotificationSeverity.FAILURE,
                title="Markdown 統合に失敗しました",
                message="Markdown 集約に失敗しました。対象パスとファイル内容を確認してください。",
                action="markdown_compile",
                status="failed",
                command_summary="compile markdown bundle",
                error_summary=str(e),
            )

    bundle = st.session_state.get("compiled_markdown_bundle")
    if not bundle:
        return

    st.json(
        {
            "generated_at": bundle.get("generated_at"),
            "file_count": bundle.get("file_count"),
            "roots": bundle.get("roots"),
        }
    )
    docs_df = pd.DataFrame(bundle.get("documents", []))
    if not docs_df.empty:
        _show_df(docs_df[["index", "relative_path", "lines", "chars"]], hide_index=True)

    compiled_md = str(bundle.get("compiled_markdown", ""))
    st.markdown("**Compiled Markdown (Rich Render)**")
    st.markdown(compiled_md)

    export_fmt = st.selectbox("export format", ["md", "json", "csv", "html"], index=0, key="md_bundle_fmt")
    if export_fmt == "md":
        content = compiled_md
        mime = "text/markdown"
        ext = "md"
    elif export_fmt == "json":
        content = json.dumps(bundle, ensure_ascii=False, indent=2)
        mime = "application/json"
        ext = "json"
    elif export_fmt == "csv":
        content = pd.DataFrame(bundle.get("documents", [])).to_csv(index=False)
        mime = "text/csv"
        ext = "csv"
    else:
        safe = html.escape(compiled_md)
        content = f"<html><body><pre>{safe}</pre></body></html>"
        mime = "text/html"
        ext = "html"
    _render_export_controls(
        content, mime, ext, filename_base="compiled_markdown_docs", key=f"compiled_md_docs_{export_fmt}"
    )


def _render_code_maps() -> None:
    st.subheader("コード解析 / Mermaid / 可視化マップ")
    st.caption("コードを解析し、Mermaid・ネットワーク・サンバーストで構造を可視化します。")

    analysis = _analyze_python_codebase(SRC_ROOT, PROJECT_ROOT / "scripts")
    st.json(
        {
            "generated_at": analysis.get("generated_at"),
            "module_count": analysis.get("module_count"),
            "function_count": analysis.get("function_count"),
            "class_count": analysis.get("class_count"),
            "edge_count": len(analysis.get("edges", [])),
        }
    )

    modules_df = pd.DataFrame(analysis.get("modules", []))
    funcs_df = pd.DataFrame(analysis.get("functions", []))
    classes_df = pd.DataFrame(analysis.get("classes", []))
    edges_df = pd.DataFrame(analysis.get("edges", []))
    calls_df = pd.DataFrame(analysis.get("top_call_names", []))

    if not modules_df.empty:
        st.markdown("**Modules**")
        _show_df(modules_df, hide_index=True)
    if not funcs_df.empty:
        st.markdown("**Functions**")
        _show_df(funcs_df.head(400), hide_index=True)
    if not classes_df.empty:
        st.markdown("**Classes**")
        _show_df(classes_df.head(200), hide_index=True)
    if not calls_df.empty:
        st.markdown("**Top Called Names**")
        _show_df(calls_df.head(50), hide_index=True)
    if not edges_df.empty:
        st.markdown("**Module Edge Table**")
        _show_df(edges_df.head(600), hide_index=True)
    else:
        st.warning(
            "module dependency edges が抽出できませんでした。Mermaid/Sankey が空の場合は import 構造を確認してください。"
        )

    st.markdown("**Mermaid: Module Dependency Flow**")
    flow_code = _module_edges_mermaid(analysis.get("edges", []))
    _render_mermaid(flow_code, key="module_flow", height=520)
    st.caption("図が表示されない場合は下の Mermaid code をコピーして外部Mermaid環境で確認してください。")
    with st.expander("Mermaid code (flow)"):
        st.code(flow_code, language="mermaid")
        _render_copy_button(flow_code, key="copy_mermaid_flow", label="Copy Mermaid Flow")

    st.markdown("**Mermaid: Execution Sequence**")
    seq_code = _default_sequence_mermaid()
    _render_mermaid(seq_code, key="sequence_flow", height=420)
    with st.expander("Mermaid code (sequence)"):
        st.code(seq_code, language="mermaid")
        _render_copy_button(seq_code, key="copy_mermaid_seq", label="Copy Mermaid Sequence")

    if PLOTLY_AVAILABLE:
        st.markdown("**Network (Sankey)**")
        fig = _edge_sankey(analysis.get("edges", []))
        if fig is None:
            st.info("表示可能なエッジがありません。")
        else:
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Sunburst (Modules by LOC)**")
        sun_mod = _sunburst_modules(analysis.get("modules", []))
        if sun_mod is not None:
            st.plotly_chart(sun_mod, width="stretch")

        st.markdown("**Sunburst (Directory by File Size)**")
        sun_dir = _sunburst_directory(PROJECT_ROOT, max_depth=4)
        if sun_dir is not None:
            st.plotly_chart(sun_dir, width="stretch")
    else:
        st.info("plotly が未導入のため Sankey/Sunburst は表示できません。")

    analysis_fmt = st.selectbox("analysis export format", ["json", "yaml", "md"], index=0)
    if analysis_fmt == "json":
        content = json.dumps(analysis, ensure_ascii=False, indent=2)
        mime = "application/json"
        ext = "json"
    elif analysis_fmt == "yaml":
        content = yaml.safe_dump(analysis, allow_unicode=True, sort_keys=False)
        mime = "application/x-yaml"
        ext = "yaml"
    else:
        lines = [
            "# Code Analysis Snapshot",
            "",
            f"- generated_at: {analysis.get('generated_at')}",
            f"- module_count: {analysis.get('module_count')}",
            f"- function_count: {analysis.get('function_count')}",
            f"- class_count: {analysis.get('class_count')}",
            "",
            "## Top Modules",
            "",
        ]
        top_mod = modules_df.sort_values("lines", ascending=False).head(30) if not modules_df.empty else pd.DataFrame()
        if not top_mod.empty:
            lines.append("| module | lines | functions | classes |")
            lines.append("|---|---:|---:|---:|")
            for r in top_mod.itertuples(index=False):
                lines.append(f"| {r.module} | {r.lines} | {r.functions} | {r.classes} |")
        content = "\n".join(lines)
        mime = "text/markdown"
        ext = "md"
    _render_export_controls(content, mime, ext, filename_base="code_analysis_snapshot", key=f"analysis_{analysis_fmt}")


def _render_artifacts_and_logs() -> None:
    st.subheader("成果物 / ログ / 変更サマリ")

    artifacts_dir = PROJECT_ROOT / "artifacts"
    logs_dir = PROJECT_ROOT / "logs"
    st.markdown("**artifacts/run_***")
    art_runs = sorted([p.name for p in artifacts_dir.glob("run_*") if p.is_dir()], reverse=True)
    if art_runs:
        run_name = st.selectbox("artifact run dir", art_runs, index=0)
        run_dir = artifacts_dir / run_name
        files = sorted([str(p.relative_to(PROJECT_ROOT)) for p in run_dir.rglob("*") if p.is_file()])
        _show_df(pd.DataFrame({"file": files}), hide_index=True)
    else:
        st.info("artifacts/run_* が見つかりません。")

    st.markdown("**logs**")
    log_files = sorted([p for p in logs_dir.rglob("*") if p.is_file()], reverse=True)
    if log_files:
        labels = [str(p.relative_to(PROJECT_ROOT)) for p in log_files]
        selected = st.selectbox("log file", labels, index=0)
        tail_n = st.slider("tail lines", min_value=50, max_value=2000, value=300, step=50)
        st.code(_tail_lines(PROJECT_ROOT / selected, n=tail_n))
    else:
        st.info("logs 配下にファイルがありません。")

    st.markdown("**dashboard event logs (jsonl)**")
    event_files = sorted(DASHBOARD_LOG_DIR.glob("events_*.jsonl"), reverse=True)
    if event_files:
        event_labels = [str(p.relative_to(PROJECT_ROOT)) for p in event_files]
        event_sel = st.selectbox("event log file", event_labels, index=0)
        event_df = _extract_jsonl(PROJECT_ROOT / event_sel, max_rows=12000)
        if event_df.empty:
            st.info("イベントログは空です。")
        else:
            if "ts" in event_df.columns:
                event_df["ts"] = pd.to_datetime(event_df["ts"], errors="coerce", utc=True)
            if "payload" in event_df.columns:
                event_df["payload"] = event_df["payload"].map(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                )
            levels = ["all"] + sorted(event_df.get("level", pd.Series(dtype=str)).astype(str).unique().tolist())
            lev = st.selectbox("level filter", levels, index=0)
            filtered = event_df.copy()
            if lev != "all" and "level" in filtered.columns:
                filtered = filtered[filtered["level"].astype(str) == lev]
            _show_df(filtered.tail(500), hide_index=True)
            if "event_type" in filtered.columns:
                ec = (
                    filtered["event_type"]
                    .astype(str)
                    .value_counts()
                    .rename_axis("event_type")
                    .reset_index(name="count")
                )
                st.bar_chart(ec.set_index("event_type")[["count"]], height=260)
    else:
        st.info("dashboard event log はまだ作成されていません。")

    st.markdown("**Directory Structure**")
    st.code(_tree_lines(PROJECT_ROOT, max_depth=4, max_entries=900))

    st.markdown("**Directory / File Diff**")
    c1, c2 = st.columns(2)
    dir_a = Path(c1.text_input("dir A", value=str(PROJECT_ROOT))).expanduser()
    dir_b = Path(c2.text_input("dir B", value=str(PROJECT_ROOT))).expanduser()
    depth = st.slider("tree diff max depth", min_value=2, max_value=8, value=4, step=1)
    if dir_a.exists() and dir_a.is_dir() and dir_b.exists() and dir_b.is_dir():
        tree_a = _tree_lines(dir_a, max_depth=depth, max_entries=1200)
        tree_b = _tree_lines(dir_b, max_depth=depth, max_entries=1200)
        tree_diff = _unified_diff_text(
            tree_a,
            tree_b,
            left_name=f"{dir_a}/tree",
            right_name=f"{dir_b}/tree",
            context_lines=3,
            ignore_ws=False,
        )
        if not tree_diff:
            st.info("Directory structure diff: no changes")
        else:
            st.code(tree_diff, language="diff")
            _render_export_controls(
                tree_diff,
                mime="text/x-diff",
                ext="diff",
                filename_base="directory_tree_diff",
                key="directory_tree_diff",
            )
    else:
        st.info("有効な2つのディレクトリを指定すると、構造差分を表示できます。")

    diff_root = Path(st.text_input("diff file scan root", value=str(PROJECT_ROOT))).expanduser()
    if not diff_root.exists() or not diff_root.is_dir():
        st.info("diff file scan root が有効なディレクトリではありません。")
    else:
        candidates = _scan_diff_files(diff_root, max_files=2500)
        if not candidates:
            st.info("差分対象のテキストファイルが見つかりません。")
        else:
            rels = [str(p.relative_to(diff_root)) for p in candidates]
            fcol1, fcol2 = st.columns(2)
            left_rel = fcol1.selectbox("left file", rels, index=0, key="diff_left")
            right_rel = fcol2.selectbox("right file", rels, index=min(1, len(rels) - 1), key="diff_right")
            ignore_ws = st.checkbox("ignore trailing whitespace", value=True)
            context_lines = st.slider("file diff context lines", min_value=1, max_value=15, value=3, step=1)

            left_text = _read_text_file(diff_root / left_rel, max_chars=1_500_000)
            right_text = _read_text_file(diff_root / right_rel, max_chars=1_500_000)
            file_diff = _unified_diff_text(
                left_text,
                right_text,
                left_name=left_rel,
                right_name=right_rel,
                context_lines=context_lines,
                ignore_ws=ignore_ws,
            )
            if not file_diff:
                st.info("File diff: no changes")
            else:
                st.code(file_diff, language="diff")
                _render_export_controls(
                    file_diff,
                    mime="text/x-diff",
                    ext="diff",
                    filename_base=f"file_diff_{_slug(left_rel)}_vs_{_slug(right_rel)}",
                    key="file_diff",
                )

    st.markdown("**Change Summary (dashboard related files)**")
    tracked = [
        PROJECT_ROOT / "scripts" / "operations_dashboard.py",
        PROJECT_ROOT / "docs" / "DEVELOPMENT_HISTORY.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "Makefile",
    ]
    rows = []
    for p in tracked:
        if p.exists():
            stat = p.stat()
            rows.append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "exists": True,
                    "size_bytes": int(stat.st_size),
                    "size_human": _format_bytes(stat.st_size),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "line_count": len(_read_text_file(p, 5_000_000).splitlines()),
                }
            )
        else:
            rows.append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "exists": False,
                    "size_bytes": 0,
                    "size_human": "0 B",
                    "mtime": "",
                    "line_count": 0,
                }
            )
    _show_df(pd.DataFrame(rows), hide_index=True)

    st.markdown("**Codegen Docs (streamlit_all_codegen.yaml)**")
    default_path = "./docs/lib_docs/streamlit_all_codegen.yaml"
    yaml_path = st.text_input("YAML path", value=default_path)
    info = _yaml_summary(yaml_path)
    st.json({k: v for k, v in info.items() if k not in {"top_modules", "top_groups"}})
    if "top_modules" in info:
        _show_df(info["top_modules"], hide_index=True)
    if "top_groups" in info:
        _show_df(info["top_groups"], hide_index=True)


def _render_guide_history() -> None:
    st.subheader("Guide / Parameter Meanings / Development History")
    st.markdown(
        """
### 実行できる主な機能
1. `Overview`: DB全体状況、主要テーブル数、最新実行状況を確認
2. `Operations`: `resources.*` 実行履歴に加え、`meta/model` の AutoModel 実行管理テーブルを確認
3. `Operations -> Runner`: `meta-automodel-create` / `run-table-pyspark` / `meta-automodel-run` / `model-save-load-analyze` / フル一括bash実行を進捗バー付きで実行
   さらに `Async XAI API` で `loop/contract/drift` の再帰実行・監視・可視化を一体運用
4. `Operations -> モデル・リソース関連`: 外生変数影響・モデル評価・リソース情報の関係分析（寄与率/ベイズ/因果代理/グラフ/ゲーム理論近似）
5. `Operations -> 実測vs予測`: `forecast.parquet` と `dataset.loto_y_ts_unified` を突合して作図
6. `Resources Analytics`: `resources.run/stage_span/resource_metric` の統計分析・可視化
7. `Schema Export`: スキーマ/テーブル情報を `json/csv/yaml/md/html` で出力し、選択対象の `backup.sh / restore.sh / manifest.json` を生成
8. `Directory Compiler`: 指定ディレクトリ内の `json/csv/yaml/md/html/mmd` を集約表示・出力・読み上げ
9. `Markdown Compiler`: 複数ディレクトリの `*.md` 資料を統合コンパイル・表示
10. `Code Maps`: コード解析、Mermaidフロー/シーケンス、ネットワーク、サンバースト表示
11. `External Targets`: `trend` / `timesfm` の構成、コンパイル、コード解析、可視化、起動
12. `Command Lab`: CLI解析、引数説明つきコマンド生成、コピー、.sh/.py保存、進捗表示付き実行
13. `Artifacts/Logs`: 実行成果物・ログ・ディレクトリ構造・差分・変更対象ファイルを確認
14. `Feature Guide`: 各タブ機能の説明と推奨ワークフロー
"""
    )

    st.markdown("### パラメータの意味")
    st.markdown(
        """
- `host/port/user/password/database`: 接続先PostgreSQL
- `row limit`: 一覧の取得上限（`resources.run` 等）
- `sample limit`: サンプル表示の最大行数
- `max files`: Directory Compiler で走査する最大ファイル数
- `export format`: 出力形式（`json/csv/yaml/md/html`）
- `analysis export format`: コード解析結果の出力形式
- `tree/file diff`: ディレクトリ構造差分とファイル差分の比較設定
- `include default values`: コマンド生成時にデフォルト値引数も明示するか
- `execution cwd`: 生成コマンド/スクリプトを実行する作業ディレクトリ
- `external app path/port`: 外部 Streamlit アプリ起動対象とポート
- `runner timeout per command`: Runner/Command Lab の1コマンド許容時間（秒）
- `use optimized fast preset`: `build-unified-dataset --fast-mode --postgres-write-mode copy` を自動適用
- `Status. Live Meta/Model Status`: `meta.nf_automodel` と `model.nf_automodel` の最新状態を表示
- `build-unified-dataset`: `dataset.loto_y_ts + dataset.loto_hist_feat + exog.*` を統合し `dataset.loto_y_ts_unified` を作成
- `meta-automodel-run`: `meta.nf_automodel` を読み、網羅/再帰実行結果を `model.nf_automodel` へ保存
- `run-table-pyspark`: PostgreSQL(JDBC) -> Spark SQL -> PostgreSQL/Parquet/CSV へ変換出力
- `model-save-load-analyze`: `NeuralForecast.save/load` 実行と保存済みモデル解析
- `run_local_nf_full_pipeline.sh`: `create -> pyspark -> grouping -> meta-run -> model save/load/analyze` を1コマンド実行
- `Async XAI API`: FastAPI (`/loops/submit`, `/tasks/{id}`, `/evaluations/{id}/contract`, `/evaluations/{id}/drift`) を直接実行/監視
"""
    )

    history_path = PROJECT_ROOT / "docs" / "DEVELOPMENT_HISTORY.md"
    st.markdown("### Development History")
    if history_path.exists():
        text_value = _read_text_file(history_path, 800_000)
        st.markdown(text_value)
        _render_copy_button(text_value, key="history_copy", label="Copy History")
        _render_read_aloud(text_value, key="history_speech")
    else:
        st.info("docs/DEVELOPMENT_HISTORY.md がありません。")


def _render_feature_explanations() -> None:
    st.subheader("Feature Explanations")
    st.caption("各タブの目的・入力・出力・主な操作をまとめています。")
    rows = [
        {
            "tab": "Overview",
            "purpose": "DB全体把握（テーブル状況/最新実行）",
            "inputs": "DB接続情報",
            "outputs": "テーブル一覧・実行概況",
            "main_actions": "カタログ確認、最新run確認",
        },
        {
            "tab": "Operations",
            "purpose": "運用データの閲覧・調査",
            "inputs": "resources/dataset/exog/meta/model テーブル",
            "outputs": "runs/spans/metrics・model/grid・meta管理テーブル",
            "main_actions": "run追跡、meta定義確認、SQL確認",
        },
        {
            "tab": "Operations.Runner",
            "purpose": "高速実行フローを進捗表示付きで実行",
            "inputs": "DB接続情報、meta/pyspark設定、API base URL、タイムアウト",
            "outputs": "実行結果、returncode、stdout/stderr tail、シーケンス要約、API契約/ドリフト可視化",
            "main_actions": "meta create / pyspark / meta run / model ops / full pipeline / live status / async xai loop-monitor",
        },
        {
            "tab": "Operations.モデル・リソース関連",
            "purpose": "外生変数・モデル評価・リソース情報の関連を統合分析",
            "inputs": "model/meta/resources テーブル",
            "outputs": "相関、寄与率、ベイズ事後、Shapley近似、因果ATE代理",
            "main_actions": "関係探索、仮説検証、優先改善点抽出",
        },
        {
            "tab": "Operations.実測vs予測",
            "purpose": "予測値と実測値を重ねて評価",
            "inputs": "artifacts/*/forecast.parquet, dataset.loto_y_ts_unified",
            "outputs": "時系列重ね描き、MAE/RMSE/MAPE",
            "main_actions": "run_id選択、予測列選択、誤差確認",
        },
        {
            "tab": "Resources Analytics",
            "purpose": "resources系の統計分析と可視化",
            "inputs": "resources.run/stage_span/resource_metric",
            "outputs": "状態分布、処理時間、メトリクス推移図",
            "main_actions": "status/day/stage/metric分析",
        },
        {
            "tab": "Schema Export",
            "purpose": "スキーマ情報とバックアップ/リストア用ファイル出力",
            "inputs": "information_schema / pg_catalog (selected schemas/tables)",
            "outputs": "snapshot(json/csv/yaml/md/html), backup.sh, restore.sh, manifest.json",
            "main_actions": "Copy/Download/Read/Script generate",
        },
        {
            "tab": "Directory Compiler",
            "purpose": "複数形式ファイルの集約確認",
            "inputs": "任意ディレクトリ",
            "outputs": "compiled payload + rich preview",
            "main_actions": "compile/export/read aloud",
        },
        {
            "tab": "Markdown Docs Compiler",
            "purpose": "md資料群の統合ドキュメント化",
            "inputs": "複数ルートの *.md",
            "outputs": "統合Markdown + エクスポート",
            "main_actions": "compile/render/export",
        },
        {
            "tab": "Code Maps",
            "purpose": "コード構造の解析可視化",
            "inputs": "src/scripts python code",
            "outputs": "Mermaid, Sankey, Sunburst, edge表",
            "main_actions": "依存把握・構造レビュー",
        },
        {
            "tab": "External Targets",
            "purpose": "trend/timesfm外部資産の探索と可視化",
            "inputs": "/mnt/e/env/ts/lib_ana/src/... ",
            "outputs": "構造・コンパイル・解析・起動",
            "main_actions": "外部app起動、資料確認",
        },
        {
            "tab": "Command Lab",
            "purpose": "CLI引数説明つきコマンド作成/実行",
            "inputs": "loto_forecast.cli argparse定義",
            "outputs": "生成コマンド/.sh/.py/進捗バー付き実行結果",
            "main_actions": "copy/download/create/live run",
        },
    ]
    _show_df(pd.DataFrame(rows), hide_index=True)

    st.markdown("### Suggested Workflows")
    st.markdown(
        """
1. `Overview` -> `Resources Analytics` で実行状況を把握
2. `Operations` で run_id / table を深掘り
3. `Directory Compiler` / `Markdown Docs Compiler` で資料を統合
4. `Code Maps` / `External Targets` で構造確認
5. `Command Lab` で再現コマンドを生成・保存・実行
"""
    )



def _render_observability_panel() -> None:
    """Render local logs, screenshots, traces, metrics, and duplicate diagnostics.

    This panel intentionally reads local artifact files only. It does not write to
    PostgreSQL and it does not run browser automation unless the user runs the
    explicit CLI script from a shell.
    """

    st.subheader("観測・診断センター")
    st.caption(
        "ブラウザスクリーンショット、console/networkログ、trace、アプリ内イベント、重複エラーをローカルで集約します。"
    )

    try:
        record_event(source="operations_dashboard", category="ui_panel", level="INFO", message="observability panel opened")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"観測イベントの保存に失敗しました: {exc}")

    limit = st.slider("読み込むイベント数", min_value=100, max_value=10000, value=2000, step=100)
    snapshot = build_observability_snapshot(limit=int(limit))
    summary = summarize_observability(load_recent_events(limit=int(limit)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("イベント数", snapshot.total_events)
    c2.metric("ブラウザRun", len(snapshot.browser_runs))
    c3.metric("スクリーンショット", snapshot.screenshot_count)
    c4.metric("エラー検出", len(snapshot.error_findings))

    if snapshot.error_findings:
        st.error("エラー候補があります。下の `エラー早期検知` を確認してください。")
    elif snapshot.duplicate_groups:
        st.warning("重複イベントがあります。下の `重複検知` を確認してください。")
    else:
        st.success("直近イベントでは重大な重複・エラー候補は検出されていません。")

    tabs = st.tabs(["概要", "エラー早期検知", "重複検知", "ブラウザ収集", "イベントログ", "運用コマンド"])

    with tabs[0]:
        st.markdown("### レベル別・カテゴリ別")
        left, right = st.columns(2)
        with left:
            level_df = pd.DataFrame(
                [{"level": key, "count": value} for key, value in sorted(snapshot.level_counts.items())]
            )
            if not level_df.empty:
                st.dataframe(level_df, use_container_width=True, hide_index=True)
                st.bar_chart(level_df.set_index("level"))
            else:
                st.info("イベントはまだありません。")
        with right:
            category_df = pd.DataFrame(
                [{"category": key, "count": value} for key, value in sorted(snapshot.category_counts.items())]
            )
            if not category_df.empty:
                st.dataframe(category_df, use_container_width=True, hide_index=True)
                st.bar_chart(category_df.set_index("category"))
            else:
                st.info("カテゴリ別イベントはまだありません。")

        st.markdown("### 保存先")
        st.code(str(OBSERVABILITY_ROOT), language="text")
        st.json(
            {
                "generated_at": snapshot.generated_at,
                "events_path": snapshot.events_path,
                "project_root": snapshot.project_root,
                "latest_event": snapshot.latest_event,
            }
        )

    with tabs[1]:
        st.markdown("### エラー早期検知")
        if snapshot.error_findings:
            err_df = pd.DataFrame([asdict(item) for item in snapshot.error_findings])
            st.dataframe(err_df, use_container_width=True, hide_index=True)
        else:
            st.success("ERROR/CRITICAL/Traceback系のイベントはありません。")
        st.caption("判定対象: Traceback、Exception、ModuleNotFoundError、timeout、failed 等を含むイベント。")

    with tabs[2]:
        st.markdown("### 重複イベント検知")
        if snapshot.duplicate_groups:
            dup_df = pd.DataFrame([asdict(item) for item in snapshot.duplicate_groups])
            st.dataframe(dup_df, use_container_width=True, hide_index=True)
        else:
            st.success("同一fingerprintの重複イベントは検出されていません。")
        st.caption("source/category/level/message/exception/path からSHA-256 fingerprintを作り、繰り返しを検出します。")

    with tabs[3]:
        st.markdown("### ブラウザ収集結果")
        if snapshot.browser_runs:
            run_df = pd.DataFrame(snapshot.browser_runs)
            st.dataframe(run_df, use_container_width=True, hide_index=True)
            latest_run = snapshot.browser_runs[0]
            st.markdown("#### 最新Run")
            latest_progress = latest_run.get("progress")
            if isinstance(latest_progress, dict):
                done = int(latest_progress.get("done", 0) or 0)
                total = max(1, int(latest_progress.get("total", 1) or 1))
                progress_value = min(1.0, max(0.0, done / total))
                st.progress(progress_value, text=f"{latest_progress.get('stage', 'progress')} / {done}/{total} ({progress_value:.1%})")
                cprog1, cprog2, cprog3, cprog4 = st.columns(4)
                cprog1.metric("クリック", int(latest_progress.get("clicks", 0) or 0))
                cprog2.metric("スクリーンショット", int(latest_progress.get("screenshots", 0) or 0))
                cprog3.metric("スキップ", int(latest_progress.get("skipped", 0) or 0))
                cprog4.metric("警告", int(latest_progress.get("warnings", 0) or 0))
                st.caption(str(latest_progress.get("message", "")))
            st.json(latest_run)
            run_path = Path(str(latest_run.get("path", "")))
            progress_path = run_path / "progress.jsonl"
            if progress_path.exists():
                try:
                    progress_items = [
                        json.loads(line)
                        for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                        if line.strip()
                    ]
                    if progress_items:
                        with st.expander("収集進捗ログ", expanded=False):
                            progress_df = pd.DataFrame(progress_items)
                            visible_progress_cols = [
                                col
                                for col in [
                                    "ts",
                                    "status",
                                    "stage",
                                    "percent",
                                    "done",
                                    "total",
                                    "clicks",
                                    "screenshots",
                                    "skipped",
                                    "warnings",
                                    "message",
                                ]
                                if col in progress_df.columns
                            ]
                            st.dataframe(progress_df[visible_progress_cols], use_container_width=True, hide_index=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"進捗ログを読み込めませんでした: {exc}")
            screenshots = sorted(run_path.rglob("*.png")) if run_path.exists() else []
            if screenshots:
                preview_count = st.slider("プレビュー枚数", min_value=1, max_value=min(20, len(screenshots)), value=min(6, len(screenshots)))
                for image_path in screenshots[:preview_count]:
                    st.image(str(image_path), caption=str(image_path.relative_to(PROJECT_ROOT)) if image_path.is_relative_to(PROJECT_ROOT) else str(image_path))
        else:
            st.info("ブラウザ収集結果はまだありません。`run_dashboard_observability.sh` を実行してください。")

    with tabs[4]:
        st.markdown("### 直近イベントログ")
        events = load_recent_events(limit=int(limit))
        if events:
            event_df = pd.DataFrame(events)
            visible_cols = [col for col in ["ts", "level", "source", "category", "message", "run_id", "fingerprint"] if col in event_df.columns]
            st.dataframe(event_df[visible_cols], use_container_width=True, hide_index=True)
            with st.expander("Raw events", expanded=False):
                st.json(events[-50:])
        else:
            st.info("イベントログはまだありません。")

    with tabs[5]:
        st.markdown("### 推奨コマンド")
        st.code(
            """# 1) dashboard + browser観測環境を作成し、起動・収集・要約まで実行
export UV_LINK_MODE=copy
./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2

# 2) 既にdashboardが起動済みの場合
PYTHONPATH=src uv run --no-sync python scripts/collect_browser_observability.py \\
  --url http://localhost:8505 --max-clicks 40 --max-depth 2

# 3) 収集済みログをJSONで要約
PYTHONPATH=src uv run --no-sync python scripts/observability_summary.py""",
            language="bash",
        )
        st.warning("safe-clicks既定では `db-init`、削除、初期化、実行系ラベルはクリックしません。 unsafe-clicks は検証環境限定です。")

def main() -> None:
    st.set_page_config(page_title="ロト予測 運用ダッシュボード", layout="wide")
    _inject_modern_theme()
    _inject_document_metadata()
    st.title("ロト予測 運用ダッシュボード")
    st.caption("運用・実行結果・分析・検定・可視化・成果物管理を統合表示します。")
    render_notification_center(drain_notifications())

    with st.sidebar:
        st.markdown('<div class="ops-sidebar-title">DB接続</div>', unsafe_allow_html=True)
        host = st.text_input("ホスト", value=settings.db_host, help="PostgreSQL host")
        port = st.number_input("ポート", min_value=1, max_value=65535, value=int(settings.db_port), step=1)
        user = st.text_input("ユーザー", value=settings.db_user)
        password_input = st.text_input(
            "パスワード",
            value="",
            type="password",
            placeholder=f"空欄時はサーバー設定の {PASSWORD_ENV_VAR_NAME} を使用",
            help=f"UI には保存済みパスワードを表示しません。空欄時はサーバー設定の {PASSWORD_ENV_VAR_NAME} を使用します。",
        )
        database = st.text_input("データベース", value=settings.db_name)
        password = _resolve_effective_password(password_input)

        with st.expander("DB接続情報まとめ / コピー", expanded=False):
            runtime_env = detect_execution_environment()
            st.caption("実行環境(OS判定)")
            st.json(runtime_env)

            conn_payload = _db_connection_payload(
                host=host, port=int(port), user=user, password=password, database=database
            )
            visible_payload = dict(conn_payload)
            visible_payload["password"] = _masked_secret(str(conn_payload.get("password", "")))
            visible_payload["password_source"] = "manual_input" if str(password_input).strip() else PASSWORD_ENV_VAR_NAME
            payload_json_visible = json.dumps(visible_payload, ensure_ascii=False, indent=2)
            st.code(payload_json_visible, language="json")
            _render_copy_button(
                payload_json_visible,
                key="copy_db_conn_json",
                label="マスク済み接続情報(JSON)をコピー",
            )

            kv_line_visible = (
                f"host={visible_payload['host']} "
                f"port={visible_payload['port']} "
                f"user={visible_payload['user']} "
                f"password={visible_payload['password']} "
                f"database={visible_payload['database']}"
            )
            kv_line_copy = (
                f"host={conn_payload['host']} "
                f"port={conn_payload['port']} "
                f"user={conn_payload['user']} "
                f"database={conn_payload['database']} "
                f"{_safe_db_env_assignment()}"
            )
            st.code(kv_line_visible, language="text")
            _render_copy_button(kv_line_copy, key="copy_db_conn_kv", label="接続情報(1行)をコピー")
            st.caption(f"実パスワードはコピーしません。必要な場合は shell で `{PASSWORD_ENV_VAR_NAME}` を設定してください。")
        row_limit = st.slider("一覧の上限行数", min_value=20, max_value=1000, value=120, step=20)
        sample_limit = st.slider("サンプル表示行数", min_value=20, max_value=1000, value=100, step=20)

        with st.expander("表示・性能・ログ設定", expanded=False):
            st.toggle("通知音 ON/OFF", value=True, key="ui_notify_beep_enabled")
            st.slider("通知音量", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="ui_notify_beep_volume")
            st.toggle("メール通知 dry-run", value=True, key="ui_notify_email_dry_run")
            st.toggle("ヘルプを表示", value=True, key="ui_show_help")
            st.toggle("高速クエリキャッシュ", value=True, key="ui_enable_query_cache")
            st.slider(
                "クエリキャッシュTTL(秒)", min_value=3, max_value=120, value=15, step=1, key="ui_query_cache_ttl_sec"
            )
            st.slider(
                "キャッシュ最大件数", min_value=50, max_value=1000, value=256, step=10, key="ui_query_cache_max_entries"
            )
            st.slider("遅延クエリ閾値(ms)", min_value=100, max_value=5000, value=800, step=50, key="ui_slow_query_ms")
            st.toggle("イベントログ保存", value=True, key="ui_enable_event_log")
            st.toggle("高速単一パネルモード", value=True, key="ui_single_panel_mode")
            st.toggle("並列クエリ実行", value=True, key="ui_enable_parallel_query")
            st.slider("並列ワーカー数", min_value=1, max_value=16, value=4, step=1, key="ui_parallel_workers")
            st.selectbox("解析計算デバイス", ["auto", "cpu", "gpu"], index=0, key="ui_compute_device")
            st.text_input(
                "コピー時cdパス(任意)",
                value=str(st.session_state.get("ui_copy_cwd", "")),
                placeholder=str(PROJECT_ROOT),
                key="ui_copy_cwd",
                help="ここにパスを入れると、コマンド系コピー時に先頭へ cd を追加します。",
            )
            if st.button("クエリキャッシュをクリア", key="clear_query_cache"):
                _clear_query_cache()
                st.success("クエリキャッシュをクリアしました。")
            st.caption("キャッシュ状態")
            st.json(_query_cache_stats_snapshot())
            st.caption("GPU状態")
            st.json(_gpu_runtime_info())
        os.environ["LF_NOTIFY_BEEP_ENABLED"] = "1" if bool(st.session_state.get("ui_notify_beep_enabled", True)) else "0"
        os.environ["LF_NOTIFY_BEEP_VOLUME"] = str(float(st.session_state.get("ui_notify_beep_volume", 0.35) or 0.35))
        os.environ["LF_NOTIFY_DRY_RUN"] = "1" if bool(st.session_state.get("ui_notify_email_dry_run", True)) else "0"
        email_status = "有効" if EmailNotifier().is_configured() else "未設定(dry-run)"
        st.caption(f"通知設定: email={email_status} / 宛先={os.getenv('LF_NOTIFY_TO', 'zakumagahiyakesita@gmail.com')}")

        with st.expander("パラメータ補助", expanded=False):
            st.markdown(
                """
`一覧の上限行数` はDB一覧系クエリの取得件数上限です。
`サンプル表示行数` はテーブルサンプル表示の行数上限です。
DB未接続でも `運用.Runner` / `ディレクトリ統合` / `コードマップ` / `外部ターゲット` / `コマンドラボ` / `成果物・ログ` は利用できます。
"""
            )

        with st.expander("Quick Launch: lib_analysis v10", expanded=False):
            default_ext_app = Path("/mnt/e/env/ts/lib_ana/src/ui/lib_analysis/v10/streamlit_app/app.py")
            ext_port = st.number_input(
                "v10 port", min_value=1, max_value=65535, value=8505, step=1, key="v10_quick_port"
            )
            launch_cmd = f"streamlit run {shlex.quote(str(default_ext_app))} --server.port {int(ext_port)}"
            st.code(launch_cmd + " &", language="bash")
            _render_copy_button(launch_cmd + " &", key="copy_quick_v10", label="v10起動コマンドをコピー")
            if st.button("v10アプリをバックグラウンド起動", key="launch_quick_v10"):
                if default_ext_app.exists() and default_ext_app.is_file():
                    st.session_state["quick_v10_launch"] = _start_background_command(
                        launch_cmd,
                        cwd=default_ext_app.parent,
                    )
                else:
                    st.error(f"not found: {default_ext_app}")
            if "quick_v10_launch" in st.session_state:
                st.json(st.session_state["quick_v10_launch"])

        if st.button("再読込", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            _log_dashboard_event("manual_refresh", {"by": "sidebar_refresh"})

    dsn = _dsn(host=host, port=int(port), user=user, password=password, database=database)
    engine = _engine_for_dsn(dsn)
    connected, err = _try_connect(engine)
    if connected:
        st.success(f"接続成功: {host}:{port}/{database}")
        _log_dashboard_event("db_connect_ok", {"host": host, "port": int(port), "database": database})
    else:
        st.warning("DB接続は未確立です。DB非依存のCockpit/観測/成果物/ログは利用できます。")
        with st.expander("DB接続失敗の詳細", expanded=False):
            st.error(err)
            st.code("cp -n .env.example .env\nchmod 600 .env\nnano .env  # DB_PASSWORD=... を設定\n./run_operations_dashboard.sh", language="bash")
        _log_dashboard_event(
            "db_connect_error",
            {"host": host, "port": int(port), "database": database, "error": err[:500]},
            level="ERROR",
        )

    if bool(st.session_state.get("ui_show_help", True)):
        with st.expander("使い方クイックガイド", expanded=False):
            st.markdown(
                """
0. 初見は `表示パネル(高速モード)` の `NeuralForecast Cockpit` から開始
1. `概要` で全体状態を確認
2. `運用` で run / meta / model の詳細を調査
3. `運用 -> メタ深層分析` で検定・再帰分析・可視化を実行
4. `運用 -> Runner` で実行系コマンドを進捗付きで運用
5. `成果物・ログ` で履歴と差分を確認
"""
            )

    panel_names = [
        "NeuralForecast Cockpit",
        "概要",
        "運用",
        "NeuralForecast 詳細ラボ",
        "リソース分析",
        "スキーマ出力",
        "DB管理/ER",
        "ディレクトリ統合",
        "Markdown統合",
        "コードマップ",
        "外部ターゲット",
        "コマンドラボ",
        "成果物・ログ",
        "観測・診断",
        "機能ガイド",
        "履歴・解説",
    ]

    if connected:
        tables = _existing_tables(engine)
    else:
        tables = set()

    single_panel_mode = bool(st.session_state.get("ui_single_panel_mode", True))
    selected_panel = st.session_state.get("ui_active_panel", panel_names[0]) if st.session_state.get("ui_active_panel", panel_names[0]) in panel_names else panel_names[0]
    if single_panel_mode:
        with st.sidebar:
            selected_panel = st.selectbox(
                "表示パネル(高速モード)",
                panel_names,
                index=panel_names.index(st.session_state.get("ui_active_panel", panel_names[0]))
                if st.session_state.get("ui_active_panel", panel_names[0]) in panel_names
                else 0,
                key="ui_active_panel",
            )
            st.caption("高速モードでは選択パネルのみ計算されるため、画面操作時の停止を抑制できます。")

    def _render_one_panel(name: str) -> None:
        if name == "概要":
            if connected:
                _render_overview(engine, tables)
            else:
                st.warning("DB未接続です。DB状態はNeuralForecast CockpitのDB接続タブで確認できます。")
                render_neuralforecast_cockpit(
                    connected=False,
                    db_error=err,
                    engine=None,
                    tables=tables,
                    row_limit=row_limit,
                    sample_limit=sample_limit,
                    host=host,
                    port=int(port),
                    user=user,
                    database=database,
                    render_legacy_lab=None,
                )
            return
        if name == "運用":
            if connected:
                sub = st.tabs(
                    [
                        "実行履歴",
                        "Exogテーブル",
                        "モデル/グリッド/メタ",
                        "メタ深層分析",
                        "モデル・リソース関連",
                        "機能動作確認",
                        "モデル解析ラボ",
                        "実測vs予測",
                        "テーブル検査",
                        "Runner",
                    ]
                )
                with sub[0]:
                    _render_runs(engine, tables, row_limit=row_limit)
                with sub[1]:
                    _render_exog_tables(engine, tables, sample_limit=sample_limit)
                with sub[2]:
                    _render_dataset_model_grid(engine, tables, row_limit=row_limit)
                with sub[3]:
                    _render_meta_deep_analysis(engine, tables, row_limit=row_limit)
                with sub[4]:
                    _render_model_resource_relationships(engine, tables, row_limit=row_limit)
                with sub[5]:
                    _render_feature_verification_lab(engine, tables, row_limit=row_limit)
                with sub[6]:
                    _render_model_analysis_lab(engine, tables, row_limit=row_limit)
                with sub[7]:
                    _render_actual_vs_forecast(engine, tables)
                with sub[8]:
                    _render_table_inspector(engine, tables, sample_limit=sample_limit)
                with sub[9]:
                    _render_operation_runner(
                        host=host,
                        port=int(port),
                        user=user,
                        password=password,
                        database=database,
                        engine=engine,
                    )
            else:
                st.warning("DB未接続ですが 機能動作確認 / モデル解析 / 実測vs予測(予測のみ) / Runner は利用できます。")
                sub = st.tabs(["機能動作確認", "モデル解析ラボ", "実測vs予測", "Runner"])
                with sub[0]:
                    _render_feature_verification_lab(None, tables, row_limit=row_limit)
                with sub[1]:
                    _render_model_analysis_lab(None, tables, row_limit=row_limit)
                with sub[2]:
                    _render_actual_vs_forecast(None, tables)
                with sub[3]:
                    _render_operation_runner(
                        host=host,
                        port=int(port),
                        user=user,
                        password=password,
                        database=database,
                        engine=None,
                    )
            return
        if name == "NeuralForecast Cockpit":
            def _legacy_nf_lab() -> None:
                _render_nf_lifecycle_lab(
                    engine=engine if connected else None,
                    tables=tables,
                    row_limit=row_limit,
                    sample_limit=sample_limit,
                    host=host,
                    port=int(port),
                    user=user,
                    database=database,
                )

            render_neuralforecast_cockpit(
                connected=connected,
                db_error=err if not connected else None,
                engine=engine if connected else None,
                tables=tables,
                row_limit=row_limit,
                sample_limit=sample_limit,
                host=host,
                port=int(port),
                user=user,
                database=database,
                render_legacy_lab=_legacy_nf_lab,
            )
            return
        if name == "NeuralForecast 詳細ラボ":
            _render_nf_lifecycle_lab(
                engine=engine if connected else None,
                tables=tables,
                row_limit=row_limit,
                sample_limit=sample_limit,
                host=host,
                port=int(port),
                user=user,
                database=database,
            )
            return
        if name == "リソース分析":
            if connected:
                _render_resources_analytics(engine, tables)
            else:
                st.warning("DB未接続のため表示できません。")
            return
        if name == "スキーマ出力":
            if connected:
                _render_schema_export(
                    engine,
                    host=host,
                    port=int(port),
                    user=user,
                    database=database,
                )
            else:
                st.warning("DB未接続のため表示できません。")
            return
        if name == "DB管理/ER":
            if connected:
                render_db_admin_panel(
                    engine=engine,
                    database=database,
                    row_limit=int(row_limit),
                    sample_limit=int(sample_limit),
                    show_df=_show_df,
                    query_df=_query_df,
                    table_columns=_table_columns,
                    sample_table=_sample_table,
                    exact_count=_exact_count,
                    clear_query_cache=_clear_query_cache,
                )
            else:
                st.warning("DB未接続のため表示できません。")
            return
        if name == "ディレクトリ統合":
            _render_directory_compiler()
            return
        if name == "Markdown統合":
            _render_markdown_compiler()
            return
        if name == "コードマップ":
            _render_code_maps()
            return
        if name == "外部ターゲット":
            _render_external_targets()
            return
        if name == "コマンドラボ":
            st.info("コマンドラボは一時的に無効化中です。未定義関数を解消後に再有効化してください。")
            return
        if name == "成果物・ログ":
            _render_artifacts_and_logs()
            return
        if name == "観測・診断":
            _render_observability_panel()
            return
        if name == "機能ガイド":
            _render_feature_explanations()
            return
        if name == "履歴・解説":
            _render_guide_history()
            return

    if single_panel_mode:
        _render_one_panel(selected_panel)
    else:
        tabs = st.tabs(panel_names)
        for i, name in enumerate(panel_names):
            with tabs[i]:
                _render_one_panel(name)


if __name__ == "__main__":
    main()
