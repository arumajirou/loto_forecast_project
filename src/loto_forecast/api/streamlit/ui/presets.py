from __future__ import annotations

from typing import Any

NF_TRAIN_PRESET_PENDING_KEY = "_nf_lab_pending_preset"
NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY = "_nf_lab_active_preset_source"
NF_TRAIN_PRESET_ACTIVE_VALUES_KEY = "_nf_lab_active_preset_values"
NF_TRAIN_PRESET_WIDGET_KEYS = {
    "nf_lab_train_model",
    "nf_lab_train_backend",
    "nf_lab_train_num_samples",
    "nf_lab_train_loss_name",
    "nf_lab_train_search_alg_choice",
    "nf_lab_train_ui_mode",
    "nf_lab_train_use_max_resources",
    "nf_lab_train_run_cv",
    "nf_lab_train_h",
    "nf_lab_train_dataset_input_method",
    "nf_lab_train_dataframe_backend",
}


def available_nf_presets(default_horizon: int) -> dict[str, dict[str, Any]]:
    return {
        "最短で試す": {
            "nf_lab_train_model": "AutoNHITS",
            "nf_lab_train_backend": "optuna",
            "nf_lab_train_num_samples": 10,
            "nf_lab_train_loss_name": "MAE",
            "nf_lab_train_search_alg_choice": "TPESampler",
            "nf_lab_train_ui_mode": "かんたん",
            "nf_lab_train_use_max_resources": True,
            "nf_lab_train_run_cv": False,
            "nf_lab_train_h": int(default_horizon),
            "nf_lab_train_dataset_input_method": "db_table",
            "nf_lab_train_dataframe_backend": "pandas",
        },
        "おすすめ設定を自動入力": {
            "nf_lab_train_model": "AutoNHITS",
            "nf_lab_train_backend": "ray",
            "nf_lab_train_num_samples": 30,
            "nf_lab_train_loss_name": "MAE",
            "nf_lab_train_search_alg_choice": "BasicVariantGenerator",
            "nf_lab_train_ui_mode": "標準",
            "nf_lab_train_use_max_resources": True,
            "nf_lab_train_run_cv": True,
            "nf_lab_train_h": int(default_horizon),
            "nf_lab_train_dataset_input_method": "db_table",
            "nf_lab_train_dataframe_backend": "pandas",
        },
    }


def queue_nf_preset(
    session_state: Any,
    preset: dict[str, Any],
    *,
    source: str = "manual",
    pending_key: str = NF_TRAIN_PRESET_PENDING_KEY,
) -> dict[str, Any]:
    payload = {
        "source": str(source),
        "values": dict(preset),
    }
    session_state[pending_key] = payload
    return payload


def apply_pending_nf_preset(
    session_state: Any,
    *,
    pending_key: str = NF_TRAIN_PRESET_PENDING_KEY,
) -> dict[str, Any]:
    payload = session_state.pop(pending_key, None)
    if not isinstance(payload, dict):
        return {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return {}
    applied: dict[str, Any] = {}
    for key, value in values.items():
        session_state[str(key)] = value
        applied[str(key)] = value
    session_state[NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY] = str(payload.get("source") or "manual")
    session_state[NF_TRAIN_PRESET_ACTIVE_VALUES_KEY] = dict(applied)
    return applied


def consume_active_nf_preset(session_state: Any) -> tuple[str, dict[str, Any]]:
    source = str(session_state.pop(NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY, "") or "")
    values = session_state.pop(NF_TRAIN_PRESET_ACTIVE_VALUES_KEY, {})
    if not isinstance(values, dict):
        values = {}
    return source, dict(values)
