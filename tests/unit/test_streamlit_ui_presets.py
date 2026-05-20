from __future__ import annotations

from loto_forecast.api.streamlit.ui.presets import (
    NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY,
    NF_TRAIN_PRESET_ACTIVE_VALUES_KEY,
    NF_TRAIN_PRESET_WIDGET_KEYS,
    apply_pending_nf_preset,
    consume_active_nf_preset,
    queue_nf_preset,
)


def test_queue_nf_preset_defers_widget_bound_keys() -> None:
    session_state = {"existing": "value"}
    preset = {
        "nf_lab_train_ui_mode": "標準",
        "nf_lab_train_backend": "ray",
        "nf_lab_train_num_samples": 30,
    }

    queue_nf_preset(session_state, preset, source="recommended")

    assert "nf_lab_train_ui_mode" not in session_state
    assert "nf_lab_train_backend" not in session_state
    assert "nf_lab_train_num_samples" not in session_state
    payload = session_state["_nf_lab_pending_preset"]
    assert payload["source"] == "recommended"
    assert payload["values"]["nf_lab_train_ui_mode"] == "標準"
    assert "nf_lab_train_ui_mode" in NF_TRAIN_PRESET_WIDGET_KEYS


def test_apply_pending_nf_preset_updates_state_before_widget_render() -> None:
    session_state = {
        "_nf_lab_pending_preset": {
            "source": "quick",
            "values": {
                "nf_lab_train_ui_mode": "かんたん",
                "nf_lab_train_backend": "optuna",
                "nf_lab_train_num_samples": 10,
            },
        }
    }

    applied = apply_pending_nf_preset(session_state)

    assert applied == {
        "nf_lab_train_ui_mode": "かんたん",
        "nf_lab_train_backend": "optuna",
        "nf_lab_train_num_samples": 10,
    }
    assert session_state["nf_lab_train_ui_mode"] == "かんたん"
    assert session_state["nf_lab_train_backend"] == "optuna"
    assert "_nf_lab_pending_preset" not in session_state
    assert session_state[NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY] == "quick"
    assert session_state[NF_TRAIN_PRESET_ACTIVE_VALUES_KEY]["nf_lab_train_num_samples"] == 10


def test_consume_active_nf_preset_clears_metadata() -> None:
    session_state = {
        NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY: "recommended",
        NF_TRAIN_PRESET_ACTIVE_VALUES_KEY: {
            "nf_lab_train_backend": "ray",
            "nf_lab_train_search_alg_choice": "BasicVariantGenerator",
        },
    }

    source, values = consume_active_nf_preset(session_state)

    assert source == "recommended"
    assert values["nf_lab_train_backend"] == "ray"
    assert NF_TRAIN_PRESET_ACTIVE_SOURCE_KEY not in session_state
    assert NF_TRAIN_PRESET_ACTIVE_VALUES_KEY not in session_state
