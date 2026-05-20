from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "src/loto_forecast/api/streamlit/operations_dashboard.py"


def _build_app_test() -> AppTest:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.session_state["ui_single_panel_mode"] = True
    at.session_state["ui_active_panel"] = "NeuralForecast 実行・検証ラボ"
    return at


def test_nf_lab_redesign_renders_wizard_and_notification_settings() -> None:
    at = _build_app_test()

    at.run(timeout=120)
    nf_menu = next(widget for widget in at.selectbox if widget.label == "NeuralForecast 実行・検証ラボ メニュー")
    nf_menu.select("学習(train)").run(timeout=120)
    train_sub_menu = next(widget for widget in at.selectbox if widget.label == "学習(train) サブメニュー")
    train_sub_menu.select("全パラメータ選択").run(timeout=120)

    texts = [getattr(widget, "value", "") for widget in at.markdown] + [getattr(widget, "value", "") for widget in at.text]
    joined = "\n".join([str(value) for value in texts])

    assert len(at.exception) == 0
    assert any(widget.label == "おすすめプリセット" for widget in at.selectbox)
    assert any(widget.label == "Run train" for widget in at.button)
    sidebar_toggle_labels = [widget.label for widget in at.sidebar.toggle]
    assert "通知音 ON/OFF" in sidebar_toggle_labels
    assert "メール通知 dry-run" in sidebar_toggle_labels
    assert "Step Wizard" in joined


def test_nf_lab_presets_apply_distinct_values_without_exception() -> None:
    at = _build_app_test()

    at.run(timeout=120)
    nf_menu = next(widget for widget in at.selectbox if widget.label == "NeuralForecast 実行・検証ラボ メニュー")
    nf_menu.select("学習(train)").run(timeout=120)
    train_sub_menu = next(widget for widget in at.selectbox if widget.label == "学習(train) サブメニュー")
    train_sub_menu.select("全パラメータ選択").run(timeout=120)

    quick_button = next(widget for widget in at.button if widget.label == "最短で試す")
    quick_button.click().run(timeout=120)
    assert at.session_state["nf_lab_train_ui_mode"] == "かんたん"
    assert at.session_state["nf_lab_train_backend"] == "optuna"
    assert at.session_state["nf_lab_train_search_alg_choice"] == "TPESampler"
    assert at.session_state["nf_lab_train_num_samples"] == 10

    recommended_button = next(widget for widget in at.button if widget.label == "おすすめ設定を自動入力")
    recommended_button.click().run(timeout=120)
    assert at.session_state["nf_lab_train_ui_mode"] == "標準"
    assert at.session_state["nf_lab_train_backend"] == "ray"
    assert at.session_state["nf_lab_train_search_alg_choice"] == "BasicVariantGenerator"
    assert at.session_state["nf_lab_train_num_samples"] == 30
    assert len(at.exception) == 0


def test_nf_lab_followup_widgets_remain_operable_after_preset_and_input_changes() -> None:
    at = _build_app_test()

    at.run(timeout=120)
    nf_menu = next(widget for widget in at.selectbox if widget.label == "NeuralForecast 実行・検証ラボ メニュー")
    nf_menu.select("学習(train)").run(timeout=120)
    train_sub_menu = next(widget for widget in at.selectbox if widget.label == "学習(train) サブメニュー")
    train_sub_menu.select("全パラメータ選択").run(timeout=120)

    next(widget for widget in at.button if widget.label == "おすすめ設定を自動入力").click().run(timeout=120)

    backend = next(widget for widget in at.selectbox if widget.label == "backend")
    search_alg = next(widget for widget in at.selectbox if widget.label == "search_alg")
    dataset_input = next(widget for widget in at.selectbox if widget.label == "dataset input method")
    dataframe_backend = next(widget for widget in at.selectbox if widget.label == "dataframe backend")
    ts_type = next(widget for widget in at.multiselect if widget.label == "ts_type")
    unique_id = next(widget for widget in at.multiselect if widget.label == "unique_id")

    assert backend.disabled is False
    assert search_alg.disabled is False
    assert dataset_input.disabled is False
    assert dataframe_backend.disabled is False
    assert ts_type.disabled is False
    assert unique_id.disabled is False

    group_mode = next(widget for widget in at.selectbox if widget.label == "学習単位")
    group_mode.select("loto+ts_type ごと (unique_id集約)").run(timeout=120)

    backend = next(widget for widget in at.selectbox if widget.label == "backend")
    search_alg = next(widget for widget in at.selectbox if widget.label == "search_alg")
    dataset_input = next(widget for widget in at.selectbox if widget.label == "dataset input method")
    dataframe_backend = next(widget for widget in at.selectbox if widget.label == "dataframe backend")
    unique_id = next(widget for widget in at.multiselect if widget.label == "unique_id")

    assert unique_id.disabled is True
    assert backend.disabled is False
    assert search_alg.disabled is False
    assert dataset_input.disabled is False
    assert dataframe_backend.disabled is False

    dataset_input.select("csv").run(timeout=120)
    dataframe_backend = next(widget for widget in at.selectbox if widget.label == "dataframe backend")
    ts_type = next(widget for widget in at.multiselect if widget.label == "ts_type")

    assert dataframe_backend.disabled is False
    assert ts_type.disabled is False
    assert len(at.exception) == 0
