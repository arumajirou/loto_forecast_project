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
    return at


def _element_values(elements: object) -> list[str]:
    values: list[str] = []
    for element in elements:
        for attr in ("value", "label"):
            raw = getattr(element, attr, None)
            if raw is not None:
                text = str(raw).strip()
                if text:
                    values.append(text)
    return values


def _tree_texts(at: AppTest) -> list[str]:
    texts: list[str] = []
    for elements in (
        at.error,
        at.warning,
        at.info,
        at.markdown,
        at.text,
        at.title,
        at.header,
        at.subheader,
    ):
        texts.extend(_element_values(elements))
    return texts


def _contains_any_text(at: AppTest, candidates: list[str]) -> bool:
    haystack = _tree_texts(at)
    return any(candidate in text for text in haystack for candidate in candidates)


def _debug_counts(at: AppTest) -> dict[str, int]:
    return {
        "exception": len(at.exception),
        "error": len(at.error),
        "warning": len(at.warning),
        "info": len(at.info),
        "title": len(at.title),
        "header": len(at.header),
        "subheader": len(at.subheader),
        "button": len(at.button),
        "selectbox": len(at.selectbox),
        "tabs": len(at.tabs),
    }


def test_operations_dashboard_initial_render_without_exception() -> None:
    at = _build_app_test()

    at.run(timeout=120)

    assert len(at.exception) == 0, _debug_counts(at)
    assert at.title or at.header, _debug_counts(at)
    assert any("ロト予測 運用ダッシュボード" in item.value for item in at.title), _debug_counts(at)
    assert _contains_any_text(at, ["DB接続"]), _debug_counts(at)
    assert "表示パネル(高速モード)" in [widget.label for widget in at.sidebar.selectbox]

    db_related_tokens = ["DB接続", "DB未接続", "DB停止中"]
    if _contains_any_text(at, db_related_tokens):
        assert _contains_any_text(at, db_related_tokens)


def test_operations_dashboard_password_input_starts_blank() -> None:
    at = _build_app_test()

    at.run(timeout=120)

    password_widgets = [widget for widget in at.text_input if widget.label == "パスワード"]
    assert password_widgets, _debug_counts(at)
    assert password_widgets[0].value == ""


def test_operations_dashboard_query_params_do_not_break_initial_state() -> None:
    at = _build_app_test()
    at.query_params["panel"] = "運用"
    at.query_params["nf_section"] = "学習(train)"

    at.run(timeout=120)

    assert len(at.exception) == 0
    assert at.query_params["panel"] == ["運用"]
    assert at.session_state["ui_active_panel"] == "概要"


def test_operations_dashboard_nf_lab_train_panel_renders_core_widgets() -> None:
    at = _build_app_test()
    at.session_state["ui_active_panel"] = "NeuralForecast 実行・検証ラボ"

    at.run(timeout=120)
    nf_menu = next(widget for widget in at.selectbox if widget.label == "NeuralForecast 実行・検証ラボ メニュー")
    nf_menu.select("学習(train)").run(timeout=120)
    train_sub_menu = next(widget for widget in at.selectbox if widget.label == "学習(train) サブメニュー")
    train_sub_menu.select("全パラメータ選択").run(timeout=120)

    assert len(at.exception) == 0, _debug_counts(at)
    assert train_sub_menu.value == "全パラメータ選択"
    labels = [widget.label for widget in at.selectbox]
    assert "NeuralForecast 実行・検証ラボ メニュー" in labels
    assert "学習(train) サブメニュー" in labels
    button_labels = [widget.label for widget in at.button]
    assert "Run train" in button_labels
    assert "最短で試す" in button_labels
    assert "おすすめ設定を自動入力" in button_labels
    assert _contains_any_text(at, ["Step Wizard", "実行前チェック", "通知設定: email="]), _debug_counts(at)


def test_operations_dashboard_nf_lab_presets_do_not_raise_widget_key_exception() -> None:
    at = _build_app_test()
    at.session_state["ui_active_panel"] = "NeuralForecast 実行・検証ラボ"

    at.run(timeout=120)
    nf_menu = next(widget for widget in at.selectbox if widget.label == "NeuralForecast 実行・検証ラボ メニュー")
    nf_menu.select("学習(train)").run(timeout=120)
    train_sub_menu = next(widget for widget in at.selectbox if widget.label == "学習(train) サブメニュー")
    train_sub_menu.select("全パラメータ選択").run(timeout=120)

    quick_button = next(widget for widget in at.button if widget.label == "最短で試す")
    quick_button.click().run(timeout=120)

    assert len(at.exception) == 0, _debug_counts(at)
    assert at.session_state["nf_lab_train_ui_mode"] == "かんたん"
    assert at.session_state["nf_lab_train_backend"] == "optuna"

    preset_selector = next(widget for widget in at.selectbox if widget.label == "おすすめプリセット")
    preset_selector.select("おすすめ設定を自動入力").run(timeout=120)

    recommended_button = next(widget for widget in at.button if widget.label == "おすすめ設定を自動入力")
    recommended_button.click().run(timeout=120)

    assert len(at.exception) == 0, _debug_counts(at)
    assert at.session_state["nf_lab_train_ui_mode"] == "標準"
    assert at.session_state["nf_lab_train_backend"] == "ray"


def test_operations_dashboard_operations_panel_gracefully_degrades_without_db() -> None:
    at = _build_app_test()
    at.session_state["ui_active_panel"] = "運用"

    at.run(timeout=120)

    assert len(at.exception) == 0, _debug_counts(at)
    tab_labels = [widget.label for widget in at.tabs]
    assert "機能動作確認" in tab_labels
    assert "モデル解析ラボ" in tab_labels
    assert "実測vs予測" in tab_labels
    assert "Runner" in tab_labels
    assert len(at.warning) + len(at.info) + len(at.error) > 0, _debug_counts(at)
    assert _contains_any_text(
        at,
        [
            "DB未接続",
            "DB停止中",
            "status query failed",
            "forecast.parquet",
            "run_id",
        ],
    )
