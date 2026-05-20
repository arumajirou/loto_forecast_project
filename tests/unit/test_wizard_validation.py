from __future__ import annotations

from loto_forecast.api.streamlit.ui.wizard import build_nf_wizard_state


def test_wizard_flags_missing_required_fields() -> None:
    state = build_nf_wizard_state(
        mode="かんたん",
        action="学習(train)",
        model="AutoNHITS",
        dataset_input_method="db_table",
        dataset_table="",
        dataset_path="",
        dataset_sql="",
        unique_ids=[],
        ts_types=[],
        horizon=0,
        backend="optuna",
        loss="MAE",
        search_alg="TPESampler",
        combo_total=0,
        combo_skip_reasons={"unique_id 候補は必須です": 3},
        command_preview="python -m loto_forecast.cli train",
    )

    assert state.can_run is False
    assert "データテーブル" in state.required_missing
    assert "ts_type" in state.required_missing
    assert any(issue.reason.startswith("有効な総当たり組合せ数が 0") for issue in state.issues)


def test_wizard_ready_path_produces_next_actions() -> None:
    state = build_nf_wizard_state(
        mode="標準",
        action="予測(predict)",
        model="AutoNHITS",
        dataset_input_method="db_table",
        dataset_table="loto_y_ts",
        dataset_path="",
        dataset_sql="",
        unique_ids=["u1"],
        ts_types=["main"],
        horizon=12,
        backend="optuna",
        loss="MAE",
        search_alg="TPESampler",
        combo_total=1,
        combo_skip_reasons={},
        command_preview="python -m loto_forecast.cli predict --run-id r1",
    )

    assert state.can_run is True
    assert state.required_missing == []
    assert state.steps[1].status == "done"
    assert "実行前チェックでエラーなしを確認する" in state.next_actions
