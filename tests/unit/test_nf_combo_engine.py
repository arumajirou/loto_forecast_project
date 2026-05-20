from __future__ import annotations

from loto_forecast.application.nf_combo_engine import ComboContext, evaluate_train_combinations


def test_evaluate_train_combinations_filters_invalid_rows_and_generates_fix_suggestions() -> None:
    axes = {
        "model": ["AutoNHITS"],
        "backend": ["optuna", "ray"],
        "search_alg": ["TPESampler", "BasicVariantGenerator"],
        "dataset_input_method": ["db_table", "csv"],
        "dataframe_backend": ["pandas"],
        "dataset_table": ["train_table"],
        "dataset_path": [""],
        "dataset_sql": [""],
        "group_mode": ["loto_unique_id_ts_type"],
        "unique_id": ["", "uid_1"],
        "ts_type": ["main"],
    }
    context = ComboContext(
        dataset_table="train_table",
        dataset_path="",
        dataset_sql="",
        default_values={"loss": "MAE", "valid_loss": "MAE"},
    )

    result = evaluate_train_combinations(axes, context=context)

    assert result.theoretical_count == 16
    assert len(result.valid_combinations) > 0
    assert len(result.excluded_combinations) > 0
    assert "backend_search_mismatch" in result.reason_summary
    assert any(row.reason_code == "backend_search_mismatch" for row in result.reason_rows)
    assert any(item.reason_ja for item in result.excluded_combinations)
    assert any("unique_id" in suggestion for suggestion in result.fix_suggestions)


def test_evaluate_train_combinations_returns_reason_rows_for_zero_valid_case() -> None:
    axes = {
        "model": ["AutoNHITS"],
        "backend": ["optuna"],
        "search_alg": ["BasicVariantGenerator"],
        "dataset_input_method": ["db_table"],
        "dataframe_backend": ["pandas"],
        "dataset_table": [""],
        "dataset_path": [""],
        "dataset_sql": [""],
        "group_mode": ["loto_unique_id_ts_type"],
        "unique_id": [""],
    }
    context = ComboContext(default_values={"loss": "MAE", "valid_loss": "MAE"})

    result = evaluate_train_combinations(axes, context=context)

    assert result.theoretical_count == 1
    assert result.valid_combinations == []
    assert len(result.excluded_combinations) == 1
    assert result.reason_rows[0].count == 1
    assert result.reason_rows[0].reason_ja
    assert result.fix_suggestions


def test_evaluate_train_combinations_excludes_fit_val_size_below_horizon() -> None:
    axes = {
        "model": ["AutoNHITS"],
        "backend": ["optuna"],
        "search_alg": ["TPESampler"],
        "dataset_input_method": ["db_table"],
        "dataframe_backend": ["pandas"],
        "dataset_table": ["train_table"],
        "group_mode": ["loto_unique_id_ts_type"],
        "unique_id": ["uid_1"],
        "ts_type": ["main"],
        "horizon": [2],
        "fit_val_size": [1, 2],
    }
    context = ComboContext(default_values={"loss": "MAE", "valid_loss": "MAE"})

    result = evaluate_train_combinations(axes, context=context)

    assert result.theoretical_count == 2
    assert len(result.valid_combinations) == 1
    assert result.valid_combinations[0]["fit_val_size"] == 2
    assert result.reason_summary["fit_val_size_too_small"] == 1
    assert any("horizon" in suggestion for suggestion in result.fix_suggestions)


def test_evaluate_train_combinations_keeps_fit_val_size_equal_to_horizon() -> None:
    axes = {
        "model": ["AutoNHITS"],
        "backend": ["optuna"],
        "search_alg": ["TPESampler"],
        "dataset_input_method": ["db_table"],
        "dataframe_backend": ["pandas"],
        "dataset_table": ["train_table"],
        "group_mode": ["loto_unique_id_ts_type"],
        "unique_id": ["uid_1"],
        "ts_type": ["main"],
        "horizon": [3],
        "fit_val_size": [3],
    }
    context = ComboContext(default_values={"loss": "MAE", "valid_loss": "MAE"})

    result = evaluate_train_combinations(axes, context=context)

    assert result.theoretical_count == 1
    assert len(result.valid_combinations) == 1
    assert result.valid_combinations[0]["fit_val_size"] == 3
    assert "fit_val_size_too_small" not in result.reason_summary
