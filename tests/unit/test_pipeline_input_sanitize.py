import pandas as pd

from loto_forecast.orchestration.pipeline import _sanitize_model_input


def test_sanitize_model_input_keeps_rows_with_sparse_exog():
    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "y": [1.0, 2.0, 3.0],
            "hist_sparse": [None, None, 1.0],
            "feat_sparse": [None, None, None],
        }
    )
    out = _sanitize_model_input(df)
    assert len(out) == 3
    assert list(out["unique_id"].unique()) == ["N1"]


def test_sanitize_model_input_drops_invalid_required_rows():
    df = pd.DataFrame(
        {
            "unique_id": ["N1", None, "N1"],
            "ds": ["2025-01-01", "invalid-date", "2025-01-03"],
            "y": [1.0, 2.0, None],
        }
    )
    out = _sanitize_model_input(df)
    assert len(out) == 1
    assert str(out.iloc[0]["ds"].date()) == "2025-01-01"
