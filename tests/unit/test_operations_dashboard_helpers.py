from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from loto_forecast.api.streamlit import operations_dashboard as dashboard
from loto_forecast.api.streamlit import operations_dashboard_helpers as helpers


def test_normalize_and_decode_optional_train_core_values() -> None:
    assert helpers.normalize_optional_train_core_value(None) is None
    assert helpers.normalize_optional_train_core_value("  ") is None
    assert helpers.normalize_optional_train_core_value("None") is None
    assert helpers.normalize_optional_train_core_value(" optuna ") == "optuna"
    assert helpers.decode_optional_train_core_choice("backend", " None ") is None
    assert helpers.decode_optional_train_core_choice("other", " raw ") == " raw "


def test_default_search_alg_for_backend() -> None:
    assert helpers.default_search_alg_for_backend("optuna", None) == "RandomSampler"
    assert helpers.default_search_alg_for_backend("ray", "OptunaSearch") == "OptunaSearch"
    assert helpers.default_search_alg_for_backend("other", "abc") == "abc"


def test_available_dataframe_backends_and_support_df(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"polars", "ray", "ray.data"}
    monkeypatch.setattr(helpers, "module_exists", lambda name: name in available)
    backends = helpers.available_dataframe_backends()
    assert backends == ["pandas", "polars", "ray"]

    support_df = helpers.dataset_loader_support_df()
    assert not support_df.empty
    assert set(support_df["input_method"]) == {"db_table", "db_sql", "csv", "parquet", "json"}


def test_supported_backend_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "available_dataframe_backends", lambda: ["pandas", "spark"])
    assert helpers.supported_backends_for_input_method("csv") == ["pandas", "spark"]
    assert helpers.is_supported_backend_for_input_method("csv", "spark") is True
    assert helpers.is_supported_backend_for_input_method("csv", "ray") is False


def test_safe_ident_csv_list_group_validation_and_tail() -> None:
    assert helpers.safe_ident("valid_name") == '"valid_name"'
    with pytest.raises(ValueError):
        helpers.safe_ident("bad-name")

    assert helpers.csv_nonempty_list(None) == []
    assert helpers.csv_nonempty_list("a, b ,,c") == ["a", "b", "c"]
    assert helpers.csv_nonempty_list(["a", "", " b "]) == ["a", "b"]

    assert helpers.group_mode_unique_id_validation_error("other", None) is None
    assert helpers.group_mode_unique_id_validation_error("loto_unique_id_ts_type", "u1") is None
    assert helpers.group_mode_unique_id_validation_error("loto_unique_id_ts_type", "") is not None
    assert helpers.safe_tail("1\n2\n3", max_lines=2) == "2\n3"


def test_stable_json_dumps_query_cache_slug_payload_and_format_bytes() -> None:
    dumped = helpers.stable_json_dumps({"b": 1, "a": 2})
    assert dumped.startswith('{"a": 2')

    engine = create_engine("sqlite://")
    key1 = helpers.query_cache_key(engine, "select 1", {"a": 1})
    key2 = helpers.query_cache_key(engine, "select 1", {"a": 1})
    assert key1 == key2

    assert helpers.slug("A/B test") == "a_b_test"
    assert helpers.db_connection_payload("h", 5432, "u", "p", "d")["port"] == 5432
    assert helpers.format_bytes(None) == "n/a"
    assert helpers.format_bytes(1024) == "1.00 KB"


def test_nf_lab_ui_state_helper_paths(tmp_path: Path) -> None:
    payload_file = tmp_path / "ui_state.json"
    payload_file.write_text(json.dumps({"payload": {"nf_lab_train_backend": "optuna"}}), encoding="utf-8")

    assert helpers.nf_lab_ui_state_persistable_key("nf_lab_train_backend", ("nf_lab_train_",)) is True
    assert helpers.nf_lab_ui_state_persistable_key("nf_lab_train_result", ("nf_lab_train_",)) is False
    assert helpers.nf_lab_ui_state_persistable_key("nf_lab_hint_btn_model", ("nf_lab_hint_open_",)) is False
    assert helpers.nf_lab_ui_state_persistable_key("nf_lab_train_submit_btn", ("nf_lab_train_",)) is False
    assert helpers.nf_lab_ui_state_persistable_key("nf_lab_bottom_combo_build_valid_only", ("nf_lab_bottom_combo_",)) is False
    assert (
        helpers.nf_lab_ui_state_storage_key("localhost", 5432, "user", "db", "operations_dashboard", "nf_scope")
        == "operations_dashboard:nf_scope:localhost:5432:user:db"
    )
    assert helpers.read_nf_lab_ui_state_file(payload_file) == {"nf_lab_train_backend": "optuna"}
    assert helpers.read_nf_lab_ui_state_file(tmp_path / "missing.json") == {}

    session_state = {
        "nf_lab_train_backend": "ray",
        "nf_lab_train_result": {"ok": True},
        "nf_lab_train_df": pd.DataFrame({"a": [1]}),
        "nf_lab_copy_button": "skip",
        "other": "skip",
    }
    collected = helpers.collect_nf_lab_ui_state_payload(session_state, prefixes=("nf_lab_train_",))
    assert collected == {"nf_lab_train_backend": "ray"}

    merged = helpers.merge_nf_lab_ui_state_payload(
        {"nf_lab_train_backend": "optuna", "nf_lab_train_result": {"ok": True}},
        {"nf_lab_axis_fixed_h": 12},
        prefixes=("nf_lab_train_", "nf_lab_axis_fixed_"),
    )
    assert merged == {"nf_lab_train_backend": "optuna", "nf_lab_axis_fixed_h": 12}


def test_normalize_df_for_streamlit() -> None:
    df = pd.DataFrame(
        {
            "uuid_col": ["x", None],
            "obj_col": [{"a": 1}, ["x", "y"]],
            "bytes_col": [b"abc", b"\xff"],
            "mixed_num": [1, 2.5],
        }
    )
    out = helpers.normalize_df_for_streamlit(df)
    assert out["obj_col"].iloc[0] == '{"a": 1}'
    assert out["bytes_col"].iloc[0] == "abc"
    assert out["bytes_col"].iloc[1] == "ff"
    assert pd.api.types.is_numeric_dtype(out["mixed_num"])


def test_normalize_status_series_handles_missing_unknown_and_aliases() -> None:
    series = dashboard._normalize_status_series(
        pd.Series([" Ready ", None, "PENDING", "queued", np.nan, ""]),
        allowed=["ready", "pending", "unknown"],
        aliases={"queued": "pending"},
        default="unknown",
    )
    assert series.tolist() == ["ready", "unknown", "pending", "pending", "unknown", "unknown"]


def test_normalize_status_series_without_pending_keeps_present_values() -> None:
    series = dashboard._normalize_status_series(
        pd.Series(["ready", "ready", "unknown"]),
        allowed=["ready", "pending", "unknown"],
        default="unknown",
    )
    order = dashboard._present_category_order(series, ["ready", "pending", "unknown"])
    assert series.tolist() == ["ready", "ready", "unknown"]
    assert order == ["ready", "unknown"]


def test_categorical_plot_helpers_support_status_groups_without_plotly_express() -> None:
    df = pd.DataFrame(
        {
            "step": ["s1", "s2", "s3"],
            "progress": [1.0, 0.35, 0.1],
            "status": ["ready", None, "mystery"],
            "run_id": ["r1", "r2", "r3"],
        }
    )
    normalized = dashboard._normalize_status_series(
        df["status"],
        allowed=["ready", "pending", "unknown"],
        aliases={"mystery": "unknown"},
        default="unknown",
    )
    df = df.assign(status=normalized)

    fig_bar = dashboard._build_categorical_bar_figure(
        df,
        x="progress",
        y="step",
        color="status",
        orientation="h",
        color_map=dashboard.STATUS_COLOR_MAP,
        color_order=dashboard._present_category_order(df["status"], dashboard.NF_LAB_STEP_STATUS_ORDER),
        title="progress",
    )
    assert [trace.name for trace in fig_bar.data] == ["ready", "unknown"]

    fig_scatter = dashboard._build_categorical_scatter_figure(
        df.assign(rows_written=[10, 20, 30], duration_sec=[1.0, 2.0, 3.0]),
        x="duration_sec",
        y="rows_written",
        color="status",
        color_map=dashboard.STATUS_COLOR_MAP,
        color_order=["ready", "pending", "unknown"],
        title="scatter",
    )
    assert [trace.name for trace in fig_scatter.data] == ["ready", "unknown"]

    fig_hist = dashboard._build_categorical_histogram_figure(
        df.assign(metric=[1.0, 2.0, 3.0]),
        x="metric",
        color="status",
        color_map=dashboard.STATUS_COLOR_MAP,
        color_order=["ready", "pending", "unknown"],
        title="hist",
    )
    assert [trace.name for trace in fig_hist.data] == ["ready", "unknown"]


def test_parameter_name_and_train_combo_validation() -> None:
    class DummyParam:
        name = "dummy"

    assert helpers.parameter_name(DummyParam()) == "dummy"
    assert helpers.parameter_name(" raw ") == "raw"
    assert helpers.is_valid_search_alg_for_backend("optuna", "TPESampler") is True
    assert helpers.is_valid_search_alg_for_backend("ray", "TPESampler") is False
    assert helpers.validate_train_combo_choice("AutoHINT", "optuna", None, None) == "AutoHINT requires backend=ray"
    assert helpers.validate_train_combo_choice("AutoNHITS", "ray", None, "TPESampler") == "invalid search_alg for ray"


def test_horizon_parsing_and_resolution() -> None:
    assert helpers.parse_horizon_axis_value(None) is None
    assert helpers.parse_horizon_axis_value(True) is None
    assert helpers.parse_horizon_axis_value("auto") is None
    assert helpers.parse_horizon_axis_value("3") == 3
    assert helpers.parse_horizon_axis_value("4.0") == 4
    assert helpers.resolve_model_horizon("Autoformer", 1) == (
        2,
        "Autoformer は h>=2 が必要なため h=1 を h=2 に自動補正しました。",
    )
    assert helpers.resolve_model_horizon("Other", 5) == (5, None)


def test_parse_json_and_flatten_json_like_value() -> None:
    assert helpers.parse_json_like('{"a": 1}') == {"a": 1}
    assert helpers.parse_json_like("[1,2]") == [1, 2]
    assert helpers.parse_json_like("plain") is None

    flat = helpers.flatten_json_like_value(
        {"a": {"b": 1}, "nums": [1, 2, 3], "rows": [{"x": 1}, {"x": 2}]},
        prefix="root",
    )
    assert flat["root.a.b"] == 1
    assert flat["root.nums.__len__"] == 3
    assert flat["root.nums.__mean__"] == 2.0
    assert flat["root.rows[0].x"] == 1


def test_flatten_json_columns_and_expand_semistructured_columns() -> None:
    df = pd.DataFrame(
        {
            "payload": ['{"a": 1, "b": {"c": 2}}', '{"a": 3}'],
            "other": ["x", "y"],
        }
    )
    out = helpers.flatten_json_columns(df, {"payload": "payload"})
    assert "payload.a" in out.columns
    assert out["payload.a"].tolist() == [1, 3]

    expanded = helpers.expand_semistructured_columns(df)
    assert "payload.a" in expanded.columns


def test_bayesian_impact_and_graph_helpers() -> None:
    df = pd.DataFrame(
        {
            "model_name": ["A", "A", "B", "B"],
            "status": ["success", "failed", "success", "success"],
        }
    )
    post = helpers.bayesian_success_posterior(df)
    assert list(post["model_name"]) == ["B", "A"]

    impact = pd.DataFrame(
        {
            "parameter": ["p1", "p2"],
            "effect_abs": [2.0, 1.0],
            "is_significant": [True, False],
        }
    )
    contrib = helpers.impact_contribution_rates(impact)
    assert pytest.approx(contrib["contribution_rate"].sum()) == 1.0

    graph = helpers.graph_centrality_from_impact(impact)
    assert graph.iloc[0]["node"] == "p1"


def test_r2_shapley_and_causal_proxy() -> None:
    df = pd.DataFrame(
        {
            "target": np.arange(30, dtype=float),
            "f1": np.arange(30, dtype=float),
            "f2": np.arange(30, dtype=float) * 2,
            "model_name": ["A"] * 15 + ["B"] * 15,
            "horizon": [1] * 30,
            "dataset_rows": [100] * 30,
            "feature_cols": [5] * 30,
        }
    )
    r2 = helpers.r2_for_features(df, "target", ["f1", "f2"])
    assert 0.0 <= r2 <= 1.0

    shap = helpers.approx_shapley_contrib(df, "target", ["f1", "f2"], n_perm=12, seed=7)
    assert set(shap["feature"]) == {"f1", "f2"}

    ate = helpers.causal_proxy_ate(df.assign(treat=np.arange(30)), "target", "treat")
    assert ate["ok"] is True
    assert ate["n"] == 30


def test_dump_selector_and_flags() -> None:
    selector = helpers.dump_selector("tables", ["dataset"], [("dataset", "t1"), ("dataset", "t2")])
    assert selector["mode"] == "tables"
    flags = helpers.dump_selector_flags(selector)
    assert "--table dataset.t1" in flags

    schema_selector = helpers.dump_selector("schemas", ["dataset", "model"], [])
    assert schema_selector["tables"] == []
    assert "--schema dataset" in helpers.dump_selector_flags(schema_selector)


def test_safe_read_json_artifact_stats_and_bundles(tmp_path: Path) -> None:
    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert helpers.safe_read_json_file(json_path) == {"a": 1}
    assert helpers.safe_read_json_file(tmp_path / "missing.json") == {}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "configuration.pkl").write_text("x", encoding="utf-8")
    (run_dir / "alias_to_model.pkl").write_text("x", encoding="utf-8")
    (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    (run_dir / "forecast.parquet").write_text("stub", encoding="utf-8")
    stats = helpers.artifact_file_stats(run_dir)
    assert stats["exists"] is True
    assert stats["file_count"] >= 4
    assert helpers.has_model_artifacts(run_dir) is True
    assert helpers.has_analysis_bundle(run_dir) is True


def test_file_scans_diff_and_summary(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md = docs_dir / "a.md"
    md.write_text("# hello", encoding="utf-8")
    jsn = docs_dir / "b.json"
    jsn.write_text('{"a": 1}', encoding="utf-8")
    pyf = docs_dir / "c.py"
    pyf.write_text("print(1)\n", encoding="utf-8")

    assert md in helpers.scan_supported_files(docs_dir)
    assert pyf in helpers.scan_diff_files(docs_dir)

    diff = helpers.unified_diff_text("a \n", "a\nb\n", "left", "right", ignore_ws=True)
    assert "right" in diff

    summary = helpers.summarize_supported_file(jsn)
    assert summary["meta"]["top_keys"] == ["a"]


def test_compile_directory_and_markdown_formats(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    md = root / "doc.md"
    md.write_text("# title\nbody", encoding="utf-8")
    csv_file = root / "table.csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")

    bundle = helpers.compile_directory_payload(root, [md, csv_file])
    assert bundle["file_count"] == 2
    md_text = helpers.compiled_to_markdown(bundle)
    assert "Directory Compile Report" in md_text
    html_text = helpers.compiled_to_html(bundle)
    assert "<html>" in html_text
    content, mime, ext = helpers.compiled_to_format(bundle, "json")
    assert mime == "application/json"
    assert ext == "json"
    assert json.loads(content)["file_count"] == 2

    files = helpers.scan_markdown_files([root])
    assert files == [md]
    md_bundle = helpers.compile_markdown_bundle([root], files)
    assert md_bundle["file_count"] == 1
    assert "Compiled Markdown Documents" in md_bundle["compiled_markdown"]


def test_module_name_and_resolve_from_import(tmp_path: Path) -> None:
    root = tmp_path / "src"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    mod = pkg / "x.py"
    mod.write_text("pass", encoding="utf-8")

    assert helpers.module_name_from_path(mod, root, "pkgroot") == "pkgroot.pkg.x"
    assert helpers.resolve_from_import("a.b.c", 1, "d.e") == "a.b.d.e"


def test_operations_dashboard_uses_helper_call_sites(tmp_path: Path) -> None:
    assert dashboard._normalize_optional_train_core_value is helpers.normalize_optional_train_core_value
    assert dashboard._default_search_alg_for_backend is helpers.default_search_alg_for_backend
    assert dashboard._dump_selector is helpers.dump_selector
    assert dashboard._compile_directory_payload is helpers.compile_directory_payload
    assert dashboard._expand_semistructured_columns is helpers.expand_semistructured_columns
    assert dashboard._safe_read_json_file is helpers.safe_read_json_file
    assert dashboard._artifact_file_stats is helpers.artifact_file_stats
    assert dashboard._has_analysis_bundle is helpers.has_analysis_bundle

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"payload": {"nf_lab_train_backend": "optuna"}}), encoding="utf-8")
    assert dashboard._read_nf_lab_ui_state_file(state_file) == {"nf_lab_train_backend": "optuna"}
    assert dashboard._nf_lab_ui_state_persistable_key("nf_lab_train_backend") is True
    assert dashboard._nf_lab_ui_state_persistable_key("nf_lab_train_result") is False
    assert dashboard._nf_lab_ui_state_persistable_key("nf_lab_bottom_combo_build_valid_only") is False
    assert dashboard._nf_lab_ui_state_storage_key("host", 5432, "user", "db").startswith("operations_dashboard:")
