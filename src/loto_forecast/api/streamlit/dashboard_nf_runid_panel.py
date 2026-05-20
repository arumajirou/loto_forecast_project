from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from loto_forecast.api.streamlit import dashboard_nf_runid_panel_analysis as panel_analysis
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_formatter as panel_formatter
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_helpers as helpers
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_state as panel_state


def _safe_pickle_summary(path: Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return {"exists": False, "path": str(p), "type": None, "keys": [], "preview": None}
    try:
        obj = pd.read_pickle(p)
    except Exception as e:
        return {"exists": True, "path": str(p), "type": "unreadable", "keys": [], "preview": str(e)}
    out: dict[str, Any] = {"exists": True, "path": str(p), "type": type(obj).__name__, "keys": [], "preview": None}
    if isinstance(obj, dict):
        out["keys"] = [str(k) for k in list(obj.keys())[:80]]
        preview: dict[str, Any] = {}
        for k in out["keys"][:20]:
            v = obj.get(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                preview[str(k)] = v
            else:
                preview[str(k)] = type(v).__name__
        out["preview"] = preview
    elif isinstance(obj, (list, tuple)):
        out["preview"] = {
            "length": int(len(obj)),
            "item_types": sorted(list({type(x).__name__ for x in list(obj)[:30]})),
        }
    else:
        out["preview"] = repr(obj)[:500]
    return out


def _to_numeric(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int_eq(a: Any, b: Any) -> bool | None:
    try:
        if a is None or b is None:
            return None
        return int(a) == int(b)
    except Exception:
        return None


def _metric_rows_from_eval(eval_obj: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(eval_obj, dict):
        return pd.DataFrame()
    metrics_obj = eval_obj.get("metrics", eval_obj)
    if not isinstance(metrics_obj, dict):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for k, v in metrics_obj.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            rows.append({"metric": str(k), "value": float(v)})
    return pd.DataFrame(rows)


def render_runid_integrated_panel(
    *,
    project_root: Path,
    run_id_options: list[str],
    run_id_to_dir: dict[str, Path],
    model_df: pd.DataFrame,
    engine: Any,
    tables: set[tuple[str, str]],
    row_limit: int,
    settings: Any,
    query_df: Callable[..., pd.DataFrame],
    show_df: Callable[..., None],
    parse_json_like: Callable[[Any], Any | None],
    safe_read_json_file: Callable[[Path], dict[str, Any]],
    has_model_artifacts: Callable[[Path], bool],
    causal_proxy_ate: Callable[..., dict[str, Any]],
    stable_json_dumps: Callable[[Any], str],
    plotly_available: bool,
    px: Any | None = None,
) -> None:
    st.caption("run-id 単位でモデル情報・設定整合性・リソース消費・精度要因を統合表示します。")
    if not run_id_options:
        st.info("解析対象 run-id が見つかりません。")
        return

    sel_run = st.selectbox("解析 run-id", run_id_options, index=0, key="nf_lab_runid_sel")
    sel_dir = run_id_to_dir.get(sel_run, Path(project_root) / "artifacts" / str(sel_run))
    sel_meta = safe_read_json_file(sel_dir / "meta.json")
    sel_eval = safe_read_json_file(sel_dir / "evaluation.json")

    sel_model, sel_model_row, sel_params, sel_metrics = panel_state.resolve_selected_model(
        model_df=model_df,
        sel_run=str(sel_run),
        parse_json_like=parse_json_like,
    )

    rid_tabs = st.tabs(["概要", "モデル詳細", "設定一致", "リソース", "精度", "相関/因果", "エクスポート"])

    with rid_tabs[0]:
        overview_metrics = panel_formatter.build_overview_metrics(
            sel_run=str(sel_run),
            sel_meta=sel_meta,
            sel_model_row=sel_model_row,
            sel_dir=sel_dir,
            sel_eval=sel_eval,
            settings=settings,
            has_model_artifacts=has_model_artifacts,
        )
        for col, metric in zip(st.columns(4), overview_metrics[:4], strict=False):
            col.metric(metric["label"], metric["value"])
        for col, metric in zip(st.columns(4), overview_metrics[4:], strict=False):
            col.metric(metric["label"], metric["value"])

        snapshot = helpers.build_run_snapshot(
            sel_run=str(sel_run),
            sel_dir=sel_dir,
            sel_meta=sel_meta,
            sel_model=sel_model,
            sel_model_row=sel_model_row,
            settings=settings,
            has_model_artifacts=has_model_artifacts,
        )
        st.markdown("**run snapshot**")
        st.json(snapshot)

    with rid_tabs[1]:
        st.markdown("**モデル保存物とプロパティ**")
        file_df = panel_state.build_file_rows(sel_dir)
        if file_df.empty:
            st.info("保存物ファイルが見つかりません。")
        else:
            file_df = file_df.sort_values("size_bytes", ascending=False)
            show_df(file_df.head(200), hide_index=True)

        cfg_summary = _safe_pickle_summary(sel_dir / "configuration.pkl")
        alias_summary = _safe_pickle_summary(sel_dir / "alias_to_model.pkl")
        st.markdown("**configuration.pkl summary**")
        st.json(cfg_summary)
        st.markdown("**alias_to_model.pkl summary**")
        st.json(alias_summary)

    with rid_tabs[2]:
        st.markdown("**モデル・設定整合性チェック**")
        config_ctx = panel_state.config_context(sel_meta, sel_model_row, sel_params)

        checks = helpers.build_config_check_rows(
            meta_model_name=config_ctx["meta_model_name"],
            db_model_name=config_ctx["db_model_name"],
            meta_h=config_ctx["meta_h"],
            db_h=config_ctx["db_h"],
            expected_backend=config_ctx["expected_backend"],
            actual_backend=config_ctx["actual_backend"],
            expected_num_samples=config_ctx["expected_num_samples"],
            actual_num_samples=config_ctx["actual_num_samples"],
            meta_pred_h=config_ctx["meta_pred_h"],
            meta_cv_h=config_ctx["meta_cv_h"],
            safe_int_eq=_safe_int_eq,
        )
        check_df = pd.DataFrame(checks)
        show_df(check_df, hide_index=True)
        mismatch_n = int((check_df["ok"] == False).sum()) if "ok" in check_df.columns else 0  # noqa: E712
        st.metric("mismatch count", mismatch_n)
        if mismatch_n > 0:
            st.warning("不一致項目があります。学習時の params-json / runtime kwargs / DB記録を再確認してください。")

        pred_h_mismatch, cv_h_mismatch = panel_state.mismatch_flags(
            meta_h=config_ctx["meta_h"],
            meta_pred_h=config_ctx["meta_pred_h"],
            meta_cv_h=config_ctx["meta_cv_h"],
            safe_int_eq=_safe_int_eq,
        )
        if pred_h_mismatch or cv_h_mismatch:
            st.error(
                "原因: `meta.json` 内の `nf_runtime_kwargs` に過去の h が残っており、"
                "`meta.h` と不一致です。predict 実行時の地平が意図とズレます。"
            )
            if st.button("meta.json の predict/cv h を meta.h に同期修正", key=f"nf_lab_runid_fix_h_{sel_run}"):
                meta_path = Path(sel_dir) / "meta.json"
                if not meta_path.exists():
                    st.error(f"meta.json が見つかりません: {meta_path}")
                else:
                    try:
                        obj = json.loads(meta_path.read_text(encoding="utf-8"))
                        if not isinstance(obj, dict):
                            raise ValueError("meta.json root must be dict")
                        target_h = int(obj.get("h") or settings.default_horizon)
                        nf_runtime = obj.get("nf_runtime_kwargs")
                        if not isinstance(nf_runtime, dict):
                            nf_runtime = {}
                        pred_kw = nf_runtime.get("nf_predict_kwargs")
                        if not isinstance(pred_kw, dict):
                            pred_kw = {}
                        pred_kw["h"] = int(target_h)
                        cv_kw = nf_runtime.get("nf_cross_validation_kwargs")
                        if not isinstance(cv_kw, dict):
                            cv_kw = {}
                        cv_kw["h"] = int(target_h)
                        nf_runtime["nf_predict_kwargs"] = pred_kw
                        nf_runtime["nf_cross_validation_kwargs"] = cv_kw
                        obj["nf_runtime_kwargs"] = nf_runtime
                        meta_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                        st.success(f"修正完了: {meta_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"meta.json 修正に失敗しました: {e}")

    with rid_tabs[3]:
        st.markdown("**run-id リソース消費**")
        if engine is None or ("resources", "run") not in tables:
            st.info("resources.* テーブル未接続のためリソース分析を表示できません。")
        else:
            run_res = query_df(
                engine,
                """
                SELECT
                  run_id::text AS run_id,
                  status,
                  started_at,
                  ended_at,
                  rows_written,
                  rows_failed,
                  EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))::double precision AS duration_sec
                FROM resources.run
                WHERE run_id::text = :run_id
                ORDER BY started_at DESC
                LIMIT 1
                """,
                {"run_id": str(sel_run)},
            )
            if run_res.empty:
                st.info("resources.run に該当run-idがありません。")
            else:
                rr = run_res.iloc[0]
                for col, metric in zip(st.columns(4), panel_formatter.build_resource_metrics(rr), strict=False):
                    col.metric(metric["label"], metric["value"])
                show_df(run_res, hide_index=True)

            if ("resources", "stage_span") in tables:
                stage_res = query_df(
                    engine,
                    """
                    SELECT
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
                    WHERE run_id::text = :run_id
                    ORDER BY started_at
                    """,
                    {"run_id": str(sel_run)},
                )
                if not stage_res.empty:
                    st.markdown("**stage_span**")
                    show_df(stage_res, hide_index=True)
                    if plotly_available and px is not None:
                        fig_stage = px.bar(
                            stage_res,
                            x="stage_name",
                            y="duration_ms",
                            color="exception_type",
                            title="stage duration (ms)",
                        )
                        fig_stage.update_layout(height=320)
                        st.plotly_chart(fig_stage, width="stretch")

            model_resource_df = query_df(
                engine,
                """
                SELECT
                  m.model_name,
                  COUNT(*)::int AS run_count,
                  AVG(EXTRACT(EPOCH FROM (COALESCE(r.ended_at, NOW()) - r.started_at)))::double precision AS avg_duration_sec,
                  AVG(COALESCE(r.rows_written, 0))::double precision AS avg_rows_written,
                  AVG(COALESCE(r.rows_failed, 0))::double precision AS avg_rows_failed
                FROM model.nf_automodel m
                LEFT JOIN resources.run r
                  ON r.run_id::text = m.run_id::text
                GROUP BY m.model_name
                ORDER BY run_count DESC, avg_duration_sec DESC NULLS LAST
                LIMIT 200
                """,
            )
            if not model_resource_df.empty:
                st.markdown("**モデル別リソース消費（run集計）**")
                model_resource_df = helpers.build_model_resource_df(model_resource_df)
                show_df(model_resource_df, hide_index=True)
                if plotly_available and px is not None:
                    fig_res = px.scatter(
                        model_resource_df,
                        x="avg_duration_sec",
                        y="avg_rows_written",
                        size="run_count",
                        color="model_name",
                        title="モデル別 リソース消費と処理量",
                    )
                    fig_res.update_layout(height=360)
                    st.plotly_chart(fig_res, width="stretch")

    with rid_tabs[4]:
        st.markdown("**予測精度（run / model 横断）**")
        if isinstance(sel_eval, dict) and sel_eval:
            eval_df = _metric_rows_from_eval(sel_eval)
            if not eval_df.empty:
                st.markdown("選択 run の評価指標")
                show_df(eval_df, hide_index=True)
        if not model_df.empty:
            metric_df = helpers.build_metric_rows(model_df, parse_json_like=parse_json_like, row_limit=row_limit)
            if not metric_df.empty:
                metric_name_options, default_metric = panel_state.default_metric_name(metric_df)
                metric_name = st.selectbox(
                    "比較する評価指標",
                    metric_name_options,
                    index=metric_name_options.index(default_metric),
                    key="nf_lab_runid_metric_name",
                )
                use_df = metric_df[metric_df["metric"].astype(str) == str(metric_name)].copy()
                agg = panel_formatter.build_accuracy_aggregate(use_df)
                show_df(agg, hide_index=True)
                if plotly_available and px is not None:
                    fig_acc = px.box(
                        use_df,
                        x="model_name",
                        y="value",
                        points="outliers",
                        title=f"{metric_name} distribution by model",
                    )
                    fig_acc.update_layout(height=380)
                    st.plotly_chart(fig_acc, width="stretch")

    with rid_tabs[5]:
        st.markdown("**精度・プロパティ・設定値の相関 / proxy因果**")
        analysis_df = helpers.build_analysis_df(model_df, parse_json_like=parse_json_like, row_limit=row_limit)
        if analysis_df.empty:
            st.info("解析対象データがありません。")
        else:
            run_feat = pd.DataFrame()
            if engine is not None and ("resources", "run") in tables:
                run_feat = query_df(
                    engine,
                    """
                    SELECT
                      run_id::text AS run_id,
                      EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))::double precision AS run_duration_sec,
                      COALESCE(rows_written, 0)::double precision AS rows_written,
                      COALESCE(rows_failed, 0)::double precision AS rows_failed
                    FROM resources.run
                    """,
                )
            stage_feat = pd.DataFrame()
            if engine is not None and ("resources", "stage_span") in tables:
                stage_feat = query_df(
                    engine,
                    """
                    SELECT
                      run_id::text AS run_id,
                      AVG(gpu_util_avg)::double precision AS gpu_util_avg,
                      AVG(gpu_mem_used_mb_avg)::double precision AS gpu_mem_avg_mb,
                      SUM(duration_ms)::double precision AS stage_total_ms
                    FROM resources.stage_span
                    GROUP BY run_id
                    """,
                )
            analysis_df = panel_analysis.merge_resource_features(
                analysis_df,
                run_feat=run_feat,
                stage_feat=stage_feat,
            )

            metric_cols = [c for c in analysis_df.columns if c.startswith("metric.")]
            if not metric_cols:
                st.info("metric.* 列が不足しており相関/因果を算出できません。")
            else:
                metric_target = panel_state.default_target_metric(metric_cols)
                assert metric_target is not None
                target_col = st.selectbox(
                    "target metric",
                    metric_cols,
                    index=metric_cols.index(metric_target),
                    key="nf_lab_runid_target_metric",
                )
                analysis_base = analysis_df.dropna(subset=[target_col]).copy()
                analysis_base[target_col] = pd.to_numeric(analysis_base[target_col], errors="coerce")
                analysis_base = analysis_base.dropna(subset=[target_col])
                if analysis_base.empty:
                    st.info("選択targetに有効データがありません。")
                else:
                    corr_df = panel_analysis.build_correlation_rows(analysis_base, target_col=target_col)
                    if corr_df.empty:
                        st.info("相関算出に十分な特徴量がありません。")
                    else:
                        show_df(corr_df.head(80), hide_index=True)
                        if plotly_available and px is not None:
                            fig_corr = px.bar(
                                corr_df.head(20),
                                x="feature",
                                y="spearman",
                                title=f"{target_col} と特徴量のSpearman相関",
                            )
                            fig_corr.update_layout(height=340)
                            st.plotly_chart(fig_corr, width="stretch")

                        treat_candidates = panel_state.treatment_candidates(corr_df)
                        if treat_candidates:
                            treat_col = st.selectbox(
                                "proxy因果 treatment", treat_candidates, index=0, key="nf_lab_runid_causal_treat"
                            )
                            ate = causal_proxy_ate(analysis_base, target_col=target_col, treatment_col=str(treat_col))
                            st.json(ate)

    with rid_tabs[6]:
        export_payload = panel_formatter.build_export_payload(
            sel_run=str(sel_run),
            sel_meta=sel_meta,
            sel_eval=sel_eval,
            sel_model_row=sel_model_row,
            sel_params=sel_params,
            sel_metrics=sel_metrics,
        )
        st.download_button(
            "Download runid_integrated_report.json",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2, default=str),
            file_name=f"runid_integrated_report_{str(sel_run)}.json",
            mime="application/json",
            key="nf_lab_runid_export_json",
        )
        st.code(panel_formatter.build_export_preview(export_payload, stable_json_dumps=stable_json_dumps), language="json")
