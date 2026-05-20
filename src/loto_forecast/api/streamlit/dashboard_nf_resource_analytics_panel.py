from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_aggregator as panel_aggregator
from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_formatter as panel_formatter
from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_helpers as helpers
from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_state as panel_state

try:
    from scipy import stats as spstats

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _parse_json_like(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            out = json.loads(s)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_mode(s: pd.Series) -> str | None:
    if s.empty:
        return None
    m = s.dropna().astype(str)
    if m.empty:
        return None
    mode = m.mode()
    return str(mode.iloc[0]) if not mode.empty else None


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    nx = int(x.size)
    ny = int(y.size)
    denom_base = float(nx + ny - 2)
    if denom_base <= 0:
        return None
    pooled = np.sqrt((((nx - 1) * vx) + ((ny - 1) * vy)) / denom_base)
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    return float((float(np.mean(x)) - float(np.mean(y))) / pooled)


def _group_robust_zscore(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    if value_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    x = _to_num(df[value_col])
    if group_col not in df.columns:
        med = float(x.median()) if x.notna().any() else np.nan
        mad = float((x - med).abs().median()) if x.notna().any() else np.nan
        denom = 1.4826 * mad if pd.notna(mad) and mad != 0 else np.nan
        return (x - med) / denom

    g = df[group_col].fillna("__NA__").astype(str)
    med = x.groupby(g).transform("median")
    mad = (x - med).abs().groupby(g).transform("median")
    denom = (1.4826 * mad).replace(0, np.nan)
    z = (x - med) / denom
    all_nan = z.isna()
    if all_nan.all() and x.notna().any():
        global_med = float(x.median())
        global_mad = float((x - global_med).abs().median())
        global_denom = 1.4826 * global_mad if global_mad != 0 else np.nan
        z = (x - global_med) / global_denom
    return z


def _group_iqr_high_flag(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    if value_col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    x = _to_num(df[value_col])
    if group_col not in df.columns:
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr):
            return pd.Series(False, index=df.index, dtype=bool)
        return x > (q3 + 1.5 * iqr)
    g = df[group_col].fillna("__NA__").astype(str)
    q1 = x.groupby(g).transform(lambda v: float(pd.Series(v).quantile(0.25)))
    q3 = x.groupby(g).transform(lambda v: float(pd.Series(v).quantile(0.75)))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return x > upper


def _recommendations(
    run_df: pd.DataFrame,
    stage_agg_df: pd.DataFrame,
    error_df: pd.DataFrame,
    anomaly_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()

    fail_rate_avg = float(_to_num(run_df["is_failed"]).mean()) if "is_failed" in run_df.columns else 0.0
    db_share_med = (
        float(_to_num(run_df.get("db_share", pd.Series(dtype=float))).median())
        if "db_share" in run_df.columns
        else np.nan
    )
    duration_med = (
        float(_to_num(run_df.get("duration_sec", pd.Series(dtype=float))).median())
        if "duration_sec" in run_df.columns
        else np.nan
    )
    gpu_util_med = (
        float(_to_num(run_df.get("gpu_util_avg", pd.Series(dtype=float))).median())
        if "gpu_util_avg" in run_df.columns
        else np.nan
    )
    gpu_mem_med = (
        float(_to_num(run_df.get("gpu_mem_used_mb_avg", pd.Series(dtype=float))).median())
        if "gpu_mem_used_mb_avg" in run_df.columns
        else np.nan
    )

    if pd.notna(db_share_med) and db_share_med >= 0.35:
        rows.append(
            {
                "priority": "high",
                "pattern": "DB時間比率が高い",
                "evidence": f"db_share median={db_share_med:.2f}",
                "suggestion": "SQL/Index/バルク書込/トランザクション幅を見直し、DB待ちの削減を優先",
            }
        )
    if not stage_agg_df.empty:
        top = stage_agg_df.iloc[0]
        top_share = float(top.get("stage_share", 0.0) or 0.0)
        if top_share >= 0.5:
            rows.append(
                {
                    "priority": "high",
                    "pattern": "特定stageが総時間を支配",
                    "evidence": f"{top.get('stage_name')} share={top_share:.2%}",
                    "suggestion": "支配stageの rows_in/out・db_rows・I/Oを確認し、並列化か処理分割を検討",
                }
            )

    if pd.notna(gpu_util_med) and pd.notna(duration_med) and gpu_util_med < 25 and duration_med > 0:
        mem_txt = f", gpu_mem median={gpu_mem_med:.1f}MB" if pd.notna(gpu_mem_med) else ""
        rows.append(
            {
                "priority": "medium",
                "pattern": "GPU util 低いのに実行時間が長い",
                "evidence": f"gpu_util median={gpu_util_med:.1f}%{mem_txt}",
                "suggestion": "データ供給/前処理/IO待ちを確認し、CPU側ボトルネックを優先解消",
            }
        )

    if fail_rate_avg >= 0.1:
        top_err = _safe_mode(error_df.get("error_type", pd.Series(dtype=str))) if not error_df.empty else None
        rows.append(
            {
                "priority": "high",
                "pattern": "失敗率が高い",
                "evidence": f"failed ratio={fail_rate_avg:.1%}, top_error={top_err or 'n/a'}",
                "suggestion": "error_event の例外型ごとに再現条件(dataset/filter/params)を固定し、再発防止手順をRunbook化",
            }
        )

    if anomaly_count > 0:
        rows.append(
            {
                "priority": "medium",
                "pattern": "期待値からの逸脱runが存在",
                "evidence": f"anomaly runs={int(anomaly_count)}",
                "suggestion": "ランキング上位runをボトルネック/エラータブで掘り、改善前後を比較・検定",
            }
        )

    if not rows:
        rows.append(
            {
                "priority": "low",
                "pattern": "顕著な異常なし",
                "evidence": "主要指標が閾値内",
                "suggestion": "現状維持。定期監視を継続し、異常発生時のみ深掘り実施",
            }
        )
    out = pd.DataFrame(rows)
    prio_order = {"high": 0, "medium": 1, "low": 2}
    out["__prio"] = out["priority"].map(prio_order).fillna(9)
    out = out.sort_values(["__prio", "pattern"]).drop(columns=["__prio"], errors="ignore")
    return out


def render_nf_resource_analytics_panel(
    *,
    engine: Any,
    tables: set[tuple[str, str]],
    row_limit: int,
    query_df: Callable[..., pd.DataFrame],
    show_df: Callable[..., None],
    plotly_available: bool,
    px: Any | None = None,
) -> None:
    st.caption(
        "schema:log (`run_history`,`error_event`) + schema:resources を run_id で結合し、"
        "消費・期待値・異常・ボトルネック・改善提案を統合表示します。"
    )
    unavailable_message = panel_state.panel_unavailable_message(engine=engine, tables=tables)
    if unavailable_message is not None:
        st.info(unavailable_message)
        return

    has_stage = ("resources", "stage_span") in tables
    has_metric = ("resources", "resource_metric") in tables
    has_metric_def = ("resources", "metric_def") in tables
    has_log_history = ("log", "run_history") in tables
    has_log_error = ("log", "error_event") in tables
    has_model = ("model", "nf_automodel") in tables

    fetch_limits = panel_state.build_fetch_limits(row_limit)
    run_fetch_limit = fetch_limits["run_fetch_limit"]
    stage_fetch_limit = fetch_limits["stage_fetch_limit"]
    metric_fetch_limit = fetch_limits["metric_fetch_limit"]
    log_fetch_limit = fetch_limits["log_fetch_limit"]

    run_df = query_df(
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
          error_summary,
          tags,
          COALESCE(tags->>'execution_os', tags->'runtime_env'->>'execution_os', 'unknown') AS execution_os,
          EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))::double precision AS duration_sec
        FROM resources.run
        ORDER BY started_at DESC
        LIMIT :limit
        """,
        {"limit": run_fetch_limit},
    )
    if run_df.empty:
        st.info("resources.run にデータがありません。")
        return

    run_df["run_id"] = run_df["run_id"].astype(str)
    run_df["started_at"] = pd.to_datetime(run_df["started_at"], errors="coerce")
    run_df["ended_at"] = pd.to_datetime(run_df["ended_at"], errors="coerce")

    if has_model:
        model_df = query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              model_name,
              status AS model_status,
              params_json,
              created_at
            FROM model.nf_automodel
            ORDER BY created_at DESC NULLS LAST
            LIMIT :limit
            """,
            {"limit": run_fetch_limit * 2},
        )
        if not model_df.empty:
            run_df = helpers.merge_run_model_metadata(run_df, model_df, parse_json_like=_parse_json_like)

    min_day, max_day, default_start = panel_state.resolve_date_bounds(run_df)
    filter_options = panel_state.build_filter_options(run_df)

    st.markdown("**分析条件**")
    f1, f2, f3 = st.columns(3)
    start_day = f1.date_input(
        "開始日", value=default_start, min_value=min_day, max_value=max_day, key="nf_lab_res_start_day"
    )
    end_day = f2.date_input("終了日", value=max_day, min_value=min_day, max_value=max_day, key="nf_lab_res_end_day")
    baseline_mode = f3.selectbox(
        "期待値ベースライン窓", ["直近N件", "直近X日"], index=0, key="nf_lab_res_baseline_mode"
    )
    start_day, end_day = panel_state.normalize_date_range(start_day, end_day)

    f4, f5, f6, f7 = st.columns(4)
    status_opts = filter_options["status_opts"]
    app_opts = filter_options["app_opts"]
    model_opts = filter_options["model_opts"]
    status_sel = f4.multiselect("status", status_opts, default=status_opts, key="nf_lab_res_status_sel")
    app_sel = f5.multiselect("app_name", app_opts, default=app_opts, key="nf_lab_res_app_sel")
    model_sel = f6.multiselect("model_name", model_opts, default=[], key="nf_lab_res_model_sel")
    command_kw = f7.text_input("command contains", value="", key="nf_lab_res_command_kw")

    group_candidates = filter_options["group_candidates"]
    if group_candidates == ["status"]:
        run_df["status"] = run_df["status"].fillna("unknown")
    group_col = st.selectbox("期待値グループキー", group_candidates, index=0, key="nf_lab_res_group_col")

    b1, b2 = st.columns(2)
    baseline_n: int | None
    baseline_days: int | None
    if baseline_mode == "直近N件":
        slider_config = panel_state.baseline_slider_config(len(run_df))
        baseline_n = int(
            b1.slider(
                "直近N件",
                min_value=slider_config["min_value"],
                max_value=slider_config["max_value"],
                value=slider_config["value"],
                step=slider_config["step"],
                key="nf_lab_res_baseline_n",
            )
        )
        baseline_days = None
    else:
        baseline_days = int(
            b2.slider("直近X日", min_value=7, max_value=365, value=60, step=1, key="nf_lab_res_baseline_days")
        )
        baseline_n = None

    filtered = helpers.filter_runs(
        run_df,
        start_day=start_day,
        end_day=end_day,
        status_sel=status_sel,
        app_sel=app_sel,
        model_sel=model_sel,
        command_kw=command_kw,
    )

    if filtered.empty:
        st.info("フィルタ後の run がありません。条件を緩めてください。")
        return

    filtered = filtered.sort_values("started_at", ascending=False).copy()
    run_ids = set(filtered["run_id"].astype(str).tolist())

    stage_df = pd.DataFrame()
    metric_df = pd.DataFrame()
    hist_df = pd.DataFrame()
    error_df = pd.DataFrame()

    if has_stage:
        stage_df = query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              stage_name,
              started_at,
              ended_at,
              duration_ms,
              rows_in,
              rows_out,
              db_time_ms,
              db_rows,
              gpu_util_avg,
              gpu_mem_used_mb_avg,
              exception_type,
              exception_msg
            FROM resources.stage_span
            ORDER BY started_at DESC
            LIMIT :limit
            """,
            {"limit": stage_fetch_limit},
        )
        if not stage_df.empty:
            stage_df = helpers.normalize_stage_frame(stage_df, run_ids=run_ids, to_num=_to_num)

    if has_metric:
        metric_df = query_df(
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
            LIMIT :limit
            """,
            {"limit": metric_fetch_limit},
        )
        if not metric_df.empty:
            metric_df = helpers.normalize_metric_frame(metric_df, run_ids=run_ids, to_num=_to_num)

    if has_log_history:
        hist_df = query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              event_ts,
              event_type,
              status,
              model_name,
              dataset_name,
              message
            FROM log.run_history
            ORDER BY event_ts DESC
            LIMIT :limit
            """,
            {"limit": log_fetch_limit},
        )
        if not hist_df.empty:
            hist_df = helpers.normalize_event_frame(hist_df, run_ids=run_ids, time_col="event_ts")

    if has_log_error:
        error_df = query_df(
            engine,
            """
            SELECT
              run_id::text AS run_id,
              event_ts,
              model_name,
              stage,
              error_type,
              error_message
            FROM log.error_event
            ORDER BY event_ts DESC
            LIMIT :limit
            """,
            {"limit": log_fetch_limit},
        )
        if not error_df.empty:
            error_df = helpers.normalize_event_frame(error_df, run_ids=run_ids, time_col="event_ts")

    filtered = panel_aggregator.merge_run_aggregates(
        filtered,
        stage_df=stage_df,
        hist_df=hist_df,
        error_df=error_df,
        safe_mode=_safe_mode,
        to_num=_to_num,
    )

    baseline_pool, baseline_stats, filtered = panel_aggregator.apply_baseline_pipeline(
        run_df,
        filtered,
        baseline_mode=baseline_mode,
        baseline_n=baseline_n,
        baseline_days=baseline_days,
        end_day=end_day,
        group_col=group_col,
        to_num=_to_num,
    )
    run_df, filtered, baseline_pool = panel_state.ensure_group_column(
        run_df,
        group_col=group_col,
        filtered=filtered,
        baseline_pool=baseline_pool,
    )
    filtered = panel_aggregator.build_anomaly_enriched(
        filtered,
        group_col=group_col,
        group_robust_zscore=_group_robust_zscore,
        group_iqr_high_flag=_group_iqr_high_flag,
    )

    stage_agg = helpers.build_stage_aggregate_summary(stage_df, to_num=_to_num)

    tabs = st.tabs(["① 概要", "② run別（ランキング）", "③ ボトルネック", "④ エラー", "⑤ 比較・検定", "⑥ 提案"])

    with tabs[0]:
        summary_metrics = panel_formatter.build_summary_metrics(filtered, to_num=_to_num)
        top_metrics = summary_metrics[:6]
        extra_metrics = summary_metrics[6:]
        for col, metric in zip(st.columns(6), top_metrics, strict=False):
            col.metric(metric["label"], metric["value"])
        for col, metric in zip(st.columns(2), extra_metrics, strict=False):
            col.metric(metric["label"], metric["value"])

        summary_cols = panel_formatter.build_summary_columns(filtered, group_col=group_col)
        st.markdown("**run別サマリ（期待値差分付き）**")
        show_df(filtered[summary_cols].head(max(200, int(row_limit))), hide_index=True)

        if plotly_available and px is not None:
            try:
                fig_sc = px.scatter(
                    filtered,
                    x="duration_sec",
                    y="rows_written",
                    color="status",
                    hover_data=["run_id", "fail_rate", "throughput", "duration_ratio_vs_expected"],
                    title="duration_sec × rows_written",
                )
                fig_sc.update_layout(height=340)
                st.plotly_chart(fig_sc, width="stretch")
            except Exception:
                pass
            try:
                grp_plot = filtered[[group_col, "duration_sec"]].dropna()
                if not grp_plot.empty and grp_plot[group_col].nunique() <= 40:
                    fig_box = px.box(
                        grp_plot,
                        x=group_col,
                        y="duration_sec",
                        points="outliers",
                        title=f"{group_col} 別 duration 分布",
                    )
                    fig_box.update_layout(height=340)
                    st.plotly_chart(fig_box, width="stretch")
            except Exception:
                pass

        st.caption(
            f"取得上限: run={run_fetch_limit}, stage={stage_fetch_limit if has_stage else 0}, "
            f"metric={metric_fetch_limit if has_metric else 0}, log={log_fetch_limit if (has_log_history or has_log_error) else 0}"
        )

    with tabs[1]:
        topn = int(st.slider("上位件数", min_value=5, max_value=200, value=20, step=5, key="nf_lab_res_rank_topn"))
        rank_cols = panel_formatter.build_rank_columns(filtered, group_col=group_col)

        st.markdown("**遅い run 上位**")
        slow_df = filtered.sort_values("duration_sec", ascending=False).head(topn)
        show_df(slow_df[rank_cols], hide_index=True)

        st.markdown("**失敗/エラー run 上位**")
        fail_df = filtered.sort_values(
            ["is_failed", "error_events", "duration_sec"], ascending=[False, False, False]
        ).head(topn)
        show_df(fail_df[rank_cols], hide_index=True)

        st.markdown("**低効率 run 上位（throughput低）**")
        low_eff_df = filtered.sort_values("throughput", ascending=True).head(topn)
        show_df(low_eff_df[rank_cols], hide_index=True)

        anomaly_df = filtered[filtered["anomaly_flag"]].sort_values("anomaly_score", ascending=False).head(topn)
        st.markdown("**アノマリ候補**")
        if anomaly_df.empty:
            st.info("閾値に該当するアノマリ run はありません。")
        else:
            show_df(anomaly_df[rank_cols], hide_index=True)
            if plotly_available and px is not None:
                try:
                    fig_anom = px.bar(
                        anomaly_df,
                        x="run_id",
                        y="anomaly_score",
                        color="status",
                        hover_data=["duration_sec", "duration_ratio_vs_expected", "throughput", "fail_rate"],
                        title="anomaly score 上位",
                    )
                    fig_anom.update_layout(height=320)
                    st.plotly_chart(fig_anom, width="stretch")
                except Exception:
                    pass

    with tabs[2]:
        if stage_df.empty:
            st.info("resources.stage_span が無いか、対象runの span がありません。")
        else:
            st.markdown("**stage寄与（Pareto）**")
            show_df(stage_agg, hide_index=True)
            if plotly_available and px is not None and not stage_agg.empty:
                try:
                    fig_stage = px.bar(stage_agg, x="stage_name", y="stage_share", title="stage share")
                    fig_stage.update_layout(height=320)
                    st.plotly_chart(fig_stage, width="stretch")
                except Exception:
                    pass

            st.markdown("**DB時間比率ランキング（stage）**")
            db_rank = panel_formatter.build_stage_db_rank(stage_agg)
            if db_rank.empty:
                st.info("db_time_ms が無く算出できません。")
            else:
                show_df(
                    db_rank[["stage_name", "total_duration_ms", "total_db_time_ms", "db_share", "exception_count"]],
                    hide_index=True,
                )

            slow_run_ids = panel_state.build_slow_run_options(filtered)
            sel_run = st.selectbox("詳細run_id", slow_run_ids, index=0, key="nf_lab_res_bottleneck_run")
            sel_stage = stage_df[stage_df["run_id"].astype(str) == str(sel_run)].copy()
            if sel_stage.empty:
                st.info("選択runの stage_span がありません。")
            else:
                ssum = panel_formatter.build_selected_stage_summary(sel_stage, to_num=_to_num)
                show_df(ssum, hide_index=True)
                if plotly_available and px is not None:
                    try:
                        fig_sel = px.bar(ssum, x="stage_name", y="duration_ms", title=f"{sel_run} stage duration")
                        fig_sel.update_layout(height=320)
                        st.plotly_chart(fig_sel, width="stretch")
                    except Exception:
                        pass

            if "gpu_util_avg" in filtered.columns:
                gpu_idle = panel_aggregator.build_gpu_idle_candidates(filtered, to_num=_to_num)
                st.markdown("**GPU遊休疑い run（長時間かつ util低）**")
                if gpu_idle.empty:
                    st.info("該当 run はありません。")
                else:
                    show_df(
                        gpu_idle[["run_id", "duration_sec", "gpu_util_avg", "gpu_mem_used_mb_avg", "status"]],
                        hide_index=True,
                    )

            if not metric_df.empty:
                st.markdown("**resource_metric 波形（選択run）**")
                mrun = metric_df[metric_df["run_id"].astype(str) == str(sel_run)].copy()
                if mrun.empty:
                    st.info("選択runの resource_metric がありません。")
                else:
                    keys = panel_state.build_metric_key_options(mrun)
                    sel_key = st.selectbox("metric_key", keys, index=0, key="nf_lab_res_metric_key")
                    km = panel_aggregator.build_selected_metric_frame(mrun, sel_run=str(sel_run), sel_key=str(sel_key))
                    show_df(
                        km[["sampled_at", "run_id", "metric_key", "metric_value", "unit"]].tail(600), hide_index=True
                    )
                    if not km.empty:
                        if plotly_available and px is not None:
                            try:
                                fig_m = px.line(km, x="sampled_at", y="metric_value", title=f"{sel_key} timeline")
                                fig_m.update_layout(height=300)
                                st.plotly_chart(fig_m, width="stretch")
                            except Exception:
                                st.line_chart(km.set_index("sampled_at")[["metric_value"]], height=280)
                        else:
                            st.line_chart(km.set_index("sampled_at")[["metric_value"]], height=280)

            if has_metric_def:
                metric_def_df = query_df(
                    engine,
                    """
                    SELECT metric_key, scope, unit, description, source_library, source_method, recommended_interval_sec
                    FROM resources.metric_def
                    ORDER BY metric_key
                    LIMIT 1000
                    """,
                )
                if not metric_def_df.empty:
                    with st.expander("resources.metric_def", expanded=False):
                        show_df(metric_def_df, hide_index=True)

    with tabs[3]:
        if error_df.empty and hist_df.empty:
            st.info("log.run_history / log.error_event が無いか、対象runにログがありません。")
        else:
            if not error_df.empty:
                st.markdown("**error_event 集計**")
                err_type, err_stage = panel_formatter.build_error_frequency_tables(error_df)
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("error_type")
                    show_df(err_type, hide_index=True)
                with e2:
                    st.markdown("stage")
                    show_df(err_stage, hide_index=True)
                if plotly_available and px is not None and not err_type.empty:
                    try:
                        fig_err = px.bar(err_type.head(20), x="error_type", y="size", title="error_type frequency")
                        fig_err.update_layout(height=320)
                        st.plotly_chart(fig_err, width="stretch")
                    except Exception:
                        pass

                if not hist_df.empty:
                    ctx_df = helpers.build_error_context(error_df, hist_df)
                    st.markdown("**エラー直前イベント**")
                    show_df(ctx_df, hide_index=True)

            if not hist_df.empty:
                st.markdown("**run_history タイムライン**")
                run_options = panel_state.build_timeline_run_options(filtered)
                if run_options:
                    sel_run = st.selectbox("timeline run_id", run_options, index=0, key="nf_lab_res_timeline_run")
                    tdf = hist_df[hist_df["run_id"].astype(str) == str(sel_run)].sort_values("event_ts")
                    show_df(tdf, hide_index=True)
                    if plotly_available and px is not None and not tdf.empty:
                        try:
                            fig_tl = px.scatter(
                                tdf,
                                x="event_ts",
                                y="event_type",
                                color="status",
                                hover_data=["message", "model_name", "dataset_name"],
                                title=f"run_history timeline: {sel_run}",
                            )
                            fig_tl.update_layout(height=340)
                            st.plotly_chart(fig_tl, width="stretch")
                        except Exception:
                            pass
                    if not stage_df.empty:
                        stage_err = stage_df[
                            (stage_df["run_id"].astype(str) == str(sel_run)) & stage_df["exception_type"].notna()
                        ][["stage_name", "started_at", "duration_ms", "exception_type", "exception_msg"]]
                        if not stage_err.empty:
                            st.markdown("**stage_span 例外**")
                            show_df(stage_err, hide_index=True)

    with tabs[4]:
        comp_metric_opts = [
            c for c in ["duration_sec", "throughput", "fail_rate", "db_share", "gpu_util_avg"] if c in filtered.columns
        ]
        if not comp_metric_opts:
            st.info("比較対象の数値メトリクスがありません。")
        else:
            comp_metric = st.selectbox("比較メトリクス", comp_metric_opts, index=0, key="nf_lab_res_comp_metric")
            comp_df = filtered[[group_col, comp_metric, "is_failed"]].copy()
            comp_df[comp_metric] = _to_num(comp_df[comp_metric])
            comp_df = comp_df.dropna(subset=[group_col, comp_metric])
            if comp_df.empty:
                st.info("比較可能なデータがありません。")
            else:
                eligible = panel_state.eligible_comparison_groups(comp_df, group_col=group_col)
                if len(eligible) < 2:
                    st.info("検定には最低2群（各5件以上）が必要です。")
                else:
                    default_groups = panel_state.default_comparison_groups(eligible)
                    sel_groups = st.multiselect(
                        "比較群", eligible, default=default_groups, key="nf_lab_res_comp_groups"
                    )
                    if len(sel_groups) < 2:
                        st.info("2群以上を選択してください。")
                    else:
                        sub = comp_df[comp_df[group_col].astype(str).isin([str(x) for x in sel_groups])].copy()
                        agg = panel_formatter.build_comparison_aggregate(sub, group_col=group_col, comp_metric=comp_metric)
                        show_df(agg, hide_index=True)
                        if plotly_available and px is not None:
                            try:
                                fig_cmp = px.violin(
                                    sub,
                                    x=group_col,
                                    y=comp_metric,
                                    box=True,
                                    points="outliers",
                                    title=f"{comp_metric} 比較",
                                )
                                fig_cmp.update_layout(height=340)
                                st.plotly_chart(fig_cmp, width="stretch")
                            except Exception:
                                pass

                        a_name, b_name = str(sel_groups[0]), str(sel_groups[1])
                        a, b = panel_aggregator.build_comparison_arrays(
                            sub,
                            group_col=group_col,
                            comp_metric=comp_metric,
                            sel_groups=sel_groups,
                            to_num=_to_num,
                        )
                        stat_payload = panel_formatter.build_stat_payload(
                            group_a=a_name,
                            group_b=b_name,
                            metric=comp_metric,
                            a=a,
                            b=b,
                            cohen_d=_cohen_d,
                            scipy_available=SCIPY_AVAILABLE,
                            spstats=spstats if SCIPY_AVAILABLE else None,
                        )
                        st.markdown("**2群比較（先頭2群）**")
                        st.json(stat_payload)

                        if SCIPY_AVAILABLE:
                            ct = pd.crosstab(sub[group_col], sub["is_failed"].astype(bool))
                            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                                try:
                                    chi2, pval, dof, _ = spstats.chi2_contingency(ct.to_numpy())
                                    st.markdown("**成否の群差 (chi-square)**")
                                    st.json({"chi2_stat": float(chi2), "p_value": float(pval), "dof": int(dof)})
                                except Exception:
                                    pass

    with tabs[5]:
        anomaly_count = int(filtered["anomaly_flag"].fillna(False).sum())
        rec_df = _recommendations(filtered, stage_agg, error_df, anomaly_count)
        st.markdown("**改善提案（ルール＋統計根拠）**")
        show_df(rec_df, hide_index=True)
        st.caption("運用フローは `docs/22_nf_resource_analytics_operations_runbook.md` を参照してください。")
