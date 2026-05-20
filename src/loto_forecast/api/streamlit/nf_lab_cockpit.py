from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


@dataclass(frozen=True, slots=True)
class CockpitIssue:
    severity: str
    title: str
    detail: str
    action: str


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OBSERVABILITY_ROOT = PROJECT_ROOT / "artifacts" / "observability"
DASHBOARD_LOG_DIR = PROJECT_ROOT / "logs" / "dashboard"


def _mask_secret(value: str | None) -> str:
    if not value:
        return "未設定"
    if len(value) <= 2:
        return "設定済み(**)"
    return f"設定済み({value[:1]}***{value[-1:]})"


def _safe_read_tail(path: Path, *, max_lines: int = 120) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []


def _recent_files(root: Path, patterns: tuple[str, ...], *, limit: int = 40) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in root.rglob(pattern) if p.is_file())
    return sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)[:limit]


def _summarize_log_lines(lines: list[str]) -> dict[str, Any]:
    joined = "\n".join(lines)
    error_patterns = {
        "traceback": r"Traceback \(most recent call last\)",
        "module_missing": r"ModuleNotFoundError|ImportError",
        "db_auth": r"fe_sendauth|password supplied|authentication failed|no password",
        "timeout": r"timed?\s*out|TimeoutError|not ready",
        "exception": r"\bException\b|OperationalError|RuntimeError|ValueError",
    }
    counts = {name: len(re.findall(pattern, joined, flags=re.IGNORECASE)) for name, pattern in error_patterns.items()}
    top_lines = [
        line
        for line in lines
        if re.search(
            r"Traceback|ERROR|Exception|OperationalError|ModuleNotFoundError|timeout|not ready|failed", line, re.I
        )
    ][-20:]
    return {"counts": counts, "top_lines": top_lines}


def _load_observability_events(limit: int = 500) -> list[dict[str, Any]]:
    events_path = OBSERVABILITY_ROOT / "events.jsonl"
    if not events_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _safe_read_tail(events_path, max_lines=limit):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows[-limit:]


def _fingerprint_event(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    parts = [
        str(event.get("source", "")),
        str(event.get("category", "")),
        str(event.get("level", "")),
        str(event.get("message", "")),
        str(payload.get("error", "")),
        str(payload.get("path", "")),
        str(payload.get("url", "")),
    ]
    normalized = "|".join(re.sub(r"\d+", "<n>", p.strip().lower()) for p in parts if p)
    return normalized[:240] if normalized else "unknown"


def build_nf_lab_health(
    *,
    connected: bool,
    db_error: str | None,
    host: str,
    port: int,
    user: str,
    database: str,
    tables: set[tuple[str, str]],
) -> tuple[list[CockpitIssue], pd.DataFrame]:
    issues: list[CockpitIssue] = []
    db_password = os.getenv("DB_PASSWORD", "")

    if not connected:
        detail = db_error or "DB接続が未確立です。"
        action = "DB_PASSWORDを.envまたは環境変数へ設定し、再読込してください。DBなしでも計画作成・観測・ログ確認は使えます。"
        severity = "error" if re.search(r"password|auth|fe_sendauth", detail, re.I) else "warning"
        issues.append(CockpitIssue(severity, "DB接続未確立", detail[:500], action))

    if not db_password:
        issues.append(
            CockpitIssue(
                "warning",
                "DB_PASSWORD未設定",
                "画面のパスワード欄が空で、環境変数DB_PASSWORDも未設定です。",
                "`.env`にDB_PASSWORDを入れるか、`export DB_PASSWORD='...'`後にdashboardを再起動してください。",
            )
        )

    required = [
        ("dataset", "loto_y_ts_unified"),
        ("meta", "nf_automodel"),
        ("model", "nf_automodel"),
        ("log", "ui_state_snapshot"),
    ]
    if connected:
        for schema, table in required:
            if (schema, table) not in tables:
                issues.append(
                    CockpitIssue(
                        "warning",
                        f"{schema}.{table} が未検出",
                        "NeuralForecastラボの一部機能が利用できない可能性があります。",
                        "db-initはdry-runで確認後、バックアップと明示承認を経て実適用してください。",
                    )
                )

    status_rows = [
        {"item": "DB接続", "status": "OK" if connected else "NG", "detail": f"{host}:{port}/{database} user={user}"},
        {"item": "DB_PASSWORD", "status": "OK" if bool(db_password) else "NG", "detail": _mask_secret(db_password)},
        {
            "item": "dataset table",
            "status": "OK" if ("dataset", "loto_y_ts_unified") in tables else "未確認",
            "detail": "dataset.loto_y_ts_unified",
        },
        {
            "item": "meta table",
            "status": "OK" if ("meta", "nf_automodel") in tables else "未確認",
            "detail": "meta.nf_automodel",
        },
        {
            "item": "model table",
            "status": "OK" if ("model", "nf_automodel") in tables else "未確認",
            "detail": "model.nf_automodel",
        },
        {
            "item": "observability",
            "status": "OK" if OBSERVABILITY_ROOT.exists() else "未作成",
            "detail": str(OBSERVABILITY_ROOT),
        },
    ]
    return issues, pd.DataFrame(status_rows)


def _render_issue_cards(issues: list[CockpitIssue]) -> None:
    if not issues:
        st.success("重大な事前検知エラーはありません。")
        return

    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(issues, key=lambda x: severity_order.get(x.severity, 9))
    for issue in issues:
        body = f"**{issue.title}**\n\n{issue.detail}\n\n次の対応: {issue.action}"
        if issue.severity == "error":
            st.error(body)
        elif issue.severity == "warning":
            st.warning(body)
        else:
            st.info(body)


def _render_db_connection_helper(*, host: str, port: int, user: str, database: str, db_error: str | None) -> None:
    st.markdown("### DB接続を先に直す")
    st.caption("パスワードは画面・コマンド履歴に残さず、`.env`または環境変数へ保存します。")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.code(
            "\n".join(
                [
                    "cd /mnt/e/env/fc/loto_forecast_project",
                    "cp -n .env.example .env",
                    "chmod 600 .env",
                    "nano .env  # DB_PASSWORD=... を設定",
                    "./run_operations_dashboard.sh",
                ]
            ),
            language="bash",
        )
    with c2:
        st.code(
            "\n".join(
                [
                    f"export DB_HOST={host}",
                    f"export DB_PORT={int(port)}",
                    f"export DB_USER={user}",
                    f"export DB_NAME={database}",
                    "export DB_PASSWORD='********'",
                    "./run_operations_dashboard.sh",
                ]
            ),
            language="bash",
        )
    if db_error:
        with st.expander("直近のDB接続エラー詳細", expanded=False):
            st.code(db_error[:2000], language="text")


def _render_goal_wizard() -> dict[str, Any]:
    st.markdown("### 目的から始める")
    goal = st.radio(
        "今日やりたいこと",
        [
            "DB接続を直して既存結果を見る",
            "学習計画を安全に作る",
            "既存run_idを比較・診断する",
            "エラー/重複/遅延を調査する",
            "スクリーンショットとブラウザ操作ログを収集する",
        ],
        horizontal=False,
        key="nf_cockpit_goal",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("操作レベル", ["かんたん", "標準", "詳細"], key="nf_cockpit_level")
    with c2:
        risk = st.selectbox(
            "実行リスク", ["読み取り/計画のみ", "dry-runのみ", "DB書き込み候補をレビュー"], key="nf_cockpit_risk"
        )
    with c3:
        next_action = st.selectbox(
            "次の画面", ["事前チェック", "実行計画", "観測・診断", "詳細ラボ"], key="nf_cockpit_next"
        )
    return {"goal": goal, "mode": mode, "risk": risk, "next_action": next_action}


def _render_execution_plan(*, connected: bool, host: str, port: int, user: str, database: str) -> None:
    st.markdown("### 実行計画レビュー")
    st.caption("この画面では、いきなり学習やDB書き込みを実行せず、コマンドと安全条件をレビューします。")
    preset = st.selectbox(
        "プリセット",
        [
            "最小 smoke: import/DB dry-run/設定確認",
            "NF train dry-run: コマンド生成のみ",
            "既存run_id診断: 読み取り専用",
            "観測収集: browser screenshots/logs",
        ],
        key="nf_cockpit_plan_preset",
    )
    plan_rows = [
        {
            "step": 1,
            "name": "DB接続確認",
            "write": "なし",
            "command": "PYTHONPATH=src uv run --no-sync python -m loto_forecast.cli db-check",
        },
        {
            "step": 2,
            "name": "db-init dry-run",
            "write": "なし",
            "command": "PYTHONPATH=src uv run --no-sync python -m loto_forecast.cli db-init --dry-run",
        },
        {"step": 3, "name": "静的検査", "write": "なし", "command": "./scripts/repair_static.sh"},
        {
            "step": 4,
            "name": "観測収集",
            "write": "artifacts/observabilityのみ",
            "command": "./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2",
        },
    ]
    df = pd.DataFrame(plan_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.code(
        "\n".join(
            [
                "cd /mnt/e/env/fc/loto_forecast_project",
                "export PYTHONPATH=src",
                "export UV_LINK_MODE=copy",
                "# まずは書き込みなし",
                "./scripts/repair_static.sh",
                "PYTHONPATH=src uv run --no-sync python -m loto_forecast.cli db-init --dry-run",
            ]
        ),
        language="bash",
    )
    if not connected:
        st.warning(f"DB未接続です。接続先 {host}:{port}/{database} user={user} を先に確認してください。")
    st.info(f"選択中プリセット: {preset}")


def _utc_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _launch_background_command(
    *,
    name: str,
    args: list[str],
    env_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Launch a local project command without shell expansion and persist logs.

    This is intentionally local-only. It does not execute DB writes unless the
    called script receives its own explicit write gate.
    """
    run_id = _utc_run_id(name)
    run_dir = PROJECT_ROOT / "artifacts" / "automation" / "app_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout_stderr.log"
    pid_path = run_dir / "pid.txt"
    manifest_path = run_dir / "manifest.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["UV_LINK_MODE"] = env.get("UV_LINK_MODE", "copy")
    if env_extra:
        env.update(env_extra)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(  # noqa: S603
            args,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "name": name,
        "pid": int(proc.pid),
        "args": args,
        "log_path": str(log_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _render_screenshot_capture_console() -> None:
    st.markdown("### 画面スクリーンショット網羅収集")
    st.caption(
        "Playwrightで画面、タブ、ボタンを安全に巡回し、スクリーンショット・console・network・traceを artifacts/observability に保存します。"
    )

    default_url = os.getenv("LOTO_DASHBOARD_URL", "http://localhost:8505")
    c1, c2 = st.columns([2, 1])
    with c1:
        url = st.text_input("対象URL", value=default_url, key="nf_screenshot_url")
    with c2:
        run_id = st.text_input("run_id", value=_utc_run_id("ui"), key="nf_screenshot_run_id")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_clicks = st.number_input("最大クリック数", min_value=0, max_value=500, value=40, step=5)
    with c2:
        max_depth = st.number_input("探索深さ", min_value=0, max_value=5, value=2, step=1)
    with c3:
        width = st.number_input("幅", min_value=640, max_value=3840, value=1440, step=80)
    with c4:
        height = st.number_input("高さ", min_value=480, max_value=2400, value=1200, step=80)

    safe_clicks = st.toggle("危険ボタンをクリックしない", value=True)
    headed = st.toggle("ブラウザを表示して実行", value=False)
    install_hint = st.toggle(
        "Chromium未導入時は先に `uv run --no-sync playwright install chromium` を実行してください", value=True
    )

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/collect_browser_observability.py",
        "--url",
        str(url),
        "--run-id",
        str(run_id),
        "--max-clicks",
        str(int(max_clicks)),
        "--max-depth",
        str(int(max_depth)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
    ]
    if safe_clicks:
        cmd.append("--safe-clicks")
    else:
        cmd.append("--unsafe-clicks")
    if headed:
        cmd.append("--headed")

    st.code(" ".join(cmd), language="bash")
    if install_hint:
        st.code(
            "LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh\nuv run --no-sync playwright install chromium",
            language="bash",
        )

    if st.button("バックグラウンドでスクリーンショット収集を開始", type="primary"):
        manifest = _launch_background_command(
            name="screenshot_capture",
            args=cmd,
            env_extra={"LOTO_UV_ENV_MODE": "browser", "LOTO_UV_CLEAR_VENV": "0"},
        )
        st.success(f"開始しました pid={manifest['pid']}")
        st.json(manifest)

    screenshot_files = _recent_files(OBSERVABILITY_ROOT / "browser_runs", ("*.png",), limit=24)
    if screenshot_files:
        st.markdown("#### 最新スクリーンショット")
        cols = st.columns(3)
        for idx, path in enumerate(screenshot_files[:12]):
            with cols[idx % 3]:
                st.image(str(path), caption=str(path.relative_to(PROJECT_ROOT)), use_container_width=True)
    else:
        st.info("スクリーンショットはまだありません。収集を開始してください。")

    manifests = _recent_files(OBSERVABILITY_ROOT / "browser_runs", ("manifest.json",), limit=10)
    if manifests:
        with st.expander("収集manifest", expanded=False):
            for path in manifests:
                st.code(path.read_text(encoding="utf-8", errors="replace")[:4000], language="json")


def _render_dataset_feature_job_panel() -> None:
    st.markdown("### データセット取得 → 特徴量生成 → DBテーブル作成")
    st.caption(
        "既定はdry-runです。DBへ書き込むには `LOTO_ALLOW_FEATURE_DB_WRITE=1` と `--yes-write` の両方が必要です。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        source_schema = st.text_input("source schema", value="dataset")
        source_table = st.text_input("source table", value="loto_y_ts_unified")
    with c2:
        target_schema = st.text_input("target schema", value="exog")
        target_table = st.text_input("target table", value="nf_feature_table_auto")
    with c3:
        limit = st.number_input("limit", min_value=1, max_value=1_000_000, value=5000, step=1000)
        yes_write = st.toggle("DBへ実書き込みする", value=False)

    if target_schema == "dataset":
        st.error(
            "datasetスキーマは読み取り専用です。target schema は exog/meta/model/resources/catalog/log を使ってください。"
        )
        yes_write = False

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/run_dataset_feature_table_job.py",
        "--source-schema",
        source_schema,
        "--source-table",
        source_table,
        "--target-schema",
        target_schema,
        "--target-table",
        target_table,
        "--limit",
        str(int(limit)),
    ]
    env_extra = {"LOTO_UV_ENV_MODE": "static", "LOTO_UV_CLEAR_VENV": "0"}
    if yes_write:
        cmd.append("--yes-write")
        env_extra["LOTO_ALLOW_FEATURE_DB_WRITE"] = "1"

    st.code(" ".join(cmd), language="bash")

    if st.button("特徴量ジョブをバックグラウンド実行", type="primary"):
        if yes_write and target_schema == "dataset":
            st.error("datasetへの書き込みは禁止です。")
        else:
            manifest = _launch_background_command(name="feature_table_job", args=cmd, env_extra=env_extra)
            st.success(f"開始しました pid={manifest['pid']}")
            st.json(manifest)

    job_manifests = _recent_files(PROJECT_ROOT / "artifacts" / "data_jobs", ("manifest.json",), limit=10)
    if job_manifests:
        st.markdown("#### 最新ジョブ")
        rows: list[dict[str, Any]] = []
        for path in job_manifests:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "run_id": payload.get("run_id"),
                    "status": payload.get("status"),
                    "mode": payload.get("mode"),
                    "rows": payload.get("rows_generated"),
                    "wrote_db": payload.get("wrote_db"),
                    "target": ".".join(
                        [
                            str(payload.get("target", {}).get("schema", "")),
                            str(payload.get("target", {}).get("table", "")),
                        ]
                    ),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### cron / WSL自動実行")
    st.code(
        "\n".join(
            [
                "./scripts/install_wsl_automation.sh --all  # dry-run preview",
                "LOTO_ALLOW_AUTOMATION_INSTALL=1 ./scripts/install_wsl_automation.sh --install --all",
                "./scripts/wsl_start_loto_app.sh",
                "./scripts/cron_run_feature_pipeline.sh",
            ]
        ),
        language="bash",
    )


def _render_observability_digest() -> None:
    st.markdown("### エラー早期検知・観測ダイジェスト")
    events = _load_observability_events()
    event_df = pd.DataFrame(events)
    log_files = _recent_files(PROJECT_ROOT, ("*.log",), limit=20)
    lines: list[str] = []
    for path in log_files[:5]:
        lines.extend(_safe_read_tail(path, max_lines=80))
    summary = _summarize_log_lines(lines)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("観測イベント", len(events))
    c2.metric("log files", len(log_files))
    c3.metric("traceback", int(summary["counts"].get("traceback", 0)))
    c4.metric("DB/auth", int(summary["counts"].get("db_auth", 0)))

    if events:
        fps = Counter(_fingerprint_event(e) for e in events)
        dup_rows = [{"fingerprint": k, "count": v} for k, v in fps.most_common(20) if v >= 2]
        if dup_rows:
            st.warning("重複イベントを検出しました。頻出順に確認してください。")
            st.dataframe(pd.DataFrame(dup_rows), use_container_width=True, hide_index=True)
        with st.expander("観測イベント recent", expanded=False):
            st.dataframe(event_df.tail(100), use_container_width=True)
    else:
        st.info(
            "観測イベントはまだありません。`./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2` を実行してください。"
        )

    if summary["top_lines"]:
        with st.expander("ログ内のエラー候補", expanded=True):
            st.code("\n".join(summary["top_lines"][-80:]), language="text")
    else:
        st.success("直近ログから重大エラー候補は検出されていません。")


def _render_duplicate_inventory() -> None:
    st.markdown("### 画面・機能の重複/棚卸")
    features = [
        ("概要", "DB接続後の全体状態", "read"),
        ("NeuralForecast Cockpit", "DBなしでも使える開始画面/計画/診断", "read/plan"),
        ("NeuralForecast 詳細ラボ", "既存の詳細実行・検証画面", "read/write candidate"),
        ("観測・診断", "スクショ・ログ・trace・メトリクス確認", "local artifacts"),
        ("成果物・ログ", "生成物とログ確認", "read"),
        ("運用.Runner", "実行系コマンド", "write candidate"),
        ("DB管理/ER", "DB管理", "write candidate"),
    ]
    df = pd.DataFrame(features, columns=["画面", "責務", "副作用"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    duplicated = [
        {
            "重複候補": "Runner / 詳細ラボ / コマンドラボ",
            "整理方針": "Cockpitで計画、詳細ラボで設定、Runnerは実行ログ専用へ分離",
        },
        {
            "重複候補": "成果物・ログ / 観測・診断",
            "整理方針": "成果物はファイル閲覧、観測はエラー検知・スクショ・trace分析へ分離",
        },
        {"重複候補": "概要 / Cockpit", "整理方針": "概要はDB状態、Cockpitは次アクション案内へ役割固定"},
    ]
    st.dataframe(pd.DataFrame(duplicated), use_container_width=True, hide_index=True)


def render_neuralforecast_cockpit(
    *,
    connected: bool,
    db_error: str | None,
    engine: Any,
    tables: set[tuple[str, str]],
    row_limit: int,
    sample_limit: int,
    host: str,
    port: int,
    user: str,
    database: str,
    render_legacy_lab: Any | None = None,
) -> None:
    st.title("NeuralForecast Cockpit")
    st.caption("DB接続、実行計画、観測・診断、重複検知を1画面に集約した再設計版ラボです。")

    issues, status_df = build_nf_lab_health(
        connected=connected,
        db_error=db_error,
        host=host,
        port=int(port),
        user=user,
        database=database,
        tables=tables,
    )

    ok_count = int((status_df["status"] == "OK").sum())
    warn_count = int(len(status_df) - ok_count)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ready", f"{ok_count}/{len(status_df)}")
    c2.metric("Warnings", warn_count)
    c3.metric("DB", "接続済み" if connected else "未接続")
    c4.metric("Mode", st.session_state.get("nf_cockpit_level", "かんたん"))

    with st.container(border=True):
        _render_issue_cards(issues)

    tabs = st.tabs(
        [
            "はじめる",
            "DB接続",
            "スクリーンショット収集",
            "データ/特徴量ジョブ",
            "実行計画",
            "観測・診断",
            "重複/棚卸",
            "詳細ラボ",
        ]
    )
    with tabs[0]:
        _render_goal_wizard()
        st.markdown("### 状態チェック")
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.info(
            "推奨順序: 1) DB_PASSWORD設定 → 2) DB dry-run → 3) 静的検査 → 4) 学習計画レビュー → 5) 観測収集 → 6) 詳細ラボで個別実行"
        )
    with tabs[1]:
        _render_db_connection_helper(host=host, port=int(port), user=user, database=database, db_error=db_error)
    with tabs[2]:
        _render_screenshot_capture_console()
    with tabs[3]:
        _render_dataset_feature_job_panel()
    with tabs[4]:
        _render_execution_plan(connected=connected, host=host, port=int(port), user=user, database=database)
    with tabs[5]:
        _render_observability_digest()
    with tabs[6]:
        _render_duplicate_inventory()
    with tabs[7]:
        st.markdown("### 既存詳細ラボ")
        st.caption("既存UIは後方互換として残します。迷った場合はCockpitの実行計画から始めてください。")
        if render_legacy_lab is None:
            st.info("詳細ラボ renderer が渡されていません。")
        else:
            render_legacy_lab()
