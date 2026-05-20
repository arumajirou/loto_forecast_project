from __future__ import annotations

from pathlib import Path

from common import ROOT, LOG_DIR, REGISTRY, read_yaml, Finding, write_json, write_md, write_csv, run


def main() -> None:
    reg = read_yaml(REGISTRY)
    findings: list[Finding] = []

    for test_path in reg.get("required_tests", []):
        p = Path(test_path)
        findings.append(
            Finding(
                "harness",
                p.name,
                "ok" if p.exists() else "missing",
                "high" if not p.exists() else "low",
                "必須テスト資産確認",
                "不足していればテストを作成する",
            )
        )

    smoke_cmds = [
        ["python", "-m", "py_compile", str(ROOT / "src/loto_forecast/api/streamlit/operations_dashboard.py")],
        ["bash", "-lc", f"cd {ROOT} && pytest --no-cov -q tests/unit/test_operations_dashboard_ui_helpers.py"],
        ["bash", "-lc", f"cd {ROOT} && pytest --no-cov -q tests/streamlit/test_operations_dashboard_apptest.py"],
    ]

    for i, cmd in enumerate(smoke_cmds, start=1):
        rc, out, err = run(cmd)
        findings.append(
            Finding(
                "harness",
                f"smoke_{i}",
                "ok" if rc == 0 else "failed",
                "high" if rc != 0 else "low",
                (out or err).strip()[:500],
                "失敗時は原因を切り分けて修正する",
            )
        )

    trace_files = [
        ROOT / "artifacts/logs/coverage_matrix.md",
        ROOT / "artifacts/logs/dynamic_trace.jsonl",
        ROOT / "artifacts/logs/dynamic_trace.csv",
        ROOT / "artifacts/logs/db_observation.json",
        ROOT / "artifacts/logs/file_observation.json",
    ]

    for p in trace_files:
        findings.append(
            Finding(
                "harness",
                p.name,
                "ok" if p.exists() else "missing",
                "medium" if not p.exists() else "low",
                "トレース器の存在確認",
                "不足なら先に空ファイルを作る",
            )
        )

    write_json(LOG_DIR / "harness_engineering_audit.json", [f.__dict__ for f in findings])
    write_md(LOG_DIR / "harness_engineering_audit.md", "Harness Engineering Audit", findings)
    write_csv(LOG_DIR / "harness_engineering_audit.csv", findings)


if __name__ == "__main__":
    main()
