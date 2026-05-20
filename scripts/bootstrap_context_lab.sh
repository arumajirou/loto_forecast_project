#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT}"
mkdir -p "$ROOT/docs/context"
mkdir -p "$ROOT/scripts/context_lab"
mkdir -p "$ROOT/artifacts/logs"

cat > "$ROOT/docs/context/best_practices_registry.yaml" <<'YAML'
version: 1
updated_at: "2026-03-31"
sources:
  - name: openai_agents_md
    category: context_engineering
    rule: "AGENTS.md を持続ルールの正本にする"
    priority: high
  - name: openai_skills_workflow
    category: context_engineering
    rule: "Skills は repo の通常ワークフローに統合し、決定的処理は scripts に寄せる"
    priority: high
  - name: openai_prompt_specificity
    category: prompt_engineering
    rule: "目的、制約、成果物、出力形式を明示し、曖昧さを減らす"
    priority: high
  - name: anthropic_context_engineering
    category: context_engineering
    rule: "context は単一プロンプトではなく、選択・順序・圧縮・再利用の設計対象とする"
    priority: high
  - name: openai_eval_skills
    category: harness_engineering
    rule: "成功条件を先に定義し、小さな eval で継続改善する"
    priority: high
required_files:
  - ${PROJECT_ROOT}/AGENTS.md
  - ${PROJECT_ROOT}/docs/context/00_context_index.md
  - ${PROJECT_ROOT}/docs/context/01_execution_contract.md
  - ${PROJECT_ROOT}/docs/context/02_decision_policy.md
  - ${PROJECT_ROOT}/docs/context/03_context_packet.md
  - ${PROJECT_ROOT}/docs/context/04_tooling_scope.md
required_logs:
  - ${PROJECT_ROOT}/artifacts/logs/coverage_matrix.md
  - ${PROJECT_ROOT}/artifacts/logs/dynamic_trace.jsonl
  - ${PROJECT_ROOT}/artifacts/logs/dynamic_trace.csv
  - ${PROJECT_ROOT}/artifacts/logs/db_observation.json
  - ${PROJECT_ROOT}/artifacts/logs/file_observation.json
required_tests:
  - ${PROJECT_ROOT}/tests/e2e/operations_dashboard_ui_check.mjs
  - ${PROJECT_ROOT}/tests/streamlit/test_operations_dashboard_apptest.py
  - ${PROJECT_ROOT}/tests/unit/test_operations_dashboard_ui_helpers.py
required_mcp:
  - chrome-devtools
  - context7
  - openaiDeveloperDocs
  - github
preferred_skills:
  - sandbox-check
  - log-triage
  - streamlit-exhaustive-testing
  - db-persistence-audit
  - artifact-generation-audit
  - dynamic-trace-recorder
  - webapp-testing
  - verification-before-completion
YAML

cat > "$ROOT/scripts/context_lab/common.py" <<'PY'
from __future__ import annotations
import csv
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import yaml

ROOT = Path("${PROJECT_ROOT}")
LOG_DIR = ROOT / "artifacts" / "logs"
CTX_DIR = ROOT / "docs" / "context"
REGISTRY = CTX_DIR / "best_practices_registry.yaml"

@dataclass
class Finding:
    category: str
    name: str
    status: str
    severity: str
    detail: str
    recommendation: str

def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def write_md(path: Path, title: str, findings: list[Finding]) -> None:
    lines = [f"# {title}", ""]
    for f in findings:
        lines += [
            f"## {f.name}",
            f"- category: {f.category}",
            f"- status: {f.status}",
            f"- severity: {f.severity}",
            f"- detail: {f.detail}",
            f"- recommendation: {f.recommendation}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "name", "status", "severity", "detail", "recommendation"])
        w.writeheader()
        for item in findings:
            w.writerow(asdict(item))
PY

cat > "$ROOT/scripts/context_lab/audit_prompt_engineering.py" <<'PY'
from __future__ import annotations
from pathlib import Path
from common import ROOT, LOG_DIR, Finding, write_json, write_md, write_csv

PROMPT_FILES = [
    ROOT / "docs" / "prompts" / "hernes.md",
    ROOT / "docs" / "prompts" / "val.md",
    ROOT / "task_prompt.md",
]

REQUIRED_SECTIONS = ["目的", "対象", "制約", "成果物", "出力形式"]

def main() -> None:
    findings: list[Finding] = []
    for path in PROMPT_FILES:
        if not path.exists():
            findings.append(Finding("prompt", path.name, "missing", "medium", "プロンプトファイルが存在しません", "必要なら作成する"))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for sec in REQUIRED_SECTIONS:
            status = "ok" if sec in text else "missing"
            findings.append(Finding(
                "prompt",
                f"{path.name}:{sec}",
                status,
                "high" if status == "missing" else "low",
                f"{sec} セクションの有無を確認",
                f"{sec} を明示する"
            ))
        if "追加質問" not in text and "確認を求めず" not in text:
            findings.append(Finding("prompt", f"{path.name}:autonomy", "missing", "high", "自律完遂条件が弱い", "追加質問なしで完遂する条件を入れる"))
    write_json(LOG_DIR / "prompt_engineering_audit.json", [f.__dict__ for f in findings])
    write_md(LOG_DIR / "prompt_engineering_audit.md", "Prompt Engineering Audit", findings)
    write_csv(LOG_DIR / "prompt_engineering_audit.csv", findings)

if __name__ == "__main__":
    main()
PY

cat > "$ROOT/scripts/context_lab/audit_context_engineering.py" <<'PY'
from __future__ import annotations
from pathlib import Path
from common import ROOT, LOG_DIR, REGISTRY, read_yaml, Finding, write_json, write_md, write_csv, run

def main() -> None:
    reg = read_yaml(REGISTRY)
    findings: list[Finding] = []

    for file_path in reg.get("required_files", []):
        p = Path(file_path)
        findings.append(Finding(
            "context",
            p.name,
            "ok" if p.exists() else "missing",
            "high" if not p.exists() else "low",
            "必須コンテキストファイル確認",
            "不足していれば生成する"
        ))

    for log_path in reg.get("required_logs", []):
        p = Path(log_path)
        findings.append(Finding(
            "context",
            p.name,
            "ok" if p.exists() else "missing",
            "medium" if not p.exists() else "low",
            "必須ログ器確認",
            "不足していれば空の器を作る"
        ))

    rc, out, _ = run(["bash", "-lc", "codex mcp list"])
    for name in reg.get("required_mcp", []):
        findings.append(Finding(
            "context",
            f"mcp:{name}",
            "ok" if name in out else "missing",
            "high" if name not in out else "low",
            "MCP 登録状況確認",
            "不足していれば codex mcp add で追加する"
        ))

    rc, out, _ = run(["bash", "-lc", "find .agents/skills -maxdepth 2 -name SKILL.md 2>/dev/null | sort"])
    skill_text = out
    rc2, out2, _ = run(["bash", "-lc", "find \"$HOME/.agents/skills\" -maxdepth 3 -name SKILL.md 2>/dev/null | sort"])
    skill_text += "\n" + out2
    for name in reg.get("preferred_skills", []):
        findings.append(Finding(
            "context",
            f"skill:{name}",
            "ok" if name in skill_text else "missing",
            "medium" if name not in skill_text else "low",
            "Skill 配置状況確認",
            "不足していれば project-local skill を作成する"
        ))

    write_json(LOG_DIR / "context_engineering_audit.json", [f.__dict__ for f in findings])
    write_md(LOG_DIR / "context_engineering_audit.md", "Context Engineering Audit", findings)
    write_csv(LOG_DIR / "context_engineering_audit.csv", findings)

if __name__ == "__main__":
    main()
PY

cat > "$ROOT/scripts/context_lab/audit_harness_engineering.py" <<'PY'
from __future__ import annotations
from pathlib import Path
from common import ROOT, LOG_DIR, REGISTRY, read_yaml, Finding, write_json, write_md, write_csv, run

def main() -> None:
    reg = read_yaml(REGISTRY)
    findings: list[Finding] = []

    for test_path in reg.get("required_tests", []):
        p = Path(test_path)
        findings.append(Finding(
            "harness",
            p.name,
            "ok" if p.exists() else "missing",
            "high" if not p.exists() else "low",
            "必須テスト資産確認",
            "不足していればテストを作成する"
        ))

    smoke_cmds = [
        ["python", "-m", "py_compile", str(ROOT / "src/loto_forecast/api/streamlit/operations_dashboard.py")],
        ["bash", "-lc", f"cd {ROOT} && pytest --no-cov -q tests/unit/test_operations_dashboard_ui_helpers.py"],
        ["bash", "-lc", f"cd {ROOT} && pytest --no-cov -q tests/streamlit/test_operations_dashboard_apptest.py"],
    ]
    for i, cmd in enumerate(smoke_cmds, start=1):
        rc, out, err = run(cmd)
        findings.append(Finding(
            "harness",
            f"smoke_{i}",
            "ok" if rc == 0 else "failed",
            "high" if rc != 0 else "low",
            (out or err).strip()[:500],
            "失敗時は原因を切り分けて修正する"
        ))

    trace_files = [
        ROOT / "artifacts/logs/coverage_matrix.md",
        ROOT / "artifacts/logs/dynamic_trace.jsonl",
        ROOT / "artifacts/logs/dynamic_trace.csv",
        ROOT / "artifacts/logs/db_observation.json",
        ROOT / "artifacts/logs/file_observation.json",
    ]
    for p in trace_files:
        findings.append(Finding(
            "harness",
            p.name,
            "ok" if p.exists() else "missing",
            "medium" if not p.exists() else "low",
            "トレース器の存在確認",
            "不足なら先に空ファイルを作る"
        ))

    write_json(LOG_DIR / "harness_engineering_audit.json", [f.__dict__ for f in findings])
    write_md(LOG_DIR / "harness_engineering_audit.md", "Harness Engineering Audit", findings)
    write_csv(LOG_DIR / "harness_engineering_audit.csv", findings)

if __name__ == "__main__":
    main()
PY

cat > "$ROOT/scripts/context_lab/build_context_packet.py" <<'PY'
from __future__ import annotations
from pathlib import Path

ROOT = Path("${PROJECT_ROOT}")
PACKET = ROOT / "docs/context/03_context_packet.md"
FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs/context/00_context_index.md",
    ROOT / "docs/context/01_execution_contract.md",
    ROOT / "docs/context/02_decision_policy.md",
    ROOT / "docs/context/04_tooling_scope.md",
    ROOT / "docs/23_harness_engineering_design.md",
    ROOT / "docs/25_context_harness_design.md",
    ROOT / "docs/operations_dashboard_debug_runbook.md",
    ROOT / "docs/operations_dashboard_operation_manual.md",
    ROOT / "docs/ui_audit/operations_dashboard_audit.md",
    ROOT / "docs/ui_audit/operations_dashboard_test_matrix.md",
    ROOT / "artifacts/logs/browser_observation.md",
    ROOT / "artifacts/logs/browser_runtime_identity.md",
    ROOT / "artifacts/logs/windows_runtime_final.md",
    ROOT / "artifacts/logs/bug_list.md",
]

def main() -> None:
    lines = ["# Context Packet", ""]
    for p in FILES:
        if p.exists():
            lines += ["---", f"## FILE: {p.relative_to(ROOT)}", "", p.read_text(encoding="utf-8", errors="ignore"), ""]
    PACKET.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
PY

cat > "$ROOT/scripts/context_lab/run_all.py" <<'PY'
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path("${PROJECT_ROOT}")
scripts = [
    ROOT / "scripts/context_lab/audit_prompt_engineering.py",
    ROOT / "scripts/context_lab/audit_context_engineering.py",
    ROOT / "scripts/context_lab/audit_harness_engineering.py",
    ROOT / "scripts/context_lab/build_context_packet.py",
]

def main() -> None:
    for s in scripts:
        subprocess.run(["python", str(s)], check=False)
    summary = ROOT / "artifacts/logs/context_engineering_summary.md"
    summary.write_text(
        "\n".join([
            "# Context Engineering Summary",
            "",
            "- prompt_engineering_audit.md を確認",
            "- context_engineering_audit.md を確認",
            "- harness_engineering_audit.md を確認",
            "- docs/context/03_context_packet.md を確認",
        ]),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
PY

chmod +x "$ROOT/scripts/bootstrap_context_lab.sh"
echo "generated: $ROOT/scripts/context_lab/* and docs/context/best_practices_registry.yaml"
