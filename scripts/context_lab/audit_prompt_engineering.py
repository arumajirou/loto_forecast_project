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
