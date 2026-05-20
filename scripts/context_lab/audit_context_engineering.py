from __future__ import annotations

from pathlib import Path

from common import LOG_DIR, REGISTRY, read_yaml, Finding, write_json, write_md, write_csv, run


def main() -> None:
    reg = read_yaml(REGISTRY)
    findings: list[Finding] = []

    for file_path in reg.get("required_files", []):
        p = Path(file_path)
        findings.append(
            Finding(
                "context",
                p.name,
                "ok" if p.exists() else "missing",
                "high" if not p.exists() else "low",
                "必須コンテキストファイル確認",
                "不足していれば生成する",
            )
        )

    for log_path in reg.get("required_logs", []):
        p = Path(log_path)
        findings.append(
            Finding(
                "context",
                p.name,
                "ok" if p.exists() else "missing",
                "medium" if not p.exists() else "low",
                "必須ログ器確認",
                "不足していれば空の器を作る",
            )
        )

    _, out, _ = run(["bash", "-lc", "codex mcp list"])
    for name in reg.get("required_mcp", []):
        findings.append(
            Finding(
                "context",
                f"mcp:{name}",
                "ok" if name in out else "missing",
                "high" if name not in out else "low",
                "MCP 登録状況確認",
                "不足していれば codex mcp add で追加する",
            )
        )

    _, out1, _ = run(["bash", "-lc", "find .agents/skills -maxdepth 2 -name SKILL.md 2>/dev/null | sort"])
    _, out2, _ = run(["bash", "-lc", "find \"$HOME/.agents/skills\" -maxdepth 3 -name SKILL.md 2>/dev/null | sort"])
    skill_text = out1 + "\n" + out2

    for name in reg.get("preferred_skills", []):
        findings.append(
            Finding(
                "context",
                f"skill:{name}",
                "ok" if name in skill_text else "missing",
                "medium" if name not in skill_text else "low",
                "Skill 配置状況確認",
                "不足していれば project-local skill を作成する",
            )
        )

    write_json(LOG_DIR / "context_engineering_audit.json", [f.__dict__ for f in findings])
    write_md(LOG_DIR / "context_engineering_audit.md", "Context Engineering Audit", findings)
    write_csv(LOG_DIR / "context_engineering_audit.csv", findings)


if __name__ == "__main__":
    main()
