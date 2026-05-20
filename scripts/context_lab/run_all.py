from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    ROOT / "scripts/context_lab/audit_prompt_engineering.py",
    ROOT / "scripts/context_lab/audit_context_engineering.py",
    ROOT / "scripts/context_lab/audit_harness_engineering.py",
    ROOT / "scripts/context_lab/build_context_packet.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run(["python", str(script)], check=False)

    summary = ROOT / "artifacts/logs/context_engineering_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Context Engineering Summary",
                "",
                "- prompt_engineering_audit.md を確認",
                "- context_engineering_audit.md を確認",
                "- harness_engineering_audit.md を確認",
                "- docs/context/03_context_packet.md を確認",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
