from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
            lines.extend(
                [
                    "---",
                    f"## FILE: {p.relative_to(ROOT)}",
                    "",
                    p.read_text(encoding="utf-8", errors="ignore"),
                    "",
                ]
            )
    PACKET.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
