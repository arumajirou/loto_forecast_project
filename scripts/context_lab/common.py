from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_md(path: Path, title: str, findings: list[Finding]) -> None:
    lines = [f"# {title}", ""]
    for item in findings:
        lines.extend(
            [
                f"## {item.name}",
                f"- category: {item.category}",
                f"- status: {item.status}",
                f"- severity: {item.severity}",
                f"- detail: {item.detail}",
                f"- recommendation: {item.recommendation}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "name",
                "status",
                "severity",
                "detail",
                "recommendation",
            ],
        )
        writer.writeheader()
        for item in findings:
            writer.writerow(asdict(item))
