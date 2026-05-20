#!/usr/bin/env python3
"""Create a complete upload-ready analysis package for loto_forecast_project.

This v18 packager is designed for support/debug uploads. It packages all
available screenshots, browser observability runs, HAR files, Playwright traces,
console/network/progress logs, launcher logs, metrics/analysis reports, repair
docs, and selected project metadata into a single ZIP.

Safety:
- Does not connect to DB.
- Does not run db-init, training, cron installation, browser capture, or writes to DB.
- Excludes .venv, node_modules, caches, Python bytecode, Windows metadata, and
  recursively generated upload package ZIPs.
- Redacts common password/token patterns in generated diagnostic text.
- De-duplicates ZIP arc names and records any skipped duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "upload_packages"

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".backup",
    ".cache",
}
EXCLUDE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}
# Avoid recursive upload packages and local binary/cache explosions by default.
EXCLUDE_REL_PREFIXES = {
    "artifacts/upload_packages/",
}
SECRET_PATTERNS = [
    re.compile(r"(?i)(DB_PASSWORD\s*=\s*)([^\s#]+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(postgresql(?:\+psycopg2| \+psycopg)?://[^:\s/]+:)([^@\s]+)(@)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[:=]\s*)([^\s,;]+)"),
]


@dataclass(frozen=True)
class PackageConfig:
    include_artifacts: bool
    include_observability: bool
    include_traces: bool
    include_har: bool
    include_screenshots: bool
    include_logs: bool
    include_docs: bool
    include_reports: bool
    include_git_diff: bool
    include_tree: bool
    max_file_mb: float
    out_dir: Path
    note: str


class ZipWriter:
    def __init__(self, zf: zipfile.ZipFile) -> None:
        self.zf = zf
        self.seen: set[str] = set()
        self.duplicates: list[str] = []
        self.manifest_files: list[dict[str, object]] = []

    def writestr(self, arcname: str, text: str) -> bool:
        arcname = arcname.replace(os.sep, "/")
        if arcname in self.seen:
            self.duplicates.append(arcname)
            return False
        self.seen.add(arcname)
        self.zf.writestr(arcname, redact_text(text))
        return True

    def write_file(self, path: Path, arc_prefix: str) -> bool:
        arcname = f"{arc_prefix}/{safe_rel(path)}".replace(os.sep, "/")
        if arcname in self.seen:
            self.duplicates.append(arcname)
            return False
        self.seen.add(arcname)
        self.zf.write(path, arcname)
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            sha = ""
        self.manifest_files.append(
            {
                "path": safe_rel(path),
                "zip_path": arcname,
                "size": path.stat().st_size,
                "sha256": sha,
            }
        )
        return True


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def redact_text(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(lambda m: f"{m.group(1)}<REDACTED>{m.group(3) if len(m.groups()) >= 3 else ''}", out)
    return out


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path).replace(os.sep, "/")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def should_exclude(path: Path, cfg: PackageConfig) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    rel = safe_rel(path)
    if any(rel.startswith(prefix) for prefix in EXCLUDE_REL_PREFIXES):
        return True
    if path.name.endswith(":Zone.Identifier"):
        return True
    if path.suffix in EXCLUDE_FILE_SUFFIXES:
        return True
    if path.is_file() and path.stat().st_size > cfg.max_file_mb * 1024 * 1024:
        return True
    return False


def iter_files(root: Path, cfg: PackageConfig) -> Iterable[Path]:
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and not should_exclude(p, cfg):
            yield p


def run_cmd(args: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return int(proc.returncode), redact_text(proc.stdout)
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_browser_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    runs_root = PROJECT_ROOT / "artifacts" / "observability" / "browser_runs"
    if not runs_root.exists():
        return runs
    for run_dir in sorted(runs_root.glob("*")):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = read_json(manifest_path)
        screenshots = sorted((run_dir / "screenshots").glob("*.png")) if (run_dir / "screenshots").exists() else []
        progress = run_dir / "progress.jsonl"
        console = run_dir / "console.jsonl"
        visited = run_dir / "visited.jsonl"
        runs.append(
            {
                "run_id": run_dir.name,
                "path": safe_rel(run_dir),
                "manifest_exists": manifest_path.exists(),
                "screenshots": len(screenshots),
                "has_trace": (run_dir / "trace.zip").exists(),
                "has_har": (run_dir / "network.har").exists(),
                "has_progress": progress.exists(),
                "has_console": console.exists(),
                "has_visited": visited.exists(),
                "manifest_click_count": manifest.get("click_count") if isinstance(manifest, dict) else None,
                "manifest_warning_count": manifest.get("warning_count") if isinstance(manifest, dict) else None,
                "manifest_error_count": len(manifest.get("errors", [])) if isinstance(manifest, dict) else None,
            }
        )
    return runs


def count_files_under(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def collect_summary(cfg: PackageConfig, selected_roots: list[Path]) -> dict[str, object]:
    observability_root = PROJECT_ROOT / "artifacts" / "observability"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "include_artifacts": cfg.include_artifacts,
            "include_observability": cfg.include_observability,
            "include_traces": cfg.include_traces,
            "include_har": cfg.include_har,
            "include_screenshots": cfg.include_screenshots,
            "include_logs": cfg.include_logs,
            "include_docs": cfg.include_docs,
            "include_reports": cfg.include_reports,
            "include_git_diff": cfg.include_git_diff,
            "include_tree": cfg.include_tree,
            "max_file_mb": cfg.max_file_mb,
            "note": cfg.note,
        },
        "selected_roots": [safe_rel(p) if is_relative_to(p, PROJECT_ROOT) else str(p) for p in selected_roots],
        "counts": {
            "observability_files": count_files_under(observability_root),
            "docs_repair_files": count_files_under(PROJECT_ROOT / "docs" / "repair"),
            "logs_files": count_files_under(PROJECT_ROOT / "logs"),
            "reports_files": count_files_under(PROJECT_ROOT / "reports"),
            "artifacts_files": count_files_under(PROJECT_ROOT / "artifacts"),
        },
        "browser_runs": collect_browser_runs(),
    }


def selected_roots(cfg: PackageConfig) -> list[Path]:
    roots: list[Path] = []
    if cfg.include_observability:
        roots.append(PROJECT_ROOT / "artifacts" / "observability")
    if cfg.include_artifacts:
        # Include additional artifacts, but EXCLUDE_REL_PREFIXES avoids recursive upload_packages.
        roots.append(PROJECT_ROOT / "artifacts")
    if cfg.include_logs:
        roots.extend([PROJECT_ROOT / "logs", PROJECT_ROOT / "artifacts" / "observability" / "launcher"])
    if cfg.include_reports:
        roots.extend([PROJECT_ROOT / "reports", PROJECT_ROOT / "outputs"])
    if cfg.include_docs:
        # Add only docs once; docs/repair is included below through docs.
        roots.append(PROJECT_ROOT / "docs")
    # De-duplicate roots while preserving order, and drop child roots if parent already exists.
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if any(root == existing or is_relative_to(root, existing) for existing in out):
            continue
        out.append(root)
    return out


def include_path(path: Path, cfg: PackageConfig) -> bool:
    rel = f"/{safe_rel(path)}"
    name = path.name
    if not cfg.include_screenshots and "/screenshots/" in rel:
        return False
    if not cfg.include_traces and name == "trace.zip":
        return False
    if not cfg.include_har and path.suffix.lower() == ".har":
        return False
    return True


def build_package(cfg: PackageConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    out_zip = cfg.out_dir / f"loto_complete_analysis_package_{stamp}.zip"
    roots = selected_roots(cfg)
    summary = collect_summary(cfg, roots)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as raw_zf:
        z = ZipWriter(raw_zf)
        z.writestr("UPLOAD_PACKAGE_README.md", f"""# loto complete analysis upload package

Created at: {summary['created_at']}

## Contents

- `summary.json`: package overview and browser run inventory
- `manifest_files.json`: packaged file list with size and sha256
- `duplicates_skipped.json`: duplicate ZIP paths that were intentionally skipped
- `project_diagnostics/`: static environment summaries
- `project_files/`: screenshots, HAR, trace, console/network/progress logs, reports, repair docs, and selected metadata

## Safety

This package does not execute DB operations, db-init, training, cron installation, or browser capture.
It excludes `.venv`, `node_modules`, caches, bytecode, Windows metadata, and recursive upload package ZIPs.
Generated command outputs are redacted for common password/token patterns.

User note:

{cfg.note or "(none)"}
""")
        z.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

        commands = {
            "pwd.txt": ["pwd"],
            "git_status.txt": ["git", "status", "--short"],
            "python_version.txt": [sys.executable, "--version"],
            "observability_tree.txt": ["bash", "-lc", "find artifacts/observability -maxdepth 5 -type f | sort 2>/dev/null || true"],
            "upload_packages_tree.txt": ["bash", "-lc", "find artifacts/upload_packages -maxdepth 2 -type f | sort 2>/dev/null || true"],
        }
        if cfg.include_tree:
            commands["project_tree_depth4.txt"] = [
                "bash",
                "-lc",
                "find . -maxdepth 4 -type f "
                "\\( -path './.venv/*' -o -path './node_modules/*' -o -path './artifacts/upload_packages/*' \\) -prune "
                "-o -type f -print | sort | sed 's#^./##' | head -4000",
            ]
        if cfg.include_git_diff:
            commands["git_diff_stat.txt"] = ["git", "diff", "--stat"]
            commands["git_diff.txt"] = ["git", "diff", "--", ".", ":(exclude)artifacts", ":(exclude)logs", ":(exclude).venv"]

        for name, cmd in commands.items():
            code, output = run_cmd(cmd)
            z.writestr(f"project_diagnostics/{name}", f"$ {' '.join(cmd)}\nexit={code}\n\n{output}")

        for root_file in [
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / ".env.example",
            PROJECT_ROOT / "Makefile",
            PROJECT_ROOT / ".python-version",
        ]:
            if root_file.exists() and not should_exclude(root_file, cfg):
                z.write_file(root_file, "project_files")

        for root in roots:
            for path in iter_files(root, cfg):
                if include_path(path, cfg):
                    z.write_file(path, "project_files")

        z.writestr("manifest_files.json", json.dumps(z.manifest_files, ensure_ascii=False, indent=2))
        z.writestr("duplicates_skipped.json", json.dumps(sorted(set(z.duplicates)), ensure_ascii=False, indent=2))

    latest = cfg.out_dir / "latest_complete_analysis_package.zip"
    latest_compat = cfg.out_dir / "latest_upload_package.zip"
    for latest_path in [latest, latest_compat]:
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            shutil.copy2(out_zip, latest_path)
        except Exception:
            pass
    return out_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a complete upload-ready ZIP of screenshots, logs, metrics, traces, and reports.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the package ZIP.")
    parser.add_argument("--max-file-mb", type=float, default=200.0, help="Skip files larger than this size.")
    parser.add_argument("--no-artifacts", action="store_true", help="Exclude generic artifacts directory except selected roots.")
    parser.add_argument("--no-observability", action="store_true", help="Exclude artifacts/observability.")
    parser.add_argument("--no-traces", action="store_true", help="Exclude Playwright trace.zip files.")
    parser.add_argument("--no-har", action="store_true", help="Exclude network.har files.")
    parser.add_argument("--no-screenshots", action="store_true", help="Exclude screenshots.")
    parser.add_argument("--no-logs", action="store_true", help="Exclude logs.")
    parser.add_argument("--no-docs", action="store_true", help="Exclude docs.")
    parser.add_argument("--no-reports", action="store_true", help="Exclude reports/outputs.")
    parser.add_argument("--git-diff", action="store_true", help="Include git diff diagnostics.")
    parser.add_argument("--tree", action="store_true", help="Include file tree diagnostics.")
    parser.add_argument("--note", default="", help="User note to include in the package README.")
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    cfg = PackageConfig(
        include_artifacts=not ns.no_artifacts,
        include_observability=not ns.no_observability,
        include_traces=not ns.no_traces,
        include_har=not ns.no_har,
        include_screenshots=not ns.no_screenshots,
        include_logs=not ns.no_logs,
        include_docs=not ns.no_docs,
        include_reports=not ns.no_reports,
        include_git_diff=bool(ns.git_diff),
        include_tree=bool(ns.tree),
        max_file_mb=float(ns.max_file_mb),
        out_dir=Path(ns.out_dir).resolve(),
        note=str(ns.note),
    )
    out_zip = build_package(cfg)
    print(out_zip)
    print(f"latest complete: {cfg.out_dir / 'latest_complete_analysis_package.zip'}")
    print(f"latest compatible: {cfg.out_dir / 'latest_upload_package.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
