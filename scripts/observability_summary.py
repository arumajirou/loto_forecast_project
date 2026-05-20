#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto_forecast.observability.store import build_observability_snapshot, write_summary_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a local observability summary JSON report.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    path = write_summary_report(output, limit=args.limit)
    snapshot = build_observability_snapshot(limit=args.limit)
    print(json.dumps({"report": str(path), "snapshot": asdict(snapshot)}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
