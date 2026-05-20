#!/usr/bin/env python
from __future__ import annotations

import asyncio
import sys


async def main() -> int:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # noqa: BLE001
        print(f"playwright import failed: {exc}", file=sys.stderr)
        return 2

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        print("Playwright chromium launch: OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("Playwright chromium launch failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("", file=sys.stderr)
        print("Install command:", file=sys.stderr)
        print("  timeout 900s uv run --no-sync playwright install chromium", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
