#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto_forecast.observability.store import (  # noqa: E402
    RUNS_DIR,
    ensure_observability_dirs,
    record_event,
    utc_now_iso,
    write_summary_report,
)

DANGEROUS_TEXT_RE = re.compile(
    r"(db-init|delete|drop|truncate|破壊|削除|初期化|実行|run|start|launch|apply|書き込み|開始|反映|保存|登録|更新|送信|submit)",
    re.IGNORECASE,
)
NON_ACTION_TEXT_RE = re.compile(
    r"(^\S+@\S+\.\S+$|^https?://|^localhost:?\d*$|^127\.0\.0\.1:?\d*$|"
    r"^Help for\s+|^Show password text$|^Show/hide columns$|^Download as CSV$|^Search$|^Fullscreen$)",
    re.IGNORECASE,
)
PROGRESS_BAR_WIDTH = 30


def _safe_name(value: str, *, fallback: str = "page") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80]
    return cleaned or fallback


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _progress_bar(done: int, total: int) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(PROGRESS_BAR_WIDTH * done / total)
    return "[" + "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled) + f"] {done}/{total} {done / total:6.1%}"


def _shorten(value: str, *, limit: int = 90) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _visible_action_candidates(page: Any, *, max_candidates: int) -> list[dict[str, Any]]:
    """Return visible, labelled, clickable-looking elements with stable temporary selectors.

    Streamlit renders many hidden buttons and repeated DOM nodes.  Clicking those
    blindly creates long timeout chains.  This helper filters on visibility,
    dimensions, disabled state, pointer-events, and non-empty labels before the
    Python click loop starts.
    """
    max_candidates = max(1, int(max_candidates))
    records = await page.evaluate(
        """(maxCandidates) => {
            const nodes = Array.from(document.querySelectorAll('button, [role="tab"], [role="button"], a'));
            const out = [];
            const now = Date.now();
            let ordinal = 0;
            for (const el of nodes) {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                const label = (
                    el.innerText ||
                    el.getAttribute("aria-label") ||
                    el.getAttribute("title") ||
                    el.getAttribute("value") ||
                    ""
                ).replace(/\\s+/g, " ").trim();
                const disabled = Boolean(
                    el.disabled ||
                    el.getAttribute("disabled") !== null ||
                    String(el.getAttribute("aria-disabled") || "").toLowerCase() === "true"
                );
                const visible = (
                    rect.width >= 4 &&
                    rect.height >= 4 &&
                    style.display !== "none" &&
                    style.visibility !== "hidden" &&
                    style.pointerEvents !== "none" &&
                    Number(style.opacity || "1") > 0
                );
                if (!visible || disabled || !label) {
                    continue;
                }
                const id = `loto-obsv-${now}-${ordinal}`;
                ordinal += 1;
                el.setAttribute("data-loto-obsv-id", id);
                out.push({
                    id,
                    label: label.slice(0, 120),
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute("role") || "",
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height),
                });
                if (out.length >= maxCandidates) {
                    break;
                }
            }
            return out;
        }""",
        max_candidates,
    )
    return list(records or [])


async def _collect(args: argparse.Namespace) -> int:
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ImportError:
        record_event(
            source="browser_observability",
            category="dependency",
            level="ERROR",
            message="playwright is not installed. Run: LOTO_UV_ENV_MODE=browser ./scripts/setup_uv.sh",
        )
        print("playwright is not installed. Run: LOTO_UV_ENV_MODE=browser ./scripts/setup_uv.sh", file=sys.stderr)
        return 2

    ensure_observability_dirs()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / run_id
    screenshot_dir = run_dir / "screenshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    console_path = run_dir / "console.jsonl"
    network_path = run_dir / "network.jsonl"
    pageerror_path = run_dir / "page_errors.jsonl"
    visited_path = run_dir / "visited.jsonl"
    progress_path = run_dir / "progress.jsonl"
    manifest_path = run_dir / "manifest.json"

    max_attempts_for_progress = int(args.max_attempts or args.max_clicks)
    estimated_total = max(1, min(max(1, max_attempts_for_progress), max(1, int(args.max_clicks))) + 1)
    progress_state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "stage": "starting",
        "done": 0,
        "total": estimated_total,
        "screenshots": 0,
        "clicks": 0,
        "processed": 0,
        "skipped": 0,
        "warnings": 0,
        "depth": 0,
        "current_url": args.url,
        "run_dir": str(run_dir),
        "progress_path": str(progress_path),
        "started_at": utc_now_iso(),
    }

    def emit_progress(stage: str, message: str = "", **updates: Any) -> None:
        progress_state.update(updates)
        progress_state["stage"] = stage
        progress_state["message"] = message
        progress_state["ts"] = utc_now_iso()
        progress_state["percent"] = round(
            min(1.0, float(progress_state.get("done", 0)) / max(1, int(progress_state.get("total", 1)))) * 100,
            2,
        )
        _write_jsonl(progress_path, dict(progress_state))
        if not args.quiet_progress:
            bar = _progress_bar(int(progress_state.get("done", 0)), int(progress_state.get("total", 1)))
            suffix = f" | {stage}"
            if message:
                suffix += f" | {_shorten(message)}"
            print(f"{bar}{suffix}", flush=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "url": args.url,
        "started_at": utc_now_iso(),
        "safe_clicks": bool(args.safe_clicks),
        "max_clicks": int(args.max_clicks),
        "max_depth": int(args.max_depth),
        "viewport": {"width": int(args.width), "height": int(args.height)},
        "screenshots": [],
        "errors": [],
        "warnings": [],
        "progress_path": str(progress_path),
    }

    record_event(
        source="browser_observability",
        category="browser_run",
        level="INFO",
        run_id=run_id,
        message=f"browser observability run started: {args.url}",
        payload={"url": args.url, "run_dir": str(run_dir), "progress_path": str(progress_path)},
    )

    emit_progress("start", f"url={args.url} run_dir={run_dir}")

    click_count = 0
    skipped_count = 0
    warning_count = 0

    async with async_playwright() as p:
        emit_progress("launch_browser", "launching chromium")
        browser = await p.chromium.launch(headless=not args.headed)
        context = await browser.new_context(
            viewport={"width": int(args.width), "height": int(args.height)},
            record_har_path=str(run_dir / "network.har") if args.har else None,
        )

        if args.trace:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)
            emit_progress("trace_started", "Playwright tracing enabled")

        page = await context.new_page()

        page.on(
            "console",
            lambda msg: _write_jsonl(
                console_path,
                {
                    "ts": utc_now_iso(),
                    "type": msg.type,
                    "text": msg.text,
                    "location": msg.location,
                },
            ),
        )
        page.on(
            "pageerror",
            lambda exc: _write_jsonl(
                pageerror_path,
                {
                    "ts": utc_now_iso(),
                    "message": str(exc),
                },
            ),
        )
        page.on(
            "requestfailed",
            lambda req: _write_jsonl(
                network_path,
                {
                    "ts": utc_now_iso(),
                    "type": "requestfailed",
                    "url": req.url,
                    "method": req.method,
                    "failure": req.failure,
                },
            ),
        )
        page.on(
            "response",
            lambda resp: _write_jsonl(
                network_path,
                {
                    "ts": utc_now_iso(),
                    "type": "response",
                    "url": resp.url,
                    "status": resp.status,
                    "ok": resp.ok,
                },
            )
            if resp.status >= 400
            else None,
        )

        async def capture(label: str, *, full_page: bool = True) -> Path:
            path = screenshot_dir / f"{len(manifest['screenshots']):03d}_{_safe_name(label)}.png"
            await page.screenshot(path=str(path), full_page=full_page)
            item = {"label": label, "path": str(path), "ts": utc_now_iso()}
            manifest["screenshots"].append(item)
            screenshot_count = len(manifest["screenshots"])
            record_event(
                source="browser_observability",
                category="screenshot",
                level="OK",
                run_id=run_id,
                message=f"screenshot captured: {label}",
                payload=item,
            )
            emit_progress(
                "screenshot",
                label,
                done=min(screenshot_count, estimated_total),
                screenshots=screenshot_count,
                clicks=click_count,
                skipped=skipped_count,
                warnings=warning_count,
                current_url=page.url,
                latest_screenshot=str(path),
            )
            return path

        try:
            emit_progress("goto", args.url)
            await page.goto(args.url, wait_until="domcontentloaded", timeout=int(args.timeout_ms))
            await page.wait_for_timeout(int(args.settle_ms))
            await capture("initial")
        except PlaywrightTimeoutError as exc:
            manifest["errors"].append({"stage": "goto", "message": str(exc)})
            record_event(
                source="browser_observability",
                category="browser_error",
                level="ERROR",
                run_id=run_id,
                message=f"browser goto timeout: {args.url}",
                exc=exc,
            )
            emit_progress("goto_timeout", str(exc), status="error", warnings=warning_count + 1)
        except Exception as exc:  # noqa: BLE001
            manifest["errors"].append({"stage": "goto", "message": str(exc)})
            record_event(
                source="browser_observability",
                category="browser_error",
                level="ERROR",
                run_id=run_id,
                message=f"browser goto failed: {args.url}",
                exc=exc,
            )
            emit_progress("goto_failed", str(exc), status="error", warnings=warning_count + 1)

        seen_labels: set[str] = set()
        processed_count = 0
        max_attempts = int(args.max_attempts or args.max_clicks)
        max_attempts = max(1, max_attempts)

        for depth in range(max(0, int(args.max_depth)) + 1):
            if click_count >= int(args.max_clicks) or processed_count >= max_attempts:
                break

            emit_progress("scan", f"depth={depth}: collecting visible labelled candidates", depth=depth)
            candidates = await _visible_action_candidates(page, max_candidates=int(args.max_candidates))
            emit_progress(
                "scan",
                f"depth={depth}: visible_candidates={len(candidates)}",
                depth=depth,
                candidates=len(candidates),
                done=min(1 + processed_count, estimated_total),
            )

            for index, record in enumerate(candidates):
                if click_count >= int(args.max_clicks) or processed_count >= max_attempts:
                    break

                label = str(record.get("label") or "").strip() or f"candidate_{depth}_{index}"
                fingerprint = f"{record.get('tag', '')}:{record.get('role', '')}:{label.lower()}"
                if fingerprint in seen_labels:
                    continue
                seen_labels.add(fingerprint)

                is_dangerous = bool(DANGEROUS_TEXT_RE.search(label))
                is_non_action = bool(NON_ACTION_TEXT_RE.search(label))
                if (args.safe_clicks and is_dangerous) or is_non_action:
                    skipped_count += 1
                    processed_count += 1
                    action = "skip_dangerous" if is_dangerous else "skip_non_action"
                    _write_jsonl(
                        visited_path,
                        {
                            "ts": utc_now_iso(),
                            "action": action,
                            "depth": depth,
                            "label": label,
                            "candidate": record,
                            "processed_count": processed_count,
                        },
                    )
                    emit_progress(
                        action,
                        f"{action}={skipped_count}: {label}",
                        done=min(1 + processed_count, estimated_total),
                        skipped=skipped_count,
                        clicks=click_count,
                        depth=depth,
                    )
                    continue

                try:
                    emit_progress(
                        "click",
                        f"attempt={processed_count + 1}/{max_attempts} success={click_count}/{args.max_clicks}: {label}",
                        done=min(1 + processed_count, estimated_total),
                        clicks=click_count,
                        skipped=skipped_count,
                        depth=depth,
                    )
                    base_locator = page.locator(f"[data-loto-obsv-id='{record['id']}']")
                    first_locator = getattr(base_locator, "first")
                    locator = first_locator() if callable(first_locator) else first_locator
                    await locator.scroll_into_view_if_needed(timeout=int(args.scroll_timeout_ms))
                    await locator.click(timeout=int(args.click_timeout_ms))
                    processed_count += 1
                    click_count += 1
                    await page.wait_for_timeout(int(args.settle_ms))
                    await capture(f"click_{click_count}_{label}", full_page=True)
                    _write_jsonl(
                        visited_path,
                        {
                            "ts": utc_now_iso(),
                            "action": "click",
                            "depth": depth,
                            "label": label,
                            "click_count": click_count,
                            "processed_count": processed_count,
                            "candidate": record,
                            "url": page.url,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    processed_count += 1
                    warning_count += 1
                    manifest["warnings"].append({"stage": "click", "label": label, "message": str(exc)})
                    record_event(
                        source="browser_observability",
                        category="browser_click_warning",
                        level="WARNING",
                        run_id=run_id,
                        message=f"click failed or skipped: {label}",
                        payload={"label": label, "depth": depth, "error": str(exc)[:500], "candidate": record},
                    )
                    emit_progress(
                        "click_warning",
                        f"attempt={processed_count}/{max_attempts}: {label}: {_shorten(str(exc), limit=120)}",
                        done=min(1 + processed_count, estimated_total),
                        warnings=warning_count,
                        clicks=click_count,
                        skipped=skipped_count,
                        depth=depth,
                    )
            if click_count >= int(args.max_clicks) or processed_count >= max_attempts:
                break

        if args.trace:
            trace_path = run_dir / "trace.zip"
            emit_progress("trace_stop", str(trace_path))
            await context.tracing.stop(path=str(trace_path))
            manifest["trace_path"] = str(trace_path)

        await context.close()
        await browser.close()

    manifest["finished_at"] = utc_now_iso()
    manifest["click_count"] = click_count
    manifest["skipped_count"] = skipped_count
    manifest["warning_count"] = warning_count
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_path = write_summary_report(limit=5000)

    final_status = "error" if manifest["errors"] else "finished"
    emit_progress(
        "finished",
        f"screenshots={len(manifest['screenshots'])} clicks={click_count} skipped={skipped_count} report={summary_path}",
        status=final_status,
        done=estimated_total,
        clicks=click_count,
        processed=locals().get("processed_count", click_count + skipped_count + warning_count),
        skipped=skipped_count,
        warnings=warning_count,
        finished_at=utc_now_iso(),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
    )

    record_event(
        source="browser_observability",
        category="browser_run",
        level="OK" if not manifest["errors"] else "ERROR",
        run_id=run_id,
        message=f"browser observability run finished: screenshots={len(manifest['screenshots'])}, clicks={click_count}",
        payload={"manifest": str(manifest_path), "screenshots": len(manifest["screenshots"]), "clicks": click_count},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if not manifest["errors"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect screenshots, console logs, traces, and network diagnostics.")
    parser.add_argument("--url", default=os.getenv("LOTO_DASHBOARD_URL", "http://localhost:8505"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--trace", action="store_true", default=True)
    parser.add_argument("--no-trace", dest="trace", action="store_false")
    parser.add_argument("--har", action="store_true", default=True)
    parser.add_argument("--no-har", dest="har", action="store_false")
    parser.add_argument("--safe-clicks", action="store_true", default=True)
    parser.add_argument("--unsafe-clicks", dest="safe_clicks", action="store_false")
    parser.add_argument("--max-clicks", type=int, default=40, help="Maximum successful clicks/screenshots to capture.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum click/skip/warning attempts. Defaults to --max-clicks so progress always advances.",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=160)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--click-timeout-ms", type=int, default=600, help="Per-candidate click timeout.")
    parser.add_argument("--scroll-timeout-ms", type=int, default=350, help="Per-candidate scroll timeout.")
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--progress-every", type=int, default=10, help="Print skip/scan progress every N skipped items.")
    parser.add_argument("--quiet-progress", action="store_true", help="Disable terminal progress bar output.")
    return parser


def main() -> int:
    return asyncio.run(_collect(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
