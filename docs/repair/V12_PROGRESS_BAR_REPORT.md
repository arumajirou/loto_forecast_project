# v12 Progress Bar Repair Report

## Conclusion
This release adds visible progress reporting for browser screenshot collection.

## What changed
- `scripts/collect_browser_observability.py`
  - Adds terminal progress bar output.
  - Writes `progress.jsonl` under each browser run directory.
  - Reports stage, current URL, screenshots, clicks, skipped dangerous actions, and warnings.
- `scripts/capture_app_screenshots.sh`
  - Prints that the collector has started and explains the progress format.
- `scripts/start_dashboard_and_capture_screenshots.sh`
  - Prints capture start parameters and shows recent artifacts after completion.
- `src/loto_forecast/observability/store.py`
  - Includes latest progress state in browser run summaries.
- `operations_dashboard.py`
  - Shows latest browser-run progress in the Observability panel.

## Safety
- DB writes are not executed.
- `dataset` remains read-only.
- Browser automation still uses safe-clicks by default.
- Dangerous labels such as delete, drop, db-init, execute, run, start, write are skipped by default.

## Verification performed
- `PYTHONPATH=src python -m compileall -q src tests tools evals scripts`
- `python scripts/collect_browser_observability.py --help`

## Not performed
- Streamlit live launch.
- Playwright browser collection.
- DB connection.
- DB writes.
- Cron/WSL automation install.
