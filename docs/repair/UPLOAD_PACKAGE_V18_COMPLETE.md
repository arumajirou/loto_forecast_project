# V18 Complete Analysis Upload Package

## Purpose

`package_analysis_upload.py` now creates a complete upload package for support/debugging.

It includes, when present:

- all `artifacts/observability` browser runs
- screenshots
- `network.har`
- Playwright `trace.zip`
- `console.jsonl`
- `progress.jsonl`
- `visited.jsonl`
- launcher logs
- `docs/repair`
- project docs
- reports and outputs
- project diagnostics
- selected project metadata

## Safety

The script does not connect to DB, does not run `db-init`, does not train models, does not install cron, and does not start browser capture.

Excluded by default:

- `.venv`
- `node_modules`
- caches
- Python bytecode
- Windows metadata
- recursive `artifacts/upload_packages`

## Command

```bash
cd /mnt/e/env/fc/loto_forecast_project

bash ./scripts/package_analysis_upload.sh \
  --tree \
  --note "complete screenshots logs metrics traces analysis package"
```

Main output:

```text
artifacts/upload_packages/latest_complete_analysis_package.zip
```

Compatibility output:

```text
artifacts/upload_packages/latest_upload_package.zip
```

## Verify

```bash
ls -lh artifacts/upload_packages/
unzip -l artifacts/upload_packages/latest_complete_analysis_package.zip | head -120
unzip -l artifacts/upload_packages/latest_complete_analysis_package.zip | grep -E 'screenshots|trace.zip|network.har|progress.jsonl|console.jsonl|manifest.json'
```
