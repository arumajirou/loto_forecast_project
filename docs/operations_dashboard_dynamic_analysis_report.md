# Operations Dashboard Dynamic Analysis Report

## Route Separation
- Linux fallback route: `8511`
- Linux connected route: `8510`
- Windows Chrome: reference-only, not mixed with current Linux results

## Observed Dynamic Behavior
- Fallback route remained interactive after DB connection failure.
- `運用` fallback mode continued to expose non-DB tabs.
- Safe local compile flows rendered summaries without crashing.
- Connected route loaded overview, NF lab, resource analytics, and schema export panels.
- NF lab navigation triggered UI-state persistence into `log.ui_state_snapshot`.
- Dashboard event logs were appended during route navigation.

## Screenshot Evidence
- Fallback:
  - `artifacts/screenshots/exhaustive/8511_home_fallback.png`
  - `artifacts/screenshots/exhaustive/8511_operations_fallback.png`
  - `artifacts/screenshots/exhaustive/8511_directory_invalid.png`
  - `artifacts/screenshots/exhaustive/8511_directory_compile.png`
  - `artifacts/screenshots/exhaustive/8511_markdown_compile.png`
  - `artifacts/screenshots/exhaustive/8511_artifacts_logs.png`
- Connected:
  - `artifacts/screenshots/exhaustive/8510_home_connected.png`
  - `artifacts/screenshots/exhaustive/8510_overview.png`
  - `artifacts/screenshots/exhaustive/8510_nf_lab_train.png`
  - `artifacts/screenshots/exhaustive/8510_nf_lab_save_load.png`
  - `artifacts/screenshots/exhaustive/8510_resources.png`
  - `artifacts/screenshots/exhaustive/8510_schema_export.png`

## Conclusion
- No new interactive defect was observed in the audited flows.
- The route split was preserved throughout the evidence set.
