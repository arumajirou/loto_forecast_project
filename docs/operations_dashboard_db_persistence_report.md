# Operations Dashboard DB Persistence Report

## Target
- Table: `log.ui_state_snapshot`
- State key: `operations_dashboard:nf_lab_fixed_grid:127.0.0.1:5432:loto:loto`

## Before
- Row count for target key: `1`
- `state_hash`: `788b12525b4d710249cde2b3389a0ccc0383382b28d9d4fffdad544eafe7e419`
- `updated_at`: `2026-03-31 18:46:05.309491+09`

## Trigger
- Connected route `8510`
- Navigate to `NeuralForecast 実行・検証ラボ`
- Open `学習(train)` then `保存/ロード`

## After
- Row count for target key: `1`
- `state_hash`: `0689dbc35c2178be5b59b86e9dd6ce8bed82593dcde3d55282a05aba75067ba5`
- `updated_at`: `2026-03-31 19:07:12.22953+09`
- Sample persisted field: `nf_lab_section_select=保存/ロード`

## Assessment
- Result: `updated in place`
- Insert count: `0`
- Update count: `1`
- `log.run_history`: `0`
- `log.error_event`: `0`

## Note
- The exact intra-navigation write edge was not separately instrumented, but the connected NF lab flow clearly updated the persisted UI-state row.
