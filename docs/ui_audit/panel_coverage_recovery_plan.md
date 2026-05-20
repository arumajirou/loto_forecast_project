# Panel Coverage Recovery Plan

## Summary Table

| File Path | Current Coverage Estimate | Large / Dense Responsibilities | Pure Helper Candidates | DI-Friendly Candidates | Easy Unit-Test Targets | Expected Coverage Gain per Change | UI Risk | Priority |
|---|---:|---|---|---|---|---:|---:|---:|
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | 3% | `render_nf_resource_analytics_panel` is ~900 lines and mixes query results normalization, filter application, baseline construction, anomaly scoring, stage aggregation, recommendations, and chart input shaping | run/model merge, filter normalization, baseline stats, run KPI derivation, anomaly flagging, stage aggregation, error context assembly | query result adapters for `run_df` / `stage_df` / `metric_df` / `hist_df` / `error_df` | dataframe transforms and recommendation inputs | High | Low-Medium | 1 |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | 5% | `render_db_admin_panel` is ~600 lines and mixes confirmation validation, SQL string building, CRUD payload normalization, ER graph shaping, and table inspection prep | create/alter/drop SQL builders, confirmation validators, row CRUD payload normalization, inspect summaries | SQL execution wrapper inputs, `query_df` result normalization | SQL builder outputs and payload validators | Medium-High | Medium | 2 |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | 5% | `render_runid_integrated_panel` is ~500 lines and mixes snapshot composition, mismatch checks, resource summaries, metric flattening, and analysis-frame assembly | snapshot rows, mismatch checks, stage summary shaping, model resource efficiency, metric flattening, analysis dataframe builders | query result shaping for `resources.run` / `stage_span` / `model_df` | check tables and flattened metric/analysis frames | Medium | Low-Medium | 3 |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | 23% | remaining dense renderer functions are panel-like composites; already helper-aliased in many low-risk areas | remaining formatter / selector / runner-summary blocks inside `_render_resources_analytics`, `_render_dataset_model_grid`, `_render_operation_runner` | renderer-local state loaders and query result normalizers | table/chart payload shaping | Medium | Medium | 4 |

## Ordering Rationale

1. `dashboard_nf_resource_analytics_panel.py`
   This file has the lowest coverage and the most pandas-heavy logic already concentrated in one renderer. It offers the highest chance to move 200+ executable lines into testable helpers without touching Streamlit widget flow.

2. `dashboard_db_admin_panel.py`
   This file has many SQL-string and payload-validation branches that are easy to isolate and verify deterministically. Coverage gain is likely better than `runid` once helper call sites are migrated.

3. `dashboard_nf_runid_panel.py`
   The renderer is smaller, but it still contains several pure data-shaping blocks that can be extracted after the first two panels.

4. `operations_dashboard.py`
   Remaining gains here are real, but marginal compared with the three low-coverage panels because the file is very large and the easy helper work has already been done.

## First Refactor Targets

### 1. Resource Analytics Panel

Target responsibilities:

- normalize and enrich `run_df`
- merge optional `model_df`
- build baseline pool and expected-value stats
- compute anomaly score / anomaly flag
- aggregate `stage_df`
- build run summary table inputs

Proposed helper file:

- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel_helpers.py`

### 2. DB Admin Panel

Target responsibilities:

- create/drop confirmation validation
- create/alter/drop SQL builders
- row CRUD payload validation and clause compilation
- inspect / summary payload formatting

Proposed helper file:

- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel_helpers.py`

### 3. RunID Panel

Target responsibilities:

- run snapshot composition
- mismatch check dataframe rows
- selected run resource summary shaping
- model metric flattening
- analysis dataframe assembly

Proposed helper file:

- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel_helpers.py`

## Success Criteria for This Recovery Pass

- no new coverage omit
- helper modules are called from the original panel renderers
- unit tests hit the helper call paths that the panel renderers now use
- AppTest and E2E smoke remain green
- overall `pytest -q` should move meaningfully upward; if `50%` is still unreachable, the remaining unextracted renderer blocks must be enumerated explicitly
