# Panel Renderer Decomposition Plan

| Target File | Function / Block | Responsibility | Pure Helper Viability | DI Separable | Unit Testability | Coverage Impact | UI Risk | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | `render_nf_resource_analytics_panel` header guards | DB connection / table availability fallback messages | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | filter controls block | default date range, option candidates, baseline bounds, group selector | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | fetch sizing block | fetch limit derivation from `row_limit` | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | run/stage/metric/log merge block | frame normalization, aggregate merge order, default error count | Medium | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | summary tab | metric card values, summary columns, caption payload | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | ranking tab | rank column selection, anomaly table shaping | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | bottleneck tab | stage ranking, selected run stage summary, metric key options | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | error tab | error aggregate tables, timeline run candidates, error context input shaping | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | compare tab | eligible group selection, aggregate rows, statistical payload | Medium | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel.py` | proposal tab | recommendation input payload / anomaly count | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | DB tab | confirmation text generation, DB SQL composition, success labels | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | schema tab | schema SQL composition, expected confirmation text, current selection validation | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | table create/alter/delete block | add/rename/drop column and table SQL, option stabilization | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | row CRUD block | payload validation, allow-all guards, CRUD execution params | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | inspect tab | schema/table selection normalization, inspect summary tables | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | ER tab | default schema selection, FK subset filtering, ER display rows | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel.py` | SQL tab | execution parameter generation, statement normalization, fallback labels | Medium | High | High | Medium | Medium | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | selection block | selected run/model/meta derivation and default snapshots | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | overview tab | overview metrics and snapshot payload shaping | High | High | High | Medium | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | model detail tab | file inventory rows and pickle summary display shaping | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | config check tab | mismatch flags, config-check rows, fix-needed messaging | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | resource tab | run resource metric cards, model resource aggregate, stage summary shaping | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | accuracy tab | default metric selection, model aggregate rows | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | correlation / causal tab | analysis frame merge, target eligibility, correlation rows, treatment options | High | High | High | High | Low | High |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel.py` | export tab | export payload compilation and preview shaping | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_nf_lifecycle_lab` state block | selector state, query param interpretation, normalization | High | High | High | High | Medium | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_nf_lifecycle_lab` command block | command preview, label/key generation, execution payload shaping | High | High | High | High | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_nf_lifecycle_lab` export/compiler blocks | markdown/html/json compile payloads | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_resources_analytics` wrapper | panel invocation guard and parameter assembly | Medium | High | Medium | Low | Low | Low |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_operation_runner` | command/result formatting and selector generation | High | High | High | Medium | Low | Medium |
| `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `_render_table_inspector` / `_render_schema_export` / compiler blocks | export formatting and selector payloads | High | High | High | Medium | Low | Medium |

## Recommended Order

1. `dashboard_nf_resource_analytics_panel.py`
2. `dashboard_db_admin_panel.py`
3. `dashboard_nf_runid_panel.py`
4. `operations_dashboard.py` only if total coverage is still below `50%`

## Planned Submodules

- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel_state.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel_formatter.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_resource_analytics_panel_aggregator.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel_sql.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel_formatter.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_db_admin_panel_validator.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel_state.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel_formatter.py`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/dashboard_nf_runid_panel_analysis.py`
