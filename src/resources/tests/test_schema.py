from resources.config import ResourcesConfig
from resources.db.schema import table_names


def test_table_names_plain() -> None:
    cfg = ResourcesConfig(schema="resources", namespace="timesfm", table_naming="plain")
    t = table_names(cfg)
    assert t["run"] == "resources.run"
    assert t["stage"] == "resources.stage_span"


def test_table_names_namespaced() -> None:
    cfg = ResourcesConfig(schema="resources", namespace="timesfm", table_naming="namespaced")
    t = table_names(cfg)
    assert t["run"] == "resources.timesfm_run"
    assert t["stage"] == "resources.timesfm_stage_span"
