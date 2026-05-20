import pytest

from resources.config import ResourcesConfig


def test_config_validate() -> None:
    cfg = ResourcesConfig(namespace="timesfm", schema="resources", table_naming="plain")
    cfg.validate()


def test_config_validate_bad_table_naming() -> None:
    cfg = ResourcesConfig(namespace="timesfm", schema="resources", table_naming="bad")
    with pytest.raises(ValueError):
        cfg.validate()
