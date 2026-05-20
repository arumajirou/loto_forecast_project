# 結合テストはローカルPostgresに依存するため、CIではskipする想定
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_INTEGRATION", "0") != "1", reason="integration tests require local DB")


def test_placeholder():
    assert True
