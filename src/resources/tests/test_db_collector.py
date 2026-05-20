from resources.collectors.db_collector import DBCollector


def test_db_collector_has_name_and_metrics() -> None:
    c = DBCollector()
    assert c.name == "db"

    snap = {"db_time_ms": 12.5, "db_rows": 3, "db_errors": 0}
    metrics = c.sample_metrics(snap)
    assert metrics
    assert metrics[0][0] == "db.query_time_ms_total"


def test_db_collector_diff() -> None:
    c = DBCollector()
    start = {"db_time_ms": 10.0, "db_rows": 2, "db_errors": 0}
    end = {"db_time_ms": 21.5, "db_rows": 8, "db_errors": 1}
    d = c.diff(start, end)
    assert d["db_time_ms"] == 11
    assert d["db_rows"] == 6
    assert d["db_errors"] == 1
