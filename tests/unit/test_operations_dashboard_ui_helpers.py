from __future__ import annotations

from pathlib import Path

from loto_forecast.api.streamlit import operations_dashboard as dashboard


def test_tree_lines_skips_hidden_and_cache_paths(tmp_path: Path) -> None:
    (tmp_path / "visible").mkdir()
    (tmp_path / "visible" / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "evidence.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: main", encoding="utf-8")
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"x")

    rendered = dashboard._tree_lines(tmp_path, max_depth=3, max_entries=50)

    assert "visible/" in rendered
    assert "app.py" in rendered
    assert "artifacts" not in rendered
    assert ".git" not in rendered
    assert ".agents" not in rendered
    assert "__pycache__" not in rendered


def test_document_metadata_html_includes_description_and_lang() -> None:
    html = dashboard._document_metadata_html()

    assert "meta" in html
    assert "description" in html
    assert "\"ja\"" in html


def test_safe_db_cli_flags_uses_env_var_instead_of_password() -> None:
    flags = dashboard._safe_db_cli_flags(
        host="127.0.0.1",
        port=5432,
        user="loto",
        database="loto",
    )

    assert "--host '127.0.0.1'" in flags or "--host 127.0.0.1" in flags
    assert "--database loto" in flags or "--database 'loto'" in flags
    assert "--password" not in flags
    assert "DB_PASSWORD=${DB_PASSWORD}" in flags
