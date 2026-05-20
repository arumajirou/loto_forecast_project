from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..config.settings import settings

ClassInfo = type[Any] | tuple[type[Any], ...]


@dataclass
class SymbolParam:
    ordinal: int
    param_name: str
    param_kind: str | None
    annotation: str | None
    has_default: bool
    default_repr: str | None

    @property
    def is_required(self) -> bool:
        return not self.has_default


@dataclass
class SymbolRecord:
    library_name: str
    module_name: str
    symbol_type: str
    symbol_name: str
    full_path: str
    parent_symbol: str | None
    role: str | None
    return_type: str | None
    event_like: bool | None
    has_code: bool
    docstring: str | None
    raw: dict[str, Any]
    params: list[SymbolParam]


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null", ""}:
        return None
    return value


def _normalize_str(value: Any) -> str | None:
    v = _clean_scalar(value)
    if v is None:
        return None
    return str(v)


def _derive_parent_symbol(symbol_type: str, full_path: str, symbol_name: str) -> str | None:
    symbol_type = (symbol_type or "").lower()
    if symbol_type in {"method", "property", "prop", "attribute"}:
        head = (
            full_path.rsplit(f".{symbol_name}", 1)[0] if symbol_name and full_path.endswith(symbol_name) else full_path
        )
        return head
    return None


def _normalize_param(raw_param: dict[str, Any], ordinal: int) -> SymbolParam:
    name = _normalize_str(raw_param.get("name")) or f"param_{ordinal}"
    has_default = bool(raw_param.get("has_default", False))
    return SymbolParam(
        ordinal=ordinal,
        param_name=name,
        param_kind=_normalize_str(raw_param.get("kind")),
        annotation=_normalize_str(raw_param.get("annotation")),
        has_default=has_default,
        default_repr=_normalize_str(raw_param.get("default_repr")),
    )


def load_codegen_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"codegen yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid yaml format: root must be mapping. path={path}")
    payload.setdefault("rows", [])
    return payload


def parse_codegen_rows(payload: dict[str, Any], default_library: str | None = None) -> list[SymbolRecord]:
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("invalid yaml format: rows must be list")

    symbols: list[SymbolRecord] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue

        symbol_name = _normalize_str(raw.get("name")) or "unknown"
        full_path = _normalize_str(raw.get("path")) or symbol_name
        module_name = _normalize_str(raw.get("module"))
        if not module_name and "." in full_path:
            module_name = full_path.rsplit(".", 1)[0]
        module_name = module_name or "unknown"

        library_name = _normalize_str(raw.get("library")) or default_library or "unknown"
        symbol_type = (_normalize_str(raw.get("type")) or "other").lower()
        raw_params = raw.get("params") or []
        params: list[SymbolParam] = []
        if isinstance(raw_params, list):
            for idx, p in enumerate(raw_params, start=1):
                if isinstance(p, dict):
                    params.append(_normalize_param(p, ordinal=idx))

        rec = SymbolRecord(
            library_name=library_name,
            module_name=module_name,
            symbol_type=symbol_type,
            symbol_name=symbol_name,
            full_path=full_path,
            parent_symbol=_derive_parent_symbol(symbol_type, full_path, symbol_name),
            role=_normalize_str(raw.get("role")),
            return_type=_normalize_str(raw.get("return_type")),
            event_like=bool(raw.get("event_like")) if raw.get("event_like") is not None else None,
            has_code=bool(_normalize_str(raw.get("code"))),
            docstring=_normalize_str(raw.get("docstring")),
            raw=raw,
            params=params,
        )
        symbols.append(rec)
    return symbols


def _count_modules(symbols: Iterable[SymbolRecord]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for s in symbols:
        key = (s.library_name, s.module_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def upsert_codegen_catalog(
    engine: Engine,
    yaml_path: str | Path,
    library_name: str | None = None,
    schema: str | None = None,
    replace_library: bool = True,
) -> dict[str, Any]:
    schema = schema or settings.catalog_schema
    payload = load_codegen_yaml(yaml_path)
    symbols = parse_codegen_rows(payload, default_library=library_name)
    if not symbols:
        raise ValueError(f"no rows found in yaml: {yaml_path}")

    actual_library = library_name or symbols[0].library_name
    module_counts = _count_modules(symbols)

    q_upsert_lib = text(f"""
    INSERT INTO {schema}.library_catalog (library_name, source_path, bundle_kind, row_count, metadata)
    VALUES (:library_name, :source_path, :bundle_kind, :row_count, CAST(:metadata AS jsonb))
    ON CONFLICT (library_name) DO UPDATE
      SET source_path = EXCLUDED.source_path,
          bundle_kind = EXCLUDED.bundle_kind,
          row_count = EXCLUDED.row_count,
          metadata = EXCLUDED.metadata,
          imported_at = now();
    """)

    q_delete_symbols = text(f"DELETE FROM {schema}.symbol_catalog WHERE library_name = :library_name")
    q_delete_modules = text(f"DELETE FROM {schema}.module_catalog WHERE library_name = :library_name")

    q_insert_module = text(f"""
    INSERT INTO {schema}.module_catalog (library_name, module_name, top_group, symbol_count, metadata)
    VALUES (:library_name, :module_name, :top_group, :symbol_count, CAST(:metadata AS jsonb))
    ON CONFLICT (library_name, module_name) DO UPDATE
      SET top_group = EXCLUDED.top_group,
          symbol_count = EXCLUDED.symbol_count,
          metadata = EXCLUDED.metadata,
          updated_at = now();
    """)

    q_insert_symbol = text(f"""
    INSERT INTO {schema}.symbol_catalog (
      library_name, module_name, symbol_type, symbol_name, full_path, parent_symbol,
      role, return_type, event_like, has_code, docstring, raw
    ) VALUES (
      :library_name, :module_name, :symbol_type, :symbol_name, :full_path, :parent_symbol,
      :role, :return_type, :event_like, :has_code, :docstring, CAST(:raw AS jsonb)
    )
    RETURNING symbol_id;
    """)

    q_insert_param = text(f"""
    INSERT INTO {schema}.symbol_param_catalog (
      symbol_id, ordinal, param_name, param_kind, annotation, has_default, default_repr, is_required
    ) VALUES (
      :symbol_id, :ordinal, :param_name, :param_kind, :annotation, :has_default, :default_repr, :is_required
    );
    """)

    with engine.begin() as conn:
        conn.execute(
            q_upsert_lib,
            {
                "library_name": actual_library,
                "source_path": str(yaml_path),
                "bundle_kind": _normalize_str(payload.get("bundle_kind")) or "code",
                "row_count": int(payload.get("count") or len(symbols)),
                "metadata": json.dumps(
                    {
                        "title": _normalize_str(payload.get("title")),
                        "imported_symbol_count": len(symbols),
                    },
                    ensure_ascii=False,
                ),
            },
        )
        if replace_library:
            conn.execute(q_delete_symbols, {"library_name": actual_library})
            conn.execute(q_delete_modules, {"library_name": actual_library})

        for (lib, module_name), cnt in module_counts.items():
            if lib != actual_library:
                continue
            top_group = module_name.split(".")[0] if module_name else None
            conn.execute(
                q_insert_module,
                {
                    "library_name": actual_library,
                    "module_name": module_name,
                    "top_group": top_group,
                    "symbol_count": cnt,
                    "metadata": json.dumps({}, ensure_ascii=False),
                },
            )

        for s in symbols:
            if s.library_name != actual_library:
                continue
            symbol_id = conn.execute(
                q_insert_symbol,
                {
                    "library_name": actual_library,
                    "module_name": s.module_name,
                    "symbol_type": s.symbol_type,
                    "symbol_name": s.symbol_name,
                    "full_path": s.full_path,
                    "parent_symbol": s.parent_symbol,
                    "role": s.role,
                    "return_type": s.return_type,
                    "event_like": s.event_like,
                    "has_code": s.has_code,
                    "docstring": s.docstring,
                    "raw": json.dumps(s.raw, ensure_ascii=False),
                },
            ).scalar_one()

            for p in s.params:
                conn.execute(
                    q_insert_param,
                    {
                        "symbol_id": int(symbol_id),
                        "ordinal": p.ordinal,
                        "param_name": p.param_name,
                        "param_kind": p.param_kind,
                        "annotation": p.annotation,
                        "has_default": p.has_default,
                        "default_repr": p.default_repr,
                        "is_required": p.is_required,
                    },
                )

    return {
        "library_name": actual_library,
        "symbol_count": len([s for s in symbols if s.library_name == actual_library]),
        "module_count": len([m for (lib, m) in module_counts if lib == actual_library]),
        "yaml_path": str(yaml_path),
    }


def list_library_symbols(
    engine: Engine, library_name: str, limit: int = 100, schema: str | None = None
) -> list[dict[str, Any]]:
    schema = schema or settings.catalog_schema
    q = text(f"""
    SELECT symbol_id, module_name, symbol_type, symbol_name, full_path, return_type
    FROM {schema}.symbol_catalog
    WHERE library_name = :library_name
    ORDER BY module_name, symbol_name
    LIMIT :limit;
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"library_name": library_name, "limit": int(limit)}).mappings().all()
    return [dict(r) for r in rows]


def _annotation_ok(annotation: str | None, value: Any) -> tuple[bool, str | None]:
    if annotation is None:
        return True, None
    ann = annotation.lower()

    primitives: dict[str, ClassInfo] = {
        "int": int,
        "float": (float, int),
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
    }
    for token, typ in primitives.items():
        if token in ann:
            ok = isinstance(value, typ)
            return ok, None if ok else f"expected {token}, got {type(value).__name__}"

    # Complex types are accepted because many annotations come from torch/pandas/custom classes.
    return True, None


def validate_call_arguments(
    engine: Engine,
    library_name: str,
    full_path: str,
    arguments: dict[str, Any],
    schema: str | None = None,
) -> dict[str, Any]:
    schema = schema or settings.catalog_schema
    q_symbol = text(f"""
    SELECT symbol_id, symbol_name, symbol_type
    FROM {schema}.symbol_catalog
    WHERE library_name = :library_name AND full_path = :full_path
    LIMIT 1;
    """)
    q_params = text(f"""
    SELECT param_name, annotation, has_default, is_required
    FROM {schema}.symbol_param_catalog
    WHERE symbol_id = :symbol_id
    ORDER BY ordinal;
    """)

    with engine.connect() as conn:
        sym = conn.execute(q_symbol, {"library_name": library_name, "full_path": full_path}).mappings().first()
        if sym is None:
            return {
                "ok": False,
                "errors": [f"symbol not found in catalog: {library_name}:{full_path}"],
                "warnings": [],
                "required": [],
                "optional": [],
            }

        rows = conn.execute(q_params, {"symbol_id": int(sym["symbol_id"])}).mappings().all()

    required: list[str] = []
    optional: list[str] = []
    known: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r["param_name"])
        info = dict(r)
        known[name] = info
        if bool(r.get("is_required", False)):
            required.append(name)
        else:
            optional.append(name)

    errors: list[str] = []
    warnings: list[str] = []

    for name in required:
        if name not in arguments:
            errors.append(f"missing required parameter: {name}")

    for name in arguments:
        if name not in known:
            errors.append(f"unknown parameter: {name}")

    for name, value in arguments.items():
        if name not in known:
            continue
        ok, why = _annotation_ok(known[name].get("annotation"), value)
        if not ok and why:
            warnings.append(f"{name}: {why}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "required": required,
        "optional": optional,
        "symbol": {"library": library_name, "path": full_path, "name": sym["symbol_name"], "type": sym["symbol_type"]},
    }
