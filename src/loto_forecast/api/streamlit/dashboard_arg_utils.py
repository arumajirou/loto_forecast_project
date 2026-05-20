from __future__ import annotations

import ast
import json
import re
import shlex
from typing import Any

import pandas as pd
import yaml


def _truncate_arg_preview(value: Any, max_len: int = 220) -> str:
    text_v = str(value)
    if len(text_v) > int(max_len):
        return text_v[: max_len - 3] + "..."
    return text_v


def _parse_jsonish_text(raw_value: str) -> Any | None:
    text_raw = str(raw_value or "").strip()
    if not text_raw:
        return None

    candidates: list[str] = [text_raw]
    if len(text_raw) >= 2 and text_raw[0] == text_raw[-1] and text_raw[0] in {"'", '"'}:
        candidates.append(text_raw[1:-1].strip())
    if '""' in text_raw:
        candidates.append(text_raw.replace('""', '"'))
    if '\\"' in text_raw:
        candidates.append(text_raw.replace('\\"', '"'))

    for cand in candidates:
        if not cand:
            continue
        try:
            loaded = json.loads(cand)
            if isinstance(loaded, str):
                inner = loaded.strip()
                if inner and inner[0] in "{[":
                    try:
                        inner_loaded = json.loads(inner)
                        if isinstance(inner_loaded, (dict, list)):
                            return inner_loaded
                    except Exception:
                        pass
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            pass
        try:
            lit = ast.literal_eval(cand)
            if isinstance(lit, str):
                inner = lit.strip()
                if inner and inner[0] in "{[":
                    try:
                        inner_loaded = json.loads(inner)
                        if isinstance(inner_loaded, (dict, list)):
                            return inner_loaded
                    except Exception:
                        pass
            if isinstance(lit, (dict, list)):
                return lit
        except Exception:
            pass
        try:
            loaded_yaml = yaml.safe_load(cand)
            if isinstance(loaded_yaml, (dict, list)):
                return loaded_yaml
        except Exception:
            pass
    return None


def flatten_json_rows_for_arg(
    *,
    argument: str,
    raw_value: str,
    max_rows: int = 240,
    max_depth: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    parsed = _parse_jsonish_text(raw_value)
    if parsed is None:
        return out

    def _walk(obj: Any, prefix: str, depth: int) -> None:
        if len(out) >= int(max_rows):
            return
        if depth > int(max_depth):
            out.append(
                {
                    "kind": "option(expanded)",
                    "argument": prefix + ".__depth_limit__",
                    "value": "<depth limit>",
                }
            )
            return
        if isinstance(obj, dict):
            if not obj:
                out.append({"kind": "option(expanded)", "argument": prefix, "value": "{}"})
                return
            for k in sorted(obj.keys(), key=lambda x: str(x)):
                kk = str(k)
                _walk(obj.get(k), f"{prefix}.{kk}" if prefix else kk, depth + 1)
            return
        if isinstance(obj, list):
            if not obj:
                out.append({"kind": "option(expanded)", "argument": prefix, "value": "[]"})
                return
            for idx, item in enumerate(obj):
                _walk(item, f"{prefix}[{idx}]", depth + 1)
                if len(out) >= int(max_rows):
                    return
            return
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            if isinstance(obj, str):
                val = obj
            else:
                try:
                    val = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    val = str(obj)
            out.append(
                {
                    "kind": "option(expanded)",
                    "argument": prefix,
                    "value": _truncate_arg_preview(val),
                }
            )
            return
        out.append(
            {
                "kind": "option(expanded)",
                "argument": prefix,
                "value": _truncate_arg_preview(repr(obj)),
            }
        )

    _walk(parsed, str(argument), 0)
    if len(out) >= int(max_rows):
        out.append(
            {
                "kind": "note",
                "argument": str(argument),
                "value": f"expanded rows capped at {int(max_rows)}",
            }
        )
    return out


def command_argument_table(command: str) -> pd.DataFrame:
    cmd = str(command or "").strip()
    if not cmd:
        return pd.DataFrame(columns=["kind", "argument", "value"])
    try:
        tokens = shlex.split(cmd)
    except Exception:
        return pd.DataFrame([{"kind": "raw", "argument": "command", "value": cmd}])

    rows: list[dict[str, Any]] = []
    positional_idx = 0
    i = 0
    while i < len(tokens):
        tok = str(tokens[i])
        if tok.startswith("--"):
            if tok.startswith("--no-"):
                rows.append({"kind": "flag", "argument": tok, "value": "false"})
                i += 1
                continue
            if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("--"):
                val = str(tokens[i + 1])
                rows.append({"kind": "option", "argument": tok, "value": _truncate_arg_preview(val)})
                looks_json_opt = tok.endswith("-json") or tok in {
                    "--params-json",
                    "--model-params-json",
                    "--param-space-json",
                    "--config-json",
                    "--arguments-json",
                    "--save-kwargs-json",
                    "--load-kwargs-json",
                    "--predict-insample-kwargs-json",
                }
                looks_json_value = bool(re.match(r"^\s*[\{\[]", val))
                if looks_json_opt or looks_json_value:
                    rows.extend(flatten_json_rows_for_arg(argument=str(tok), raw_value=str(val)))
                i += 2
            else:
                rows.append({"kind": "flag", "argument": tok, "value": "true"})
                i += 1
            continue
        positional_idx += 1
        rows.append({"kind": "positional", "argument": f"arg{positional_idx}", "value": tok})
        i += 1
    return pd.DataFrame(rows)
