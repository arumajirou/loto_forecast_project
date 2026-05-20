from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loto_forecast.api.streamlit import operations_dashboard_helpers as dashboard_helpers


@dataclass(slots=True)
class ComboContext:
    dataset_table: str = ""
    dataset_path: str = ""
    dataset_sql: str = ""
    default_values: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComboIssue:
    reason_code: str
    reason_ja: str


@dataclass(slots=True)
class ExcludedCombination:
    values: dict[str, Any]
    reason_code: str
    reason_ja: str


@dataclass(slots=True)
class ComboReasonSummary:
    reason_code: str
    reason_ja: str
    count: int


@dataclass(slots=True)
class ComboEvaluation:
    theoretical_count: int
    valid_combinations: list[dict[str, Any]]
    excluded_combinations: list[ExcludedCombination]
    reason_summary: dict[str, int]
    reason_rows: list[ComboReasonSummary]
    fix_suggestions: list[str]


def evaluate_train_combinations(
    axes: Mapping[str, Sequence[Any]],
    *,
    context: ComboContext,
    row_normalizer: Callable[[dict[str, Any], ComboContext], dict[str, Any]] | None = None,
    runtime_validator: Callable[[str, int | None], str | None] | None = None,
) -> ComboEvaluation:
    axis_map = {str(key): list(values) for key, values in axes.items() if list(values)}
    if not axis_map:
        return ComboEvaluation(
            theoretical_count=0,
            valid_combinations=[],
            excluded_combinations=[],
            reason_summary={},
            reason_rows=[],
            fix_suggestions=[],
        )

    keys = list(axis_map.keys())
    theoretical_count = 1
    for values in axis_map.values():
        theoretical_count *= max(1, len(values))

    valid_rows: list[dict[str, Any]] = []
    excluded_rows: list[ExcludedCombination] = []
    seen: set[str] = set()
    reason_counts: Counter[str] = Counter()

    for combo_values in itertools.product(*[axis_map[key] for key in keys]):
        row = {key: value for key, value in zip(keys, combo_values, strict=False)}
        for key, value in context.default_values.items():
            row.setdefault(key, value)
        if row_normalizer is not None:
            row = dict(row_normalizer(dict(row), context))
        issue = _validate_combo_row(row, context=context, runtime_validator=runtime_validator)
        if issue is not None:
            reason_counts[issue.reason_code] += 1
            excluded_rows.append(
                ExcludedCombination(values=dict(row), reason_code=issue.reason_code, reason_ja=issue.reason_ja)
            )
            continue
        dedup_key = _stable_json_dumps(row)
        if dedup_key in seen:
            reason_counts["duplicate_combination"] += 1
            excluded_rows.append(
                ExcludedCombination(
                    values=dict(row),
                    reason_code="duplicate_combination",
                    reason_ja="同一内容の設定組合せが重複したため除外しました。",
                )
            )
            continue
        seen.add(dedup_key)
        valid_rows.append(dict(row))

    return ComboEvaluation(
        theoretical_count=int(theoretical_count),
        valid_combinations=valid_rows,
        excluded_combinations=excluded_rows,
        reason_summary=dict(reason_counts),
        reason_rows=_build_reason_rows(excluded_rows, dict(reason_counts)),
        fix_suggestions=_build_fix_suggestions(dict(reason_counts)),
    )


def _validate_combo_row(
    row: dict[str, Any],
    *,
    context: ComboContext,
    runtime_validator: Callable[[str, int | None], str | None] | None,
) -> ComboIssue | None:
    model = str(row.get("model") or "").strip()
    backend = row.get("backend")
    valid_loss = row.get("valid_loss")
    search_alg = row.get("search_alg")
    group_mode = str(row.get("group_mode") or "").strip()
    unique_id = row.get("unique_id")
    dataset_input_method = str(row.get("dataset_input_method") or "db_table").strip()
    dataframe_backend = str(row.get("dataframe_backend") or "pandas").strip()
    dataset_table = str(row.get("dataset_table") or context.dataset_table).strip()
    dataset_path = str(row.get("dataset_path") or context.dataset_path).strip()
    dataset_sql = str(row.get("dataset_sql") or context.dataset_sql).strip()
    horizon = dashboard_helpers.parse_horizon_axis_value(row.get("horizon") or row.get("h"))
    fit_val_size = row.get("fit_val_size")

    unique_id_error = dashboard_helpers.group_mode_unique_id_validation_error(group_mode, unique_id)
    if unique_id_error:
        return ComboIssue(
            reason_code="missing_unique_id",
            reason_ja="学習単位が unique_id 必須のため、unique_id 候補が空の組合せを除外しました。",
        )

    combo_error = dashboard_helpers.validate_train_combo_choice(model, backend, valid_loss, search_alg)
    if combo_error:
        if "invalid search_alg" in combo_error:
            return ComboIssue(
                reason_code="backend_search_mismatch",
                reason_ja="backend と search_alg の整合性が取れないため除外しました。",
            )
        if "requires backend=ray" in combo_error:
            return ComboIssue(
                reason_code="model_backend_requirement",
                reason_ja="選択モデルの必須 backend 条件を満たさないため除外しました。",
            )
        return ComboIssue(reason_code="invalid_train_choice", reason_ja=str(combo_error))

    if dataset_input_method == "db_table" and not dataset_table:
        return ComboIssue(
            reason_code="missing_dataset_table",
            reason_ja="dataset_input_method=db_table では dataset table が必須のため除外しました。",
        )
    if dataset_input_method == "db_sql" and not dataset_sql:
        return ComboIssue(
            reason_code="missing_dataset_sql",
            reason_ja="dataset_input_method=db_sql では dataset SQL が必須のため除外しました。",
        )
    if dataset_input_method in {"csv", "parquet", "json"} and not dataset_path:
        return ComboIssue(
            reason_code="missing_dataset_path",
            reason_ja=f"dataset_input_method={dataset_input_method} では dataset path が必須のため除外しました。",
        )

    if not dashboard_helpers.is_supported_backend_for_input_method(dataset_input_method, dataframe_backend):
        return ComboIssue(
            reason_code="unsupported_dataframe_backend",
            reason_ja=f"{dataset_input_method} では dataframe backend={dataframe_backend} を使えないため除外しました。",
        )

    try:
        fit_val_size_int = int(fit_val_size) if fit_val_size is not None else None
    except Exception:
        fit_val_size_int = None
    if horizon is not None and fit_val_size_int is not None and fit_val_size_int not in {0} and fit_val_size_int < int(horizon):
        return ComboIssue(
            reason_code="fit_val_size_too_small",
            reason_ja="fit.val_size が horizon 未満のため除外しました。",
        )

    if runtime_validator is not None:
        runtime_error = runtime_validator(model, horizon)
        if runtime_error:
            return ComboIssue(
                reason_code="runtime_prerequisite",
                reason_ja=str(runtime_error),
            )
    return None


def _build_fix_suggestions(reason_summary: dict[str, int]) -> list[str]:
    ordered = sorted(reason_summary.items(), key=lambda item: (-int(item[1]), item[0]))
    suggestions: list[str] = []
    mapping = {
        "missing_unique_id": "group_mode が unique_id 必須なら、unique_id 候補を1件以上選択してください。",
        "backend_search_mismatch": "backend に対応する search_alg の候補だけを選ぶと有効件数が増えます。",
        "model_backend_requirement": "モデル固有の backend 制約を満たす候補だけを残してください。",
        "missing_dataset_table": "db_table を使う場合は dataset table を指定してください。",
        "missing_dataset_sql": "db_sql を使う場合は dataset SQL を入力してください。",
        "missing_dataset_path": "csv/parquet/json を使う場合は dataset path を入力してください。",
        "unsupported_dataframe_backend": "dataset_input_method ごとの対応 dataframe backend に揃えてください。",
        "fit_val_size_too_small": "fit.val_size は 0 または horizon 以上の値にしてください。",
        "runtime_prerequisite": "モデルの実行前提を満たす候補に絞ってください。",
        "duplicate_combination": "同一内容の候補が重複しているため、候補配列を整理してください。",
    }
    for reason_code, _count in ordered:
        suggestion = mapping.get(str(reason_code))
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
    return suggestions[:5]


def _build_reason_rows(
    excluded_rows: Sequence[ExcludedCombination], reason_summary: Mapping[str, int]
) -> list[ComboReasonSummary]:
    reason_map: dict[str, str] = {}
    for row in excluded_rows:
        reason_map.setdefault(str(row.reason_code), str(row.reason_ja))
    ordered = sorted(reason_summary.items(), key=lambda item: (-int(item[1]), item[0]))
    return [
        ComboReasonSummary(
            reason_code=str(reason_code),
            reason_ja=reason_map.get(str(reason_code), str(reason_code)),
            count=int(count),
        )
        for reason_code, count in ordered
    ]


def _stable_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)
