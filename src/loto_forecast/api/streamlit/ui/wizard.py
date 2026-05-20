from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WizardIssue:
    reason: str
    impact: str
    fix: str


@dataclass(slots=True)
class WizardStep:
    title: str
    status: str
    summary: str


@dataclass(slots=True)
class WizardState:
    mode: str
    action: str
    can_run: bool
    required_missing: list[str] = field(default_factory=list)
    issues: list[WizardIssue] = field(default_factory=list)
    steps: list[WizardStep] = field(default_factory=list)
    command_preview: str = ""
    result_summary: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)


def build_nf_wizard_state(
    *,
    mode: str,
    action: str,
    model: str,
    dataset_input_method: str,
    dataset_table: str,
    dataset_path: str,
    dataset_sql: str,
    unique_ids: list[str],
    ts_types: list[str],
    horizon: int,
    backend: str,
    loss: str,
    search_alg: str,
    combo_total: int | None = None,
    combo_skip_reasons: dict[str, int] | None = None,
    command_preview: str = "",
) -> WizardState:
    missing: list[str] = []
    issues: list[WizardIssue] = []
    if not str(model).strip():
        missing.append("モデル")
    if dataset_input_method == "db_table" and not str(dataset_table).strip():
        missing.append("データテーブル")
    if dataset_input_method == "db_sql" and not str(dataset_sql).strip():
        missing.append("データSQL")
    if dataset_input_method in {"csv", "parquet", "json"} and not str(dataset_path).strip():
        missing.append("データパス")
    if not list(ts_types):
        missing.append("ts_type")
    if not list(unique_ids):
        issues.append(
            WizardIssue(
                reason="unique_id が未選択です。",
                impact="系列単位での学習や h 自動算出が期待どおり動かない可能性があります。",
                fix="最低1つの unique_id を選ぶか、学習単位を loto+ts_type に変更してください。",
            )
        )
    if combo_total is not None and combo_total <= 0:
        top_reason = ""
        if combo_skip_reasons:
            top_reason = max(combo_skip_reasons.items(), key=lambda item: item[1])[0]
        issues.append(
            WizardIssue(
                reason="有効な総当たり組合せ数が 0 です。",
                impact="meta 反映も実行も開始できません。",
                fix=top_reason or "backend / search_alg / unique_id / データ入力条件の不整合を修正してください。",
            )
        )

    steps = [
        WizardStep("Step 1: 何をしたいか選ぶ", "done" if action else "pending", f"選択中: {action or '-'}"),
        WizardStep(
            "Step 2: 必須入力だけ埋める",
            "done" if not missing else "warning",
            "必須不足: " + (", ".join(missing) if missing else "なし"),
        ),
        WizardStep(
            "Step 3: 自動補完の確認",
            "done" if horizon > 0 else "warning",
            f"h={int(horizon)} / backend={backend} / loss={loss} / search_alg={search_alg}",
        ),
        WizardStep(
            "Step 4: 実行プレビュー",
            "done" if command_preview else "pending",
            command_preview or "コマンド未生成",
        ),
    ]
    result_summary = [
        f"目的: {action}",
        f"データ入力: {dataset_input_method}",
        f"モデル: {model}",
    ]
    next_actions = (
        ["不足項目を埋めてから再度プレビューを確認する"]
        if missing
        else ["実行前チェックでエラーなしを確認する", "実行ボタンを押す", "結果サマリを確認する"]
    )
    return WizardState(
        mode=mode,
        action=action,
        can_run=not missing and not any(issue.reason.startswith("有効な総当たり") for issue in issues),
        required_missing=missing,
        issues=issues,
        steps=steps,
        command_preview=command_preview,
        result_summary=result_summary,
        next_actions=next_actions,
    )
