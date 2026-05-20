#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT}
bash scripts/context_preflight.sh

cat <<'PROMPT'
AGENTS.md
docs/context/03_context_packet.md
docs/context/01_execution_contract.md
docs/context/02_decision_policy.md
docs/context/04_tooling_scope.md
task_prompt.md
を前提に、追加質問なしで単独判断して完遂してください。

最優先:
1. coverage matrix を埋める
2. dynamic trace を更新する
3. 全画面・全主要操作・DB保存・生成物生成・異常系・境界値・組み合わせを段階的に検証する
4. 不具合があれば修正→再実行→再検証→資料化まで行う
5. Windows Chrome と Linux fallback の結果は混同しない
6. DB書き込み系は必ず前後 SELECT を取る
7. 生成物系は before/after の path/size/mtime/exists を残す
PROMPT

echo
echo "↑ このプロンプトを Codex に貼り付けて実行してください"
