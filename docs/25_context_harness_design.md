# コンテキスト・ハーネスエンジニアリング設計書

> 参照: garrytan/gstack + karpathy/autoresearch の設計思想を本プロジェクトに適用した設計書

---

## 1. 設計思想の引用元と適用方針

### garrytan/gstack から学んだこと

| gstack パターン | このプロジェクトへの適用 |
|---------------|----------------------|
| **SKILL.md as system prompt** | `.claude/skills/*.md` — 各操作の完全な手順書 |
| **ETHOS.md 哲学文書** | `ETHOS.md` — プロジェクト固有の判断基準 |
| **CLAUDE.md にテストコマンドを記載** | `CLAUDE.md` のクイックスタートセクション |
| **/ship 品質パイプライン** | `.claude/skills/ship.md` — bisectable commit 順序 |
| **/investigate Iron Law** | `.claude/skills/investigate.md` — 根本原因なしに修正しない |
| **/autoplan 多角レビュー** | `.claude/skills/autoplan.md` — Phase5実装前に必須 |
| **/cso セキュリティ監査** | `.claude/skills/cso.md` — git考古学 + bandit + SQL injection |
| **/retro 振り返り** | `.claude/skills/retro.md` — 実験メトリクス + コードホットスポット |
| **Adversarial scaling** | diff サイズに応じてレビュー深度を自動変更 |
| **Bisectable commits** | DB → ORM → pipeline → CLI → tests の順序 |

### karpathy/autoresearch から学んだこと

| autoresearch パターン | このプロジェクトへの適用 |
|--------------------|----------------------|
| **program.md 自律ループ指示書** | `program.md` — AutoModel実験エージェントの指示書 |
| **results.tsv flat-file実験ログ** | `results.tsv` — DB死亡時のサイドカー |
| **固定時間予算 (5分/実験)** | 30分/実験のタイムアウト設計 |
| **git-as-experiment-log** | `git commit -m "exp: ..."` 形式 |
| **Simplicity Criterion** | `ETHOS.md` §4 — 同精度なら削除を選ぶ |
| **1ファイル変更制約** | 1実験 = 1変更のみ (因果関係の明確化) |
| **val_bpb 正規化指標** | MASE を主要指標として採用 (系列間比較可能) |

---

## 2. コンテキストエンジニアリング設計

### 2.1 コンテキスト注入の優先順位

```
優先度1 (常時): MEMORY.md → CLAUDE.md → AGENTS.md
優先度2 (タスク別): 該当 skill file → 関連 docs/claude/*.md
優先度3 (必要時): 実際のソースファイル (Grep → 部分Read)
```

### 2.2 コンテキスト汚染を防ぐ設計

| 汚染源 | 対策 |
|--------|------|
| cli.py (26k tokens) | Grep で cmd_* を特定 → 部分Read のルール |
| lib_docs/*.yaml (巨大) | catalog-validate コマンド経由のみ |
| artifacts/lightning_logs/ | .gitignore + Read 禁止 |
| 過去の実験メタ | results.tsv の tail で最近のみ確認 |
| 全DBテーブルスキャン | docs/claude/db_schemas.md を先に読む |

### 2.3 コンテキスト使用量の目安

| 操作 | 想定トークン消費 | 最適化方法 |
|------|--------------|-----------|
| 通常の質問回答 | < 5k | MEMORY.md + CLAUDE.md のみ |
| コード修正 | < 20k | skill + 対象ファイルのみ |
| デバッグ | < 30k | skill + ログ + 対象モジュール |
| 新機能実装 | < 50k | autoplan + 関連モジュール |
| cli.py 全読み | ~26k (禁止) | Grep で特定後、部分Read |

---

## 3. スキル設計詳細

### 3.1 スキルマップ (全14スキル)

```
【即実行型】
  /db-check   → read-only SQL確認 (安全)
  /lint       → ruff + mypy
  /test       → pytest
  /retro      → 振り返り

【確認付き実行型】
  /train      → 学習 (長時間)
  /grid       → グリッドサーチ (長時間 + 確認必須)
  /evaluate   → 評価・診断
  /catalog    → カタログ操作

【破壊的 (承認必須)】
  /db-init    → DBスキーマ再作成

【品質保証】
  /python-review  → ruff + mypy + bandit
  /ship           → コミット前パイプライン
  /cso            → セキュリティ監査

【設計支援】
  /autoplan   → 実装前多角レビュー
  /investigate → 根本原因デバッグ
```

### 3.2 スキル間の依存関係

```
/investigate → /lint (エラーの静的確認)
/ship → /python-review (品質確認後にコミット)
/ship → /test (テスト確認後にコミット)
/autoplan → 実装 → /ship (計画→実装→品質確認)
/retro → /autoplan (振り返り→次の計画)
program.md → /train (実験ループ)
program.md → /retro (実験振り返り)
```

---

## 4. Hooks 設計詳細

### 4.1 現在設定済みの Hooks

| Hook | タイプ | 条件 | 動作 | 目的 |
|------|-------|------|------|------|
| PreToolUse(Bash) | 警告 | `cli db-init` | stderr に警告 | 破壊的操作防止 |
| PreToolUse(Bash) | ブロック | `--password xxx` | exit 1 | 認証情報漏洩防止 |
| PreToolUse(Bash) | 情報 | `cli grid-run` | stderr に警告 | 長時間実行の認識 |
| PostToolUse(Bash) | 情報 | `cli train` | run_id 確認促進 | 実験追跡 |

### 4.2 将来追加すべき Hooks (Phase C)

| Hook | タイミング | 動作 | 実装難易度 |
|------|----------|------|---------|
| PostEdit(*.py) | py ファイル編集後 | `ruff check` 自動実行 | 中 |
| PreCommit | コミット前 | `/ship` パイプライン実行 | 高 (pre-commit で代替) |
| PostTrain | 学習完了後 | results.tsv 自動更新 | 中 |
| PreWrite(sql/) | sql/ 書き込み前 | マイグレーション提案 | 低 |

---

## 5. 実験管理アーキテクチャ

### 5.1 二層実験ログ (autoresearch パターン)

```
Layer 1 (Primary): PostgreSQL model.nf_automodel
  → 完全なパラメータ・メトリクス・エラー情報
  → run_id でトレース可能
  → Streamlit Dashboard から参照

Layer 2 (Sidecar): results.tsv
  → DB接続死亡時でも参照可能
  → git log で変遷が見える
  → grep で即座に検索可能
  → 軽量 (1行 = 1実験)
```

### 5.2 実験ブランチ戦略

```
main
├── autoexp/20260326   ← 実験ブランチ (program.md の自律ループ)
│   ├── exp: AutoNHITS num_samples=10 → MAE=0.123 (keep)
│   ├── exp: AutoNBEATS num_samples=10 → MAE=0.145 (discard)
│   └── exp: AutoNHITS num_samples=50 → MAE=0.118 (keep)
└── feat/phase5-async  ← 機能ブランチ (autoplan で設計確認後)
```

### 5.3 実験メトリクスの選択理由

| 指標 | 選択理由 | 対応 autoresearch |
|------|---------|-----------------|
| MASE | スケール非依存、baseline比較可能 | val_bpb (語彙サイズ非依存) |
| MAE | 解釈しやすい絶対値 | raw loss |
| sMAPE | パーセンテージで直感的 | - |

---

## 6. ファイル構成 (完全版)

```
loto_forecast_project/
├── AGENTS.md          ★ 9つの必須ルール
├── CLAUDE.md          ★ 背景知識 + クイックスタート (gstack パターン)
├── ETHOS.md           ★ エンジニアリング哲学 (gstack ETHOS.md 適用)
├── program.md         ★ 自律実験ループ指示書 (autoresearch pattern)
├── results.tsv        ★ 実験ログサイドカー (autoresearch pattern)
│
├── .claude/
│   ├── settings.local.json  ← MCP + permissions + hooks
│   └── skills/
│       ├── train.md         /train
│       ├── grid.md          /grid
│       ├── evaluate.md      /evaluate
│       ├── db-check.md      /db-check
│       ├── db-init.md       /db-init
│       ├── catalog.md       /catalog
│       ├── lint.md          /lint
│       ├── python-review.md /python-review
│       ├── test.md          /test
│       ├── ship.md          /ship         ← gstack /ship
│       ├── investigate.md   /investigate  ← gstack /investigate
│       ├── autoplan.md      /autoplan     ← gstack /autoplan
│       ├── retro.md         /retro        ← gstack /retro
│       └── cso.md           /cso          ← gstack /cso
│
├── docs/
│   ├── 23_harness_engineering_design.md  ← 5フェーズ設計
│   ├── 24_toolchain_setup.md             ← MCP/plugin/tool設定
│   ├── 25_context_harness_design.md      ← 本ドキュメント
│   └── claude/
│       ├── arch.md           アーキテクチャ/フロー
│       ├── db_schemas.md     DBスキーマ + 権限
│       ├── exog_convention.md 外生変数命名規約
│       └── commands_ref.md   CLIリファレンス
│
├── evals/
│   ├── suites/
│   │   ├── should_trigger.csv     10ケース
│   │   └── should_not_trigger.csv 10ケース
│   └── run_checks.py              自動判定スクリプト
│
├── pyproject.toml     ← ruff/mypy/pytest/bandit 設定
└── .pre-commit-config.yaml ← 11種類のコミット前チェック
```

---

## 7. 次のステップ (Phase B → C)

### 即実行 (Phase B 完了後)

```bash
# 現在のカバレッジ計測
pytest tests/unit/ --cov=src/loto_forecast --cov-report=term-missing -q

# ハーネス採点
# Claude Code 内で: /harness-audit

# セキュリティ監査
# Claude Code 内で: /cso
```

### Phase C: CI 回帰検知

```yaml
# .github/workflows/quality.yml (将来)
on: [push, pull_request]
jobs:
  quality:
    steps:
      - run: ruff check src/ tests/
      - run: mypy src/loto_forecast/ --ignore-missing-imports
      - run: bandit -r src/ -c pyproject.toml
      - run: pytest tests/unit/ -q
      - run: python evals/run_checks.py --suite should_trigger should_not_trigger
```

### Phase D: 高度化

```
- PostgreSQL MCP を project-level から user-level に昇格 (複数プロジェクト共有)
- program.md の自律ループを GitHub Actions に組み込み (nightly実験)
- results.tsv → 可視化スクリプト (analysis.ipynb 追加、autoresearch パターン)
- /ship に adversarial review の多モデル版 (Claude + 別モデル) を追加
```
