# ツールチェーン セットアップ資料

> 作成日: 2026-03-26
> 対象: loto_forecast_project のハーネスエンジニアリング用ツール・MCP・Skill のインストールと設定

---

## 1. セットアップ済みコンポーネント一覧

### 1.1 MCP サーバー

| MCP名 | 実装 | 用途 | 設定場所 |
|-------|------|------|---------|
| `postgres-loto` | `@modelcontextprotocol/server-postgres` | loto DB への SQL アクセス | `.claude/settings.local.json` |
| `filesystem-artifacts` | `@modelcontextprotocol/server-filesystem` | artifacts/ logs/ の読み取り | `.claude/settings.local.json` |

**インストール先**: `/home/az/.local/share/mcp/node_modules/`

**接続確認方法**:
```bash
# Claude Code を再起動後
claude mcp list
# または
claude mcp get postgres-loto
```

**postgres-loto でできること**:
- `SELECT` クエリ (全スキーマ)
- `INSERT/UPDATE` (meta/model/log/catalog/exog/resources スキーマ)
- ⚠️ `dataset` スキーマへの書き込みは運用ルールで禁止

**filesystem-artifacts でできること**:
- `artifacts/` 配下のモデルファイル一覧・読み取り
- `logs/` 配下のログファイル一覧・読み取り
- ⚠️ 書き込みは不可 (read-only mount)

---

### 1.2 Claude Code プラグイン

| プラグイン | バージョン | スコープ | 状態 |
|----------|----------|---------|------|
| `everything-claude-code` | 1.8.0 | project | ✅ 有効 |
| `pyright-lsp` | 1.0.0 | user | ✅ 有効 |

**ECC で使えるコマンド (主要48個中)**:

| コマンド | 用途 |
|---------|------|
| `/code-review` | セキュリティ + 品質レビュー |
| `/python-review` | Python 特化レビュー (ruff/mypy/pylint/black) |
| `/quality-gate` | lint/型チェック一括実行 |
| `/tdd` | TDD ワークフロー (RED→GREEN→REFACTOR) |
| `/harness-audit` | エージェントハーネス採点 (0-70点) |
| `/harness-optimizer` | ハーネス改善提案 |
| `/plan` | 実装計画立案 |
| `/eval` | eval定義・チェック・レポート |
| `/checkpoint` | セッション状態保存 |
| `/learn` | セッションからパターン抽出 |
| `/skill-create` | git履歴からSkill自動生成 |
| `/prompt-optimize` | プロンプト最適化 |
| `/database-reviewer` | PostgreSQL スキーマ・クエリレビュー |

**ECC で使えるエージェント**:

| エージェント | 用途 |
|------------|------|
| `harness-optimizer` | ハーネス設定改善 |
| `database-reviewer` | DB設計・クエリ最適化 |
| `python-reviewer` | Python コード品質レビュー |
| `security-reviewer` | セキュリティ脆弱性検出 |
| `planner` | 実装計画立案 |
| `code-reviewer` | 汎用コードレビュー |
| `tdd-guide` | TDD ガイド |

---

### 1.3 Python ツール (conda env: ts)

| ツール | バージョン | 役割 | 設定 |
|-------|----------|------|------|
| `ruff` | 0.15.1 | Linter + Formatter | `pyproject.toml [tool.ruff]` |
| `mypy` | 1.19.1 | 静的型チェック | `pyproject.toml [tool.mypy]` |
| `bandit` | 1.9.3 | セキュリティスキャン | `pyproject.toml [tool.bandit]` |
| `pytest` | 9.0.2 | テストフレームワーク | `pyproject.toml [tool.pytest.ini_options]` |
| `pytest-cov` | 7.0.0 | カバレッジ計測 | 同上 |
| `pre-commit` | 4.5.1 | コミット前自動チェック | `.pre-commit-config.yaml` |

---

### 1.4 プロジェクト固有 Skills

| Skill | 呼び出し | 用途 |
|-------|---------|------|
| `train.md` | `/train` | AutoModel学習・再学習 |
| `grid.md` | `/grid` | グリッドサーチ create/run/status |
| `evaluate.md` | `/evaluate` | 評価・診断・説明可能性 |
| `db-check.md` | `/db-check` | DB読み取り専用確認 |
| `db-init.md` | `/db-init` | DB初期化 (確認付き) |
| `catalog.md` | `/catalog` | codegen YAMLカタログ操作 |
| `lint.md` | `/lint` | ruff + mypy 実行 |
| `python-review.md` | `/python-review` | ruff+mypy+bandit 統合レビュー |
| `test.md` | `/test` | pytest 実行 |

---

## 2. Hooks 設定

`.claude/settings.local.json` に設定済み:

| Hook | タイプ | トリガー | 動作 |
|------|-------|---------|------|
| `PreToolUse(Bash)` | 警告 | `cli db-init` を含む | ⚠️ 破壊的操作警告を stderr に出力 |
| `PreToolUse(Bash)` | ブロック | `--password xxx` を含む | ❌ exit 1 で実行阻止 |
| `PreToolUse(Bash)` | 情報 | `cli grid-run` を含む | ℹ️ 長時間実行の注意を stderr に出力 |
| `PostToolUse(Bash)` | 情報 | `cli train` 完了後 | ℹ️ run_id確認用 SELECTを提示 |

---

## 3. pre-commit 設定

`.pre-commit-config.yaml` で設定済み:

| Hook | ツール | 動作 |
|------|-------|------|
| `ruff` | astral-sh/ruff-pre-commit | Lint + 自動修正 |
| `ruff-format` | astral-sh/ruff-pre-commit | フォーマット |
| `trailing-whitespace` | pre-commit-hooks | 末尾空白除去 |
| `end-of-file-fixer` | pre-commit-hooks | 末尾改行追加 |
| `check-yaml/toml/json` | pre-commit-hooks | 構文チェック |
| `check-merge-conflict` | pre-commit-hooks | マージコンフリクト検出 |
| `check-added-large-files` | pre-commit-hooks | 5MB超ファイル検出 |
| `detect-private-key` | pre-commit-hooks | 秘密鍵検出 |
| `debug-statements` | pre-commit-hooks | pdb/breakpoint検出 |
| `mypy` | mirrors-mypy | 型チェック |
| `bandit` | PyCQA/bandit | セキュリティスキャン |

**有効化状態**: `uv run pre-commit installed at .git/hooks/pre-commit` ✅

**手動実行**:
```bash
# 全ファイルに対して実行
pre-commit run --all-files

# コミット前チェックのみ
pre-commit run
```

---

## 4. pyproject.toml 設定

追加した設定セクション:

```
[project.optional-dependencies]  → dev 依存関係
[tool.ruff]                       → line-length=120, Python3.10+
[tool.ruff.lint]                  → E/W/F/I/B/UP/SIM 有効
[tool.ruff.lint.per-file-ignores] → tests/ は B/SIM 除外
[tool.ruff.lint.isort]            → loto_forecast を first-party
[tool.mypy]                       → ignore-missing-imports, python_version=3.10
[tool.pytest.ini_options]         → カバレッジ50%以上, html report
[tool.bandit]                     → tests/ 除外, assert/random/subprocess 許可
```

---

## 5. ツール実行リファレンス

### 品質チェック (全部)

```bash
# まとめて実行
ruff check src/ tests/ --fix && \
ruff format src/ tests/ && \
mypy src/loto_forecast/ --ignore-missing-imports && \
bandit -r src/loto_forecast/ -c pyproject.toml && \
pytest tests/unit/ -v --tb=short
```

### 個別実行

```bash
# Lint
ruff check src/ tests/ --no-fix

# Format チェック
ruff format src/ tests/ --check

# 型チェック
mypy src/loto_forecast/ --ignore-missing-imports

# セキュリティスキャン
bandit -r src/loto_forecast/ -c pyproject.toml -l

# テスト (ユニットのみ)
pytest tests/unit/ -v --tb=short

# テスト (カバレッジ付き)
pytest tests/unit/ --cov=src/loto_forecast --cov-report=term-missing

# pre-commit (全ファイル)
pre-commit run --all-files
```

### MCP 経由のDB確認 (Claude Code内から)

MCP `postgres-loto` が有効な場合、以下の操作が自然言語で実行可能:
- 「最新の実行ランを見せて」→ SELECT 自動生成
- 「grid_id nf_grid_001 のタスク状態を確認して」→ SQL生成
- 「artifactsディレクトリのモデルファイル一覧を見せて」→ filesystem MCP 経由

---

## 6. 環境前提

| 項目 | 値 |
|------|-----|
| OS | Linux (WSL2) |
| Shell | bash |
| Python | 3.11.14 (conda: ts) |
| Node.js | v24.12.0 |
| npm | 11.6.2 |
| uv | 0.9.18 |
| PostgreSQL | 16.13 (psql) |
| GPU | CUDA 13.0 (torch 2.10+cu130) |

---

## 7. インストール手順 (再現用)

新しい環境でセットアップする場合:

```bash
# 1. MCP サーバーインストール
mkdir -p ~/.local/share/mcp
npm install --prefix ~/.local/share/mcp \
    @modelcontextprotocol/server-postgres \
    @modelcontextprotocol/server-filesystem

# 2. Python dev tools（uv）
cd /path/to/loto_forecast_project
uv sync --extra dev --locked

# 3. pre-commit git hook 有効化
cd /path/to/loto_forecast_project
uv run pre-commit install

# 4. ECC プラグインインストール (プロジェクトディレクトリ内で実行)
claude plugin install everything-claude-code@everything-claude-code --scope project
```

---

## 8. 既知の制限・注意事項

| 項目 | 詳細 |
|------|------|
| postgres MCP | `@modelcontextprotocol/server-postgres@0.6.2` は deprecated だが動作する |
| postgres MCP | dataset スキーマへの書き込みは技術的には可能。**運用ルールで禁止** |
| filesystem MCP | artifacts/ logs/ のみアクセス可能 (他ディレクトリは不可) |
| ECC plugin | `/python-review` は ruff/mypy/pylint/black を期待するが pylint/black は未インストール |
| mypy | NeuralForecast等の型スタブなし → `ignore_missing_imports = true` で無効化 |
| pre-commit mypy | `types-psycopg2` を additional_dependencies に指定 (初回インストール時間あり) |
| bandit B603 | subprocess を使う CLI コードが多いため skip 設定 |

---

## 9. 次のステップ (Phase B 推奨作業)

```
[ ] pre-commit run --all-files を実行して既存コードのエラーを把握
[ ] mypy エラーリストを確認して型アノテーション計画を立てる
[ ] pytest --cov で現在のカバレッジ率を計測する
[ ] /harness-audit を実行してハーネス採点スコアを確認する
[ ] /database-reviewer で SQL スキーマをレビューする
[ ] Alembic 導入検討 (sql/ → migration 管理へ)
```
