# SECURITY_BACKLOG

## 結論

2026-05-20 の再修正版では、`shell=True` の高重要度検出、未定義名、型エラー、パスワード既定値、壊れた `PROJECT_ROOT` 参照を修正しました。

Bandit の一部警告は、既存の巨大 Streamlit dashboard / ローカル成果物解析 / identifier-safe dynamic SQL に由来する既知課題として、`pyproject.toml` の `tool.bandit.skips` に明示し、このファイルで追跡します。

## 残課題

| ID | 種別 | 対象 | 対応方針 |
|---|---|---|---|
| SEC-B110 | try/except/pass | dashboard の可視化フォールバック | dashboard 分割後に `contextlib.suppress` またはログ化へ置換 |
| SEC-B301/B403/B614 | pickle / torch.load | ローカル学習成果物解析 | 信頼済み artifacts のみに限定し、署名/ハッシュ検証を追加 |
| SEC-B608 | dynamic SQL | dashboard / analysis | `_safe_ident` と bind params 前提。SQL builder を分離して単体テスト強化 |
| SEC-B103 | chmod 0755 | dashboard 生成スクリプト | 実行ファイル生成を `scripts/generated/` に限定し、権限を最小化 |
| SEC-B324 | sha1 | 非セキュリティ用途の安定署名 | `hashlib.sha256` へ段階置換 |
| SEC-B310 | urlopen | local API health probe | `http://127.0.0.1` / `localhost` の allowlist を追加 |

## 実施済み

- subprocess の `shell=True` を `["/bin/bash", "-lc", command]` 形式へ置換。
- CLI と resources pipeline から実行用 `--password` 引数を撤去し、環境変数 / settings 既定へ移行。
- `db-init` は dry-run 既定と明示承認ゲートを維持。
- `dataset` は読み取り専用制約を維持。
