# v10 Screenshot / Dataset Feature / Automation

## 目的

v10では次を追加します。

1. アプリ内からの網羅的スクリーンショット収集
2. Playwrightによる console / network / page error / trace / HAR 保存
3. データセット取得 → 特徴量生成 → DBテーブル作成ジョブ
4. cron / WSL起動時の自動実行ファイル
5. アプリ起動用ファイル

## 安全方針

- `dataset` スキーマは読み取り専用。
- 特徴量テーブルの書き込み先既定は `exog.nf_feature_table_auto`。
- DB書き込みには `--yes-write` と `LOTO_ALLOW_FEATURE_DB_WRITE=1` の両方が必要。
- crontab導入はdry-run既定。
- crontabを実際に入れるには `LOTO_ALLOW_AUTOMATION_INSTALL=1` と `--install` が必要。
- スクリーンショット巡回は `--safe-clicks` 既定で、破壊的ラベルのボタンをクリックしない。

## アプリ内UI

`表示パネル(高速モード)` → `NeuralForecast Cockpit`

追加タブ:

- `スクリーンショット収集`
- `データ/特徴量ジョブ`

## 主要ファイル

```text
scripts/capture_app_screenshots.sh
scripts/run_dataset_feature_table_job.py
scripts/run_dataset_feature_table_job.sh
scripts/cron_run_feature_pipeline.sh
scripts/wsl_start_loto_app.sh
scripts/install_wsl_automation.sh
start_loto_app.sh
```

## スクリーンショット収集

```bash
LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium
./scripts/capture_app_screenshots.sh --url http://localhost:8505 --max-clicks 80 --max-depth 3
```

保存先:

```text
artifacts/observability/browser_runs/<run_id>/
```

## 特徴量ジョブ dry-run

```bash
./scripts/run_dataset_feature_table_job.sh \
  --source-schema dataset \
  --source-table loto_y_ts_unified \
  --target-schema exog \
  --target-table nf_feature_table_auto \
  --limit 5000
```

## 特徴量ジョブ DB書き込み

```bash
export LOTO_ALLOW_FEATURE_DB_WRITE=1
./scripts/run_dataset_feature_table_job.sh \
  --source-schema dataset \
  --source-table loto_y_ts_unified \
  --target-schema exog \
  --target-table nf_feature_table_auto \
  --limit 5000 \
  --yes-write
unset LOTO_ALLOW_FEATURE_DB_WRITE
```

## WSLアプリ起動

```bash
./scripts/wsl_start_loto_app.sh
```

## cron dry-run

```bash
./scripts/install_wsl_automation.sh --all
```

## cron install

```bash
LOTO_ALLOW_AUTOMATION_INSTALL=1 ./scripts/install_wsl_automation.sh --install --all
```
