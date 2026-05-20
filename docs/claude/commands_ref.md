# CLIコマンドリファレンス

実行形式: `python -m loto_forecast.cli <command> [options]`

## DB初期化

```bash
python -m loto_forecast.cli db-init
```
⚠️ 破壊的操作。事前確認必須。

## 学習・再学習・予測・評価

```bash
# 学習
python -m loto_forecast.cli train \
  --model AutoNHITS \
  --h 28 \
  [--params-json '{"num_samples":10,"seed":1}']

# 再学習
python -m loto_forecast.cli retrain \
  --base-run-id <RUN_ID>

# 予測
python -m loto_forecast.cli predict \
  --run-id <RUN_ID> \
  --h 28

# 評価
python -m loto_forecast.cli evaluate \
  --run-id <RUN_ID>

# 説明可能性 (Permutation)
python -m loto_forecast.cli explain \
  --run-id <RUN_ID> \
  --method permutation

# 説明可能性 (Granger)
python -m loto_forecast.cli explain \
  --run-id <RUN_ID> \
  --method granger \
  --maxlag 8 \
  --top-k 20
```

## グリッドサーチ

```bash
# グリッド定義
python -m loto_forecast.cli grid-create \
  --grid-id nf_grid_001 \
  --adapter neuralforecast_auto \
  --model AutoNHITS \
  --h 28 \
  --param-space-json '{"num_samples":[10,20],"seed":[1,2]}'

# グリッド実行 (⚠️ 長時間・バックグラウンド推奨)
python -m loto_forecast.cli grid-run --grid-id nf_grid_001

# グリッド状態確認 (read-only・安全)
python -m loto_forecast.cli grid-status --grid-id nf_grid_001
```

## カタログ管理

```bash
# 取り込み
python -m loto_forecast.cli catalog-import \
  --library neuralforecast \
  --yaml-path ./docs/lib_docs/neuralforecast_all_codegen.yaml

# 一覧
python -m loto_forecast.cli catalog-list --library neuralforecast --limit 20

# 検証
python -m loto_forecast.cli catalog-validate \
  --library neuralforecast \
  --full-path neuralforecast.auto.AutoNHITS.__init__ \
  --arguments-json '{"h":28,"num_samples":10}'
```

## アダプタ確認

```bash
python -m loto_forecast.cli adapters
```

## 外生変数生成

```bash
# 標準 exog (NeuralForecast用)
python -m loto_forecast.cli build-exog \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table loto_y_ts_exog \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --parallel-workers 4 --enable-gpu-compute

# TimesFM埋め込み exog
python -m loto_forecast.cli build-exog-timesfm \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table timesfm \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y

# UNI2TS埋め込み exog
python -m loto_forecast.cli build-exog-uni2ts \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table uni2ts \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --context-length 128 --embedding-dim 256 --batch-size 512

# Chronos埋め込み exog
python -m loto_forecast.cli build-exog-chronos \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table chronos \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y
```

## ⚠️ セキュリティ注意事項

- `--password` 引数はコマンドラインに直接書かない
- 代わりに環境変数 `DB_PASSWORD=<set-in-env>` を設定するか `.env` ファイルを使用
- 上記コマンド例から `` を意図的に省略している

## 利用可能なモデル (neuralforecast_auto adapter)

- AutoNHITS
- AutoNBEATS
- AutoPatchTST
- AutoTFT
- AutoLSTM
- AutoGRU
- AutoDeepAR
- AutoTimesNet
- AutoAutoformer
- AutoFEDformer

`python -m loto_forecast.cli adapters` で最新一覧を確認。
