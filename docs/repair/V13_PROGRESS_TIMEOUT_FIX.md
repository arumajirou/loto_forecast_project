# V13 Screenshot Collector Progress/Timeout Fix

## 結論

v12では進捗バーは表示されたが、Streamlitの隠れた/押せないDOM候補を大量にクリックしようとして、`candidate_*` の timeout warning が連続した。v13では候補抽出と進捗計算を修正し、収集が前に進んでいることを正しく表示する。

## 変更内容

- `scripts/collect_browser_observability.py`
  - 可視・有効・ラベルありの候補だけをJavaScript側で抽出。
  - 空ラベル候補をクリック対象から除外。
  - email / URL / localhost風テキストを非アクションとしてスキップ。
  - `反映`、`保存`、`登録`、`更新`、`送信` を危険操作候補としてskip対象へ追加。
  - `--max-attempts` を追加。
  - 失敗・skipも processed として進捗に反映。
  - `--click-timeout-ms` と `--scroll-timeout-ms` を追加し、1候補あたりの待ち時間を短縮。
- `scripts/start_dashboard_and_capture_screenshots.sh`
  - help例に `--max-attempts` を追加。
- `scripts/capture_app_screenshots.sh`
  - progress説明を processed/total に変更。

## 推奨実行例

```bash
./scripts/start_dashboard_and_capture_screenshots.sh   --max-clicks 80   --max-attempts 80   --max-depth 2   --click-timeout-ms 600   --scroll-timeout-ms 350
```

より速く棚卸する場合:

```bash
./scripts/start_dashboard_and_capture_screenshots.sh   --max-clicks 30   --max-attempts 40   --max-depth 1   --click-timeout-ms 400
```

## 未実行

- 実ブラウザ収集
- DB接続
- DB書き込み
- cron導入
