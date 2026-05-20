# V16 Playwright Locator Fix Report

## 結論

v15ではブラウザ収集が起動し、進捗表示と成果物保存までは動作しましたが、クリックがすべて失敗し、初期スクリーンショット1枚だけの取得になっていました。

原因は `scripts/collect_browser_observability.py` で Playwright の `Locator.first` をメソッドとして `first()` と呼んでいたことです。現在のPlaywright Pythonでは `first` はプロパティであるため、`'Locator' object is not callable` が発生していました。

## 変更

- `Locator.first()` を互換処理に変更
  - `first` がプロパティならそのまま使用
  - メソッド型なら呼び出す
- Help / Search / CSV download / Fullscreen などUIクローム操作を非アクションとしてスキップ
- 最終進捗が `1/31` に戻らないよう、完了時は100%表示へ変更

## 安全性

- DB接続、DB書き込み、`db-init`、dataset書き込み、学習、E2Eは未実行。
- ブラウザ収集は既定で safe-clicks のままです。
