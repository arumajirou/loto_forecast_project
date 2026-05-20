# scripts/analyze_nf_artifact.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from loto_forecast.analysis.nf_artifact_analysis import (
    evaluate_single_series_holdout,
    load_artifact_bundle,
    load_dataset_for_analysis,
    load_model_from_bundle,
    plot_forecast_vs_actual,
    plot_residuals,
    standardize_to_neuralforecast_format,
    summarize_bundle,
)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralForecast artifact をロードして予測/精度/診断/検定をまとめて実行"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "src"
            / "loto_forecast"
            / "api"
            / "artifacts"
            / "run_autoautoformer_optuna_mae_mae_randomsampler_dataset_loto_y_ts_unified_loto_ts_type_b_20260226_091800_80e03c113c"
        ),
        help="artifact ディレクトリ（*.ckpt, meta.json, configuration.pkl, alias_to_model.pkl を含む）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "resource_analytics_local" / "analyze_nf_artifact"),
        help="結果出力ディレクトリ",
    )
    parser.add_argument("--h", type=int, default=30, help="予測ホライズン(horizon=先のステップ数)")
    parser.add_argument("--test-size", type=int, default=30, help="ホールドアウト評価に使う末尾データ長")
    parser.add_argument("--freq", type=str, default="D", help="ds の頻度（例: D=日, W=週）")

    # データ指定（推測に失敗した場合の保険）
    parser.add_argument("--dataset-path", type=str, default=None, help="CSV/Parquet の明示パス（任意）")
    parser.add_argument("--sqlite-path", type=str, default=None, help="SQLite の明示パス（任意）")
    parser.add_argument("--sqlite-table", type=str, default=None, help="SQLite テーブル名（任意）")
    parser.add_argument("--row-limit", type=int, default=None, help="データの上限行数（任意）")

    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) artifact 読み込み
    bundle = load_artifact_bundle(artifact_dir)
    bundle_summary = summarize_bundle(bundle)
    _write_json(out_dir / "bundle_summary.json", bundle_summary)

    # 2) モデル読み込み（複数方式を自動で試す）
    model = load_model_from_bundle(bundle)
    _write_json(
        out_dir / "model_summary.json",
        {
            "kind": model.kind,
            "model_name": model.model_name,
            "extra": model.extra,
            "repr_head": repr(model.obj)[:4000],
        },
    )

    # 3) データ読み込み（推測→失敗したら引数で指定）
    raw_df = load_dataset_for_analysis(
        bundle=bundle,
        dataset_path=args.dataset_path,
        sqlite_path=args.sqlite_path,
        sqlite_table=args.sqlite_table,
        row_limit=args.row_limit,
    )
    raw_df.to_csv(out_dir / "dataset_raw_head.csv", index=False)

    # 4) NeuralForecast 形式に整形（unique_id, ds, y）
    df_nf = standardize_to_neuralforecast_format(raw_df)
    df_nf.to_csv(out_dir / "dataset_nf.csv", index=False)

    # 5) 予測＋精度＋診断＋検定
    #    ※まずは “単一系列” を動かす（unique_id が複数なら、先にフィルタしてから回す）
    result = evaluate_single_series_holdout(
        model=model,
        df_nf=df_nf,
        h=args.h,
        test_size=args.test_size,
        freq=args.freq,
    )

    # 6) 可視化
    p1 = plot_forecast_vs_actual(result, out_dir, title_prefix=model.model_name)
    p2 = plot_residuals(result, out_dir, title_prefix=model.model_name)

    # 7) レポート保存（数値）
    report = {
        "artifact_dir": str(artifact_dir),
        "out_dir": str(out_dir),
        "model": {"kind": model.kind, "model_name": model.model_name, "extra": model.extra},
        "eval": {
            "uid": result["uid"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "used_eval_len": result["used_eval_len"],
            "metrics": result["metrics"],
            "baseline_metrics": result["baseline_metrics"],
            "ljung_box": result["ljung_box"],
            "diebold_mariano_vs_naive": result["diebold_mariano_vs_naive"],
        },
        "plots": {"forecast_vs_actual": str(p1), "residuals": str(p2)},
    }
    _write_json(out_dir / "report.json", report)

    # 8) 画面用に stdout も短く
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
