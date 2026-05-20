# src/loto_forecast/analysis/nf_artifact_analysis.py
from __future__ import annotations

import importlib
import inspect
import json
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ArtifactBundle:
    artifact_dir: Path
    ckpt_path: Path
    meta: dict[str, Any]
    config: Any
    alias_to_model: Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LoadedModel:
    """Holds the loaded model-like object and metadata we can infer."""

    obj: Any
    kind: str  # "neuralforecast" / "lightning" / "torch" / "unknown"
    model_name: str
    extra: dict[str, Any]


# -----------------------------
# Utilities
# -----------------------------
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _find_ckpt(artifact_dir: Path) -> Path:
    ckpts = sorted(artifact_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"*.ckpt が見つかりません: {artifact_dir}")
    # だいたい1個。複数なら最新(名前順)を選ぶ
    return ckpts[-1]


def guess_dataset_id_from_artifact_dir(artifact_dir: Path) -> str | None:
    """
    例:
      run_autoautoformer_optuna_..._dataset_loto_y_ts_unified_loto_ts_type_b_20260226_091800_80e03c113c
    -> dataset_loto_y_ts_unified_loto_ts_type_b
    """
    name = artifact_dir.name
    m = re.search(r"(_dataset_[A-Za-z0-9_]+)_\d{8}_\d{6}_[a-f0-9]{6,}$", name)
    if m:
        return m.group(1).lstrip("_")
    # フォールバック：dataset_～ が含まれるならそこから末尾近くまで
    if "_dataset_" in name:
        s = name.split("_dataset_", 1)[1]
        # 末尾の日時+hashっぽい部分を削る
        s = re.sub(r"_\d{8}_\d{6}_[a-f0-9]{6,}$", "", s)
        return "dataset_" + s
    return None


def safe_to_datetime(s: Any) -> pd.Timestamp | None:
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        return pd.to_datetime(s)
    except Exception:
        return None


# -----------------------------
# Load artifact bundle
# -----------------------------
def load_artifact_bundle(artifact_dir: str | Path) -> ArtifactBundle:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact_dir が存在しません: {artifact_dir}")

    ckpt_path = _find_ckpt(artifact_dir)
    meta = _read_json(artifact_dir / "meta.json")
    config = _read_pickle(artifact_dir / "configuration.pkl")
    alias_to_model = _read_pickle(artifact_dir / "alias_to_model.pkl")

    return ArtifactBundle(
        artifact_dir=artifact_dir,
        ckpt_path=ckpt_path,
        meta=meta,
        config=config,
        alias_to_model=alias_to_model,
    )


# -----------------------------
# Model loading strategies
# -----------------------------
def _try_import_class(class_path: str):
    """
    class_path:
      - "package.module:ClassName"
      - "package.module.ClassName"
    """
    if ":" in class_path:
        module, cls = class_path.split(":", 1)
    else:
        module, cls = class_path.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, cls)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _lightning_available() -> bool:
    try:
        import pytorch_lightning  # noqa: F401

        return True
    except Exception:
        return False


def _load_ckpt_raw(ckpt_path: Path) -> dict[str, Any]:
    if not _torch_available():
        raise RuntimeError("torch が import できません（環境に torch が必要です）")
    import torch

    return torch.load(str(ckpt_path), map_location="cpu")


def load_model_from_bundle(bundle: ArtifactBundle) -> LoadedModel:
    """
    可能性の高い順にロードを試みる：
      1) alias_to_model.pkl が「学習済みモデル本体」を含む
      2) configuration.pkl に class_path / init_kwargs がある
      3) NeuralForecast.load / LightningModule.load_from_checkpoint などを反射で試す
      4) ckpt の state_dict を torch.nn.Module に流し込む（最後の手段）
    """
    meta = bundle.meta or {}
    config = bundle.config
    alias_to_model = bundle.alias_to_model

    # 1) alias_to_model が「モデル実体っぽい」ケース（最も確実）
    if alias_to_model is not None:
        # dict(alias->model_obj) あるいは model_obj 単体
        if isinstance(alias_to_model, dict) and len(alias_to_model) > 0:
            # 最初の要素を採用（best の情報があればそれ優先にしてもOK）
            alias, obj = next(iter(alias_to_model.items()))
            kind, name, extra = infer_model_properties(obj)
            extra["alias"] = str(alias)
            return LoadedModel(obj=obj, kind=kind, model_name=name, extra=extra)
        else:
            obj = alias_to_model
            kind, name, extra = infer_model_properties(obj)
            return LoadedModel(obj=obj, kind=kind, model_name=name, extra=extra)

    # 2) config にクラス情報があるケース
    # よくあるキー名候補を広めに当てに行く
    class_path = None
    init_kwargs = None

    if isinstance(config, dict):
        for k in ["class_path", "model_class", "model_class_path", "pl_module", "module_class"]:
            if k in config and isinstance(config[k], str):
                class_path = config[k]
                break
        for k in ["init_kwargs", "model_kwargs", "hparams", "params"]:
            if k in config and isinstance(config[k], dict):
                init_kwargs = config[k]
                break

    if class_path:
        cls = _try_import_class(class_path)
        # 2-a) Lightningなら load_from_checkpoint があるかも
        if hasattr(cls, "load_from_checkpoint"):
            try:
                obj = cls.load_from_checkpoint(str(bundle.ckpt_path), **(init_kwargs or {}))
                kind, name, extra = infer_model_properties(obj)
                extra["class_path"] = class_path
                return LoadedModel(obj=obj, kind=kind, model_name=name, extra=extra)
            except Exception:
                pass

        # 2-b) 普通にインスタンス化して state_dict を入れる
        try:
            obj = cls(**(init_kwargs or {}))
            ckpt = _load_ckpt_raw(bundle.ckpt_path)
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("state") or None
            if state_dict and hasattr(obj, "load_state_dict"):
                obj.load_state_dict(state_dict, strict=False)
            kind, name, extra = infer_model_properties(obj)
            extra["class_path"] = class_path
            return LoadedModel(obj=obj, kind=kind, model_name=name, extra=extra)
        except Exception:
            pass

    # 3) NeuralForecast のロードAPIがあるか反射で試す（環境差対応）
    try:
        nf_cls = _try_import_class("neuralforecast.core:NeuralForecast")
        if hasattr(nf_cls, "load"):
            try:
                obj = nf_cls.load(str(bundle.artifact_dir))
                kind, name, extra = infer_model_properties(obj)
                extra["loader"] = "neuralforecast.core.NeuralForecast.load"
                return LoadedModel(obj=obj, kind=kind, model_name=name, extra=extra)
            except Exception:
                pass
    except Exception:
        pass

    # 4) 最後の手段：ckpt を生で読む（モデルの形が分からない場合は情報だけ返す）
    ckpt = _load_ckpt_raw(bundle.ckpt_path)
    return LoadedModel(
        obj=ckpt,
        kind="unknown",
        model_name=str(meta.get("model_name") or meta.get("alias") or bundle.ckpt_path.name),
        extra={"note": "モデルのクラスを復元できませんでした。ckpt辞書を返します。"},
    )


def infer_model_properties(obj: Any) -> tuple[str, str, dict[str, Any]]:
    """
    モデルの“プロパティ情報”を雑にでも抜き出す（検索/調査の入口）。
    """
    extra: dict[str, Any] = {}
    name = obj.__class__.__name__ if hasattr(obj, "__class__") else "unknown"
    kind = "unknown"

    # NeuralForecast 本体か？
    if obj is not None and obj.__class__.__name__ == "NeuralForecast":
        kind = "neuralforecast"
        # 内部に models を持つことが多い
        if hasattr(obj, "models"):
            try:
                extra["n_models"] = len(obj.models)
                extra["models"] = [m.__class__.__name__ for m in obj.models]
            except Exception:
                pass
        return kind, name, extra

    # PyTorch / Lightning 系
    if _torch_available():
        import torch

        if isinstance(obj, torch.nn.Module):
            kind = "torch"
            try:
                n_params = sum(p.numel() for p in obj.parameters())
                n_trainable = sum(p.numel() for p in obj.parameters() if p.requires_grad)
                extra["n_params"] = int(n_params)
                extra["n_trainable_params"] = int(n_trainable)
            except Exception:
                pass
            extra["device"] = str(next(obj.parameters()).device) if any(True for _ in obj.parameters()) else "unknown"
            return kind, name, extra

    # dict(ckpt) の場合
    if isinstance(obj, dict):
        kind = "ckpt_dict"
        for k in ["epoch", "global_step"]:
            if k in obj:
                extra[k] = obj[k]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            extra["state_dict_keys_sample"] = list(obj["state_dict"].keys())[:20]
        return kind, name, extra

    return kind, name, extra


# -----------------------------
# Dataset loading (heuristic)
# -----------------------------
def _read_table_sqlite(sqlite_path: Path, table: str, limit: int | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        q = f"SELECT * FROM {table}"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()


def _list_tables_sqlite(sqlite_path: Path) -> list[str]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def load_dataset_for_analysis(
    bundle: ArtifactBundle,
    dataset_path: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    sqlite_table: str | None = None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """
    まずは「存在するものを掴む」方針：
      1) dataset_path (csv/parquet) が指定されていれば読む
      2) config/meta にパス/テーブルの手掛かりがあればそれを使う
      3) sqlite_path があればテーブル名推測して読む
      4) それでもダメなら例外
    """
    # 1) 明示パス
    if dataset_path is not None:
        p = Path(dataset_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"dataset_path が存在しません: {p}")
        if p.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(p)
        elif p.suffix.lower() in [".parquet"]:
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"未対応の拡張子: {p.suffix}")
        return df

    # 2) config/meta から推測
    meta = bundle.meta or {}
    config = bundle.config

    def _extract_path_from_obj(o: Any) -> str | None:
        if isinstance(o, dict):
            for k in ["dataset_path", "data_path", "path", "csv_path", "parquet_path"]:
                v = o.get(k)
                if isinstance(v, str) and v:
                    return v
        return None

    guessed_path = _extract_path_from_obj(meta) or _extract_path_from_obj(config)
    if guessed_path:
        p = Path(guessed_path).expanduser()
        if p.exists():
            if p.suffix.lower() in [".csv", ".tsv"]:
                return pd.read_csv(p)
            if p.suffix.lower() in [".parquet"]:
                return pd.read_parquet(p)

    # 3) sqlite から推測（ユーザ環境の registry.sqlite を想定）
    if sqlite_path is None:
        # 典型: <project-root>/data/registry.sqlite
        # （存在しなければスキップ）
        default = PROJECT_ROOT / "data" / "registry.sqlite"
        sqlite_path = default if default.exists() else None

    if sqlite_path is not None:
        sqlite_path = Path(sqlite_path).expanduser().resolve()
        if not sqlite_path.exists():
            raise FileNotFoundError(f"sqlite_path が存在しません: {sqlite_path}")

        tables = _list_tables_sqlite(sqlite_path)

        # table 指定があれば最優先
        if sqlite_table and sqlite_table in tables:
            return _read_table_sqlite(sqlite_path, sqlite_table, limit=row_limit)

        # dataset_id を artifact名から推測してテーブル探索
        dsid = guess_dataset_id_from_artifact_dir(bundle.artifact_dir)
        if dsid:
            # 完全一致優先、次に部分一致
            if dsid in tables:
                return _read_table_sqlite(sqlite_path, dsid, limit=row_limit)
            cand = [t for t in tables if dsid in t]
            if cand:
                # 長い方（より具体的）を採用
                cand = sorted(cand, key=lambda x: len(x))
                return _read_table_sqlite(sqlite_path, cand[-1], limit=row_limit)

        # 最後：列名候補から “時系列っぽい” テーブルを探す
        # (ds/date, y/target, unique_id 等がありそうなもの)
        for t in tables:
            try:
                sample = _read_table_sqlite(sqlite_path, t, limit=50)
            except Exception:
                continue
            cols = {c.lower() for c in sample.columns}
            if ({"ds", "date", "datetime", "timestamp"} & cols) and ({"y", "target", "value"} & cols):
                return _read_table_sqlite(sqlite_path, t, limit=row_limit)

    raise RuntimeError(
        "データセットを推測できませんでした。--dataset-path か --sqlite-path/--sqlite-table を指定してください。"
    )


def standardize_to_neuralforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    NeuralForecast でよく使う列名:
      unique_id, ds(datetime), y(float)
    入力dfの列名が違っても、ある程度は推測して整形する。
    """
    df = df.copy()

    # 推測候補
    col_lower = {c.lower(): c for c in df.columns}
    ds_col = None
    y_col = None
    uid_col = None

    for k in ["ds", "date", "datetime", "timestamp", "time"]:
        if k in col_lower:
            ds_col = col_lower[k]
            break
    for k in ["y", "target", "value"]:
        if k in col_lower:
            y_col = col_lower[k]
            break
    for k in ["unique_id", "uid", "series", "id"]:
        if k in col_lower:
            uid_col = col_lower[k]
            break

    if ds_col is None or y_col is None:
        raise ValueError(f"ds/date系列と y/target系列が見つかりません。columns={list(df.columns)}")

    out = pd.DataFrame()
    out["ds"] = pd.to_datetime(df[ds_col])
    out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    if uid_col is None:
        out["unique_id"] = "series_0"
    else:
        out["unique_id"] = df[uid_col].astype(str)

    out = out.dropna(subset=["ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out


# -----------------------------
# Forecast, metrics, tests
# -----------------------------
def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-12, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape_denom = np.where(y_true == 0, 1e-12, np.abs(y_true))
    mape = float(np.mean(np.abs(err) / mape_denom) * 100.0)
    smape = _smape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "sMAPE(%)": smape}


def naive_baseline_last_value(y: np.ndarray, h: int) -> np.ndarray:
    """
    直近値を h ステップ先までコピーする単純ベースライン。
    """
    last = float(y[-1])
    return np.full(shape=(h,), fill_value=last, dtype=float)


def ljung_box_test(residuals: np.ndarray, lags: int = 20) -> dict[str, Any]:
    """
    Ljung–Box 検定（リュング＝ボックス）：残差に自己相関が残っているか？
    """
    residuals = np.asarray(residuals, dtype=float)
    out: dict[str, Any] = {"available": False}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        df = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        out["available"] = True
        out["lags"] = int(lags)
        out["lb_stat"] = float(df["lb_stat"].iloc[0])
        out["lb_pvalue"] = float(df["lb_pvalue"].iloc[0])
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> dict[str, Any]:
    """
    Diebold–Mariano 検定（予測比較の検定）：
      e1: model errors
      e2: baseline errors
    d_t = |e1|^power - |e2|^power を使い、平均が 0 かを検定。
    HAC(Newey-West) で自己相関補正（h>1 を意識）。
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d = d[~np.isnan(d)]
    out: dict[str, Any] = {"available": False, "n": int(d.size)}
    if d.size < 10:
        out["note"] = "データが少なすぎて検定が不安定です（n<10）"
        return out

    try:
        import statsmodels.api as sm

        # 定数項のみ回帰して、HACでt値を出す
        X = np.ones((d.size, 1))
        model = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
        t = float(model.tvalues[0])
        p = float(model.pvalues[0])
        out.update({"available": True, "dm_t": t, "dm_pvalue": p, "mean_d": float(np.mean(d))})
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _call_predict_like(obj: Any, df: pd.DataFrame | None, h: int) -> Any:
    """
    ライブラリ差を吸収する “予測呼び出し”。
    返り値は DataFrame / dict / np.ndarray など色々あり得るので、後段で整形する。
    """
    # 1) predict(df=...) 型
    if hasattr(obj, "predict"):
        try:
            sig = inspect.signature(obj.predict)
            if "df" in sig.parameters and df is not None:
                return obj.predict(df=df)
            if len(sig.parameters) == 0:
                return obj.predict()
            # predict(df) のような positional
            if df is not None:
                return obj.predict(df)
        except Exception:
            pass

    # 2) forecast(df=..., h=...) 型
    for method_name in ["forecast", "predict_future", "forward"]:
        if hasattr(obj, method_name):
            fn = getattr(obj, method_name)
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "df" in sig.parameters and df is not None:
                    kwargs["df"] = df
                if "h" in sig.parameters:
                    kwargs["h"] = h
                return fn(**kwargs)
            except Exception:
                continue

    raise RuntimeError("予測メソッドを呼び出せませんでした（predict/forecast等が見つからない）")


def extract_point_forecast(pred_obj: Any, model_name_hint: str = "model") -> pd.DataFrame:
    """
    予測の返り値から “点予測(point forecast)” を DataFrame に揃える。
    期待形：
      columns: unique_id, ds, y_hat
    """
    # NeuralForecast系は DataFrame で列がモデル名になりがち
    if isinstance(pred_obj, pd.DataFrame):
        df = pred_obj.copy()
        # まずは標準キーを探す
        cols = set(df.columns)
        uid = "unique_id" if "unique_id" in cols else None
        ds = "ds" if "ds" in cols else None
        # 点予測列：よくある候補
        yhat_col = None
        for c in [model_name_hint, "y_hat", "yhat", "pred", "prediction"]:
            if c in cols:
                yhat_col = c
                break
        if yhat_col is None:
            # “モデル名っぽい列” を推測（キー以外の最初）
            candidates = [c for c in df.columns if c not in ["unique_id", "ds", "y", "cutoff"]]
            if candidates:
                yhat_col = candidates[0]

        if uid and ds and yhat_col:
            out = df[[uid, ds, yhat_col]].rename(columns={yhat_col: "y_hat"})
            out["unique_id"] = out["unique_id"].astype(str)
            out["ds"] = pd.to_datetime(out["ds"])
            out["y_hat"] = pd.to_numeric(out["y_hat"], errors="coerce")
            return out.dropna(subset=["ds", "y_hat"]).reset_index(drop=True)

    # numpy array の場合：1系列想定
    if isinstance(pred_obj, np.ndarray):
        arr = pred_obj.astype(float).reshape(-1)
        # ds が無いので後段で付与する
        return pd.DataFrame({"unique_id": ["series_0"] * arr.size, "ds": pd.NaT, "y_hat": arr})

    # dict の場合：ありがちなキーを探す
    if isinstance(pred_obj, dict):
        for k in ["y_hat", "yhat", "pred", "prediction"]:
            if k in pred_obj:
                arr = np.asarray(pred_obj[k], dtype=float).reshape(-1)
                return pd.DataFrame({"unique_id": ["series_0"] * arr.size, "ds": pd.NaT, "y_hat": arr})

    raise ValueError("予測結果から点予測を抽出できませんでした（返り値形式が未対応）")


def build_future_ds(last_ds: pd.Timestamp, h: int, freq: str = "D") -> list[pd.Timestamp]:
    """
    将来の ds を生成。freq は 'D'（日）など。
    """
    if last_ds is None or pd.isna(last_ds):
        return [pd.NaT] * h
    start = last_ds + pd.tseries.frequencies.to_offset(freq)
    return list(pd.date_range(start=start, periods=h, freq=freq))


def evaluate_single_series_holdout(
    model: LoadedModel,
    df_nf: pd.DataFrame,
    h: int,
    test_size: int,
    freq: str = "D",
) -> dict[str, Any]:
    """
    単一系列（unique_id 1個）を想定した簡易ホールドアウト評価。
    ※ multi-series でも groupby して回せるが、まずは “動く最小” を優先。
    """
    df_nf = df_nf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    uids = df_nf["unique_id"].unique().tolist()
    if len(uids) != 1:
        raise ValueError(f"この関数は unique_id=1 を想定しています（現在 {len(uids)}）")

    series = df_nf[df_nf["unique_id"] == uids[0]].reset_index(drop=True)
    if series.shape[0] < (test_size + 5):
        raise ValueError(f"データが少なすぎます: n={series.shape[0]}, test_size={test_size}")

    train = series.iloc[:-test_size].copy()
    test = series.iloc[-test_size:].copy()

    # 予測：ライブラリ差があるので “まずは呼ぶ” を優先
    pred_raw = _call_predict_like(model.obj, df=train, h=h)

    # 予測抽出
    pred_df = extract_point_forecast(pred_raw, model_name_hint=model.model_name)

    # ds が欠ける場合、train末尾から future ds を作って埋める
    if pred_df["ds"].isna().all():
        last_ds = pd.to_datetime(train["ds"].iloc[-1])
        future = build_future_ds(last_ds=last_ds, h=pred_df.shape[0], freq=freq)
        pred_df["ds"] = future

    # 評価：test の先頭 h と pred の先頭 h を合わせる（不足は切る）
    y_true = test["y"].to_numpy(dtype=float)
    # pred が test_size より短い可能性があるので最小長に揃える
    L = min(y_true.size, pred_df.shape[0])
    y_true = y_true[:L]
    y_pred = pred_df["y_hat"].to_numpy(dtype=float)[:L]

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    residuals = y_pred - y_true
    lb = ljung_box_test(residuals=residuals, lags=min(20, max(5, L // 3)))

    # baseline：直近値コピー（train末尾）
    base_pred = naive_baseline_last_value(train["y"].to_numpy(dtype=float), h=L)
    base_metrics = compute_metrics(y_true=y_true, y_pred=base_pred)

    dm = diebold_mariano_test(
        e1=(y_pred - y_true),
        e2=(base_pred - y_true),
        h=1,
        power=2,
    )

    return {
        "uid": uids[0],
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "used_eval_len": int(L),
        "metrics": metrics,
        "baseline_metrics": base_metrics,
        "ljung_box": lb,
        "diebold_mariano_vs_naive": dm,
        "pred_df": pred_df.iloc[:L].copy(),
        "test_df": test.iloc[:L].copy(),
    }


# -----------------------------
# Visualization
# -----------------------------
def plot_forecast_vs_actual(
    result: dict[str, Any],
    out_dir: str | Path,
    title_prefix: str = "",
) -> Path:
    """
    actual vs forecast の基本プロットを PNG で保存。
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df: pd.DataFrame = result["pred_df"]
    test_df: pd.DataFrame = result["test_df"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(test_df["ds"], test_df["y"], label="actual")
    ax.plot(pred_df["ds"], pred_df["y_hat"], label="forecast")

    title = f"{title_prefix} forecast vs actual"
    ax.set_title(title)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.legend()

    out_path = out_dir / "forecast_vs_actual.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residuals(
    result: dict[str, Any],
    out_dir: str | Path,
    title_prefix: str = "",
) -> Path:
    """
    残差（forecast-actual）を可視化（時系列）。
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df: pd.DataFrame = result["pred_df"]
    test_df: pd.DataFrame = result["test_df"]

    residuals = pred_df["y_hat"].to_numpy(dtype=float) - test_df["y"].to_numpy(dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(test_df["ds"], residuals, label="residuals")
    ax.axhline(0.0, linewidth=1.0)

    ax.set_title(f"{title_prefix} residuals (forecast-actual)")
    ax.set_xlabel("ds")
    ax.set_ylabel("residual")
    ax.legend()

    out_path = out_dir / "residuals.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -----------------------------
# Reporting helpers
# -----------------------------
def summarize_bundle(bundle: ArtifactBundle) -> dict[str, Any]:
    """
    config/meta を “検索可能” な形で要約（キー一覧や型など）。
    """
    meta = bundle.meta or {}
    config = bundle.config
    alias = bundle.alias_to_model

    out: dict[str, Any] = {
        "artifact_dir": str(bundle.artifact_dir),
        "ckpt_path": str(bundle.ckpt_path),
        "meta_keys": sorted(list(meta.keys()))[:200],
        "config_type": str(type(config)),
        "alias_to_model_type": str(type(alias)),
    }

    if isinstance(config, dict):
        out["config_keys"] = sorted(list(config.keys()))[:400]
    else:
        # dict 以外なら repr を少しだけ
        out["config_repr_head"] = repr(config)[:4000]

    if isinstance(alias, dict):
        out["alias_keys"] = [str(k) for k in list(alias.keys())[:50]]
        out["alias_values_types"] = [str(type(v)) for v in list(alias.values())[:10]]
    else:
        out["alias_repr_head"] = repr(alias)[:2000]

    return out
