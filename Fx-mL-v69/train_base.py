import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None

from config import config
from utils import (
    parse_end_date_arg,
    parse_start_date_arg,
    load_data,
    get_run_dir,
    make_run_dirs,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train per-regime LightGBM base models")
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Final bar used for training (YYYY-MM-DD)",
    )
    parser.add_argument("--end_date", type=str, required=False, help="YYYY-MM-DD last date for predictions")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    return parser.parse_args()


def time_split(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def build_sequences(df: pd.DataFrame, features: list[str], seq_len: int):
    """Create sliding window sequences from dataframe."""
    arr = df[features].values
    target = ((df["label_long"] == 1) | (df["label_short"] == 1)).astype(int).values

    if len(df) < seq_len:
        return np.empty((0, seq_len, len(features))), np.array([]), pd.Index([])

    windows = np.lib.stride_tricks.sliding_window_view(arr, seq_len, axis=0)
    X_seq = windows.swapaxes(1, 2)
    y_seq = target[seq_len - 1 :]
    idx = df.index[seq_len - 1 :]
    return X_seq, y_seq, pd.Index(idx)


def build_cnn(input_shape: tuple[int, int]):
    if tf is None:
        raise RuntimeError("TensorFlow is not available")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(32, 3, activation="relu", input_shape=input_shape),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def search_best_params(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    """Run parameter grid search with cross-validation."""
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    best_params: dict | None = None
    best_prec = -1.0
    best_rec = 0.0
    for params in ParameterGrid(param_grid):
        prec_scores: list[float] = []
        rec_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if y_train.sum() == 0:
                continue
            pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1.0
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val)
            all_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "scale_pos_weight": pos_weight,
                **params,
            }
            model = lgb.train(
                all_params,
                lgb_train,
                num_boost_round=50,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
            )
            preds = model.predict(X_val)
            if len(np.unique(y_val)) > 1:
                prec_scores.append(
                    precision_score(y_val, preds >= 0.5, zero_division=0)
                )
                rec_scores.append(
                    recall_score(y_val, preds >= 0.5, zero_division=0)
                )
            else:
                prec_scores.append(0.0)
                rec_scores.append(0.0)
        if prec_scores:
            avg_prec = float(np.mean(prec_scores))
            avg_rec = float(np.mean(rec_scores))
        else:
            avg_prec = 0.0
            avg_rec = 0.0
        if avg_prec > best_prec:
            best_prec = avg_prec
            best_rec = avg_rec
            best_params = params
    return best_params or {}, {"precision": best_prec, "recall": best_rec}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    run_dir = args.run if args.run else get_run_dir()
    dirs = make_run_dirs(run_dir)

    start_date = parse_start_date_arg(args.start_date)
    end_date = parse_end_date_arg(args.end_date)
    train_end_date = parse_end_date_arg(args.train_end_date) or end_date
    labeled_path = Path(run_dir) / "data" / "labeled.csv"
    if not labeled_path.exists():
        raise SystemExit(f"{labeled_path} not found. Run label.py first")
    df_all = load_data(
        str(labeled_path), end_date, start_date, strict=args.strict
    )
    feature_cols = config.get("signal.features")
    df_all = df_all.dropna(subset=feature_cols)
    df_all = df_all.sort_index()

    df_train = df_all[df_all.index <= train_end_date] if train_end_date else df_all

    model_dir = Path(run_dir) / "models" / "base"
    ensure_dir(model_dir)
    probs = pd.DataFrame(index=df_all.index)
    for side in ["long", "short"]:
        probs[f"prob_{side}_lgbm"] = np.nan
        for regime in sorted(df_train["market_regime"].unique()):
            df_reg_train = df_train[df_train["market_regime"] == regime]
            y = (df_reg_train[f"label_{side}"] == 1).astype(int)
            X = df_reg_train[feature_cols]
            if y.sum() == 0:
                continue
            best_params, cv_metrics = search_best_params(X, y)
            with open(model_dir / f"params_reg{regime}_{side}.json", "w") as f:
                json.dump(best_params, f)
            pos_weight = len(y) / (2 * y.sum()) if y.sum() > 0 else 1.0
            final_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "scale_pos_weight": pos_weight,
                **best_params,
            }
            model = lgb.train(
                final_params,
                lgb.Dataset(X, y),
                num_boost_round=50,
                callbacks=[lgb.log_evaluation(0)],
            )
            prec = cv_metrics.get("precision", 0.0)
            rec = cv_metrics.get("recall", 0.0)
            model_path = model_dir / f"model_reg{regime}_{side}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(model_dir / f"metrics_reg{regime}_{side}.json", "w") as f:
                json.dump({"precision": prec, "recall": rec}, f)
            df_reg_all = df_all[df_all["market_regime"] == regime]
            preds_all = model.predict(df_reg_all[feature_cols])
            probs.loc[df_reg_all.index, f"prob_{side}_lgbm"] = preds_all

    # ----- CNN training -----
    seq_len = config.get("signal.sequence_length")
    X_seq_all, y_seq_all, idx_seq_all = build_sequences(df_all, feature_cols, seq_len)
    probs["prob_cnn"] = np.nan
    if len(X_seq_all) > 0 and tf is not None:
        train_mask = idx_seq_all <= train_end_date if train_end_date else np.array([True] * len(idx_seq_all))
        X_seq_train = X_seq_all[train_mask]
        y_seq_train = y_seq_all[train_mask]
        split = int(len(X_seq_train) * (1 - args.test_split))
        X_train, X_val = X_seq_train[:split], X_seq_train[split:]
        y_train, y_val = y_seq_train[:split], y_seq_train[split:]
        cnn = build_cnn((seq_len, len(feature_cols)))
        es = tf.keras.callbacks.EarlyStopping(
            patience=config.get("signal.patience"), restore_best_weights=True
        )
        cnn.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.get("signal.epochs"),
            batch_size=config.get("signal.batch_size"),
            callbacks=[es],
            verbose=0,
        )
        val_pred = cnn.predict(X_val).ravel() if len(X_val) else np.array([])
        if len(val_pred) and len(np.unique(y_val)) > 1:
            prec = precision_score(y_val, val_pred >= 0.5, zero_division=0)
            rec = recall_score(y_val, val_pred >= 0.5, zero_division=0)
        else:
            prec = 0.0
            rec = 0.0
        cnn.save(model_dir / "cnn_model.h5")
        with open(model_dir / "cnn_metrics.json", "w") as f:
            json.dump({"precision": float(prec), "recall": float(rec)}, f)
        all_pred = cnn.predict(X_seq_all).ravel()
        probs.loc[idx_seq_all, "prob_cnn"] = all_pred
    else:
        # TensorFlow unavailable or not enough data; create placeholder files
        (model_dir / "cnn_model.h5").touch()
        with open(model_dir / "cnn_metrics.json", "w") as f:
            json.dump({"precision": 0.0, "recall": 0.0}, f)
    probs_path = Path(run_dir) / "data" / "probs_base.csv"
    probs.to_csv(probs_path)
    print(f"Wrote {probs_path}")


if __name__ == "__main__":
    main()
