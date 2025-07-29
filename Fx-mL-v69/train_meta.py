import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from config import config
from utils import (
    load_data,
    parse_end_date_arg,
    parse_start_date_arg,
    get_run_dir,
    make_run_dirs,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic GBM meta model")
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
    parser.add_argument("--search", choices=["grid", "bayes"], default=None, help="Hyperparameter search method")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of CV folds")
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


def search_best_params(X: pd.DataFrame, y: pd.Series, method: str | None, n_splits: int) -> tuple[dict, float]:
    """Return best params and AUC via CV search."""
    if method is None:
        return {}, 0.0
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_estimators": 50,
        "scale_pos_weight": len(y) / (2 * y.sum()) if y.sum() > 0 else 1.0,
    }
    if method == "grid":
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31],
        }
        best_params: dict | None = None
        best_auc = -1.0
        for params in ParameterGrid(param_grid):
            auc_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = lgb.LGBMClassifier(**base_params, **params)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    auc_scores.append(roc_auc_score(y_val, preds))
            if auc_scores:
                mean_auc = float(np.mean(auc_scores))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = params
        return best_params or {}, best_auc
    elif method == "bayes":
        search_space = {
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "num_leaves": Integer(15, 63),
        }
        estimator = lgb.LGBMClassifier(**base_params)
        bayes = BayesSearchCV(
            estimator,
            search_space,
            cv=tscv,
            scoring="roc_auc",
            n_iter=10,
            n_jobs=1,
            random_state=0,
        )
        bayes.fit(X, y)
        return bayes.best_params_, bayes.best_score_
    else:
        return {}, 0.0


def train_side(df, side, args, feature_cols, model_dir):
    y = (df[f"label_{side}"] == 1).astype(int)
    X = df[feature_cols]
    if y.sum() == 0:
        return None, None

    best_params, cv_auc = search_best_params(X, y, args.search, args.cv_folds)

    X_train, X_val, y_train, y_val = time_split(X, y, args.test_split)
    pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1.0
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)
    params = {
        "objective": "binary",
        "verbosity": -1,
        "metric": "auc",
        "scale_pos_weight": pos_weight,
        **(best_params or {}),
    }
    if "learning_rate" not in params:
        params["learning_rate"] = 0.05
    if "num_leaves" not in params:
        params["num_leaves"] = 31

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(10)],
    )
    preds = model.predict(X_val)
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, preds)
    else:
        auc = 0.5
    with open(model_dir / f"meta_{side}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / f"meta_{side}_metrics.json", "w") as f:
        json.dump({"roc_auc": auc}, f)
    if best_params:
        with open(model_dir / f"meta_{side}_params.json", "w") as f:
            json.dump(best_params, f)
    metric_to_log = cv_auc if args.search else auc
    return model, metric_to_log


def main():
    args = parse_args()
    np.random.seed(args.seed)
    run_dir = args.run if args.run else get_run_dir()
    make_run_dirs(run_dir)
    start_date = parse_start_date_arg(args.start_date)
    end_date = parse_end_date_arg(args.end_date)
    train_end_date = parse_end_date_arg(args.train_end_date) or end_date

    labeled = load_data(
        str(Path(run_dir) / "data" / "labeled.csv"), end_date, start_date, strict=args.strict
    )
    probs = load_data(
        str(Path(run_dir) / "data" / "probs_base.csv"), end_date, start_date, strict=args.strict
    )
    df_all = labeled.join(probs, how="inner")
    df_train = df_all[df_all.index <= train_end_date] if train_end_date else df_all
    feature_cols = [c for c in probs.columns] + ["atr_pct", "hour", "market_regime"]
    model_dir = Path(run_dir) / "models"
    ensure_dir(model_dir)
    models = {}
    metrics = {}
    for side in ["long", "short"]:
        model, auc = train_side(df_train, side, args, feature_cols, model_dir)
        if model is not None:
            models[side] = model
            metrics[side] = auc
    with open(model_dir / "meta.pkl", "wb") as f:
        pickle.dump(models, f)
    with open(model_dir / "meta_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Meta models trained")


if __name__ == "__main__":
    main()
