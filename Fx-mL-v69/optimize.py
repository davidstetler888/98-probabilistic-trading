import argparse
import importlib
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config import config
from utils import make_run_dirs

# Default hyperparameter grid used when optimizing via ``search_configs``.
DEFAULT_GRID = {
    'label.threshold': [0.0008, 0.0010, 0.0012],
    'sl_tp_grid.tp_multipliers': [[1.8, 2.0, 2.2], [2.0, 2.2, 2.4]],
    'signal.sequence_length': [16, 32],
    'ranker.target_trades_per_week': [25, 35, 45],
    'ranker.min_trades_per_week': [20, 25, 30],
    'ranker.max_trades_per_week': [45, 50, 55],
}

# Smaller grids for quicker experimentation
FAST_GRID = {
    'label.threshold': [0.0010],
    'sl_tp_grid.tp_multipliers': [[2.0, 2.2, 2.4]],
    'signal.sequence_length': [16],
    'ranker.target_trades_per_week': [35],
    'ranker.min_trades_per_week': [25],
    'ranker.max_trades_per_week': [50],
}

MID_GRID = {
    'label.threshold': [0.0008, 0.0010],
    'sl_tp_grid.tp_multipliers': [[1.8, 2.0, 2.2], [2.0, 2.2, 2.4]],
    'signal.sequence_length': [16, 32],
    'ranker.target_trades_per_week': [25, 35, 45],
    'ranker.min_trades_per_week': [20, 25, 30],
    'ranker.max_trades_per_week': [45, 50, 55],
}

GRID_MAP = {
    'fast': FAST_GRID,
    'mid': MID_GRID,
    'full': DEFAULT_GRID,
}


def apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``base`` updated with dotted-key overrides."""
    conf = deepcopy(base)
    for key, val in overrides.items():
        parts = key.split('.')
        d = conf
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return conf


def run_module(module_name: str, args: list[str]) -> None:
    """Import ``module_name`` fresh and run its main() with args."""
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        module = sys.modules[module_name]
    else:
        module = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module.__file__] + args
    try:
        try:
            module.main()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv


def run_pipeline(
    run_dir: str,
    start: str,
    train_end: str,
    end: str,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    os.environ['RUN_ID'] = run_dir
    make_run_dirs(run_dir)

    cache_used = False
    cache_key = f"{start}_{train_end}_{end}"
    train_base_needed = True
    train_meta_needed = True
    sltp_needed = True
    if cache_dir:
        cache_base = Path(cache_dir) / cache_key
        data_cache = cache_base / "data"
        models_cache = cache_base / "models"
        prepared_cache = data_cache / "prepared.csv"
        labeled_cache = data_cache / "labeled.csv"
        if prepared_cache.exists() and labeled_cache.exists():
            shutil.copy(prepared_cache, Path(run_dir) / "data" / "prepared.csv")
            if (models_cache / "scaler.joblib").exists():
                shutil.copy(
                    models_cache / "scaler.joblib",
                    Path(run_dir) / "models" / "scaler.joblib",
                )
            for fname in [
                "regime_scaler.pkl",
                "regime_pca.pkl",
                "regime_kmeans.pkl",
            ]:
                src = models_cache / fname
                if src.exists():
                    shutil.copy(src, Path(run_dir) / "models" / fname)
            shutil.copy(labeled_cache, Path(run_dir) / "data" / "labeled.csv")
            cache_used = True

        # Check for cached model artifacts
        base_cache = models_cache / "probs_base.csv"
        if base_cache.exists():
            shutil.copy(base_cache, Path(run_dir) / "data" / "probs_base.csv")
            train_base_needed = False
        meta_cache = models_cache / "meta.pkl"
        if meta_cache.exists():
            shutil.copy(meta_cache, Path(run_dir) / "models" / "meta.pkl")
            train_meta_needed = False
        sltp_cache_files = list(models_cache.glob("sltp_regime*_*.pkl"))
        if sltp_cache_files:
            for src in sltp_cache_files:
                shutil.copy(src, Path(run_dir) / "models" / src.name)
            sltp_needed = False

    if not cache_used:
        run_module('prepare', ['--start_date', start, '--train_end_date', train_end, '--end_date', end])
        run_module(
            'label',
            ['--run', run_dir, '--start_date', start, '--train_end_date', train_end, '--end_date', end],
        )
        if cache_dir:
            cache_base.mkdir(parents=True, exist_ok=True)
            data_cache.mkdir(parents=True, exist_ok=True)
            models_cache.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(run_dir) / "data" / "prepared.csv", prepared_cache)
            shutil.copy(Path(run_dir) / "data" / "labeled.csv", labeled_cache)
            if (Path(run_dir) / "models" / "scaler.joblib").exists():
                shutil.copy(
                    Path(run_dir) / "models" / "scaler.joblib",
                    models_cache / "scaler.joblib",
                )
            for fname in [
                "regime_scaler.pkl",
                "regime_pca.pkl",
                "regime_kmeans.pkl",
            ]:
                src = Path(run_dir) / "models" / fname
                if src.exists():
                    shutil.copy(src, models_cache / fname)
    if train_base_needed:
        run_module(
            'train_base',
            [
                '--run', run_dir,
                '--start_date', start,
                '--train_end_date', train_end,
                '--end_date', end,
            ],
        )
        if cache_dir:
            models_cache.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                Path(run_dir) / 'data' / 'probs_base.csv',
                models_cache / 'probs_base.csv',
            )

    if train_meta_needed:
        run_module(
            'train_meta',
            [
                '--run', run_dir,
                '--start_date', start,
                '--train_end_date', train_end,
                '--end_date', end,
            ],
        )
        if cache_dir:
            models_cache.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                Path(run_dir) / 'models' / 'meta.pkl',
                models_cache / 'meta.pkl',
            )

    if sltp_needed:
        run_module(
            'sltp',
            [
                '--run', run_dir,
                '--start_date', start,
                '--train_end_date', train_end,
                '--end_date', end,
            ],
        )
        if cache_dir:
            models_cache.mkdir(parents=True, exist_ok=True)
            for src in Path(run_dir).glob('models/sltp_regime*_*.pkl'):
                shutil.copy(src, models_cache / src.name)
    run_module(
        'train_ranker',
        [
            '--run', run_dir,
            '--start_date', start,
            '--train_end_date', train_end,
            '--end_date', end,
        ],
    )
    run_module(
        'simulate',
        [
            '--run', run_dir,
            '--start_date', train_end,
            '--end_date', end,
        ],
    )
    metrics_path = Path(run_dir) / 'artifacts' / 'sim_metrics.json'
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        pass
    return metrics


def search_configs(
    param_grid: Dict[str, list[Any]],
    start: str,
    train_end: str,
    end: str,
    base_dir: str,
    max_configs: int | None = None,
    cache_dir: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    best_conf: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {}
    goal_min, goal_max = config.get("goals.trades_per_week_range", [25, 50])

    for idx, combo in enumerate(__import__("itertools").product(*values)):
        if max_configs is not None and idx >= max_configs:
            break
        overrides = dict(zip(keys, combo))
        cfg = apply_overrides(base_config, overrides)
        config._config = cfg
        run_dir = str(Path(base_dir) / f"run_{idx}")
        metrics = run_pipeline(run_dir, start, train_end, end, cache_dir)
        trades = metrics.get('trades_per_wk', 0)
        if trades < goal_min or trades > goal_max:
            continue
        if not best_metrics or metrics.get('win_rate', 0) > best_metrics.get('win_rate', 0) or (
            metrics.get('win_rate', 0) == best_metrics.get('win_rate', 0) and metrics.get('avg_rr', 0) > best_metrics.get('avg_rr', 0)
        ):
            best_metrics = metrics
            best_conf = overrides
    return best_conf, best_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Optimize configuration settings')
    p.add_argument('--start_date', required=True)
    p.add_argument('--train_end_date', required=True)
    p.add_argument('--end_date', required=True)
    p.add_argument('--output_dir', default='opt_runs')
    p.add_argument(
        '--max_configs',
        type=int,
        default=None,
        help='Stop search after this many parameter combinations',
    )
    p.add_argument(
        '--grid',
        choices=['fast', 'mid', 'full'],
        default='full',
        help='Parameter grid size to search',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base_config = deepcopy(config._config)
    grid = GRID_MAP.get(args.grid, DEFAULT_GRID)
    best_conf, best_metrics = search_configs(
        grid,
        args.start_date,
        args.train_end_date,
        args.end_date,
        args.output_dir,
        args.max_configs,
    )
    config._config = base_config
    print('Best configuration:')
    print(best_conf)
    print('Metrics:')
    print(best_metrics)
