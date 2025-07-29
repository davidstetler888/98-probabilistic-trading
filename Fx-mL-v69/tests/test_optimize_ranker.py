import os
import json
import shutil
from copy import deepcopy
from pathlib import Path

from utils import make_run_dirs

import optimize
from config import config


def test_default_grid_has_ranker_params():
    assert 'ranker.target_trades_per_week' in optimize.DEFAULT_GRID
    assert 'ranker.min_trades_per_week' in optimize.DEFAULT_GRID
    assert 'ranker.max_trades_per_week' in optimize.DEFAULT_GRID


def test_search_configs_applies_ranker(monkeypatch, tmp_path):
    recorded = []

    def dummy_run_pipeline(run_dir, start, train_end, end, cache_dir=None):
        recorded.append(
            (
                config.get('ranker.target_trades_per_week'),
                config.get('ranker.min_trades_per_week'),
                config.get('ranker.max_trades_per_week'),
            )
        )
        artifacts = Path(run_dir) / 'artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        metrics = {'win_rate': 0.6, 'avg_rr': 2.0, 'trades_per_wk': 40}
        with open(artifacts / 'sim_metrics.json', 'w') as f:
            json.dump(metrics, f)
        with open(artifacts / 'sim_metrics.json') as f:
            loaded = json.load(f)
        shutil.rmtree(run_dir)
        return loaded

    monkeypatch.setattr(optimize, 'run_pipeline', dummy_run_pipeline)
    optimize.base_config = deepcopy(config._config)
    grid = {
        'ranker.target_trades_per_week': [41],
        'ranker.min_trades_per_week': [36],
        'ranker.max_trades_per_week': [46],
    }
    optimize.search_configs(
        grid,
        '2021-01-01',
        '2021-01-02',
        '2021-01-03',
        str(tmp_path),
        cache_dir=str(tmp_path / 'cache'),
    )
    assert recorded and recorded[0] == (41, 36, 46)
    assert not (tmp_path / 'run_0').exists()


def test_search_configs_honors_max(monkeypatch, tmp_path):
    calls = []

    def dummy_run_pipeline(run_dir, start, train_end, end, cache_dir=None):
        calls.append(run_dir)
        artifacts = Path(run_dir) / 'artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        metrics = {'win_rate': 0.5, 'avg_rr': 1.0, 'trades_per_wk': 40}
        with open(artifacts / 'sim_metrics.json', 'w') as f:
            json.dump(metrics, f)
        with open(artifacts / 'sim_metrics.json') as f:
            loaded = json.load(f)
        shutil.rmtree(run_dir)
        return loaded

    monkeypatch.setattr(optimize, 'run_pipeline', dummy_run_pipeline)
    optimize.base_config = deepcopy(config._config)
    grid = {
        'label.threshold': [0.0008, 0.0010, 0.0012],
        'signal.sequence_length': [16, 32],
    }
    optimize.search_configs(
        grid,
        '2021-01-01',
        '2021-01-02',
        '2021-01-03',
        str(tmp_path),
        max_configs=2,
        cache_dir=str(tmp_path / 'cache'),
    )
    assert len(calls) == 2
    assert not (tmp_path / 'run_0').exists()
    assert not (tmp_path / 'run_1').exists()


def test_run_pipeline_caches(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    calls = []

    def dummy_run_module(module, args):
        calls.append(module)
        run_dir = Path(os.environ["RUN_ID"])
        make_run_dirs(str(run_dir))
        if module == "prepare":
            (run_dir / "data" / "prepared.csv").write_text("prep")
            (run_dir / "models" / "scaler.joblib").write_text("scaler")
        elif module == "label":
            (run_dir / "data" / "labeled.csv").write_text("label")
        elif module == "train_base":
            (run_dir / "data" / "probs_base.csv").write_text("probs")
        elif module == "train_meta":
            (run_dir / "models" / "meta.pkl").write_text("meta")
        elif module == "sltp":
            (run_dir / "models" / "sltp_regime0_long.pkl").write_text("sltp")
        elif module == "simulate":
            art = run_dir / "artifacts"
            art.mkdir(parents=True, exist_ok=True)
            with open(art / "sim_metrics.json", "w") as f:
                json.dump({"win_rate": 0.5, "avg_rr": 1.0, "trades_per_wk": 40}, f)

    monkeypatch.setattr(optimize, "run_module", dummy_run_module)

    optimize.run_pipeline(
        str(tmp_path / "run1"),
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        cache_dir=str(cache),
    )
    assert {"prepare", "label", "train_base", "train_meta", "sltp"}.issubset(set(calls))

    calls.clear()
    optimize.run_pipeline(
        str(tmp_path / "run2"),
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        cache_dir=str(cache),
    )
    assert "prepare" not in calls and "label" not in calls
    assert "train_base" not in calls and "train_meta" not in calls and "sltp" not in calls
