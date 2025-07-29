import pandas as pd
import pytz
import pytest
import simulate
from config import config

def test_non_overlapping_trades():
    times = pd.date_range("2020-01-01", periods=4, freq="5min")
    df = pd.DataFrame({
        'timestamp': times,
        'price': [1.0]*4,
        'high': [1.0001, 1.0001, 1.0001, 1.0030],
        'low': [0.9995]*4,
        'side': ['long']*4,
        'sl_points': [1.0]*4,
        'tp_points': [2.0]*4,
    })
    equity, metrics = simulate.simulate_df(df, cooldown_min=0, max_positions=1)
    assert metrics['total_trades'] == 2


def test_prepare_trade_df_left_join():
    import importlib

    # Reload simulate to undo any patches from other tests
    importlib.reload(simulate)

    times = pd.date_range("2020-01-01", periods=3, freq="5min")
    prepared = pd.DataFrame(
        {
            "close": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
        },
        index=times,
    )
    signals = pd.DataFrame(
        {"side": ["long"], "sl_pips": [10.0], "tp_pips": [20.0]}, index=[times[1]]
    )
    df = simulate.prepare_trade_df(prepared, signals)
    assert len(df) == len(prepared)
    assert df.index.equals(prepared.index)
    assert not df.loc[times[0], "has_signal"]
    assert df.loc[times[1], "has_signal"]


def test_max_positions_allows_concurrent_trades():
    times = pd.date_range("2020-01-01", periods=5, freq="5min")
    df = pd.DataFrame({
        'timestamp': times,
        'price': [1.0]*5,
        'high': [1.0001, 1.0001, 1.0001, 1.0001, 1.0030],
        'low': [0.9995]*5,
        'side': ['long']*5,
        'sl_points': [1.0]*5,
        'tp_points': [2.0]*5,
        'has_signal': [True]*5,
    })

    _, metrics1 = simulate.simulate_df(df, cooldown_min=0, max_positions=1)
    _, metrics2 = simulate.simulate_df(df, cooldown_min=0, max_positions=2)

    assert metrics1['total_trades'] == 2
    assert metrics2['total_trades'] > metrics1['total_trades']


def _base_df(times):
    return pd.DataFrame(
        {
            'timestamp': times,
            'price': [1.0] * len(times),
            'high': [1.001] * len(times),
            'low': [0.999] * len(times),
            'side': ['long'] * len(times),
            'sl_points': [1.0] * len(times),
            'tp_points': [2.0] * len(times),
        }
    )


def test_trade_details_timestamp_naive():
    times = pd.date_range('2020-01-01', periods=3, freq='5min')
    df = _base_df(times)
    _, _, trades = simulate.simulate_df(
        df, cooldown_min=0, max_positions=1, return_trades=True
    )
    assert trades['timestamp'].dt.tz is None


def test_trade_details_timestamp_tz_aware():
    times = pd.date_range('2020-01-01', periods=3, freq='5min', tz='UTC')
    df = _base_df(times)
    _, _, trades = simulate.simulate_df(
        df, cooldown_min=0, max_positions=1, return_trades=True
    )
    assert trades['timestamp'].dt.tz is None


def test_simulate_df_localizes_input():
    times = pd.date_range('2020-01-01', periods=2, freq='5min')
    df = _base_df(times)
    simulate.simulate_df(df, cooldown_min=0, max_positions=1)
    tz = pytz.timezone(config.get('market.timezone'))
    assert df['timestamp'].dt.tz == tz


def test_validate_simulation_config_invalid(monkeypatch):
    monkeypatch.setitem(config._config['simulation'], 'risk_per_trade', -0.1)
    with pytest.raises(ValueError):
        simulate.validate_simulation_config()


def test_side_specific_cooldown():
    times = pd.date_range('2020-01-01', periods=2, freq='5min')
    df = pd.DataFrame(
        {
            'timestamp': times,
            'price': [1.0, 1.0],
            'high': [1.0, 1.0],
            'low': [0.999, 0.999],
            'side': ['long', 'short'],
            'sl_points': [1.0, 1.0],
            'tp_points': [2.0, 2.0],
        }
    )
    _, metrics = simulate.simulate_df(df, cooldown_min=10, max_positions=1)
    assert metrics['total_trades'] == 2

