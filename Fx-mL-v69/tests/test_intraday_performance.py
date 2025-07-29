import pandas as pd
from monitor_win_rate import analyze_intraday_performance


def test_analyze_intraday_performance_basic():
    times = pd.date_range('2021-01-04', periods=4, freq='H')
    trades = pd.DataFrame({'timestamp': times, 'profit': [1, -1, 1, 1]})
    result = analyze_intraday_performance(trades)
    by_hour = result['by_hour']
    by_weekday = result['by_weekday']
    assert by_hour.loc[0] == 1.0
    assert by_hour.loc[1] == 0.0
    assert abs(by_weekday.loc['Monday'] - 0.75) < 1e-9


def test_analyze_intraday_performance_empty():
    trades = pd.DataFrame(columns=['timestamp', 'profit'])
    result = analyze_intraday_performance(trades)
    assert result['by_hour'].empty
    assert result['by_weekday'].empty
