# Intraday Performance Analysis

`analyze_intraday_performance()` in `monitor_win_rate.py` inspects how win rates vary by hour of day and by weekday.

```python
from monitor_win_rate import analyze_intraday_performance
from simulate import simulate_df

# run a simulation and capture trade details
_, _, trades = simulate_df(df, return_trades=True)

# print intraday win rate stats
analyze_intraday_performance(trades)
```

The function groups executed trades by their timestamp hour and weekday, then prints the average win rate for each group. It also returns the aggregated series for further analysis.
