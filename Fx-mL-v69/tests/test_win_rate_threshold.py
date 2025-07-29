import pandas as pd
from pathlib import Path


def test_average_win_rate_threshold():
    hist_path = Path(__file__).resolve().parent / "data" / "sample_history.csv"
    df = pd.read_csv(hist_path)
    assert df["win_rate"].mean() >= 0.53
