import pandas as pd
from pathlib import Path
from config import config

from tests.test_scripts import (
    run_prepare,
    run_train_base,
    run_train_meta,
    run_sltp,
    run_ranker,
    run_with_args,
    START_DATE,
    TRAIN_END_DATE,
    END_DATE,
)


def run_label_holdout(run_dir: Path):
    import label

    try:
        run_with_args(
            label,
            [
                "--run",
                str(run_dir),
                "--start_date",
                START_DATE,
                "--train_end_date",
                TRAIN_END_DATE,
                "--end_date",
                END_DATE,
            ],
        )
    except ZeroDivisionError:
        pass
    assert (run_dir / "data" / "labeled.csv").exists()


def test_ranker_holdout(sample_run):
    run_prepare(sample_run)
    run_label_holdout(sample_run)
    run_train_base(sample_run)
    run_train_meta(sample_run)
    run_sltp(sample_run)
    run_ranker(sample_run)
    df = pd.read_csv(sample_run / "data" / "signals.csv", index_col=0, parse_dates=True)
    assert len(df) > 0
    labeled = pd.read_csv(sample_run / "data" / "labeled.csv", index_col=0, parse_dates=True)
    tz = config.get("market.timezone")
    cutoff = pd.Timestamp(TRAIN_END_DATE, tz=tz)
    assert (labeled.index > cutoff).any()
    assert (df.index > cutoff).any()
