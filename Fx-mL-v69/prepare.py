import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import joblib
import argparse
import sys
from utils import (
    get_run_id,
    make_run_dirs,
    prep_summary,
    parse_end_date_arg,
    parse_start_date_arg,
    load_data as load_csv,
)
from config import config
from walkforward import load_raw_index

# === Script Configuration ===
SPREAD = config.get('sl_tp_grid.spread', 0.00013)  # Use config, fallback to old default

# Default number of market regimes. Used when dynamic selection is not enabled.
DEFAULT_N_REGIMES = (
    int(config.get("prepare.n_clusters", 4))
    if isinstance(config.get("prepare.n_clusters", 4), int)
    else 4
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare trading data with features and regimes"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format",
        required=False,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Earliest bar to include (YYYY-MM-DD)",
        required=False,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of raw CSV files (overrides config 'data.input_dir')",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Final bar used for fitting scalers (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed rows when loading raw CSVs",
        default=False,
    )
    return parser.parse_args()

# === Load & Parse ===

def limit_to_recent_years(df, years=3):
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def compute_dynamic_n_clusters(df) -> int:
    """Return an appropriate cluster count based on the date range."""
    days = (df.index.max() - df.index.min()).days
    if days < 30:
        return 2
    elif days < 90:
        return 3
    else:
        return DEFAULT_N_REGIMES

def calculate_indicators_and_lags(df):
    """Calculate technical indicators and their lags."""
    # === EMAs ===
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # === RSI ===
    for period in [14, 28]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # === MACD ===
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # === Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line)
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    df['ichimoku_tenkan'] = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line)
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['ichimoku_kijun'] = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A)
    df['ichimoku_senkouA'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
    
    # Senkou Span B (Leading Span B)
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['ichimoku_senkouB'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span)
    df['ichimoku_chikou'] = df['close'].shift(26)
    
    # === ATR ===
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # === ADX ===
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = true_range
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
    minus_di = abs(100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean()))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # === Bollinger Bands ===
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # === Stochastic Oscillator ===
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stochastic_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
    
    # === CMF (Chaikin Money Flow) ===
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # === MFI (Money Flow Index) ===
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    positive_flow[typical_price > typical_price.shift(1)] = money_flow
    negative_flow[typical_price < typical_price.shift(1)] = money_flow
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    df['mfi'] = mfi
    
    # === Bar Volatility ===
    df['bar_volatility'] = (df['high'] - df['low']) / df['close'] * 100
    
    # === Relative Tick Volume ===
    df['relative_tick_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # === Spread Features ===
    df['spread_pips'] = SPREAD * 10000
    df['bar_range_minus_spread_pips'] = (
        (df['high'] - df['low'] - SPREAD).clip(lower=0) / 0.0001
    )
    
    # === Returns ===
    for period in [1, 3, 5, 10]:
        df[f'return_{period}'] = df['close'].pct_change(period) * 100
    
    # === Time-based Features ===
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    
    # Trading sessions
    df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['session_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    # Add lags for all technical indicators
    lag_periods = [1, 3, 5, 10]
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 
                                                             'hour', 'weekday', 'session_asian', 
                                                             'session_london', 'session_ny', 
                                                             'session_overlap']]
    
    lag_series = []
    for col in indicator_cols:
        for lag in lag_periods:
            lag_series.append(df[col].shift(lag).rename(f'{col}_lag_{lag}'))

    if lag_series:
        df = pd.concat([df] + lag_series, axis=1)

    return df


def prepare_multi_timeframe(df):
    """Generate features for multiple timeframes"""
    
    # Original 5-minute data
    df_5min = df.copy()
    
    # 15-minute aggregation
    df_15min = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # 30-minute aggregation
    df_30min = df.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate trends for higher timeframes
    df_15min['trend'] = df_15min['close'].pct_change()
    df_30min['trend'] = df_30min['close'].pct_change()
    
    # Calculate momentum indicators for higher timeframes
    df_15min['momentum'] = df_15min['close'].pct_change(4)  # 1 hour momentum
    df_30min['momentum'] = df_30min['close'].pct_change(4)  # 2 hour momentum
    
    # Calculate volatility for higher timeframes
    df_15min['volatility'] = df_15min['close'].pct_change().rolling(8).std() * np.sqrt(8)  # 2 hour volatility
    df_30min['volatility'] = df_30min['close'].pct_change().rolling(8).std() * np.sqrt(8)  # 4 hour volatility
    
    # Add higher timeframe features to 5-minute data
    df_5min['htf_15min_trend'] = df_15min['trend'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_trend'] = df_30min['trend'].reindex(df_5min.index, method='ffill')
    df_5min['htf_15min_momentum'] = df_15min['momentum'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_momentum'] = df_30min['momentum'].reindex(df_5min.index, method='ffill')
    df_5min['htf_15min_volatility'] = df_15min['volatility'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_volatility'] = df_30min['volatility'].reindex(df_5min.index, method='ffill')
    
    # Add higher timeframe price levels
    df_5min['htf_15min_close'] = df_15min['close'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_close'] = df_30min['close'].reindex(df_5min.index, method='ffill')
    
    # Calculate price distance from higher timeframe levels
    df_5min['price_distance_15min'] = (df_5min['close'] - df_5min['htf_15min_close']) / df_5min['htf_15min_close']
    df_5min['price_distance_30min'] = (df_5min['close'] - df_5min['htf_30min_close']) / df_5min['htf_30min_close']
    
    # Calculate multi-timeframe alignment score
    df_5min['mtf_alignment'] = 0.0
    
    # Trend alignment component
    trend_5min = df_5min['return_1']
    trend_15min = df_5min['htf_15min_trend']
    trend_30min = df_5min['htf_30min_trend']
    
    # Calculate alignment based on trend directions
    trend_alignment = (
        (trend_5min > 0) & (trend_15min > 0) & (trend_30min > 0) |
        (trend_5min < 0) & (trend_15min < 0) & (trend_30min < 0)
    ).astype(int)
    
    df_5min['mtf_alignment'] = trend_alignment
    
    print(f"Added multi-timeframe features to {len(df_5min)} bars")
    print(f"15-minute timeframe: {len(df_15min)} bars")
    print(f"30-minute timeframe: {len(df_30min)} bars")
    
    return df_5min, df_15min, df_30min


# === Normalize & Clean ===
def clean_and_normalize(df, train_end_date: str | None = None):
    """Clean and normalize features.

    The scaler is fitted on rows up to ``train_end_date`` if provided and then
    applied to the full dataframe.
    """
    df = df.dropna().copy()
    
    # Store raw values before normalization
    raw_cols = ["atr", "adx", "rsi_14", "rsi_28"]
    raw_values = {col: df[col].copy() for col in raw_cols}
    
    # Remove rows with invalid indicator values (-1.0)
    all_invalid_cols = ['atr', 'adx', 'cmf', 'mfi', 'plus_di', 'minus_di', 'relative_tick_volume']
    invalid_cols = [col for col in all_invalid_cols if col in df.columns]
    initial_rows = len(df)
    if invalid_cols:
        df = df[~df[invalid_cols].isin([-1.0]).any(axis=1)]
    removed_rows = initial_rows - len(df)
    print(f"\nRemoved {removed_rows:,} rows with invalid indicator values")
    
    # Normalize features
    feature_cols = df.columns.difference(["open", "high", "low", "close", "volume"])
    df[feature_cols] = df[feature_cols].astype(float)
    scaler = StandardScaler()
    if train_end_date:
        te_ts = pd.Timestamp(train_end_date, tz=df.index.tz)
        train_mask = df.index <= te_ts
        scaler.fit(df.loc[train_mask, feature_cols])
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols]).astype("float32")
    else:
        df.loc[:, feature_cols] = scaler.fit_transform(df[feature_cols]).astype("float32")
    
    # Restore raw values efficiently to avoid fragmentation
    raw_df = pd.concat(
        [raw_values[col].rename(f"{col}_raw") for col in raw_cols], axis=1
    )
    df = pd.concat([df, raw_df], axis=1)
    
    return df, scaler

# === Market Regime Clustering ===
def add_market_regimes(
    df,
    config,
    models_dir: Path | None = None,
    train_end_date: str | None = None,
):
    """Add market regime labels and calculate statistics.

    If ``models_dir`` is provided the scaler, PCA and KMeans objects used for
    regime clustering are persisted there using ``joblib``.
    """
    # Calculate raw statistics
    raw_stats = {
        "avg_atr_pips": df["atr_raw"].mean() * 10000,  # Convert to pips
        "atr_std_pips": df["atr_raw"].std() * 10000,
        "avg_adx": df["adx_raw"].mean(),
        "avg_rsi": df["rsi_14_raw"].mean()
    }
    
    # Prepare data for clustering
    regime_features = ["atr", "rsi_14", "adx", "cmf", "volume"]
    regime_df = df[regime_features].copy()

    # Determine number of clusters
    n_clusters_setting = config.get('prepare', {}).get('n_clusters', DEFAULT_N_REGIMES)
    auto_clusters = isinstance(n_clusters_setting, str) and n_clusters_setting == 'auto'
    if auto_clusters:
        n_clusters = compute_dynamic_n_clusters(df)
    else:
        n_clusters = int(n_clusters_setting)

    # Preprocess regime features for better clustering
    regime_scaler = StandardScaler()
    if train_end_date:
        te_ts = pd.Timestamp(train_end_date, tz=df.index.tz)
        mask = df.index <= te_ts
        regime_scaler.fit(regime_df.loc[mask])
        regime_scaled = regime_scaler.transform(regime_df)
    else:
        regime_scaled = regime_scaler.fit_transform(regime_df)

    pca = PCA(n_components=3, random_state=config.get('seed', 42))
    if train_end_date:
        pca.fit(regime_scaled[mask])
        regime_pca = pca.transform(regime_scaled)
    else:
        regime_pca = pca.fit_transform(regime_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.get('seed', 42))
    if train_end_date:
        kmeans.fit(regime_pca[mask])
        df["market_regime"] = kmeans.predict(regime_pca)
    else:
        df["market_regime"] = kmeans.fit_predict(regime_pca)
    df["market_regime"] = df["market_regime"].astype(int)

    # Reassign regime labels so that Regime 0 has the lowest ATR
    regime_atr = df.groupby("market_regime")["atr_raw"].mean()
    regime_order = regime_atr.sort_values().index
    regime_map = {old: new for new, old in enumerate(regime_order)}
    df["market_regime"] = df["market_regime"].map(regime_map)

    # Ensure expected cluster count when fixed value provided
    if not auto_clusters:
        found = df["market_regime"].nunique()
        if found != n_clusters:
            raise ValueError(
                f"KMeans produced {found} clusters, expected {n_clusters}"
            )

    # Check cluster balance
    regime_counts = df["market_regime"].value_counts()
    min_cluster_pct = (regime_counts.min() / len(df)) * 100
    if min_cluster_pct < 5:
        print(f"\nWARNING: Unbalanced clusters detected. Smallest cluster is {min_cluster_pct:.1f}% of data")
        print("Cluster distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(df)) * 100
            print(f"Regime {regime}: {count:,} bars ({pct:.1f}%)")
    
    # Drop raw columns before saving
    df = df.drop(columns=[col for col in df.columns if col.endswith('_raw')])

    # Add lags for market_regime
    lag_series = [
        df['market_regime'].shift(lag).rename(f'market_regime_lag_{lag}')
        for lag in [1, 3, 5, 10]
    ]
    df = pd.concat([df] + lag_series, axis=1)

    # Save regime models if a models_dir is provided
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(regime_scaler, models_dir / "regime_scaler.pkl")
        joblib.dump(pca, models_dir / "regime_pca.pkl")
        joblib.dump(kmeans, models_dir / "regime_kmeans.pkl")

    return df

def main():
    """Main function."""
    args = parse_args()
    run_id = get_run_id()
    dirs = make_run_dirs(run_id)
    lookback_years = config.get("prepare.lookback_years", 3)

    raw_start, _ = load_raw_index()
    tz = config.get("market.timezone")
    if raw_start.tzinfo is None:
        raw_start = raw_start.tz_localize(tz)
    else:
        raw_start = raw_start.tz_convert(tz)

    start_date = parse_start_date_arg(args.start_date)
    end_date = parse_end_date_arg(args.end_date)
    train_end_date = parse_end_date_arg(args.train_end_date)

    if end_date and not start_date:
        start_date = (
            pd.to_datetime(end_date) - pd.DateOffset(years=lookback_years)
        ).strftime("%Y-%m-%d")

    if start_date:
        sd_ts = pd.Timestamp(start_date, tz=tz)
        if sd_ts < raw_start:
            start_date = raw_start.strftime("%Y-%m-%d")

    input_dir = args.input_dir or config.get("data.input_dir")
    path = glob.glob(os.path.join(input_dir, "*.csv"))[0]
    df = load_csv(
        path,
        start_date=start_date,
        end_date=end_date,
        strict=args.strict,
    )
    if not end_date and not start_date:
        df = limit_to_recent_years(df, years=lookback_years)
    # Add features
    df = calculate_indicators_and_lags(df)
    
    # Add multi-timeframe features
    print("Adding multi-timeframe features...")
    df, df_15min, df_30min = prepare_multi_timeframe(df)
    
    df, scaler = clean_and_normalize(df, train_end_date=train_end_date)
    df = add_market_regimes(df, config, dirs["models"], train_end_date=train_end_date)
    # Save prepared data and scaler
    output_path = dirs["data"] / "prepared.csv"
    scaler_path = dirs["models"] / "scaler.joblib"
    df.to_csv(output_path)
    joblib.dump(scaler, scaler_path)
    # Print one-liner summary
    prep_summary(df)

if __name__ == "__main__":
    main()
