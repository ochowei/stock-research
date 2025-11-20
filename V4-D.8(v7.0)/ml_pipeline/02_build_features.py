import pandas as pd
import numpy as np
import os
import pandas_ta as ta

def load_data():
    """Loads the raw 60m and daily data from the data_acquisition directory."""
    # Get the directory of the current script to build robust paths
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to the data files, now in the same directory
    data_60m_path = os.path.join(data_dir, 'raw_60m.parquet')
    data_daily_path = os.path.join(data_dir, 'raw_daily.parquet')

    # Read the parquet files
    try:
        data_60m = pd.read_parquet(data_60m_path)
        data_daily = pd.read_parquet(data_daily_path)
        print("Successfully loaded raw_60m.parquet and raw_daily.parquet.")
        return data_60m, data_daily
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Could not find data files in {data_dir}.")
        print("Please make sure you have run '01_get_data.py' first.")
        return None, None

def calculate_base_metrics(data_60m):
    """
    Calculates the 9 base metrics for the 60m data, grouped by symbol.
    """
    if data_60m is None:
        print("Input data is None. Cannot calculate base metrics.")
        return None

    # Ensure the index is a DatetimeIndex for time-based operations
    if not isinstance(data_60m.index.get_level_values('timestamp'), pd.DatetimeIndex):
        data_60m.index = data_60m.index.set_levels(pd.to_datetime(data_60m.index.get_level_values('timestamp')), level='timestamp')

    # Define a function to apply to each group
    def calculate_metrics_for_group(group):
        group = group.copy()
        # Sort by timestamp to ensure correct rolling calculations
        group.sort_values('timestamp', inplace=True)
        symbol = group['symbol'].iloc[0] if 'symbol' in group.columns and not group.empty else 'Unknown'

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            group[col] = pd.to_numeric(group[col], errors='coerce')

        # --- FIX: Add forward fill for price data to handle small gaps ---
        price_cols = ['Open', 'High', 'Low', 'Close']
        group[price_cols] = group[price_cols].ffill(limit=2) # Limit to 2 periods to avoid excessive propagation
        # --- END FIX ---

        # Skip calculation if the group is too small for the largest window (20)
        if len(group) < 20:
            for metric in ['RSI', 'ATR', 'MFI', 'Vol_Ratio', 'Body_Pct_ATR', 'Upper_Wick_Pct_ATR', 'Lower_Wick_Pct_ATR', 'Z_Score_20_60m', 'BBWidth_20_60m']:
                group[metric] = np.nan
            return group

        # Individual try-except blocks for each metric
        try:
            group['RSI'] = ta.rsi(group['Close'], length=14)
        except Exception as e:
            print(f"Could not calculate RSI for {symbol}: {e}")
            group['RSI'] = np.nan

        try:
            group['ATR'] = ta.atr(group['High'], group['Low'], group['Close'], length=14)
        except Exception as e:
            print(f"Could not calculate ATR for {symbol}: {e}")
            group['ATR'] = np.nan

        try:
            # Strategy B: MFI/Indicators - Use a "safe" volume to avoid NaN from pandas_ta
            group['Volume'] = group['Volume'].fillna(0)
            safe_volume = group['Volume'].replace(0, 1) # Replace 0 with 1 to avoid division by zero
            group['MFI'] = ta.mfi(group['High'], group['Low'], group['Close'], safe_volume, length=14)
        except Exception as e:
            print(f"Could not calculate MFI for {symbol}: {e}")
            group['MFI'] = np.nan

        try:
            # Strategy A: Vol_Ratio - Use conditional logic for division by zero
            vol_ma = group['Volume'].rolling(window=20).mean()
            group['Vol_Ratio'] = np.where(vol_ma > 0, group['Volume'] / vol_ma, 0)
        except Exception as e:
            print(f"Could not calculate Vol_Ratio for {symbol}: {e}")
            group['Vol_Ratio'] = np.nan

        try:
            group['Body_Pct_ATR'] = (group['Close'] - group['Open']).abs() / group['ATR']
        except Exception as e:
            print(f"Could not calculate Body_Pct_ATR for {symbol}: {e}")
            group['Body_Pct_ATR'] = np.nan

        try:
            group['Upper_Wick_Pct_ATR'] = (group['High'] - np.maximum(group['Open'], group['Close'])) / group['ATR']
        except Exception as e:
            print(f"Could not calculate Upper_Wick_Pct_ATR for {symbol}: {e}")
            group['Upper_Wick_Pct_ATR'] = np.nan

        try:
            group['Lower_Wick_Pct_ATR'] = (np.minimum(group['Open'], group['Close']) - group['Low']) / group['ATR']
        except Exception as e:
            print(f"Could not calculate Lower_Wick_Pct_ATR for {symbol}: {e}")
            group['Lower_Wick_Pct_ATR'] = np.nan

        try:
            rolling_mean_20 = group['Close'].rolling(window=20).mean()
            rolling_std_20 = group['Close'].rolling(window=20).std()
            group['Z_Score_20_60m'] = (group['Close'] - rolling_mean_20) / (rolling_std_20 + 1e-8)
        except Exception as e:
            # More specific error logging
            print(f"Could not calculate Z_Score_20_60m for {symbol}. Error: {e}. Group head:\n{group.head()}")
            group['Z_Score_20_60m'] = np.nan

        try:
            bbands = ta.bbands(group['Close'], length=20)
            if isinstance(bbands, pd.DataFrame) and bbands.shape[1] >= 3:
                # --- FIX: Use positional indexing to avoid column name issues ---
                bbl = bbands.iloc[:, 0]  # Lower band
                bbm = bbands.iloc[:, 1]  # Middle band
                bbu = bbands.iloc[:, 2]  # Upper band

                # Avoid division by zero
                group['BBWidth_20_60m'] = np.where(
                    bbm != 0,
                    (bbu - bbl) / (bbm + 1e-8),
                    0
                )
            else:
                group['BBWidth_20_60m'] = np.nan
        except Exception as e:
            print(f"Could not calculate BBWidth_20_60m for {symbol}: {e}")
            group['BBWidth_20_60m'] = np.nan

        return group

    # Reset index to make 'symbol' a column for easier processing
    data_reset = data_60m.reset_index()

    # Apply the calculation to each symbol group
    print("Calculating base metrics for all symbols...")

    # Use a try-except block to catch errors within the apply function
    try:
        base_metrics_list = []
        for name, group in data_reset.groupby('symbol'):
            base_metrics_list.append(calculate_metrics_for_group(group))

        base_metrics = pd.concat(base_metrics_list)

        # Restore the original index
        base_metrics = base_metrics.set_index(['symbol', 'timestamp'])
        print("Base metrics calculation complete.")
    except TypeError as e:
        print(f"A TypeError occurred during base metrics calculation: {e}")
        # Here you could add more detailed logging, e.g., which symbol caused it
        return None

    return base_metrics

def apply_scheme_c(base_metrics):
    """
    Applies Scheme C logic to separate RTH/ETH and Full/Partial bars.
    """
    if base_metrics is None:
        print("Input data is None. Cannot apply Scheme C.")
        return None, None, None

    # Make sure timestamp is a DatetimeIndex
    if not isinstance(base_metrics.index.get_level_values('timestamp'), pd.DatetimeIndex):
        base_metrics.index.set_levels(pd.to_datetime(base_metrics.index.get_level_values('timestamp')), level='timestamp', inplace=True)

    # 1. Tag RTH vs ETH
    # RTH is 09:30 to 16:00 ET. The data is in US/Eastern timezone.
    time = base_metrics.index.get_level_values('timestamp').time
    base_metrics['session'] = np.where(
        (time >= pd.to_datetime('09:30').time()) & (time < pd.to_datetime('16:00').time()),
        'RTH',
        'ETH'
    )

    # 2. Tag Full vs Partial bars (新邏輯)
    base_metrics['bar_type'] = 'Full' # 預設全部為 Full

    # 找出每天最後一根 ETH K 棒
    base_metrics_reset = base_metrics.reset_index()
    eth_bars = base_metrics_reset[base_metrics_reset['session'] == 'ETH']

    if not eth_bars.empty:
        # 按 (symbol, date) 分組，找到最後的時間戳
        last_eth_indices = eth_bars.groupby(
            [pd.Grouper(key='symbol'), pd.Grouper(key='timestamp', freq='D')]
        )['timestamp'].idxmax() # 找出最後一根 K 棒的索引

        # 將這些索引在 'bar_type' 欄位中標記為 'Partial'
        # 我們需要使用 .loc 來安全地修改原始 DataFrame
        base_metrics.loc[base_metrics_reset.loc[last_eth_indices].set_index(['symbol', 'timestamp']).index, 'bar_type'] = 'Partial'

    print("Scheme C (v2) applied. RTH bars are Full, last ETH bar is Partial.")

    # 3. Split the data
    rth_full_bars = base_metrics[(base_metrics['session'] == 'RTH') & (base_metrics['bar_type'] == 'Full')]
    eth_full_bars = base_metrics[(base_metrics['session'] == 'ETH') & (base_metrics['bar_type'] == 'Full')]
    eth_partial_bars = base_metrics[(base_metrics['session'] == 'ETH') & (base_metrics['bar_type'] == 'Partial')]

    print("Scheme C applied. Data split into RTH/ETH and Full/Partial bars.")

    return rth_full_bars, eth_full_bars, eth_partial_bars

def _calculate_aggregated_features(data, session_suffix):
    """
    Helper function to calculate aggregated features for a given session type (RTH or ETH).
    """
    if data is None or data.empty:
        print(f"No data provided for {session_suffix} session; skipping feature calculation.")
        return pd.DataFrame()

    base_metrics_cols = ['RSI', 'ATR', 'MFI', 'Vol_Ratio', 'Body_Pct_ATR', 'Upper_Wick_Pct_ATR', 'Lower_Wick_Pct_ATR', 'Z_Score_20_60m', 'BBWidth_20_60m']

    # Resample to daily frequency, aggregating the base metrics
    # The key is to group by symbol and the date part of the timestamp
    daily_aggregated_features = data.groupby([pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq='D')])[base_metrics_cols].agg(['mean', 'min', 'max'])

    # Flatten the multi-level column index and rename columns to match the spec
    agg_map = {'mean': 'AVG', 'min': 'MIN', 'max': 'MAX'}
    new_cols = []
    for metric, agg in daily_aggregated_features.columns.values:
        agg_name = agg_map.get(agg, agg.upper())
        # session_suffix will be 'RTH_Full' or 'ETH_Full'
        session_name = session_suffix.replace('_Full', '')
        new_cols.append(f"X_T1_{metric}_60m_{session_name}_{agg_name}_Full")
    daily_aggregated_features.columns = new_cols

    print(f"Calculated aggregated features for {session_suffix} session.")
    return daily_aggregated_features

def calculate_feature_group_a(rth_full_bars):
    """
    Calculates the 27 features for RTH full bars by aggregating daily.
    """
    return _calculate_aggregated_features(rth_full_bars, 'RTH_Full')

def calculate_feature_group_b(eth_full_bars):
    """
    Calculates the 27 features for ETH full bars by aggregating daily.
    """
    return _calculate_aggregated_features(eth_full_bars, 'ETH_Full')

def calculate_feature_group_c(eth_partial_bars):
    """
    Calculates the 9 features for ETH partial bars by taking the last value of the day.
    """
    if eth_partial_bars is None or eth_partial_bars.empty:
        print("No data provided for ETH partial bars; skipping feature calculation.")
        return pd.DataFrame()

    base_metrics_cols = ['RSI', 'ATR', 'MFI', 'Vol_Ratio', 'Body_Pct_ATR', 'Upper_Wick_Pct_ATR', 'Lower_Wick_Pct_ATR', 'Z_Score_20_60m', 'BBWidth_20_60m']

    # Get the last partial bar for each day
    last_partial_features = eth_partial_bars.groupby([pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq='D')])[base_metrics_cols].last()

    # Rename columns to match the required format X_T1_{metric}_60m_ETH_Last_Partial
    last_partial_features.columns = [f"X_T1_{col}_60m_ETH_Last_Partial" for col in last_partial_features.columns]

    print("Calculated features for ETH partial bars.")
    return last_partial_features

def _calculate_beta(data_daily, market_symbol='VOO', window=126):
    """Calculates 6-month (126-day) beta."""
    market_returns = data_daily.loc[market_symbol]['Adj Close'].pct_change()

    betas = {}
    for symbol in data_daily.index.get_level_values('symbol').unique():
        if symbol == market_symbol:
            continue

        asset_returns = data_daily.loc[symbol]['Adj Close'].pct_change()
        rolling_cov = asset_returns.rolling(window=window).cov(market_returns)
        rolling_var = market_returns.rolling(window=window).var()
        beta = rolling_cov / rolling_var
        betas[symbol] = beta

    beta_s = pd.concat(betas)
    beta_s.name = 'X_34_Beta_6M'
    return beta_s

def _calculate_momentum(data_daily, window=126, lag=21):
    """Calculates 6-month momentum, lagged by 1 month."""
    momentum = data_daily.groupby('symbol')['Adj Close'].pct_change(periods=window).shift(lag)
    momentum.name = 'X_35_Momentum_6_1M'
    return momentum

def _calculate_z_score_daily(data_daily, window=126):
    """Calculates 126-day Z-Score."""
    rolling_mean = data_daily.groupby('symbol')['Adj Close'].rolling(window=window).mean().reset_index(0, drop=True)
    rolling_std = data_daily.groupby('symbol')['Adj Close'].rolling(window=window).std().reset_index(0, drop=True)
    z_score = ((data_daily['Adj Close'] - rolling_mean) / rolling_std)
    z_score.name = 'X_36_Z_Score_126_Daily'
    return z_score

def _calculate_amihud_liquidity(data_daily):
    """Calculates Amihud's illiquidity measure."""
    daily_returns = data_daily.groupby('symbol')['Adj Close'].pct_change().abs()
    dollar_volume = data_daily['Adj Close'] * data_daily['Volume']
    amihud = daily_returns / dollar_volume
    amihud.name = 'X_37_Liquidity_Amihud'
    return amihud

def calculate_feature_group_g(data_daily):
    """
    Calculates the 4 contextual features from daily data.
    """
    if data_daily is None or data_daily.empty:
        print("No daily data provided; skipping G-group feature calculation.")
        return pd.DataFrame()

    # Calculate each feature and collect them in a list
    beta = _calculate_beta(data_daily)
    momentum = _calculate_momentum(data_daily)
    z_score = _calculate_z_score_daily(data_daily)
    amihud = _calculate_amihud_liquidity(data_daily)

    # Combine all G-group features into a single DataFrame
    features_g = pd.concat([beta, momentum, z_score, amihud], axis=1)
    features_g.index.names = ['symbol', 'timestamp'] # Ensure index names are set

    print("Calculated G-group features.")
    return features_g

def build_features():
    """
    Main function to build the features.
    """
    # Load data
    data_60m, data_daily = load_data()

    # Calculate base metrics
    base_metrics = calculate_base_metrics(data_60m)

    # Apply Scheme C
    rth_full_bars, eth_full_bars, eth_partial_bars = apply_scheme_c(base_metrics)

    # Calculate feature groups
    features_a = calculate_feature_group_a(rth_full_bars)
    features_b = calculate_feature_group_b(eth_full_bars)
    features_c = calculate_feature_group_c(eth_partial_bars)
    features_g = calculate_feature_group_g(data_daily)

    # Combine features
    # Merge A, B, and C which are already daily
    features_abc = features_a.join(features_b, how='outer').join(features_c, how='outer')

    # Shift G features to align with T-1 timestamp
    features_g_shifted = features_g.groupby(level='symbol').shift(1)

    # --- Bug 修復：在 Join 之前標準化所有索引 ---

    # 1. 標準化 ABC
    if not features_abc.empty:
        # The index names from groupby are 'symbol' and 'timestamp'
        asset_abc = features_abc.index.get_level_values('symbol')
        ts_abc = pd.to_datetime(features_abc.index.get_level_values('timestamp')).date
        features_abc.index = pd.MultiIndex.from_arrays(
            [asset_abc, ts_abc],
            names=['asset', 'T-1_timestamp']
        )

    # 2. 標準化 G
    if not features_g_shifted.empty:
        asset_g = features_g_shifted.index.get_level_values('symbol')
        ts_g = pd.to_datetime(features_g_shifted.index.get_level_values('timestamp')).date
        features_g_shifted.index = pd.MultiIndex.from_arrays(
            [asset_g, ts_g],
            names=['asset', 'T-1_timestamp'] # 確保索引名稱一致
        )
    # --- 修復結束 ---

    # Now, join with G-group features
    final_features = features_abc.join(features_g_shifted, how='left')

    # --- Feature Merge Diagnostics ---
    g_cols = ['X_34_Beta_6M', 'X_35_Momentum_6_1M', 'X_36_Z_Score_126_Daily', 'X_37_Liquidity_Amihud']
    abc_cols = [col for col in final_features.columns if col not in g_cols]

    shape_abc = features_abc.shape
    shape_g = features_g_shifted.shape
    shape_final = final_features.shape

    # Identify rows with any NaNs in each feature group
    nan_in_g_group = final_features[g_cols].isnull().any(axis=1)
    nan_in_abc_group = final_features[abc_cols].isnull().any(axis=1)

    # Calculate the intersection of NaN conditions
    X = (~nan_in_abc_group & nan_in_g_group).sum()  # NaNs only in G
    Y = (nan_in_abc_group & ~nan_in_g_group).sum()  # NaNs only in ABC
    Z = (nan_in_abc_group & nan_in_g_group).sum()   # NaNs in both

    # W is the remainder of rows, which have no NaNs in either group
    W = shape_final[0] - (X + Y + Z)

    # D is the count of rows where ALL columns are NaN
    rows_to_be_dropped = final_features.isnull().all(axis=1).sum()

    print("\n--- Feature Merge Diagnostics (Step 02) ---")
    print(f"Shape of ABC Features (60m-derived): {shape_abc}")
    print(f"Shape of G Features (Daily-derived): {shape_g}")
    print(f"Shape of Final Merged (pre-dropna 'all'): {shape_final}")
    print(f"\nTotal Merged Rows: {shape_final[0]}")
    print(f"Rows with NaNs ONLY in G-Group (X_34-X_37): {X}")
    print(f"Rows with NaNs ONLY in ABC-Group (X_T1...): {Y}")
    print(f"Rows with NaNs in BOTH groups: {Z}")
    print(f"Rows with NO NaNs (Complete): {W}")

    # Verification check
    check_sum = X + Y + Z + W
    print(f"(Check: X + Y + Z + W should equal C) -> {check_sum} == {shape_final[0]} -> {check_sum == shape_final[0]}")

    print(f"\nRows to be dropped by 'dropna(how='all')': {rows_to_be_dropped}")
    print("--- End of Report ---\n")
    # --- End of Diagnostics ---

    # Drop rows with all NaN values, which can result from joins with no matching data
    final_features.dropna(how='all', inplace=True)

    # Final check on the index and sorting
    # (The index names 'asset' and 'T-1_timestamp' are now set before the join)
    final_features.sort_index(inplace=True)


    # Save features
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'features_X_T-1.parquet')
    final_features.to_parquet(output_path)
    print(f"Successfully saved final features to {output_path}")
    print(f"Final features shape: {final_features.shape}")

if __name__ == "__main__":
    build_features()
