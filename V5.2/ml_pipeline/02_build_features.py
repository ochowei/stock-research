
import os
import pandas as pd
import pandas_ta as ta

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def calculate_stock_features(df):
    """
    Calculates technical indicators (SMA_200, RSI_2, ATR_14) for each stock.
    This version uses a robust iteration strategy for ATR to avoid complex
    MultiIndex issues.
    """
    # Use transform for indicators that operate on a single series (performant).
    df['SMA_200'] = df.groupby(level='symbol')['close'].transform(lambda x: ta.sma(x, length=200))
    df['RSI_2'] = df.groupby(level='symbol')['close'].transform(lambda x: ta.rsi(x, length=2))

    # For ATR, which requires multiple columns, we iterate through groups to avoid
    # complex and sometimes buggy `groupby.apply()` index alignment issues.
    # This is more explicit and robust.
    all_atr = []
    # Iterate through each symbol group. `df` is already sorted by index.
    for symbol, group in df.groupby(level='symbol'):
        # Calculate ATR for the current symbol's group. The index is preserved.
        atr = ta.atr(high=group['high'], low=group['low'], close=group['close'], length=14)
        all_atr.append(atr)

    # Concatenate all the resulting ATR series. Because the index of each series
    # is a slice of the original, concat stitches them back together perfectly.
    if all_atr:
        atr_series = pd.concat(all_atr)
        df['ATR_14'] = atr_series
    else:
        df['ATR_14'] = pd.NA

    return df

def calculate_market_breadth(df):
    """
    Calculates the market breadth (percentage of stocks with Close > SMA_200).
    """
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df.groupby(level='symbol')['close'].transform(lambda x: ta.sma(x, length=200))

    df['above_sma200'] = (df['close'] > df['SMA_200']).astype(int)

    breadth = df.groupby(level='timestamp')['above_sma200'].mean().to_frame()
    breadth.columns = ['market_breadth']
    return breadth

def main():
    """
    Generates stock features for both 'custom' and 'index' tracks,
    and calculates market breadth from the 'index' track.
    """
    script_dir = get_script_dir()
    v5_2_dir = os.path.abspath(os.path.join(script_dir, '..'))

    data_dir = os.path.join(v5_2_dir, 'data')
    features_dir = os.path.join(v5_2_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    stock_features_output_path = os.path.join(features_dir, 'stock_features.parquet')
    market_breadth_output_path = os.path.join(features_dir, 'market_breadth.parquet')

    all_features = []

    for track in ['custom', 'index']:
        print(f"--- Processing features for track: {track} ---")
        universe_path = os.path.join(data_dir, track, 'universe_daily.parquet')

        if not os.path.exists(universe_path):
            print(f"Warning: Universe data not found for track '{track}' at {universe_path}")
            continue

        df = pd.read_parquet(universe_path)

        # Ensure the dataframe is sorted for predictable processing
        df.sort_index(inplace=True)

        features = calculate_stock_features(df)
        all_features.append(features)

        if track == 'index':
            print("Calculating market breadth from index data...")
            market_breadth = calculate_market_breadth(features)
            print(f"Saving market breadth data to: {market_breadth_output_path}")
            market_breadth.to_parquet(market_breadth_output_path)
            print("Market breadth data saved successfully.")

    if all_features:
        combined_features = pd.concat(all_features)
        combined_features = combined_features[~combined_features.index.duplicated(keep='first')]
        print(f"Saving combined stock features to: {stock_features_output_path}")
        combined_features.to_parquet(stock_features_output_path)
        print("Stock features saved successfully.")

if __name__ == '__main__':
    main()
