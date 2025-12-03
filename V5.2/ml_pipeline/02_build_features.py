
import os
import pandas as pd
import pandas_ta as ta

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def calculate_stock_features(df):
    """
    Calculates technical indicators (SMA_200, RSI_2, ATR_14) for each stock
    using a performant groupby().transform() approach.
    """
    df['SMA_200'] = df.groupby('symbol')['close'].transform(lambda x: ta.sma(x, length=200))
    df['RSI_2'] = df.groupby('symbol')['close'].transform(lambda x: ta.rsi(x, length=2))
    # ATR requires High, Low, and Close, so we can't use a simple transform.
    # We will use groupby().apply() just for this one.
    atr = df.groupby('symbol').apply(lambda x: ta.atr(high=x['high'], low=x['low'], close=x['close'], length=14)).rename('ATR_14')
    df = df.join(atr, on='symbol')
    return df

def calculate_market_breadth(df):
    """
    Calculates the market breadth (percentage of stocks with Close > SMA_200).
    """
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df.groupby('symbol')['close'].transform(lambda x: ta.sma(x, length=200))

    df['above_sma200'] = (df['close'] > df['SMA_200']).astype(int)

    breadth = df.groupby('timestamp')['above_sma200'].mean().to_frame()
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
