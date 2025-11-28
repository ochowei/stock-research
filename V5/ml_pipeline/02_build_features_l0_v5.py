import pandas as pd
import pandas_ta as ta
import numpy as np
import os

def load_data(data_dir):
    """Loads standardized parquet data."""
    universe_path = os.path.join(data_dir, 'universe_daily.parquet')
    market_path = os.path.join(data_dir, 'market_indicators.parquet')
    
    if not os.path.exists(universe_path) or not os.path.exists(market_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Run 01_format_data_v5.py first.")
        
    print(f"Loading data from {data_dir}...")
    universe_df = pd.read_parquet(universe_path)
    market_df = pd.read_parquet(market_path)
    return universe_df, market_df

def build_market_features(market_df):
    """
    Calculates macro features for L1 Regime Identification (HMM/IsoForest).
    Key Features: IWO Volatility, SPY-IWO Divergence, VIX Changes.
    """
    print("Building Market Features (L0)...")
    
    # Pivot to wide format for easier calculation between assets
    # Index: timestamp, Columns: (symbol, field) -> we want just Close for now
    closes = market_df['Close'].unstack(level='symbol')
    
    # 1. Extract Key Assets
    spy = closes['SPY']
    iwo = closes['IWO']
    vix = closes['^VIX']
    tnx = closes['^TNX']
    
    features = pd.DataFrame(index=closes.index)
    
    # 2. Volatility Features
    # IWO (Russell 2000 Growth) represents high-beta/risk appetite
    features['IWO_Ret'] = iwo.pct_change()
    features['IWO_Vol_21d'] = features['IWO_Ret'].rolling(window=21).std()
    
    features['SPY_Ret'] = spy.pct_change()
    features['SPY_Vol_21d'] = features['SPY_Ret'].rolling(window=21).std()
    
    # VIX Dynamics
    features['VIX_Close'] = vix
    features['VIX_Change_1d'] = vix.diff()
    features['VIX_MA_50'] = vix.rolling(window=50).mean()
    features['VIX_Gap'] = vix - features['VIX_MA_50'] # Positive means elevated fear
    
    # TNX (Rates) Dynamics
    features['TNX_Close'] = tnx
    features['TNX_Change_5d'] = tnx.diff(5)
    
    # 3. Divergence Features (The "Canary in the Coal Mine")
    # When SPY (Large Caps) rises but IWO (Small Growth) falls -> Market Breadth Weakening
    # Calculate rolling cumulative returns for divergence
    features['SPY_CumRet_21d'] = spy.pct_change(21)
    features['IWO_CumRet_21d'] = iwo.pct_change(21)
    features['SPY_IWO_Div_21d'] = features['SPY_CumRet_21d'] - features['IWO_CumRet_21d']
    
    # Drop NaN (initial rolling windows)
    features.dropna(inplace=True)
    
    print(f"  - Market Features shape: {features.shape}")
    return features

def build_stock_features(universe_df, market_features):
    """
    Calculates stock-level features for L2 Strategy (Mean Reversion) & L3 Meta-Labeling.
    Key Features: RSI, SMA Distance, Bollinger Bands, Relative Strength.
    """
    print("Building Stock Features (L0)...")
    
    # Ensure sorted by symbol then timestamp for correct rolling calc
    universe_df = universe_df.sort_index(level=['symbol', 'timestamp'])
    
    # Use groupby apply for efficiency on large dataset? 
    # Actually, direct pandas-ta on groupby object is cleaner.
    
    feature_list = []
    
    # We process by symbol
    # Note: For very large datasets, we might optimize this. For ~100 tickers, loop/groupby is fine.
    grouped = universe_df.groupby(level='symbol')
    
    for symbol, group in grouped:
        # Avoid SettingWithCopyWarning
        df = group.copy().reset_index(level='symbol', drop=True)
        
        # --- Mean Reversion Indicators ---
        
        # 1. RSI (2) - Extreme short term
        df['RSI_2'] = ta.rsi(df['Close'], length=2)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        # 2. SMA 200 - Trend Filter
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['Dist_SMA_200'] = (df['Close'] / df['SMA_200']) - 1
        
        # 3. Bollinger Bands (20, 2)
        bbands = ta.bbands(df['Close'], length=20, std=2)
        # pandas_ta returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        # BBP is %B (Percent Bandwidth), which is what we want
        if bbands is not None:
             df = pd.concat([df, bbands], axis=1)
             # Rename strictly needed columns for clarity
             df.rename(columns={'BBP_20_2.0': 'BB_PctB'}, inplace=True)

        # 4. Volatility (ATR)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Norm'] = df['ATR_14'] / df['Close'] # Normalized ATR
        
        # 5. Volume
        df['Vol_MA_20'] = ta.sma(df['Volume'], length=20)
        df['Rel_Vol'] = df['Volume'] / df['Vol_MA_20']
        
        # --- Add Context ---
        df['symbol'] = symbol
        feature_list.append(df)
    
    # Combine all
    full_features = pd.concat(feature_list)
    
    # Restore Index
    full_features = full_features.reset_index().set_index(['symbol', 'timestamp']).sort_index()
    
    # Keep only computed features + OHLCV needed for strategy
    cols_to_keep = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI_2', 'RSI_14', 
        'SMA_200', 'Dist_SMA_200', 
        'BB_PctB', 
        'ATR_14', 'ATR_Norm',
        'Rel_Vol'
    ]
    # Filter columns ensuring they exist
    existing_cols = [c for c in cols_to_keep if c in full_features.columns]
    full_features = full_features[existing_cols]
    
    # Drop initial NaNs (from SMA200 mostly)
    # Note: This might drop first 200 days for every ticker.
    full_features.dropna(subset=['SMA_200'], inplace=True)
    
    print(f"  - Stock Features shape: {full_features.shape}")
    return full_features

def main():
    # --- Setup Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    FEATURES_DIR = os.path.join(SCRIPT_DIR, 'features')
    
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # 1. Load Data
    universe_df, market_df = load_data(DATA_DIR)
    
    # 2. Build Market Features (For L1)
    market_features = build_market_features(market_df)
    market_out_path = os.path.join(FEATURES_DIR, 'market_features_L0.parquet')
    market_features.to_parquet(market_out_path)
    print(f"Saved Market Features to {market_out_path}")
    
    # 3. Build Stock Features (For L2/L3)
    stock_features = build_stock_features(universe_df, market_features)
    stock_out_path = os.path.join(FEATURES_DIR, 'stock_features_L0.parquet')
    stock_features.to_parquet(stock_out_path)
    print(f"Saved Stock Features to {stock_out_path}")
    
    print("\nStep 1: L0 Feature Engineering Complete.")

if __name__ == "__main__":
    main()