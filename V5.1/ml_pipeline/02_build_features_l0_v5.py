import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import json

# --- V5.1 Sector Mapping Configuration ---
SECTOR_MAP = {
    # Technology (XLK)
    'NVDA': 'XLK', 'AMD': 'XLK', 'MSFT': 'XLK', 'AAPL': 'XLK', 'PLTR': 'XLK',
    'QCOM': 'XLK', 'TSM': 'XLK', 'MU': 'XLK', 'LRCX': 'XLK', 'ON': 'XLK',
    'ARM': 'XLK', 'AVGO': 'XLK', 'KLAC': 'XLK', 'INTC': 'XLK', 'PANW': 'XLK',
    'CRWD': 'XLK', 'DDOG': 'XLK', 'NET': 'XLK', 'SNOW': 'XLK', 'FI': 'XLK',
    'ORCL': 'XLK', 'IBM': 'XLK', 'NOW': 'XLK', 'ADBE': 'XLK', 'ANET': 'XLK',
    
    # Consumer Discretionary (XLY)
    'TSLA': 'XLY', 'AMZN': 'XLY', 'MCD': 'XLY', 'SBUX': 'XLY', 'NKE': 'XLY',
    'HD': 'XLY', 'ABNB': 'XLY', 'BKNG': 'XLY', 'UBER': 'XLY', 'DASH': 'XLY',
    'GRAB': 'XLY', 'CAVA': 'XLY', 'BROS': 'XLY',
    
    # Communication Services (Mapped to XLK/XLY/XLU proxy)
    'GOOG': 'XLK', 'GOOGL': 'XLK', 'META': 'XLK', 'NFLX': 'XLY', 'DIS': 'XLY',
    'SPOT': 'XLY', 'T': 'XLU', 'VZ': 'XLU',
    
    # Financials (XLF)
    'JPM': 'XLF', 'BAC': 'XLF', 'V': 'XLF', 'MA': 'XLF', 'GS': 'XLF',
    'MS': 'XLF', 'WFC': 'XLF', 'BLK': 'XLF', 'PYPL': 'XLF', 'SQ': 'XLF',
    'COIN': 'XLF', 'HOOD': 'XLF', 'SOFI': 'XLF', 'AFRM': 'XLF',
    
    # Healthcare (XLV)
    'LLY': 'XLV', 'UNH': 'XLV', 'JNJ': 'XLV', 'PFE': 'XLV', 'MRK': 'XLV',
    'ABBV': 'XLV', 'TMO': 'XLV', 'DHR': 'XLV', 'ISRG': 'XLV', 'VRTX': 'XLV',
    'HIMS': 'XLV', 'TMDX': 'XLV', 'RXRX': 'XLV', 'SDGR': 'XLV',
    
    # Energy (XLE) & Materials (XLB)
    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'OXY': 'XLE', 'SLB': 'XLE',
    'LIN': 'XLB', 'FCX': 'XLB', 'NEM': 'XLB', 'AA': 'XLB', 'CCJ': 'XLB',
    'LAC': 'XLB', 'MP': 'XLB',
    
    # Industrials (XLI)
    'CAT': 'XLI', 'DE': 'XLI', 'BA': 'XLI', 'HON': 'XLI', 'GE': 'XLI',
    'LMT': 'XLI', 'RTX': 'XLI', 'KTOS': 'XLI', 'JOBY': 'XLI', 'ACHR': 'XLI',
    
    # Utilities (XLU)
    'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'VST': 'XLU', 'CEG': 'XLU'
}

def load_data(data_dir):
    """Loads standardized parquet data."""
    universe_path = os.path.join(data_dir, 'universe_daily.parquet')
    market_path = os.path.join(data_dir, 'market_indicators.parquet')
    sector_path = os.path.join(data_dir, 'sector_daily.parquet') # V5.1
    
    if not os.path.exists(universe_path) or not os.path.exists(market_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Run 01_format_data_v5.py first.")
        
    print(f"Loading data from {data_dir}...")
    universe_df = pd.read_parquet(universe_path)
    market_df = pd.read_parquet(market_path)
    
    sector_df = None
    if os.path.exists(sector_path):
        sector_df = pd.read_parquet(sector_path)
        print("  - Sector data loaded.")
    else:
        print("  - Warning: Sector data NOT found. Orthogonal features will be skipped.")
        
    return universe_df, market_df, sector_df

def build_market_features(market_df):
    """
    Calculates macro features for L1 Regime Identification (HMM/IsoForest).
    """
    print("Building Market Features (L0)...")
    
    # Ensure numeric types for market data
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in market_df.columns:
            market_df[col] = pd.to_numeric(market_df[col], errors='coerce')

    # Pivot to wide format for specific columns
    closes = market_df['Close'].unstack(level='symbol')
    
    # Extract Key Assets
    spy = closes['SPY'] if 'SPY' in closes else pd.Series(dtype=float)
    iwo = closes['IWO'] if 'IWO' in closes else pd.Series(dtype=float)
    vix = closes['^VIX'] if '^VIX' in closes else pd.Series(dtype=float)
    tnx = closes['^TNX'] if '^TNX' in closes else pd.Series(dtype=float)
    
    features = pd.DataFrame(index=closes.index)
    
    # 1. Volatility & Returns
    if not iwo.empty:
        features['IWO_Ret'] = iwo.pct_change()
        features['IWO_Vol_21d'] = features['IWO_Ret'].rolling(window=21).std()
        features['IWO_CumRet_21d'] = iwo.pct_change(21)
        
    if not spy.empty:
        features['SPY_Ret'] = spy.pct_change()
        features['SPY_Vol_21d'] = features['SPY_Ret'].rolling(window=21).std()
        features['SPY_CumRet_21d'] = spy.pct_change(21)
    
    # 2. VIX Dynamics
    if not vix.empty:
        features['VIX_Close'] = vix
        features['VIX_Change_1d'] = vix.diff()
        features['VIX_MA_50'] = vix.rolling(window=50).mean()
        features['VIX_Gap'] = vix - features['VIX_MA_50']
        
    # 3. TNX (Rates) Dynamics
    if not tnx.empty:
        features['TNX_Close'] = tnx
        features['TNX_Change_5d'] = tnx.diff(5)
        
    # 4. Divergence (SPY vs IWO)
    if 'SPY_CumRet_21d' in features.columns and 'IWO_CumRet_21d' in features.columns:
        features['SPY_IWO_Div_21d'] = features['SPY_CumRet_21d'] - features['IWO_CumRet_21d']
    
    features.dropna(inplace=True)
    print(f"  - Market Features shape: {features.shape}")
    return features

def build_stock_features(universe_df, market_features, sector_df=None):
    """
    Calculates stock-level features for L2 (Strategy) & L3 (Ranking).
    Includes V5.1 Orthogonal Sector Features.
    """
    print("Building Stock Features (L0)...")
    
    # Pre-process Sector Data if available
    sector_metrics = {}
    if sector_df is not None:
        print("  - Pre-computing Sector RSI and Returns...")
        
        # [Fix] Ensure Sector Data is Numeric
        for col in ['Close']:
            if col in sector_df.columns:
                sector_df[col] = pd.to_numeric(sector_df[col], errors='coerce')

        # Unstack sector close prices
        sec_closes = sector_df['Close'].unstack(level='symbol')
        
        for col in sec_closes.columns:
            # Calculate RSI for each sector ETF
            try:
                rsi = ta.rsi(sec_closes[col], length=14)
                ret = sec_closes[col].pct_change()
                sector_metrics[col] = {'RSI': rsi, 'Ret': ret}
            except Exception as e:
                print(f"    ! Warning: Failed to calc sector metrics for {col}: {e}")
            
    # Process Stock Features
    universe_df = universe_df.sort_index(level=['symbol', 'timestamp'])
    grouped = universe_df.groupby(level='symbol')
    
    feature_list = []
    
    for symbol, group in grouped:
        # Avoid SettingWithCopyWarning
        df = group.copy().reset_index(level='symbol', drop=True)
        
        # --- [CRITICAL FIX] Ensure Numeric Types ---
        # yfinance sometimes returns objects. This fixes 'TypeError: ufunc isnan not supported'
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # -------------------------------------------
        
        # --- 1. Base Strategy Indicators (V5) ---
        try:
            df['RSI_2'] = ta.rsi(df['Close'], length=2)
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['Dist_SMA_200'] = (df['Close'] / df['SMA_200']) - 1
            
            # Bollinger Bands %B
            bbands = ta.bbands(df['Close'], length=20, std=2)
            if bbands is not None:
                df = pd.concat([df, bbands], axis=1)
                bb_p_cols = [c for c in bbands.columns if c.startswith('BBP_')]
                if bb_p_cols:
                    df.rename(columns={bb_p_cols[0]: 'BB_PctB'}, inplace=True)
            
            # Volatility (ATR)
            df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_Norm'] = df['ATR_14'] / df['Close']
            
            # Volume
            df['Vol_MA_20'] = ta.sma(df['Volume'], length=20)
            df['Rel_Vol'] = df['Volume'] / df['Vol_MA_20']
            
            # --- 2. V5.1 Microstructure Features ---
            # Volume Structure: Is volume expanding on down moves?
            is_down = (df['Close'] < df['Open']).astype(int)
            df['Down_Vol_Prop'] = (df['Volume'] * is_down) / (df['Vol_MA_20'] + 1)
            
            # --- 3. V5.1 Orthogonal Sector Features ---
            if sector_metrics:
                sec_ticker = SECTOR_MAP.get(symbol)
                if sec_ticker and sec_ticker in sector_metrics:
                    sec_data = sector_metrics[sec_ticker]
                    
                    if sec_data['RSI'] is not None:
                         # Use index alignment (safe for daily data)
                        df['Sector_RSI_14'] = sec_data['RSI']
                        df['RSI_Divergence'] = df['RSI_14'] - df['Sector_RSI_14']
                        
                        stock_ret = df['Close'].pct_change()
                        df['Rel_Strength_Daily'] = stock_ret - sec_data['Ret']
                    else:
                        df['Sector_RSI_14'] = np.nan
                        df['RSI_Divergence'] = np.nan
                        df['Rel_Strength_Daily'] = np.nan
                else:
                    df['Sector_RSI_14'] = np.nan
                    df['RSI_Divergence'] = np.nan
                    df['Rel_Strength_Daily'] = np.nan

        except Exception as e:
            # Catch individual ticker errors to prevent whole pipeline crash
            print(f"Error calculating features for {symbol}: {e}")
            continue
        
        # --- Final Cleanup ---
        df['symbol'] = symbol
        feature_list.append(df)
        
    # Combine
    full_features = pd.concat(feature_list)
    full_features = full_features.reset_index().set_index(['symbol', 'timestamp']).sort_index()
    
    # Define columns to keep
    base_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI_2', 'RSI_14', 
        'SMA_200', 'Dist_SMA_200', 
        'BB_PctB', 
        'ATR_14', 'ATR_Norm',
        'Rel_Vol'
    ]
    v5_1_cols = [
        'Down_Vol_Prop',
        'Sector_RSI_14', 'RSI_Divergence', 'Rel_Strength_Daily'
    ]
    
    # Filter only existing columns
    all_cols = base_cols + v5_1_cols
    existing_cols = [c for c in all_cols if c in full_features.columns]
    full_features = full_features[existing_cols]
    
    # Drop initial NaNs (SMA 200 causes first 200 rows to be NaN)
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
    universe_df, market_df, sector_df = load_data(DATA_DIR)
    
    # 2. Build Market Features (For L1)
    market_features = build_market_features(market_df)
    market_out_path = os.path.join(FEATURES_DIR, 'market_features_L0.parquet')
    market_features.to_parquet(market_out_path)
    print(f"Saved Market Features to {market_out_path}")
    
    # 3. Build Stock Features (For L2/L3)
    stock_features = build_stock_features(universe_df, market_features, sector_df)
    stock_out_path = os.path.join(FEATURES_DIR, 'stock_features_L0.parquet')
    stock_features.to_parquet(stock_out_path)
    print(f"Saved Stock Features to {stock_out_path}")
    
    print("\nStep 1: L0 Feature Engineering Complete (V5.1).")

if __name__ == "__main__":
    main()