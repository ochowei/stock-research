import os
import pandas as pd
import numpy as np
import pandas_ta as ta

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def calculate_macro_features(df):
    """
    Calculates L1 Macro Features (Hybrid Defense).
    - Junk_Bond_Stress: HYG / IEF (Falling = Stress)
    - Risk_Off_Flow: IEF / SPY (Rising = Fear)
    """
    # 處理可能的多層索引或重複索引，轉為寬表格 (Index=Timestamp, Cols=Symbol)
    # 假設 df 包含 'close' 欄位
    try:
        # 重置索引以確保 'symbol' 和 'timestamp' 都是欄位 (如果不是的話)
        df_reset = df.reset_index()
        if 'symbol' not in df_reset.columns: # 可能是舊格式，嘗試從 columns 判斷
             return pd.DataFrame()
        
        # Pivot: 取得各標的的收盤價
        closes = df_reset.pivot_table(index='timestamp', columns='symbol', values='close')
    except Exception as e:
        print(f"  [Macro] Error pivoting macro data: {e}")
        return pd.DataFrame()

    features = pd.DataFrame(index=closes.index)
    
    # 1. Junk Bond Stress (Credit Risk)
    # HYG (High Yield) vs IEF (Treasury). Ratio DOWN = Stress UP.
    if 'HYG' in closes.columns and 'IEF' in closes.columns:
        features['Junk_Bond_Stress'] = closes['HYG'] / closes['IEF']
        features['Junk_Bond_Stress_MA20'] = features['Junk_Bond_Stress'].rolling(20).mean()
    else:
        features['Junk_Bond_Stress'] = np.nan
        features['Junk_Bond_Stress_MA20'] = np.nan

    # 2. Risk Off Flow (Sentiment)
    # IEF (Treasury) vs SPY (Equity). Ratio UP = Fear UP.
    if 'IEF' in closes.columns and 'SPY' in closes.columns:
        features['Risk_Off_Flow'] = closes['IEF'] / closes['SPY']
        features['Risk_Off_Flow_MA20'] = features['Risk_Off_Flow'].rolling(20).mean()
    else:
        features['Risk_Off_Flow'] = np.nan
        features['Risk_Off_Flow_MA20'] = np.nan

    return features.dropna(how='all')

def calculate_stock_features(df):
    """
    Calculates technical indicators + V5.3 Microstructure Features.
    """
    # 確保按標的分組
    # Use transform for indicators that operate on a single series
    df['SMA_200'] = df.groupby(level='symbol')['close'].transform(lambda x: ta.sma(x, length=200))
    df['RSI_2'] = df.groupby(level='symbol')['close'].transform(lambda x: ta.rsi(x, length=2))

    # --- V5.3 New Features ---
    # 1. Amihud Illiquidity (Price Impact)
    # Formula: Abs(Ret) / (Price * Volume)
    # 我們取 20 日平均來平滑
    def calc_amihud(group):
        ret = group['close'].pct_change().abs()
        dollar_vol = group['close'] * group['volume']
        # 避免除以零
        dollar_vol = dollar_vol.replace(0, np.nan)
        amihud = (ret / dollar_vol)
        return amihud.rolling(20).mean()

    df['Amihud_Illiquidity'] = df.groupby(level='symbol', group_keys=False).apply(calc_amihud)

    # 2. Down Volume Proportion (Distribution Pressure)
    # Formula: Sum(Vol where Close < Open) / Sum(Total Vol) over 10 days
    def calc_down_vol_prop(group):
        is_down = (group['close'] < group['open']).astype(int)
        down_vol = group['volume'] * is_down
        down_sum = down_vol.rolling(10).sum()
        total_sum = group['volume'].rolling(10).sum()
        return down_sum / total_sum.replace(0, np.nan)

    df['Down_Vol_Prop'] = df.groupby(level='symbol', group_keys=False).apply(calc_down_vol_prop)

    # ATR Calculation (Robust Loop)
    all_atr = []
    for symbol, group in df.groupby(level='symbol'):
        atr = ta.atr(high=group['high'], low=group['low'], close=group['close'], length=14)
        all_atr.append(atr)

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
    print("=== V5.3 Step 2.2: Feature Engineering (L1 Macro + L3 Micro) ===")
    script_dir = get_script_dir()

    data_tracks = ['custom', 'index'] # Process both tracks

    for track in data_tracks:
        print(f"\n--- Processing Track: {track} ---")
        
        # Paths
        base_data_dir = os.path.join(script_dir, 'data', track)
        features_dir = os.path.join(script_dir, 'features') # Common features dir? Or per track?
        # V5.2 似乎是共用 features 資料夾，但這樣會導致不同 track 覆蓋彼此的 stock_features
        # 為了支援雙軌回測，我們應該將 features 也分開存，或者在檔名加上 track
        # 這裡為了相容性，我們將 stock_features 存回 data/{track} 目錄下，或者分開命名
        # 根據 V5.2 結構，features 是共用的，這其實是個潛在問題 (Index 會覆蓋 Custom)
        # V5.3 我們修正這個邏輯：將 features 存入 data/{track}/features/
        
        track_features_dir = os.path.join(base_data_dir, 'features')
        os.makedirs(track_features_dir, exist_ok=True)

        universe_path = os.path.join(base_data_dir, 'universe_daily.parquet')
        market_path = os.path.join(base_data_dir, 'market_indicators.parquet')

        # Outputs
        stock_feat_path = os.path.join(track_features_dir, 'stock_features.parquet')
        macro_feat_path = os.path.join(track_features_dir, 'macro_features.parquet')
        breadth_path = os.path.join(track_features_dir, 'market_breadth.parquet')

        # 1. Process Macro Features (L1)
        if os.path.exists(market_path):
            print("Calculating L1 Macro Features...")
            market_df = pd.read_parquet(market_path)
            macro_features = calculate_macro_features(market_df)
            if not macro_features.empty:
                macro_features.to_parquet(macro_feat_path)
                print(f"Saved: {macro_feat_path}")
                # 簡單檢查
                if 'Junk_Bond_Stress' in macro_features.columns:
                    last_val = macro_features['Junk_Bond_Stress'].iloc[-1]
                    print(f"  - Latest Junk Bond Stress: {last_val:.4f}")
        else:
            print(f"Warning: Market indicators not found at {market_path}")

        # 2. Process Stock Features (L2/L3)
        if os.path.exists(universe_path):
            print("Calculating L3 Microstructure & Stock Features...")
            universe_df = pd.read_parquet(universe_path)
            universe_df.sort_index(inplace=True)

            stock_features = calculate_stock_features(universe_df)
            
            # Save Stock Features
            stock_features.to_parquet(stock_feat_path)
            print(f"Saved: {stock_feat_path}")

            # 3. Calculate Breadth (Context)
            # Breadth is meaningful mostly for Index track, but we calc for both
            print("Calculating Market Breadth...")
            breadth = calculate_market_breadth(stock_features)
            breadth.to_parquet(breadth_path)
            print(f"Saved: {breadth_path}")
            
        else:
            print(f"Warning: Universe data not found at {universe_path}")

    print("\nFeature Engineering Complete.")

if __name__ == '__main__':
    main()