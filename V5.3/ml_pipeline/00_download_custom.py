# V5.2/ml_pipeline/00_download_custom.py

import os
import pandas as pd
import yfinance as yf
# [New] 引入 DataLoader
from data_loader import DataLoader 

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def download_data(tickers, start_date, end_date):
    """Downloads 1d OHLCV data for a list of tickers."""
    # [Optimized] 加入 group_by='ticker' 確保多標的格式統一，雖 yf 預設即是 column，但明確指定較佳
    return yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        timeout=30,
        threads=True
    )

def main():
    # --- Configuration ---
    START_DATE = '2015-01-01'
    END_DATE = '2025-11-30'
    
    # 模式設定：'NORMAL', 'TOXIC', 或 'MERGED' (全部)
    DOWNLOAD_MODE = 'MERGED' 

    script_dir = get_script_dir()
    
    # [New] 初始化 DataLoader
    loader = DataLoader(script_dir)

    # 根據模式選擇清單
    if DOWNLOAD_MODE == 'NORMAL':
        target_tickers = loader.get_normal_tickers()
        print(f"--- Mode: Normal ({len(target_tickers)} tickers) ---")
    elif DOWNLOAD_MODE == 'TOXIC':
        target_tickers = loader.get_toxic_tickers()
        print(f"--- Mode: Toxic Stress Test ({len(target_tickers)} tickers) ---")
    else: # MERGED
        target_tickers = loader.get_all_tickers()
        print(f"--- Mode: MERGED ({len(target_tickers)} tickers) ---")

    # Output paths (維持存入 data/custom，讓後續腳本無縫接軌)
    output_dir = os.path.join(script_dir, 'data', 'custom')
    tickers_output_path = os.path.join(output_dir, 'raw_tickers.pkl')
    macro_output_path = os.path.join(output_dir, 'raw_macro.pkl')
    os.makedirs(output_dir, exist_ok=True)

    # --- Download Ticker Data ---
    print(f"Downloading data for {len(target_tickers)} tickers...")
    daily_tickers_df = download_data(target_tickers, START_DATE, END_DATE)

    # Package and Save
    output_tickers_data = {'daily': daily_tickers_df}
    pd.to_pickle(output_tickers_data, tickers_output_path)
    print(f"Saved to: {tickers_output_path}")

    # --- Download Macro Data (不變) ---
    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX']
    print(f"Downloading macro data...")
    macro_df = download_data(macro_tickers, START_DATE, END_DATE)
    pd.to_pickle(macro_df, macro_output_path)
    print("Macro data saved.")

if __name__ == '__main__':
    main()
