import os
import pandas as pd
import yfinance as yf
# 引入 DataLoader
from data_loader import DataLoader 

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def download_data(tickers, start_date, end_date):
    """Downloads 1d OHLCV data for a list of tickers."""
    # [Optimized] 加入 group_by='ticker' 確保多標的格式統一
    return yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        timeout=30,
        threads=True,
        group_by='ticker' 
    )

def main():
    print("=== V5.3 Step 2.1: Data Expansion (Macro & Tickers) ===")
    
    # --- Configuration ---
    START_DATE = '2015-01-01'
    END_DATE = '2025-11-30'
    
    # 模式設定：'NORMAL', 'TOXIC', 或 'MERGED' (全部)
    # V5.3 核心開發建議使用 'MERGED' 或 'NORMAL' 確保數據完整
    DOWNLOAD_MODE = 'MERGED' 

    script_dir = get_script_dir()
    
    # 初始化 DataLoader
    # V5.3 Update: 這裡應該讀取清洗後的 Final 清單，確保下載的是高品質標的
    # 但為了相容性，若 Final 不存在，DataLoader 預設會讀取 Standard (asset_pool.json)
    # 建議先確認 00_audit_data.py 是否已執行並產生 final_*.json
    
    # 我們嘗試明確指定讀取 final 清單，若無則 fallback 到預設
    final_normal_path = os.path.join(script_dir, 'final_asset_pool.json')
    if os.path.exists(final_normal_path):
        print(">> Detected Final (Cleaned) Asset Pools. Using them.")
        loader = DataLoader(script_dir, normal_file='final_asset_pool.json', toxic_file='final_toxic_asset_pool.json')
    else:
        print(">> Final Asset Pools not found. Using Standard (Uncleaned) pools.")
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

    # Output paths (維持存入 data/custom)
    output_dir = os.path.join(script_dir, 'data', 'custom')
    tickers_output_path = os.path.join(output_dir, 'raw_tickers.pkl')
    macro_output_path = os.path.join(output_dir, 'raw_macro.pkl')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Download Ticker Data ---
    print(f"\n[1/2] Downloading Stock Data for {len(target_tickers)} tickers...")
    daily_tickers_df = download_data(target_tickers, START_DATE, END_DATE)

    # Package and Save
    output_tickers_data = {'daily': daily_tickers_df}
    pd.to_pickle(output_tickers_data, tickers_output_path)
    print(f"Saved stocks to: {tickers_output_path}")

    # --- 2. Download Macro Data (V5.3 Expanded) ---
    # 新增 HYG (高收益債) 與 IEF (7-10年公債) 用於 L1 混合防禦
    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX', 'HYG', 'IEF']
    
    print(f"\n[2/2] Downloading Macro Data ({len(macro_tickers)} symbols)...")
    macro_df = download_data(macro_tickers, START_DATE, END_DATE)
    
    pd.to_pickle(macro_df, macro_output_path)
    print(f"Saved macro to: {macro_output_path}")
    
    print("\nData download complete.")

if __name__ == '__main__':
    main()