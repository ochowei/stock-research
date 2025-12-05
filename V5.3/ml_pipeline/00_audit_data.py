import pandas as pd
import numpy as np
import os
import json
import shutil
import matplotlib.pyplot as plt
from data_loader import DataLoader

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def audit_ticker(df, ticker, start_year=2015, end_year=2025):
    """審計單一標的的數據品質"""
    
    # 1. 篩選個股數據
    # df 已經在 main 被 reset_index，所以 symbol 是 column
    ticker_data = df[df['symbol'] == ticker].copy()
    
    if ticker_data.empty:
        return {'valid': False, 'reason': 'No Data', 'days': 0}
    
    # [Fix] 確保 timestamp 是索引，以便使用 resample
    if 'timestamp' in ticker_data.columns:
        ticker_data = ticker_data.set_index('timestamp')
    
    # 檢查是否為 DatetimeIndex (防呆)
    if not isinstance(ticker_data.index, pd.DatetimeIndex):
        # 嘗試轉換，若無法轉換則報錯
        try:
            ticker_data.index = pd.to_datetime(ticker_data.index)
        except:
            return {'valid': False, 'reason': 'Invalid Index', 'days': 0}

    # 2. 檢查覆蓋率 (Coverage)
    # 計算每年的交易日數量
    yearly_counts = ticker_data.resample('YE').size() # pandas 2.2+ use 'YE', older use 'Y'
    if yearly_counts.empty:
         yearly_counts = ticker_data.resample('Y').size()

    # 規則：必須在最近 3 年 (2023-2025) 都有足夠數據，且總天數 > 500 (約2年)
    recent_years = [2023, 2024, 2025]
    
    # 檢查最近年份是否都有數據 (>200天)
    has_recent_data = True
    current_year = pd.Timestamp.now().year
    
    for y in recent_years:
        if y > current_year: continue # 未來年份跳過
        # 構建該年年底的時間戳
        try:
            ts = pd.Timestamp(f'{y}-12-31')
            # 檢查該年是否存在於 yearly_counts
            # 注意: resample index 通常是該年最後一天
            # 我們用較寬鬆的方式檢查：該年是否有數據
            count = yearly_counts[yearly_counts.index.year == y].sum()
            if count <= 200: 
                has_recent_data = False
                break
        except:
            pass
    
    total_valid_days = len(ticker_data)
    
    if not has_recent_data:
        return {'valid': False, 'reason': 'Missing Recent Data', 'days': total_valid_days}
    
    if total_valid_days < 500:
        return {'valid': False, 'reason': 'Insufficient History (<500 days)', 'days': total_valid_days}

    # 3. 檢查流動性 (Liquidity) - 簡易版：檢查是否有過多 0 Volume
    # 確保 volume 欄位存在 (01_format_data 轉成了小寫)
    vol_col = 'volume' if 'volume' in ticker_data.columns else 'Volume'
    
    if vol_col in ticker_data.columns:
        zero_vol_days = (ticker_data[vol_col] == 0).sum()
        if zero_vol_days / total_valid_days > 0.1: # 超過 10% 日子無量
            return {'valid': False, 'reason': 'Illiquid (>10% Zero Vol)', 'days': total_valid_days}
    else:
        print(f"Warning: No volume column found for {ticker}")

    return {'valid': True, 'reason': 'Pass', 'days': total_valid_days}

def audit_and_save_pool(universe_df, script_dir, loader_func, output_filename, pool_name):
    """通用函數：審計特定資產池並儲存結果"""
    print(f"\n--- Auditing {pool_name} Pool ---")
    
    try:
        raw_assets = loader_func()
        if not raw_assets:
            print(f"Warning: {pool_name} pool is empty or not found. Skipping.")
            return [], 0, 0
    except Exception as e:
        print(f"Error loading {pool_name} pool: {e}. Skipping.")
        return [], 0, 0

    print(f"Auditing {len(raw_assets)} tickers from {output_filename}...")

    cleaned_pool = []
    audit_results = []

    for ticker in raw_assets:
        res = audit_ticker(universe_df, ticker)
        res['ticker'] = ticker
        audit_results.append(res)
        
        if res['valid']:
            cleaned_pool.append(ticker)
        else:
            print(f"  [Reject] {ticker}: {res['reason']} (Days: {res['days']})")

    # 存檔
    output_path = os.path.join(script_dir, output_filename)
    
    if os.path.exists(output_path):
        backup_path = os.path.join(script_dir, f"{output_filename}.bak")
        shutil.copy(output_path, backup_path)
        
    with open(output_path, 'w') as f:
        # 讀取原始 json (包含交易所前綴) 以保留格式
        original_json_path = output_path.replace('.json', '.json').replace('_cleaned', '') # Robust way to find original
        if not os.path.exists(original_json_path): original_json_path = output_path
        
        try:
            with open(original_json_path, 'r') as fr:
                original_data = json.load(fr)

            # 建立一個 map (cleaned_ticker -> original_format)
            original_map = {t.split(':')[-1].replace('.', '-'): t for t in original_data}

            # 依據清理後的 ticker list，寫回原始格式
            final_list = [original_map[t] for t in cleaned_pool if t in original_map]
            json.dump(final_list, f, indent=2)
        except Exception:
            # Fallback
            json.dump(cleaned_pool, f, indent=2)

    print(f"\n{pool_name} Audit Complete.")
    print(f"Original: {len(raw_assets)}")
    print(f"Cleaned : {len(cleaned_pool)}")
    print(f"Removed : {len(raw_assets) - len(cleaned_pool)}")
    print(f"Cleaned pool saved to {output_path}")
    
    # Return the detailed results for report generation
    return audit_results, len(raw_assets), len(cleaned_pool)


def main():
    print("=== V5.3 Step 1.1: Dual-Track Data Audit & Cleaning ===")
    script_dir = get_script_dir()

    # --- 1. 載入全量數據 ---
    parquet_path = os.path.join(script_dir, 'data', 'custom', 'universe_daily.parquet')
    print(f"Loading data from {parquet_path}...")

    if not os.path.exists(parquet_path):
        print("Error: universe_daily.parquet not found. Run 01_format_data.py first.")
        return

    universe_df = pd.read_parquet(parquet_path).reset_index()
    print(f"Data Loaded. Shape: {universe_df.shape}")

    # --- 2. 審計 Group B (Cleaned / Target) ---
    loader_group_b = DataLoader(script_dir,
                                normal_file='asset_pool.json',
                                toxic_file='toxic_asset_pool.json')

    all_audit_results = []

    # 審計 Normal Pool
    results_normal, _, _ = audit_and_save_pool(universe_df, script_dir,
                                                 loader_group_b.get_normal_tickers,
                                                 'asset_pool.json',
                                                 'Normal (Group B)')
    for r in results_normal: r['pool'] = 'Normal'
    all_audit_results.extend(results_normal)

    # 審計 Toxic Pool
    results_toxic, _, _ = audit_and_save_pool(universe_df, script_dir,
                                                loader_group_b.get_toxic_tickers,
                                                'toxic_asset_pool.json',
                                                'Toxic (Group B)')
    for r in results_toxic: r['pool'] = 'Toxic'
    all_audit_results.extend(results_toxic)

    # --- 3. 產出合併報告 ---
    if all_audit_results:
        report_df = pd.DataFrame(all_audit_results)
        report_path = os.path.join(script_dir, 'analysis', 'data_audit_report.csv')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report_df.to_csv(report_path, index=False)
        print(f"\nDetailed audit report for Group B saved to {report_path}")

    print("\n--- Audit Summary ---")
    print("Group A (origin_*.json) files remain untouched.")
    print("Group B (*.json) files have been audited and overwritten.")

if __name__ == "__main__":
    main()