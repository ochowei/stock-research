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
    ticker_data = df[df['symbol'] == ticker].copy()
    
    if ticker_data.empty:
        return {'valid': False, 'reason': 'No Data', 'days': 0}
    
    # [Fix] 確保 timestamp 是索引
    if 'timestamp' in ticker_data.columns:
        ticker_data = ticker_data.set_index('timestamp')
    
    if not isinstance(ticker_data.index, pd.DatetimeIndex):
        try:
            ticker_data.index = pd.to_datetime(ticker_data.index)
        except:
            return {'valid': False, 'reason': 'Invalid Index', 'days': 0}

    # 2. 檢查覆蓋率 (Coverage)
    yearly_counts = ticker_data.resample('YE').size() # pandas 2.2+ use 'YE'
    if yearly_counts.empty:
         yearly_counts = ticker_data.resample('Y').size()

    # 規則：最近 3 年 (2023-2025) 需有足夠數據，且總天數 > 500
    recent_years = [2023, 2024, 2025]
    has_recent_data = True
    current_year = pd.Timestamp.now().year
    
    for y in recent_years:
        if y > current_year: continue
        try:
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

    # 3. 檢查流動性 (Liquidity)
    vol_col = 'volume' if 'volume' in ticker_data.columns else 'Volume'
    if vol_col in ticker_data.columns:
        zero_vol_days = (ticker_data[vol_col] == 0).sum()
        if zero_vol_days / total_valid_days > 0.1:
            return {'valid': False, 'reason': 'Illiquid (>10% Zero Vol)', 'days': total_valid_days}

    return {'valid': True, 'reason': 'Pass', 'days': total_valid_days}

def audit_and_save_pool(universe_df, script_dir, loader_func, input_filename, output_filename, pool_name):
    """
    通用函數：審計特定資產池並儲存結果
    input_filename: 原始清單檔名 (用於讀取原始格式)
    output_filename: 輸出清單檔名 (儲存審計後的結果)
    """
    print(f"\n--- Auditing {pool_name} Pool ---")
    print(f"Input: {input_filename} -> Output: {output_filename}")
    
    try:
        # loader_func 會讀取 input_filename (在 main 設定)
        raw_assets = loader_func()
        if not raw_assets:
            print(f"Warning: {pool_name} pool is empty. Skipping.")
            return [], 0, 0
    except Exception as e:
        print(f"Error loading {pool_name} pool: {e}. Skipping.")
        return [], 0, 0

    print(f"Auditing {len(raw_assets)} tickers...")
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

    # 存檔處理
    output_path = os.path.join(script_dir, output_filename)
    input_path = os.path.join(script_dir, input_filename)
    
    # 如果輸出檔已存在，先備份
    if os.path.exists(output_path):
        backup_path = os.path.join(script_dir, f"{output_filename}.bak")
        shutil.copy(output_path, backup_path)
        
    with open(output_path, 'w') as f:
        # 嘗試讀取原始 Input json 以保留交易所前綴格式 (如 NYSE:MP)
        try:
            if os.path.exists(input_path):
                with open(input_path, 'r') as fr:
                    original_data = json.load(fr)
                # 建立 map: clean_ticker -> original_ticker
                original_map = {t.split(':')[-1].replace('.', '-'): t for t in original_data}
                # 轉換
                final_list = [original_map[t] for t in cleaned_pool if t in original_map]
                # 補漏：如果有 clean ticker 沒對應到 (極少見)，則直接用 clean ticker
                for t in cleaned_pool:
                    if t not in original_map:
                        final_list.append(t) # Fallback
            else:
                final_list = cleaned_pool
                
            json.dump(final_list, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to preserve formatting ({e}), saving cleaned tickers only.")
            json.dump(cleaned_pool, f, indent=2)

    print(f"Cleaned pool saved to {output_path}")
    return audit_results, len(raw_assets), len(cleaned_pool)

# --- 生成覆蓋率圖表 ---
def generate_coverage_chart(universe_df, valid_tickers, output_dir):
    print("\nGenerating data coverage chart...")
    
    df_valid = universe_df[universe_df['symbol'].isin(valid_tickers)].copy()
    if df_valid.empty:
        print("No valid data to plot.")
        return

    if 'timestamp' in df_valid.columns:
        df_valid = df_valid.set_index('timestamp')
    df_valid.index = pd.to_datetime(df_valid.index)

    # 計算每年有多少檔標的是「活躍」的
    yearly_counts = df_valid.groupby(df_valid.index.year)['symbol'].nunique()
    
    plt.figure(figsize=(10, 6))
    bars = yearly_counts.plot(kind='bar', color='#4c72b0', zorder=3)
    
    plt.title('Data Coverage Over Time (Valid Tickers)', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Count of Active Tickers')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.xticks(rotation=45)
    
    for i, v in enumerate(yearly_counts):
        plt.text(i, v + 1, str(v), ha='center', va='bottom')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'data_coverage_over_time.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Chart saved to {output_path}")

# --- 生成 Markdown 報告 ---
def generate_markdown_report(audit_df, output_dir):
    print("\nGenerating markdown report...")
    
    total = len(audit_df)
    valid_count = audit_df['valid'].sum()
    invalid_count = total - valid_count
    reasons = audit_df[~audit_df['valid']]['reason'].value_counts()
    
    md_content = f"""# V5.3 Data Selection Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Status:** #generated #report

## 1. Executive Summary

* **Total Tickers Audited:** {total}
* **Final Valid Tickers:** {valid_count} ({(valid_count/total)*100:.1f}%)
* **Rejected Tickers:** {invalid_count} ({(invalid_count/total)*100:.1f}%)

## 2. Pool Breakdown

| Pool | Total | Valid | Rejected | Pass Rate |
| :--- | :---: | :---: | :---: | :---: |
"""
    
    for pool in audit_df['pool'].unique():
        pool_stats = audit_df[audit_df['pool'] == pool]
        p_total = len(pool_stats)
        p_valid = pool_stats['valid'].sum()
        p_reject = p_total - p_valid
        p_rate = (p_valid / p_total) * 100 if p_total > 0 else 0
        md_content += f"| **{pool}** | {p_total} | {p_valid} | {p_reject} | {p_rate:.1f}% |\n"

    md_content += """
## 3. Rejection Analysis

| Reason | Count | Share |
| :--- | :---: | :---: |
"""
    for reason, count in reasons.items():
        share = (count / invalid_count) * 100 if invalid_count > 0 else 0
        md_content += f"| {reason} | {count} | {share:.1f}% |\n"

    md_content += "\n## 4. Detailed Rejection List\n"
    
    for pool in audit_df['pool'].unique():
        md_content += f"\n### {pool} Pool - Rejected\n"
        rejects = audit_df[(audit_df['pool'] == pool) & (~audit_df['valid'])]
        if rejects.empty:
            md_content += "*None*\n"
        else:
            for _, row in rejects.iterrows():
                md_content += f"* **{row['ticker']}**: {row['reason']} (Data Days: {row['days']})\n"

    report_path = os.path.join(output_dir, 'data_selection_report.md')
    with open(report_path, 'w') as f:
        f.write(md_content)
    print(f"Report saved to {report_path}")

def main():
    print("=== V5.3 Step 1.1: Dual-Track Data Audit & Cleaning ===")
    script_dir = get_script_dir()
    output_dir = os.path.join(script_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 載入全量數據
    parquet_path = os.path.join(script_dir, 'data', 'custom', 'universe_daily.parquet')
    if not os.path.exists(parquet_path):
        print("Error: universe_daily.parquet not found. Run 01_format_data.py first.")
        return

    universe_df = pd.read_parquet(parquet_path).reset_index()
    print(f"Data Loaded. Shape: {universe_df.shape}")

    # 2. 執行審計 (Input -> Output)
    # 初始化 DataLoader，指向「Input」檔案
    input_normal = 'asset_pool.json'
    input_toxic = 'toxic_asset_pool.json'
    
    output_normal = 'final_asset_pool.json'
    output_toxic = 'final_toxic_asset_pool.json'

    # 這裡我們使用 DataLoader 僅作為讀取 Input 檔案的工具
    loader = DataLoader(script_dir, normal_file=input_normal, toxic_file=input_toxic)

    all_audit_results = []
    valid_tickers_all = []

    # 審計 Normal Pool
    results_normal, _, _ = audit_and_save_pool(
        universe_df, script_dir,
        loader.get_normal_tickers, 
        input_normal,   # Input File (Source of Truth for formatting)
        output_normal,  # Output File (Final list)
        'Normal (Custom)'
    )
    for r in results_normal: 
        r['pool'] = 'Normal'
        if r['valid']: valid_tickers_all.append(r['ticker'])
    all_audit_results.extend(results_normal)

    # 審計 Toxic Pool
    results_toxic, _, _ = audit_and_save_pool(
        universe_df, script_dir,
        loader.get_toxic_tickers, 
        input_toxic,    # Input File
        output_toxic,   # Output File
        'Toxic (Custom)'
    )
    for r in results_toxic: 
        r['pool'] = 'Toxic'
        if r['valid']: valid_tickers_all.append(r['ticker'])
    all_audit_results.extend(results_toxic)

    # 3. 產出報告與圖表
    if all_audit_results:
        report_df = pd.DataFrame(all_audit_results)
        csv_path = os.path.join(output_dir, 'data_audit_report.csv')
        report_df.to_csv(csv_path, index=False)
        print(f"\nDetailed CSV audit report saved to {csv_path}")
        
        generate_markdown_report(report_df, output_dir)
        generate_coverage_chart(universe_df, valid_tickers_all, output_dir)

    print("\n--- Audit Summary ---")
    print(f"Processed: {input_normal} -> {output_normal}")
    print(f"Processed: {input_toxic} -> {output_toxic}")
    print(f"Check {output_dir} for reports and charts.")

if __name__ == "__main__":
    main()