import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import utils

def calculate_calmar(cagr, mdd):
    """計算 Calmar Ratio"""
    if mdd == 0: return np.nan
    return cagr / abs(mdd)

def run_individual_stock_analysis(group_name, tickers):
    print(f"\n>>> Analyzing Individual Stocks for {group_name}...")
    
    # 1. 抓取資料
    data_map = utils.fetch_data(tickers)
    
    results = []
    
    for ticker, df in tqdm(data_map.items()):
        try:
            df = df.copy()
            
            # --- 2. 準備數據 ---
            # Hold Return (Buy & Hold)
            df['Ret_Hold'] = df['Close'].pct_change()
            
            # Gap Return = (Open - Prev_Close) / Prev_Close
            df['Prev_Close'] = df['Close'].shift(1)
            df['Ret_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
            
            # Gap Filter Mask: Gap > 0.5%
            # 當天如果大幅跳空，則視為過熱，執行「賣開盤、買收盤」
            mask_gap = df['Ret_Gap'] > 0.005
            
            # --- 3. 計算策略報酬 ---
            # Strat A: 
            # - 觸發條件 (True): 當日報酬 = Ret_Gap (只賺隔夜跳空，避開日內)
            # - 未觸發 (False): 當日報酬 = Ret_Hold (續抱)
            strat_a_ret = np.where(mask_gap, df['Ret_Gap'], df['Ret_Hold'])
            strat_a_ret = pd.Series(strat_a_ret, index=df.index).dropna()
            
            hold_ret = df['Ret_Hold'].dropna()
            
            # --- 4. 計算績效指標 (包含所有 EXP-03 的 Metrics) ---
            
            # (A) Benchmark: Buy & Hold
            perf_bh = utils.calculate_performance_metrics(hold_ret, "B&H")
            total_ret_bh = perf_bh.get('Total Return', 0)
            cagr_bh = perf_bh.get('CAGR', 0)
            mdd_bh = perf_bh.get('Max Drawdown', 0)
            calmar_bh = calculate_calmar(cagr_bh, mdd_bh)
            # 手動計算勝率
            wr_bh = (hold_ret > 0).mean()
            
            # (B) Strategy: Gap Filter (Strat A)
            perf_strat = utils.calculate_performance_metrics(strat_a_ret, "Strat A")
            total_ret_strat = perf_strat.get('Total Return', 0)
            cagr_strat = perf_strat.get('CAGR', 0)
            mdd_strat = perf_strat.get('Max Drawdown', 0)
            calmar_strat = calculate_calmar(cagr_strat, mdd_strat)
            # 手動計算勝率
            wr_strat = (strat_a_ret > 0).mean()
            
            # (C) 差異分析 (Delta / Improvement)
            total_ret_delta = total_ret_strat - total_ret_bh
            cagr_delta = cagr_strat - cagr_bh
            mdd_improv = mdd_strat - mdd_bh  # MDD 是負值，若由 -0.5 變 -0.3，相減為 +0.2 (改善)
            wr_delta = wr_strat - wr_bh
            
            # 統計迴避資訊
            avoided_days = mask_gap.sum()
            total_days = len(df)
            avoided_pct = (avoided_days / total_days) * 100 if total_days > 0 else 0
            
            results.append({
                'Group': group_name,
                'Ticker': ticker,
                'Total Days': total_days,
                'Avoided Days': avoided_days,
                'Avoided %': round(avoided_pct, 1),
                
                # --- Total Return ---
                'Total Ret (B&H)': total_ret_bh,
                'Total Ret (Strat)': total_ret_strat,
                'Total Ret Delta': total_ret_delta,
                
                # --- CAGR ---
                'CAGR (B&H)': cagr_bh,
                'CAGR (Strat)': cagr_strat,
                'CAGR Delta': cagr_delta,
                
                # --- Max Drawdown ---
                'MDD (B&H)': mdd_bh,
                'MDD (Strat)': mdd_strat,
                'MDD Improv': mdd_improv,
                
                # --- Win Rate ---
                'Win Rate (B&H)': wr_bh,
                'Win Rate (Strat)': wr_strat,
                'Win Rate Delta': wr_delta,
                
                # --- Calmar Ratio ---
                'Calmar (B&H)': calmar_bh,
                'Calmar (Strat)': calmar_strat
            })
            
        except Exception as e:
            print(f"[Error] Failed to process {ticker}: {e}")
            continue
            
    return pd.DataFrame(results)

def main():
    print(">>> Starting Individual Stock Analysis for Strat A (Gap > 0.5%)")
    
    # 建立輸出目錄
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    # 載入清單
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    
    # 執行分析
    res_a = run_individual_stock_analysis("Group A (Final)", pool_a)
    res_b = run_individual_stock_analysis("Group B (Toxic)", pool_b)
    
    # 合併
    all_res = pd.concat([res_a, res_b], ignore_index=True)
    
    # 排序：按照 Calmar Ratio 的改善幅度排序 (這樣可以找出風險回報比提升最多的股票)
    # 或者您也可以改用 'Total Ret Delta' 來排序
    all_res['Calmar Delta'] = all_res['Calmar (Strat)'] - all_res['Calmar (B&H)']
    all_res = all_res.sort_values(by='Calmar Delta', ascending=False)
    
    # 設定欄位順序 (方便閱讀)
    cols = [
        'Group', 'Ticker', 'Avoided Days', 'Avoided %',
        'Total Ret (B&H)', 'Total Ret (Strat)', 'Total Ret Delta',
        'MDD (B&H)', 'MDD (Strat)', 'MDD Improv',
        'Calmar (B&H)', 'Calmar (Strat)',
        'CAGR (B&H)', 'CAGR (Strat)',
        'Win Rate (B&H)', 'Win Rate (Strat)'
    ]
    # 確保只輸出存在的欄位
    cols = [c for c in cols if c in all_res.columns]
    
    # 儲存 CSV
    output_path = os.path.join(config.OUTPUT_DIR, 'exp_03_individual_stock_report.csv')
    all_res[cols].to_csv(output_path, index=False)
    
    print(f"\n>>> Individual Report saved to: {output_path}")
    
    # 顯示前 10 名改善最多的股票 (基於總回報差異)
    print("\nTop 10 Stocks by Total Return Improvement:")
    top_10 = all_res.sort_values(by='Total Ret Delta', ascending=False).head(10)
    print(top_10[['Group', 'Ticker', 'Total Ret (B&H)', 'Total Ret (Strat)', 'Total Ret Delta', 'MDD Improv']])

if __name__ == '__main__':
    main()