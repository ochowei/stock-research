import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import utils

# 設定繪圖風格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def prepare_gap_strategy_data(df):
    """
    準備 EXP-03 所需資料：
    1. Prev_IBS (T-1 訊號)
    2. Night_Ret (T 開盤 Gap)
    3. Day_Ret (T 日內績效 - Target)
    """
    # 取得基礎報酬 (Day_Ret, Night_Ret)
    decomposed = utils.calculate_decomposed_returns(df)
    
    # 計算 IBS 並位移 (取得 T-1 IBS)
    ibs = utils.calculate_ibs(df)
    prev_ibs = ibs.shift(1).rename('Prev_IBS')
    
    # 合併
    data = decomposed.join(prev_ibs)
    
    # 移除空值 (第一筆資料)
    data.dropna(subset=['Day_Ret', 'Night_Ret', 'Prev_IBS'], inplace=True)
    
    return data

def run_gap_analysis(group_name, tickers):
    print(f"\n=== Processing {group_name} (Count: {len(tickers)}) ===")
    
    data_map = utils.fetch_data(tickers)
    if not data_map:
        return None, None

    # 收集所有個股的數據
    all_data_list = []
    
    for ticker, df in data_map.items():
        try:
            processed_df = prepare_gap_strategy_data(df)
            processed_df['Ticker'] = ticker # 標記股票代碼
            all_data_list.append(processed_df)
        except Exception:
            continue
            
    if not all_data_list:
        return None, None

    # 合併成一個巨大的 Panel Data (Date x Ticker)
    # 這裡我們採用「每日平均」的方式來模擬投資組合
    # 先將所有資料垂直合併，再按日期 Groupby
    full_df = pd.concat(all_data_list)
    
    # 定義過濾條件 (Vectorized Filters)
    # 基礎信號: 昨日殺尾盤
    base_signal = full_df['Prev_IBS'] <= 0.2
    
    # Gap 條件
    cond_gap_down = full_df['Night_Ret'] < -0.005  # Gap < -0.5%
    cond_gap_up   = full_df['Night_Ret'] > 0.005   # Gap > +0.5%
    
    # 建立策略 Mask
    mask_gap_down = base_signal & cond_gap_down
    mask_gap_up   = base_signal & cond_gap_up
    mask_all_ibs  = base_signal # 不管 Gap，只要 IBS 低就買 (對照組)
    
    # 計算每日投資組合報酬 (Day_Ret)
    # Groupby Level=0 (Date) -> Mean
    
    # 1. Strategy: Low IBS + Gap Down
    port_gap_down = full_df[mask_gap_down]['Day_Ret'].groupby(level=0).mean().fillna(0)
    
    # 2. Strategy: Low IBS + Gap Up
    port_gap_up = full_df[mask_gap_up]['Day_Ret'].groupby(level=0).mean().fillna(0)
    
    # 3. Strategy: Low IBS (Any Gap) - 用於對比濾網效果
    port_base = full_df[mask_all_ibs]['Day_Ret'].groupby(level=0).mean().fillna(0)
    
    # 對齊日期索引 (確保畫圖一致)
    common_idx = port_base.index
    port_gap_down = port_gap_down.reindex(common_idx).fillna(0)
    port_gap_up = port_gap_up.reindex(common_idx).fillna(0)
    
    # 計算績效
    metrics_down = utils.calculate_performance_metrics(port_gap_down, 'Low IBS + Gap Down (<-0.5%)')
    metrics_up = utils.calculate_performance_metrics(port_gap_up, 'Low IBS + Gap Up (>0.5%)')
    metrics_base = utils.calculate_performance_metrics(port_base, 'Low IBS (Base)')
    
    # 補上 Win Rate (修正版：排除沒交易的日子)
    # 這裡的 Win Rate 定義為：在有進場的日子裡，當日 Day_Ret > 0 的比例
    def calc_true_win_rate(series):
        traded_days = series[series != 0]
        if len(traded_days) == 0: return 0
        return (traded_days > 0).mean()

    metrics_df = pd.DataFrame([metrics_down, metrics_up, metrics_base])
    metrics_df['Group'] = group_name
    metrics_df['Win Rate'] = [
        calc_true_win_rate(port_gap_down),
        calc_true_win_rate(port_gap_up),
        calc_true_win_rate(port_base)
    ]
    
    # 準備曲線
    equity_curves = pd.DataFrame({
        'Gap Down (Washout)': (1 + port_gap_down).cumprod(),
        'Gap Up (Chase)': (1 + port_gap_up).cumprod(),
        'Base (No Filter)': (1 + port_base).cumprod()
    })
    
    return metrics_df, equity_curves

def main():
    print(">>> Starting Experiment EXP-V6.0-03: Gap & Intraday Reversion")
    
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_c = config.BENCHMARK_TICKER
    
    results_a, curves_a = run_gap_analysis("Group A (Final Pool)", pool_a)
    results_b, curves_b = run_gap_analysis("Group B (Toxic Pool)", pool_b)
    results_c, curves_c = run_gap_analysis("Group C (SPY Benchmark)", pool_c)
    
    # 彙整
    all_results = pd.concat([results_a, results_b, results_c], ignore_index=True)
    
    csv_path = os.path.join(config.OUTPUT_DIR, 'exp_02_1_gap_summary.csv')
    all_results.to_csv(csv_path, index=False)
    
    print("\n>>> Performance Report:")
    # 顯示重點欄位
    print(all_results[['Group', 'Strategy', 'CAGR', 'Sharpe Ratio', 'Win Rate', 'Total Return']])
    
    # 繪圖
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    if curves_a is not None:
        curves_a.plot(ax=axes[0], title='Group A: Gap Strategies (Intraday Return)')
        axes[0].set_ylabel('Normalized Wealth')
        
    if curves_b is not None:
        curves_b.plot(ax=axes[1], title='Group B: Gap Strategies (Intraday Return)')
        axes[1].set_ylabel('Normalized Wealth')
        
    if curves_c is not None:
        curves_c.plot(ax=axes[2], title='Group C: SPY Gap Strategies (Intraday Return)')
        axes[2].set_ylabel('Normalized Wealth')
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_02_1_gap_equity.png'))
    print("\n>>> Charts saved.")

if __name__ == '__main__':
    main()