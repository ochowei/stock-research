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
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft JhengHei']  # 嘗試支援中文顯示
plt.rcParams['axes.unicode_minus'] = False

def run_ibs_analysis(group_name, tickers):
    print(f"\n=== Processing {group_name} (Count: {len(tickers)}) ===")
    
    # 1. 下載資料
    data_map = utils.fetch_data(tickers)
    
    if not data_map:
        print(f"[Error] No data fetched for {group_name}")
        return None, None

    # 2. 建構策略矩陣
    # 我們需要兩個大的 DataFrame:
    # - night_ret_df: 所有股票每日的隔夜報酬
    # - prev_ibs_df:  所有股票每日的「前一日 IBS」
    
    night_ret_dict = {}
    prev_ibs_dict = {}

    for ticker, df in data_map.items():
        try:
            strategy_data = utils.prepare_ibs_strategy_data(df)
            night_ret_dict[ticker] = strategy_data['Night_Ret']
            prev_ibs_dict[ticker] = strategy_data['Prev_IBS']
        except Exception as e:
            # 某些股票可能資料太短無法計算，略過
            continue
            
    night_ret_df = pd.DataFrame(night_ret_dict)
    prev_ibs_df = pd.DataFrame(prev_ibs_dict)
    
    # 對齊日期索引 (取聯集)
    common_index = night_ret_df.index.union(prev_ibs_df.index)
    night_ret_df = night_ret_df.reindex(common_index)
    prev_ibs_df = prev_ibs_df.reindex(common_index)

    # 3. 定義過濾條件 (Masks)
    # Condition 1: Weak Tail (IBS <= 0.2) -> 預期隔夜反彈
    mask_weak = prev_ibs_df <= 0.2
    
    # Condition 2: Strong Tail (IBS >= 0.8) -> 預期隔夜動能耗盡或反轉
    mask_strong = prev_ibs_df >= 0.8
    
    # 4. 計算每日投資組合報酬 (等權重)
    # 邏輯：當天只持有符合條件的股票
    
    # 策略 A: 專買殺尾盤 (Weak Tail)
    # 使用 DataFrame 的 mask 功能，不符合條件設為 NaN，然後算 mean (自動忽略 NaN)
    # fillna(0) 是假設如果當天沒有任何股票符合條件，則報酬為 0 (空手)
    portfolio_weak = night_ret_df[mask_weak].mean(axis=1).fillna(0)
    
    # 策略 B: 專買拉尾盤 (Strong Tail)
    portfolio_strong = night_ret_df[mask_strong].mean(axis=1).fillna(0)
    
    # 基準: 持有全部隔夜 (All Night)
    portfolio_benchmark = night_ret_df.mean(axis=1).fillna(0)
    
    # 5. 計算績效指標
    metrics_weak = utils.calculate_performance_metrics(portfolio_weak, 'Weak Tail (IBS<=0.2)')
    metrics_strong = utils.calculate_performance_metrics(portfolio_strong, 'Strong Tail (IBS>=0.8)')
    metrics_base = utils.calculate_performance_metrics(portfolio_benchmark, 'Benchmark (All)')
    
    metrics_df = pd.DataFrame([metrics_weak, metrics_strong, metrics_base])
    metrics_df['Group'] = group_name
    
    # 準備回測曲線資料 (用於繪圖)
    equity_curves = pd.DataFrame({
        'Weak Tail (Buy Dip)': (1 + portfolio_weak).cumprod(),
        'Strong Tail (Chase)': (1 + portfolio_strong).cumprod(),
        'Benchmark (All)': (1 + portfolio_benchmark).cumprod()
    })
    
    # 計算勝率 (Win Rate)
    win_rate_weak = (portfolio_weak > 0).mean()
    win_rate_strong = (portfolio_strong > 0).mean()
    metrics_df.loc[metrics_df['Strategy'] == 'Weak Tail (IBS<=0.2)', 'Win Rate'] = win_rate_weak
    metrics_df.loc[metrics_df['Strategy'] == 'Strong Tail (IBS>=0.8)', 'Win Rate'] = win_rate_strong
    
    return metrics_df, equity_curves

def main():
    print(">>> Starting Experiment EXP-V6.0-02: Tail-End Reversion (IBS Effect)")
    
    # --- 1. 載入資產清單 ---
    # 假設這些檔案存在於 config 指定的路徑
    pool_a_tickers = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b_tickers = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_c_tickers = config.BENCHMARK_TICKER
    
    # --- 2. 執行分析 ---
    results_a, curves_a = run_ibs_analysis("Group A (Final Pool)", pool_a_tickers)
    results_b, curves_b = run_ibs_analysis("Group B (Toxic Pool)", pool_b_tickers)
    results_c, curves_c = run_ibs_analysis("Group C (SPY Benchmark)", pool_c_tickers)
    
    # --- 3. 彙整報表 ---
    all_results = pd.concat([results_a, results_b, results_c], ignore_index=True)
    
    # 輸出 CSV
    csv_path = os.path.join(config.OUTPUT_DIR, 'exp_02_ibs_summary.csv')
    all_results.to_csv(csv_path, index=False)
    print(f"\n>>> Performance Report saved to: {csv_path}")
    print(all_results[['Group', 'Strategy', 'CAGR', 'Sharpe Ratio', 'Win Rate', 'Total Return']])

    # --- 4. 視覺化 ---
    
    # 4.1 策略走勢圖 (Equity Curves)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    if curves_a is not None:
        curves_a.plot(ax=axes[0], title='Group A: Final Asset Pool - IBS Strategies')
        axes[0].set_ylabel('Normalized Wealth')
    
    if curves_b is not None:
        curves_b.plot(ax=axes[1], title='Group B: Toxic Asset Pool - IBS Strategies')
        axes[1].set_ylabel('Normalized Wealth')
        
    if curves_c is not None:
        curves_c.plot(ax=axes[2], title='Group C: SPY Benchmark - IBS Strategies')
        axes[2].set_ylabel('Normalized Wealth')
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_02_reversion_equity.png'))
    
    # 4.2 平均隔夜報酬對比 (Bar Chart)
    # 我們想看不同 IBS 分組下的 "平均每日報酬 (Average Daily Return)"
    # 這裡大略用 CAGR / 252 近似，或者直接重算平均值
    plt.figure(figsize=(10, 6))
    sns.barplot(data=all_results, x='Group', y='Sharpe Ratio', hue='Strategy', palette='coolwarm')
    plt.title('Sharpe Ratio Comparison: Weak Tail vs Strong Tail')
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_02_sharpe_comparison.png'))
    
    print("\n>>> All charts saved to output directory.")

if __name__ == '__main__':
    main()