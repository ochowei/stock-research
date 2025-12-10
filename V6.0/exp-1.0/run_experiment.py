import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import utils

# 設定繪圖風格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 避免中文亂碼問題，使用通用字體

def run_group_analysis(group_name, tickers):
    print(f"\n=== Processing {group_name} (Count: {len(tickers)}) ===")
    
    # 1. 下載資料
    data_map = utils.fetch_data(tickers)
    
    if not data_map:
        print(f"[Error] No data fetched for {group_name}")
        return None, None

    # 2. 計算每一檔股票的 Night/Day/Total Returns
    all_night_rets = pd.DataFrame()
    all_day_rets = pd.DataFrame()
    all_hold_rets = pd.DataFrame()

    for ticker, df in data_map.items():
        res = utils.calculate_decomposed_returns(df)
        all_night_rets[ticker] = res['Night_Ret']
        all_day_rets[ticker] = res['Day_Ret']
        all_hold_rets[ticker] = res['Total_Ret']
    
    # 3. 建構等權重投資組合 (Equal Weighted Portfolio)
    # 取每日所有股票報酬的平均值，代表持有整個 Pool 的績效
    portfolio_night = all_night_rets.mean(axis=1).fillna(0)
    portfolio_day = all_day_rets.mean(axis=1).fillna(0)
    portfolio_hold = all_hold_rets.mean(axis=1).fillna(0)
    
    # 4. 計算績效指標
    metrics_night = utils.calculate_performance_metrics(portfolio_night, 'Night Only')
    metrics_day = utils.calculate_performance_metrics(portfolio_day, 'Day Only')
    metrics_hold = utils.calculate_performance_metrics(portfolio_hold, 'Buy & Hold')
    
    metrics_df = pd.DataFrame([metrics_night, metrics_day, metrics_hold])
    metrics_df['Group'] = group_name
    
    # 準備回測曲線資料 (用於繪圖)
    equity_curves = pd.DataFrame({
        'Night Only': (1 + portfolio_night).cumprod(),
        'Day Only': (1 + portfolio_day).cumprod(),
        'Buy & Hold': (1 + portfolio_hold).cumprod()
    })
    
    return metrics_df, equity_curves

def main():
    print(">>> Starting Experiment EXP-V6.0-01: Overnight Anomaly Verification")
    
    # --- 1. 載入資產清單 ---
    pool_a_tickers = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b_tickers = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_c_tickers = config.BENCHMARK_TICKER
    
    # --- 2. 執行分析 ---
    results_a, curves_a = run_group_analysis("Group A (Final Pool)", pool_a_tickers)
    results_b, curves_b = run_group_analysis("Group B (Toxic Pool)", pool_b_tickers)
    results_c, curves_c = run_group_analysis("Group C (SPY Benchmark)", pool_c_tickers)
    
    # --- 3. 彙整報表 ---
    all_results = pd.concat([results_a, results_b, results_c], ignore_index=True)
    
    # 調整欄位順序
    cols = ['Group', 'Strategy', 'CAGR', 'Sharpe Ratio', 'Volatility (Ann.)', 'Max Drawdown', 'Total Return']
    final_report = all_results[cols]
    
    # 輸出 CSV
    csv_path = os.path.join(config.OUTPUT_DIR, 'exp_01_performance_summary.csv')
    final_report.to_csv(csv_path, index=False)
    print(f"\n>>> Performance Report saved to: {csv_path}")
    print(final_report)

    # --- 4. 視覺化 ---
    
    # 4.1 權益曲線圖 (Equity Curves)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    if curves_a is not None:
        curves_a.plot(ax=axes[0], title='Group A: Final Asset Pool - Equity Curves')
        axes[0].set_ylabel('Normalized Wealth')
    
    if curves_b is not None:
        curves_b.plot(ax=axes[1], title='Group B: Toxic Asset Pool - Equity Curves')
        axes[1].set_ylabel('Normalized Wealth')
        
    if curves_c is not None:
        curves_c.plot(ax=axes[2], title='Group C: SPY Benchmark - Equity Curves')
        axes[2].set_ylabel('Normalized Wealth')
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_01_equity_curves.png'))
    
    # 4.2 波動率對比圖 (Volatility Comparison)
    plt.figure(figsize=(10, 6))
    vol_data = all_results[all_results['Strategy'].isin(['Night Only', 'Day Only'])]
    sns.barplot(data=vol_data, x='Group', y='Volatility (Ann.)', hue='Strategy', palette='viridis')
    plt.title('Volatility Comparison: Day vs Night')
    plt.ylabel('Annualized Volatility')
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_01_volatility_analysis.png'))

    # 4.3 最大回撤對比圖 (Max Drawdown Comparison)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=all_results, x='Group', y='Max Drawdown', hue='Strategy', palette='rocket')
    plt.title('Max Drawdown Comparison')
    plt.ylabel('Max Drawdown (Negative is deeper)')
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_01_drawdown_analysis.png'))
    
    print("\n>>> All charts saved to output directory.")

if __name__ == '__main__':
    main()
