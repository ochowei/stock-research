import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import utils

# 設定繪圖風格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def run_signal_backtest(group_name, tickers):
    print(f"\n=== Backtesting {group_name} (RSI < 10 Strategies) ===")
    
    data_map = utils.fetch_data(tickers)
    if not data_map:
        return None, None

    # 用來儲存所有股票的加總報酬 (Portfolio Level)
    # 我們假設每當有訊號時，資金是等分投入該股票 (Simplified Equal Weight)
    # 這裡我們簡單將所有股票的報酬率平均，模擬持有整個 Pool
    
    all_dates = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='B')
    portfolio_rets = pd.DataFrame(index=all_dates, columns=['Ret_MOO', 'Ret_MOC', 'Ret_Ideal', 'Ret_Limit'])
    portfolio_rets = portfolio_rets.fillna(0.0)
    
    signal_counts = pd.Series(0, index=all_dates)
    
    print("Calculating strategies...")
    for ticker, df in data_map.items():
        res = utils.backtest_strategies(df, limit_buffer=0.98) # Limit = 98% of Prev Close
        
        # 將個股結果加總到 Portfolio (這裡做簡單的累加，最後再除以當天有訊號的股票數，或者直接觀察累加效果)
        # 為了模擬真實 Portfolio，我們假設資金是分散的。
        # 簡單做法：將個股報酬依照日期對齊後取平均 (Mean Return of Active Signals)
        
        # 這裡我們只取有訊號的日子
        res = res.set_index(df.loc[res.index].index) # 確保索引是對的
        
        for col in ['Ret_MOO', 'Ret_MOC', 'Ret_Ideal', 'Ret_Limit']:
            # 注意：這裡使用 add 是為了處理多檔股票同一天有訊號的情況
            # 但若要精確計算 "Portfolio Return"，應該是 Sum(Rets) / Total_Capital_Allocation
            # 這裡簡化指標：我們看 "Average Opportunity Return" (平均每次機會的報酬)
            pass

    # 為了更直觀的比較，我們將所有發生過的交易 (All Trades) 收集起來分析
    # 而不是做 Time Series 的 Equity Curve (因為 RSI < 10 不是每天都有，時間軸會很多空窗)
    
    all_trades = []
    
    for ticker, df in data_map.items():
        res = utils.backtest_strategies(df)
        # 只保留有交易的日子
        active_trades = res[res['Has_Signal'] == 1].copy()
        if not active_trades.empty:
            active_trades['Ticker'] = ticker
            all_trades.append(active_trades)
            
    if not all_trades:
        print("No signals found.")
        return None, None
        
    combined_trades = pd.concat(all_trades)
    
    # 計算各策略的統計數據
    stats = []
    strategies = {
        '1. Benchmark (MOO)': 'Ret_MOO',
        '2. Delayed (MOC)': 'Ret_MOC', 
        '3. Ideal (Night T)': 'Ret_Ideal',
        '4. Limit Buy (0.98)': 'Ret_Limit'
    }
    
    equity_curves = {}
    
    for label, col in strategies.items():
        # 計算績效
        perf = utils.calculate_performance_summary(combined_trades[col])
        perf['Strategy'] = label
        stats.append(perf)
        
        # 模擬資金曲線 (假設每次交易投入固定金額，不複利，單純看累加報酬)
        # 依照時間排序
        sorted_trades = combined_trades.sort_index()
        # 這裡用 cumsum 代表單利累加 (Points)
        equity_curves[label] = sorted_trades[col].cumsum()
        
    stats_df = pd.DataFrame(stats)
    stats_df['Group'] = group_name
    
    # 整理曲線資料
    curves_df = pd.DataFrame(equity_curves)
    # 因為這是交易序列 (Trade Sequence) 而非時間序列，我們 reset index 方便畫圖 (x軸為交易次數)
    curves_df = curves_df.reset_index(drop=True)
    
    return stats_df, curves_df

def main():
    print(">>> Starting Experiment EXP-1.0.1: Signal Decay Verification")
    
    # 載入清單
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    
    # 執行回測
    res_a, curves_a = run_signal_backtest("Group A (Final Pool)", pool_a)
    res_b, curves_b = run_signal_backtest("Group B (Toxic Pool)", pool_b)
    
    # 彙整報告
    if res_a is not None and res_b is not None:
        final_report = pd.concat([res_a, res_b], ignore_index=True)
        
        # 調整欄位順序
        cols = ['Group', 'Strategy', 'Win Rate', 'Avg Trade %', 'Total Return', 'Max Drawdown', 'Trades']
        final_report = final_report[cols]
        
        csv_path = os.path.join(config.OUTPUT_DIR, 'exp_1_0_1_report.csv')
        final_report.to_csv(csv_path, index=False)
        print(f"\n>>> Report saved to: {csv_path}")
        print(final_report)
        
        # 視覺化
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        curves_a.plot(ax=axes[0], title='Group A: Strategy Cumulative Returns (Per Trade)')
        axes[0].set_ylabel('Cumulative Return (Points)')
        axes[0].set_xlabel('Trade Count')
        
        curves_b.plot(ax=axes[1], title='Group B: Strategy Cumulative Returns (Per Trade)')
        axes[1].set_ylabel('Cumulative Return (Points)')
        axes[1].set_xlabel('Trade Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_1_0_1_chart.png'))
        print("\n>>> Chart saved.")

if __name__ == '__main__':
    main()