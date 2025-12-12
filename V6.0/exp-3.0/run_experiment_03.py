import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 假設 utils 和 config 位於相同目錄或是 Python路徑中
import config
import utils

# --- 設定繪圖風格 ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_calmar_ratio(cagr, max_drawdown):
    """計算 Calmar Ratio"""
    if max_drawdown == 0:
        return np.nan
    return cagr / abs(max_drawdown)

def prepare_smart_hold_data(df):
    """
    計算實驗所需的特徵與報酬
    """
    df = df.copy()
    
    # 1. 基礎報酬 (Buy and Hold)
    # Ret_Hold = (Close_t - Close_{t-1}) / Close_{t-1}
    df['Ret_Hold'] = df['Close'].pct_change()
    
    # 2. 隔夜跳空報酬 (Gap Return)
    # Ret_Gap = (Open_t - Close_{t-1}) / Close_{t-1}
    # 這是如果我們在開盤賣出所獲得的收益
    df['Prev_Close'] = df['Close'].shift(1)
    df['Ret_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # 3. IBS (前一日)
    # 用於判斷 T-1 日是否收盤過強
    # IBS = (Close - Low) / (High - Low)
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['Prev_IBS'] = df['IBS'].shift(1)
    
    return df

def run_smart_hold_analysis(group_name, tickers):
    print(f"\n=== Processing {group_name} (Count: {len(tickers)}) ===")
    
    # 1. 下載資料
    data_map = utils.fetch_data(tickers)
    
    if not data_map:
        print(f"[Error] No data fetched for {group_name}")
        return None, None

    # 2. 建立策略需要的 DataFrame
    # 我們需要對齊所有股票的日期，因此先收集所有 Series
    
    hold_returns = {}
    gap_returns = {}
    prev_ibs = {}
    
    for ticker, df in data_map.items():
        try:
            processed_df = prepare_smart_hold_data(df)
            hold_returns[ticker] = processed_df['Ret_Hold']
            gap_returns[ticker] = processed_df['Ret_Gap']
            prev_ibs[ticker] = processed_df['Prev_IBS']
        except Exception as e:
            continue
            
    # 轉為 DataFrame (Rows: Date, Cols: Tickers)
    df_hold = pd.DataFrame(hold_returns)
    df_gap = pd.DataFrame(gap_returns)
    df_ibs = pd.DataFrame(prev_ibs)
    
    # 對齊索引
    common_index = df_hold.index.intersection(df_gap.index).intersection(df_ibs.index)
    df_hold = df_hold.reindex(common_index)
    df_gap = df_gap.reindex(common_index)
    df_ibs = df_ibs.reindex(common_index)
    
    # 3. 定義策略遮罩 (Masks)
    # True 代表「觸發迴避條件」(當天只賺 Gap，避開日內)
    
    # Strategy A: Gap Filter (Gap > 0.5%)
    # 如果 Gap > 0.5%，則 Action = Sell Open (Return = Gap)，否則 Hold
    mask_gap_filter = df_gap > 0.005
    
    # Strategy B: Smart Filter (Prev IBS > 0.8 AND Gap > 0%)
    # 強勢收盤後又跳空高開 -> 視為過熱，避開日內
    mask_smart_filter = (df_ibs > 0.8) & (df_gap > 0.0)
    
    # 4. 計算每日投資組合報酬 (等權重)
    # 對於每一檔股票，根據 Mask 決定當日報酬是 Gap 還是 Hold
    
    # 基礎策略 (B&H)
    # fillna(0) 處理停牌或缺值
    port_bh = df_hold.mean(axis=1).fillna(0)
    
    # 策略 A (Gap Filter)
    # 如果 Mask 為 True，取 Gap Return；否則取 Hold Return
    returns_strat_a = np.where(mask_gap_filter, df_gap, df_hold)
    # 將 numpy array 轉回 dataframe 以便計算 mean
    returns_strat_a = pd.DataFrame(returns_strat_a, index=df_hold.index, columns=df_hold.columns)
    port_strat_a = returns_strat_a.mean(axis=1).fillna(0)
    
    # 策略 B (Smart Filter)
    returns_strat_b = np.where(mask_smart_filter, df_gap, df_hold)
    returns_strat_b = pd.DataFrame(returns_strat_b, index=df_hold.index, columns=df_hold.columns)
    port_strat_b = returns_strat_b.mean(axis=1).fillna(0)
    
    # 5. 統計迴避次數 (Avoidance Stats)
    # 計算平均每檔股票被「迴避」了幾天
    avg_avoid_days_a = mask_gap_filter.sum().mean()
    avg_avoid_days_b = mask_smart_filter.sum().mean()
    
    # 6. 計算績效指標
    strategies = {
        'Benchmark (B&H)': port_bh,
        'Strat A (Gap>0.5%)': port_strat_a,
        'Strat B (Smart Filter)': port_strat_b
    }
    
    results_list = []
    equity_curves = pd.DataFrame()
    
    for name, returns in strategies.items():
        # [Fix] 使用 utils.calculate_performance_metrics 並傳入策略名稱
        # 因為你的 utils.py 版本可能沒有 calculate_performance_summary
        perf = utils.calculate_performance_metrics(returns, name) 
        
        # [Fix] 手動計算 Win Rate (因為新的 utils 沒有回傳這個欄位)
        n_trades = (returns != 0).sum()
        win_rate = (returns > 0).sum() / n_trades if n_trades > 0 else 0

        # 安全獲取指標 (使用 get 避免 KeyError)
        total_ret = perf.get('Total Return', 0)
        cagr = perf.get('CAGR', 0)
        mdd = perf.get('Max Drawdown', 0)
        
        # 補充 Calmar Ratio
        calmar = calculate_calmar_ratio(cagr, mdd)
        
        # 紀錄
        row = {
            'Pool': group_name,
            'Strategy': name,
            'Total Return': total_ret,
            'CAGR': cagr,
            'Max Drawdown': mdd,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Avg Avoided Days': 0
        }
        
        if name == 'Strat A (Gap>0.5%)':
            row['Avg Avoided Days'] = avg_avoid_days_a
        elif name == 'Strat B (Smart Filter)':
            row['Avg Avoided Days'] = avg_avoid_days_b
            
        results_list.append(row)
        
        # 權益曲線
        equity_curves[name] = (1 + returns).cumprod()
        
    return pd.DataFrame(results_list), equity_curves

def plot_drawdown_curves(equity_curves, group_name, output_dir):
    """繪製水下曲線 (Underwater Plot)"""
    drawdowns = equity_curves / equity_curves.cummax() - 1
    
    plt.figure(figsize=(12, 6))
    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], label=col, linewidth=1.5, alpha=0.8)
        
    plt.title(f'{group_name} - Drawdown Profile')
    plt.ylabel('Drawdown %')
    plt.legend()
    plt.fill_between(drawdowns.index, 0, -1, color='gray', alpha=0.1) # 增加背景對比
    plt.ylim(drawdowns.min().min() * 1.1, 0.05)
    
    filename = f"exp_03_drawdown_{group_name.split(' ')[0]}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    print(">>> Starting Experiment EXP-V6.0-03: Smart Hold & Intraday Avoidance")
    
    # 確保輸出目錄存在
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # --- 1. 載入資產清單 ---
    # 使用與 EXP-01/02 相同的配置
    pool_a_tickers = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b_tickers = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_c_tickers = config.BENCHMARK_TICKER # 例如 ['SPY']
    
    # --- 2. 執行分析 ---
    results_a, curves_a = run_smart_hold_analysis("Group A (Final Pool)", pool_a_tickers)
    results_b, curves_b = run_smart_hold_analysis("Group B (Toxic Pool)", pool_b_tickers)
    results_c, curves_c = run_smart_hold_analysis("Group C (SPY Benchmark)", pool_c_tickers)
    
    # --- 3. 彙整報表 ---
    all_results = pd.concat([results_a, results_b, results_c], ignore_index=True)
    
    # 格式化輸出
    cols_order = ['Pool', 'Strategy', 'Total Return', 'Max Drawdown', 'Calmar Ratio', 'CAGR', 'Win Rate', 'Avg Avoided Days']
    all_results = all_results[cols_order]
    
    csv_path = os.path.join(config.OUTPUT_DIR, 'exp_03_smart_hold_summary.csv')
    all_results.to_csv(csv_path, index=False)
    print(f"\n>>> Performance Report saved to: {csv_path}")
    print(all_results)

    # --- 4. 視覺化 ---
    
    # 4.1 權益曲線比較 (Equity Curves)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    if curves_a is not None:
        curves_a.plot(ax=axes[0], title='Group A: Smart Hold Performance')
        axes[0].set_ylabel('Normalized Wealth')
        plot_drawdown_curves(curves_a, "Group A", config.OUTPUT_DIR)
    
    if curves_b is not None:
        curves_b.plot(ax=axes[1], title='Group B: Smart Hold Performance (Toxic)')
        axes[1].set_ylabel('Normalized Wealth')
        plot_drawdown_curves(curves_b, "Group B", config.OUTPUT_DIR)
        
    if curves_c is not None:
        curves_c.plot(ax=axes[2], title='Group C: Smart Hold Performance (Benchmark)')
        axes[2].set_ylabel('Normalized Wealth')
        plot_drawdown_curves(curves_c, "Group C", config.OUTPUT_DIR)
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_03_equity_comparison.png'))
    plt.close()
    
    print("\n>>> All charts saved.")

if __name__ == '__main__':
    main()