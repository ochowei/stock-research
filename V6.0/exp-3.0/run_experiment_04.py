import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
import utils

# --- 設定繪圖風格 ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 實驗參數與黑名單 ---
# 這是目前實盤腳本 (daily_gap_signal_generator.py) 使用的黑名單
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI'
]

def calculate_calmar(cagr, mdd):
    """計算 Calmar Ratio"""
    if mdd == 0: return np.nan
    return cagr / abs(mdd)

def calculate_profit_factor(returns):
    """計算獲利因子 (Gross Profit / Gross Loss)"""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0: return np.inf if gains > 0 else 0
    return gains / losses

def prepare_data(df):
    """計算實驗所需的特徵與報酬"""
    df = df.copy()
    
    # 1. 基礎報酬 (Buy and Hold)
    df['Ret_Hold'] = df['Close'].pct_change()
    
    # 2. 隔夜跳空報酬 (Gap Return)
    # Ret_Gap = (Open_t - Close_{t-1}) / Close_{t-1}
    # 這是如果我們在開盤賣出所獲得的收益 (避開日內)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Ret_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # 3. IBS (前一日)
    # IBS = (Close - Low) / (High - Low)
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    # 處理分母為0的情況 (High=Low)
    df['IBS'] = df['IBS'].fillna(0.5)
    df['Prev_IBS'] = df['IBS'].shift(1)
    
    return df

def run_portfolio_test(group_name, all_tickers):
    """
    執行投資組合層級的回測
    1. 過濾黑名單
    2. 下載資料
    3. 計算三種策略的每日報酬 (Portfolio Level)
    """
    # --- 1. 過濾黑名單 ---
    # 為了公平比較，所有策略都在「排除黑名單後」的有效池中運行
    valid_tickers = [t for t in all_tickers if t not in MOMENTUM_BLACKLIST]
    removed_count = len(all_tickers) - len(valid_tickers)
    
    print(f"\n{'='*60}")
    print(f"Processing: {group_name}")
    print(f"Original Count: {len(all_tickers)}")
    print(f"Blacklisted:    {removed_count} (Ex: NVDA, TSLA...)")
    print(f"Effective Pool: {len(valid_tickers)}")
    print(f"{'='*60}")
    
    if not valid_tickers:
        print("No valid tickers after filtering.")
        return None, None

    # --- 2. 下載資料 ---
    data_map = utils.fetch_data(valid_tickers)
    if not data_map: return None, None

    # --- 3. 整理數據矩陣 (Panel Data) ---
    # 我們需要對齊日期
    hold_returns = {}
    gap_returns = {}
    prev_ibs = {}
    
    for ticker, df in data_map.items():
        try:
            processed = prepare_data(df)
            hold_returns[ticker] = processed['Ret_Hold']
            gap_returns[ticker] = processed['Ret_Gap']
            prev_ibs[ticker] = processed['Prev_IBS']
        except Exception:
            continue
            
    df_hold = pd.DataFrame(hold_returns)
    df_gap = pd.DataFrame(gap_returns)
    df_ibs = pd.DataFrame(prev_ibs)
    
    # 對齊索引
    common_idx = df_hold.index.intersection(df_gap.index).intersection(df_ibs.index)
    df_hold = df_hold.reindex(common_idx)
    df_gap = df_gap.reindex(common_idx)
    df_ibs = df_ibs.reindex(common_idx)
    
    # --- 4. 定義策略邏輯 (Vectorized) ---
    
    # (A) Benchmark: Buy & Hold
    # 每日持有，承擔日內波動
    # 使用 fillna(0) 處理停牌，計算平均報酬
    port_bh = df_hold.mean(axis=1).fillna(0)
    
    # (B) Strategy A: Current Live (Gap > 0.5%)
    # 邏輯：Gap > 0.5% -> 賺 Gap (賣開盤)；否則 -> 賺 Hold (續抱)
    mask_live = df_gap > 0.005
    # 根據 Mask 選擇報酬來源
    returns_live = np.where(mask_live, df_gap, df_hold)
    returns_live = pd.DataFrame(returns_live, index=common_idx, columns=df_hold.columns)
    port_live = returns_live.mean(axis=1).fillna(0)
    
    # (C) Strategy B: Live + Smart Filter (Gap > 0.5% AND IBS > 0.8)
    # 邏輯：只有在 Gap > 0.5% 且 昨日收盤很強 (IBS>0.8) 時才賣出；否則續抱
    # 這會過濾掉「殺尾盤後的反彈跳空」，避免過早賣出
    mask_smart = (df_gap > 0.005) & (df_ibs > 0.8)
    returns_smart = np.where(mask_smart, df_gap, df_hold)
    returns_smart = pd.DataFrame(returns_smart, index=common_idx, columns=df_hold.columns)
    port_smart = returns_smart.mean(axis=1).fillna(0)
    
    # --- 5. 計算績效指標 ---
    strategies = {
        '1. Benchmark (B&H)': port_bh,
        '2. Strategy A (Live)': port_live,
        '3. Strategy B (Smart Filter)': port_smart
    }
    
    summary_list = []
    equity_curves = pd.DataFrame()
    
    for name, ret_series in strategies.items():
        # 基礎指標 (來自 utils)
        # 注意：需確保 utils 有 calculate_performance_metrics，若無則需手動計算
        try:
            perf = utils.calculate_performance_metrics(ret_series, name)
        except AttributeError:
            # Fallback if utils doesn't support the new function signature
            equity = (1 + ret_series).cumprod()
            total_ret = equity.iloc[-1] - 1
            cagr = equity.iloc[-1]**(252/len(equity)) - 1
            mdd = (equity / equity.cummax() - 1).min()
            vol = ret_series.std() * np.sqrt(252)
            sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(252)
            perf = {
                'Total Return': total_ret,
                'CAGR': cagr,
                'Max Drawdown': mdd,
                'Volatility (Ann.)': vol,
                'Sharpe Ratio': sharpe
            }

        # 進階指標
        calmar = calculate_calmar(perf['CAGR'], perf['Max Drawdown'])
        pf = calculate_profit_factor(ret_series)
        win_rate = (ret_series > 0).mean()
        
        # 統計觸發次數 (Avg per stock)
        # 對於 Portfolio 來說，我們計算平均每天有多少比例的股票觸發了「賣出開盤」
        if name == '2. Strategy A (Live)':
            trigger_pct = mask_live.mean().mean() * 100 # 平均每日觸發率 %
        elif name == '3. Strategy B (Smart Filter)':
            trigger_pct = mask_smart.mean().mean() * 100
        else:
            trigger_pct = 0.0

        summary_list.append({
            'Group': group_name,
            'Strategy': name,
            'Total Return': perf['Total Return'],
            'CAGR': perf['CAGR'],
            'Max Drawdown': perf['Max Drawdown'],
            'Sharpe Ratio': perf['Sharpe Ratio'],
            'Calmar Ratio': calmar,
            'Profit Factor': pf,
            'Win Rate': win_rate,
            'Avg Avoidance %': trigger_pct  # 平均每天有多少 % 的股票被執行了「賣出開盤」
        })
        
        equity_curves[name] = (1 + ret_series).cumprod()
        
    return pd.DataFrame(summary_list), equity_curves

def main():
    print(">>> Starting Experiment EXP-V6.0-04: Live Filter Validation")
    
    # 1. 載入清單
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_all = list(set(pool_a + pool_b))
    
    # 2. 執行測試
    results = []
    
    # Test 1: Group A (Final Pool)
    res_a, curves_a = run_portfolio_test("Group A (Final)", pool_a)
    if res_a is not None:
        results.append(res_a)
        # 繪圖
        fig, ax = plt.subplots(figsize=(12, 6))
        curves_a.plot(ax=ax, title='Group A: Live Strategy vs Filter Validation')
        ax.set_ylabel('Normalized Wealth')
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_04_equity_GroupA.png'))
        plt.close()

    # Test 2: Group B (Toxic Pool)
    res_b, curves_b = run_portfolio_test("Group B (Toxic)", pool_b)
    if res_b is not None:
        results.append(res_b)
        fig, ax = plt.subplots(figsize=(12, 6))
        curves_b.plot(ax=ax, title='Group B: Live Strategy vs Filter Validation')
        ax.set_ylabel('Normalized Wealth')
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_04_equity_GroupB.png'))
        plt.close()

    # Test 3: All Combined
    res_all, curves_all = run_portfolio_test("All Combined", pool_all)
    if res_all is not None:
        results.append(res_all)
        fig, ax = plt.subplots(figsize=(12, 6))
        curves_all.plot(ax=ax, title='All Combined: Live Strategy vs Filter Validation')
        ax.set_ylabel('Normalized Wealth')
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'exp_04_equity_All.png'))
        plt.close()

    # 3. 輸出總表
    if results:
        final_df = pd.concat(results, ignore_index=True)
        
        # 調整欄位顯示格式
        cols = ['Group', 'Strategy', 'Total Return', 'Max Drawdown', 'Sharpe Ratio', 
                'Calmar Ratio', 'Profit Factor', 'Win Rate', 'Avg Avoidance %']
        
        # 存檔
        csv_path = os.path.join(config.OUTPUT_DIR, 'exp_04_portfolio_summary.csv')
        final_df[cols].to_csv(csv_path, index=False)
        
        print("\n" + "="*80)
        print("EXPERIMENT 04 RESULTS SUMMARY")
        print("="*80)
        print(final_df[cols].to_string(index=False))
        print("-" * 80)
        print(f"Detailed report saved to: {csv_path}")
        print("Check 'output/' folder for equity charts.")

if __name__ == '__main__':
    main()