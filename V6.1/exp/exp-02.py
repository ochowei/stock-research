import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 1. 實驗配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 假設資源檔在 V6.0/resource，請根據實際路徑調整
RESOURCE_DIR = os.path.join(BASE_DIR, '..', '..', 'V6.0', 'resource') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 時間設定 (保留 2023 作為指標暖身期)
DATA_START = '2023-01-01'
TEST_START = '2024-01-01'
TEST_END   = '2025-12-31'

# 策略參數
THRESHOLDS = {
    'Fixed 0.5% (Baseline)': {'type': 'fixed', 'value': 0.005},
    'Fixed 1.0% (High)':     {'type': 'fixed', 'value': 0.010},
    'Dynamic ATR (k=0.2)':   {'type': 'dynamic', 'k': 0.2},
    'Dynamic ATR (k=0.3)':   {'type': 'dynamic', 'k': 0.3}
}

# --- 2. 工具函數 ---

def load_tickers():
    """讀取 V6.0 的資產池 (Naive Portfolio 概念)"""
    files = ['2025_final_asset_pool.json', '2025_final_toxic_asset_pool.json']
    tickers = []
    
    for f in files:
        path = os.path.join(RESOURCE_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                raw = json.load(json_file)
                # 清洗 "NYSE:MP" -> "MP"
                clean = [t.split(':')[-1].strip() for t in raw]
                tickers.extend(clean)
        else:
            print(f"[Warning] Resource file not found: {path}")
    
    # 去重並清洗 (BRK.B -> BRK-B)
    clean_tickers = [t.replace('.', '-').strip() for t in tickers]
    # 排除大盤指數，專注於個股
    indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']
    return sorted(list(set([t for t in clean_tickers if t not in indices])))

def fetch_data(tickers):
    """下載 OHLCV 數據"""
    print(f"Downloading data for {len(tickers)} tickers ({DATA_START} ~ {TEST_END})...")
    try:
        data = yf.download(
            tickers, 
            start=DATA_START, 
            end=TEST_END, 
            interval='1d', 
            auto_adjust=True, 
            progress=True,
            timeout=60,
            threads=True
        )
    except Exception as e:
        print(f"[Critical Error] Batch download failed: {e}")
        return {}
    
    # 處理 yfinance 格式 (MultiIndex columns)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Pandas 2.x
            data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        except TypeError:
            # Pandas 1.x
            data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            
        data_dict = {}
        for ticker, group in data.groupby('Ticker'):
            df = group.set_index('Date').sort_index()
            # 簡單過濾
            if df.empty or df['Close'].isna().all(): continue
            data_dict[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        return data_dict
    return {}

def calculate_strategy_returns(df, strategies):
    """
    計算單一股票在不同策略下的每日報酬
    """
    df = df.copy()
    
    # 1. 基礎指標
    df['Prev_Close'] = df['Close'].shift(1)
    df['Ret_Hold'] = df['Close'].pct_change()
    df['Ret_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # 2. 計算 ATR (14)
    # 使用 pandas_ta
    try:
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR'] / df['Prev_Close'] # 相對於昨收的波動率 %
    except Exception:
        # 若數據不足無法計算 ATR
        df['ATR_Pct'] = np.nan

    # 3. 僅保留測試期數據
    mask_test = (df.index >= TEST_START) & (df.index <= TEST_END)
    df_test = df.loc[mask_test].copy()
    
    if df_test.empty: return None

    # 4. 計算各策略報酬
    results = pd.DataFrame(index=df_test.index)
    
    # Benchmark: Buy & Hold
    results['Buy & Hold'] = df_test['Ret_Hold']
    
    for name, params in strategies.items():
        # 決定觸發門檻
        if params['type'] == 'fixed':
            threshold = params['value']
        elif params['type'] == 'dynamic':
            # 動態門檻 = k * ATR%
            threshold = params['k'] * df_test['ATR_Pct']
        else:
            threshold = 999 # Should not happen
            
        # 判斷訊號: Gap > Threshold -> Sell Open (Earn Gap Only)
        # 注意: 若 threshold 是 Series (動態)，pandas 會自動對齊 index 比較
        is_signal = df_test['Ret_Gap'] > threshold
        
        # 策略邏輯:
        # Signal True  -> Return = Ret_Gap (賣開盤，避開日內)
        # Signal False -> Return = Ret_Hold (續抱)
        strat_ret = np.where(is_signal, df_test['Ret_Gap'], df_test['Ret_Hold'])
        
        results[name] = strat_ret
        
        # 紀錄觸發次數 (用於統計)
        results[f'{name}_Trigger'] = is_signal.astype(int)

    return results

def calculate_metrics(equity_curve):
    """計算 CAGR, Sharpe, MaxDD"""
    total_ret = equity_curve.iloc[-1] - 1
    
    # CAGR
    n_years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
    
    # MaxDD
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Sharpe (Simplified, Rf=0)
    daily_ret = equity_curve.pct_change().fillna(0)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    return total_ret, cagr, max_dd, sharpe

# --- 3. 主程式 ---

def main():
    print(f"=== EXP-V6.1-02: Gap Threshold Analysis ({TEST_START} ~ {TEST_END}) ===")
    
    # 1. 準備數據
    tickers = load_tickers()
    if not tickers:
        print("Error: No tickers found. Check resource directory.")
        return
        
    data_map = fetch_data(tickers)
    print(f"Valid Data: {len(data_map)} tickers")
    
    # 2. 執行回測
    # 我們將所有股票的報酬加總平均 (Equal Weight Portfolio)
    
    portfolio_returns = {k: [] for k in ['Buy & Hold'] + list(THRESHOLDS.keys())}
    trigger_counts = {k: [] for k in THRESHOLDS.keys()}
    
    valid_ticker_count = 0
    
    for ticker, df in data_map.items():
        res = calculate_strategy_returns(df, THRESHOLDS)
        if res is None: continue
        
        valid_ticker_count += 1
        
        # 收集每日報酬 (之後做平均)
        for col in portfolio_returns.keys():
            portfolio_returns[col].append(res[col])
            
        # 收集觸發次數
        for name in THRESHOLDS.keys():
            trigger_counts[name].append(res[f'{name}_Trigger'].sum())

    print(f"Backtested on {valid_ticker_count} tickers.")
    
    # 3. 聚合投資組合 (Portfolio Aggregation)
    # 將 List of Series 轉為 DataFrame (Cols=Tickers, Rows=Date) 然後取 Mean
    
    final_stats = []
    equity_curves = pd.DataFrame()
    
    for name in portfolio_returns.keys():
        # Concat all tickers' returns for this strategy
        all_rets = pd.concat(portfolio_returns[name], axis=1)
        # Average across tickers (Equal Weight)
        port_daily_ret = all_rets.mean(axis=1).fillna(0)
        
        # 建立權益曲線
        equity = (1 + port_daily_ret).cumprod()
        equity_curves[name] = equity
        
        # 計算指標
        tot, cagr, mdd, sharpe = calculate_metrics(equity)
        
        # 計算平均觸發率 (Trigger %)
        avg_trigger_pct = 0
        if name in trigger_counts:
            total_days = len(port_daily_ret)
            # Sum of triggers across all stocks / (Num Stocks * Total Days)
            total_triggers = sum(trigger_counts[name])
            avg_trigger_pct = (total_triggers / (valid_ticker_count * total_days)) * 100
            
        final_stats.append({
            'Strategy': name,
            'Total Return': tot,
            'CAGR': cagr,
            'Max Drawdown': mdd,
            'Sharpe Ratio': sharpe,
            'Calmar Ratio': cagr / abs(mdd) if mdd < 0 else 0,
            'Avg Trigger %': avg_trigger_pct
        })
        
    # 4. 輸出報表
    df_stats = pd.DataFrame(final_stats).sort_values('Calmar Ratio', ascending=False)
    
    # 格式化
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS (OOS 2024-2025)")
    print("="*80)
    print(df_stats.round(4).to_string(index=False))
    
    csv_path = os.path.join(OUTPUT_DIR, 'exp_02_threshold_comparison.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"\nReport saved to: {csv_path}")
    
    # 5. 繪圖
    plt.figure(figsize=(12, 7))
    
    # 設定線條樣式
    styles = {
        'Buy & Hold':            {'ls': '--', 'color': 'gray', 'alpha': 0.6},
        'Fixed 0.5% (Baseline)': {'ls': '-',  'color': 'blue', 'lw': 1.5},
        'Fixed 1.0% (High)':     {'ls': '-',  'color': 'cyan', 'lw': 1.5},
        'Dynamic ATR (k=0.2)':   {'ls': '-',  'color': 'orange', 'lw': 2},
        'Dynamic ATR (k=0.3)':   {'ls': '-',  'color': 'red', 'lw': 2}
    }
    
    for col in equity_curves.columns:
        style = styles.get(col, {})
        plt.plot(equity_curves.index, equity_curves[col], label=col, **style)
        
    plt.title('EXP-V6.1-02: Gap Threshold Analysis (Fixed vs Dynamic ATR)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    png_path = os.path.join(OUTPUT_DIR, 'exp_02_equity_curves.png')
    plt.savefig(png_path)
    print(f"Chart saved to: {png_path}")

if __name__ == '__main__':
    main()