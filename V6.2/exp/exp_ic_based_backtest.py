import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import datetime

# --- 設定 ---
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "output"
START_DATE = "2020-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_ticker(ticker):
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    tickers = []
    # 載入 Final Pool
    path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
            
    # 加入重點測試標的 (包含指數與妖股)
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'EOSE', 'SMR', 'OKLO']
    tickers.extend(default)
    return list(set([clean_ticker(t) for t in tickers]))

def calculate_ic(df):
    """計算單一標的的 Inertia IC"""
    try:
        # X: Past Trend
        net_change = (df['Close'] - df['Close'].shift(5)).abs()
        sum_abs = df['Close'].diff().abs().rolling(5).sum()
        x_er = net_change / (sum_abs + 1e-9)
        
        # Y: Future Trend
        future_net = (df['Close'].shift(-5) - df['Close']).abs()
        future_sum = df['Close'].diff().abs().rolling(5).sum().shift(-5)
        y_er = future_net / (future_sum + 1e-9)
        
        df_calc = pd.concat([x_er, y_er], axis=1).dropna()
        if len(df_calc) < 100: return 0
        
        ic, _ = spearmanr(df_calc.iloc[:, 0], df_calc.iloc[:, 1])
        return ic
    except:
        return 0

def run_group_backtest(tickers):
    print(f"🚀 開始 IC 分組對抗回測 (Pool: {len(tickers)})...")
    
    # 1. 計算每個 Ticker 的 IC 並分組
    ticker_ics = {}
    valid_tickers = []
    
    print("   Computing IC for all tickers...")
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            if len(df) < 250: continue
            ic = calculate_ic(df)
            ticker_ics[t] = ic
            valid_tickers.append(t)
        except:
            pass
            
    # 分組
    sorted_tickers = sorted(ticker_ics.items(), key=lambda x: x[1])
    n = len(sorted_tickers)
    
    # Group A: Negative IC (均值回歸組 - 預期 V6.1 表現最好)
    group_reversion = [t[0] for t in sorted_tickers[:int(n*0.3)]]
    
    # Group B: Positive IC (動能慣性組 - 預期 V6.1 表現最差)
    group_momentum = [t[0] for t in sorted_tickers[-int(n*0.3):]]
    
    # Group C: Random (中間)
    group_random = [t[0] for t in sorted_tickers[int(n*0.3):-int(n*0.3)]]
    
    print(f"\n📊 分組完成:")
    print(f"   Group Reversion (Mean IC: {np.mean([ticker_ics[t] for t in group_reversion]):.3f}): {len(group_reversion)} symbols")
    print(f"     Examples: {group_reversion[:5]}")
    print(f"   Group Momentum  (Mean IC: {np.mean([ticker_ics[t] for t in group_momentum]):.3f}): {len(group_momentum)} symbols")
    print(f"     Examples: {group_momentum[-5:]}")
    
    # 2. 執行 V6.1 回測 (Gap Fade)
    results = []
    
    for group_name, group_tickers in [("Reversion", group_reversion), ("Momentum", group_momentum), ("Random", group_random)]:
        group_pnl = []
        
        for t in group_tickers:
            try:
                df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
                
                # V6.1 Logic
                df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
                df['R_day'] = (df['Close'] - df['Open']) / df['Open']
                
                # Signal: Gap > 0.5% Short, Gap < -0.5% Long
                df['PnL'] = 0.0
                df.loc[df['Gap'] > 0.005, 'PnL'] = -1 * df['R_day']
                df.loc[df['Gap'] < -0.005, 'PnL'] = df['R_day']
                
                # 簡單累積
                group_pnl.append(df['PnL'].fillna(0))
            except:
                pass
        
        # 合併該組 PnL (Equal Weight)
        if group_pnl:
            portfolio_pnl = pd.concat(group_pnl, axis=1).mean(axis=1).fillna(0)
            cum_ret = portfolio_pnl.cumsum()
            sharpe = portfolio_pnl.mean() / (portfolio_pnl.std() + 1e-9) * np.sqrt(252)
            
            results.append({
                'Group': group_name,
                'Total_Return': cum_ret.iloc[-1],
                'Sharpe': sharpe,
                'Equity_Curve': cum_ret
            })
            
    # 3. 輸出報告與繪圖
    print("\n🏆 --- 回測結果 (V6.1 Strategy Performance by IC Group) ---")
    res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='Equity_Curve'} for r in results])
    print(res_df)
    
    plt.figure(figsize=(12, 7))
    for res in results:
        plt.plot(res['Equity_Curve'].index, res['Equity_Curve'].values, label=f"{res['Group']} (Sharpe: {res['Sharpe']:.2f})")
    
    plt.title("V6.1 Performance: Reversion vs Momentum vs Random Tickers")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_ic_group_comparison.png"))
    print(f"\n📈 對比圖已儲存至 {OUTPUT_DIR}/exp_ic_group_comparison.png")
    
    # 4. 儲存分類後的清單，供未來使用
    with open(os.path.join(OUTPUT_DIR, "group_reversion_tickers.json"), "w") as f:
        json.dump(group_reversion, f)
    with open(os.path.join(OUTPUT_DIR, "group_momentum_tickers.json"), "w") as f:
        json.dump(group_momentum, f)

if __name__ == "__main__":
    tickers = load_tickers()
    run_group_backtest(tickers)