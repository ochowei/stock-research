import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr

# --- 設定 ---
RESOURCE_DIR = "../resource"
START_DATE = "2023-01-01" # 近兩年數據即可
END_DATE = "2025-12-31"

def clean_ticker(ticker):
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    tickers = []
    path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'GME', 'AMC', 'COIN', 'HOOD', 'ONDS', 'GLD']
    tickers.extend(default)
    return list(set([clean_ticker(t) for t in tickers]))

def calculate_er(series, window=5):
    net = (series - series.shift(window)).abs()
    sum_abs = series.diff().abs().rolling(window).sum()
    return net / (sum_abs + 1e-9)

def analyze_conditional_probability(tickers):
    print("🚀 分析「震盪後」的變盤機率...")
    
    group_stats = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            if len(df) < 200: continue
            
            # 計算 IC 決定組別
            df['Past_ER'] = calculate_er(df['Close'], 5)
            df['Future_ER'] = calculate_er(df['Close'].shift(-5), 5) # Future label
            
            clean_df = df.dropna()
            if len(clean_df) < 50: continue
            
            ic, _ = spearmanr(clean_df['Past_ER'], clean_df['Future_ER'])
            
            # 定義 "Currently Safe" (目前是震盪)
            # 取 ER 最低的 40% 作為震盪期
            safe_threshold = clean_df['Past_ER'].quantile(0.40)
            is_currently_safe = clean_df['Past_ER'] <= safe_threshold
            
            # 在這些 Safe 的時刻，未來變成 Trend (ER > 0.6) 的機率是多少？
            future_is_trend = clean_df.loc[is_currently_safe, 'Future_ER'] > 0.6
            blow_up_prob = future_is_trend.mean()
            
            group_stats.append({
                'Ticker': ticker,
                'IC': ic,
                'Group': 'Momentum' if ic > 0 else 'Reversion',
                'BlowUp_Prob': blow_up_prob # 從震盪變成強趨勢的機率
            })
            
        except Exception:
            pass

    res_df = pd.DataFrame(group_stats)
    
    # 彙總分析
    print("\n📊 --- 分組條件機率分析 (Conditional Probability) ---")
    print("定義: BlowUp_Prob = 當現在很平靜 (Low ER) 時，未來 5 天突然噴出 (High ER) 的機率")
    
    grp = res_df.groupby('Group')['BlowUp_Prob'].agg(['mean', 'count', 'std'])
    print(grp)
    
    # 檢驗顯著性
    rev_prob = res_df[res_df['Group']=='Reversion']['BlowUp_Prob'].mean()
    mom_prob = res_df[res_df['Group']=='Momentum']['BlowUp_Prob'].mean()
    
    print(f"\n💡 洞察: Reversion 組的爆倉機率是 Momentum 組的 {rev_prob/mom_prob:.2f} 倍")
    
    if rev_prob > mom_prob:
        print("✅ 驗證成功：Reversion (Negative IC) 組確實更容易在平靜後突然爆炸。")
    else:
        print("❌ 驗證失敗：需要重新思考。")

if __name__ == "__main__":
    tickers = load_tickers()
    analyze_conditional_probability(tickers)