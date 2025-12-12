import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOXIC_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_toxic_asset_pool.json')

# 載入有毒池
with open(TOXIC_POOL_FILE, 'r') as f:
    raw_list = json.load(f)
    tickers = [t.split(':')[-1].strip().replace('.', '-') for t in raw_list]

CRYPTO_TICKER = "ETH-USD"

print(f"啟動深度分析：{len(tickers)} 檔股票 vs {CRYPTO_TICKER} (2023-2025)")

# 1. 下載數據 (增加錯誤重試機制)
# 使用 1h 數據
df_crypto = yf.download(CRYPTO_TICKER, interval="1h", period="730d", progress=False)
df_stocks = yf.download(tickers, interval="1h", period="730d", group_by='ticker', progress=False)

# 時區處理
if df_crypto.index.tz is None:
    df_crypto.index = df_crypto.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df_crypto.index = df_crypto.index.tz_convert('America/New_York')

if df_stocks.index.tz is None:
    df_stocks.index = df_stocks.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df_stocks.index = df_stocks.index.tz_convert('America/New_York')

# 2. 建立交易日誌 (Trade Log)
trade_logs = []
mondays = [d for d in df_crypto.index.normalize().unique() if d.weekday() == 0]

for mon in mondays:
    # 簡易推算上週五
    prev_fri = mon - pd.Timedelta(days=3)
    
    # 定位時間點
    ts_fri_close = prev_fri + pd.Timedelta(hours=16)
    ts_mon_open = mon + pd.Timedelta(hours=9, minutes=30)
    
    # 計算 ETH 週末漲幅
    try:
        idx_start = df_crypto.index.get_indexer([ts_fri_close], method='nearest')[0]
        idx_end = df_crypto.index.get_indexer([ts_mon_open], method='nearest')[0]
        
        # 檢查時間誤差是否過大 (超過 4 小時視為數據缺失)
        time_diff = abs(df_crypto.index[idx_end] - ts_mon_open)
        if time_diff > pd.Timedelta(hours=4): continue

        eth_start = float(df_crypto['Close'].iloc[idx_start])
        eth_end = float(df_crypto['Open'].iloc[idx_end])
        eth_ret = (eth_end - eth_start) / eth_start
    except:
        continue

    # 遍歷股票
    for ticker in tickers:
        try:
            stock_data = df_stocks[ticker]
            day_data = stock_data[stock_data.index.normalize() == mon]
            fri_data = stock_data[stock_data.index.normalize() == prev_fri]
            
            if day_data.empty or fri_data.empty: continue

            stock_open = float(day_data['Open'].iloc[0])
            stock_close = float(day_data['Close'].iloc[-1])
            stock_prev_close = float(fri_data['Close'].iloc[-1])
            
            # Gap 計算
            gap_pct = (stock_open - stock_prev_close) / stock_prev_close
            
            # 策略：Gap > 0.5% 賣出
            if gap_pct > 0.005:
                # 做空損益：(Open - Close) / Open
                # Open 100, Close 110 -> (100-110)/100 = -0.10 (-10%)
                trade_ret = (stock_open - stock_close) / stock_open
                
                trade_logs.append({
                    'Date': mon.date(),
                    'Ticker': ticker,
                    'ETH_Weekend_Ret': eth_ret,
                    'Stock_Gap': gap_pct,
                    'Strategy_Ret': trade_ret
                })
        except:
            continue

df_trades = pd.DataFrame(trade_logs)
print(f"\n共蒐集到 {len(df_trades)} 筆符合 Gap > 0.5% 的潛在交易。")

if df_trades.empty:
    exit()

# === 分析 1: 門檻敏感度 (Sensitivity Analysis) ===
print("\n=== 分析 1: ETH 門檻敏感度測試 ===")
print(f"{'Threshold':<10} {'Avg Ret (Bull)':<15} {'Avg Ret (Normal)':<15} {'Diff':<10} {'Count (Bull)'}")
print("-" * 70)

thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
best_threshold = 0.05
max_divergence = 0

stats_list = []

for th in thresholds:
    mask = df_trades['ETH_Weekend_Ret'] > th
    bull_trades = df_trades[mask]
    norm_trades = df_trades[~mask]
    
    avg_bull = bull_trades['Strategy_Ret'].mean()
    avg_norm = norm_trades['Strategy_Ret'].mean()
    
    # 差異：正常情況比暴漲情況好多少 (數值越大代表過濾越有效)
    diff = avg_norm - avg_bull
    
    if diff > max_divergence:
        max_divergence = diff
        best_threshold = th
        
    print(f"{th:>6.0%}    {avg_bull:>12.2%}   {avg_norm:>12.2%}   {diff:>9.2%}   {len(bull_trades):>5}")
    
    stats_list.append({'Threshold': th, 'Avg_Bull': avg_bull, 'Avg_Norm': avg_norm})

print(f"\n[結論] 最佳區隔門檻似乎在 {best_threshold:.0%} (差異 {max_divergence:.2%})")


# === 分析 2: 風險分佈 (Risk Distribution) at Best Threshold ===
print(f"\n=== 分析 2: 風險分佈 (Threshold = {best_threshold:.0%}) ===")
mask_best = df_trades['ETH_Weekend_Ret'] > best_threshold
df_bull = df_trades[mask_best]
df_norm = df_trades[~mask_best]

def get_risk_metrics(df, name):
    if df.empty: return
    avg = df['Strategy_Ret'].mean()
    win_rate = (df['Strategy_Ret'] > 0).mean()
    worst_trade = df['Strategy_Ret'].min()
    tail_risk_5 = df['Strategy_Ret'].quantile(0.05) # 5% VaR
    
    print(f"[{name}]")
    print(f"  筆數: {len(df)}")
    print(f"  平均報酬: {avg:.2%}")
    print(f"  勝率: {win_rate:.2%}")
    print(f"  最慘交易: {worst_trade:.2%}")
    print(f"  尾部風險 (5% VaR): {tail_risk_5:.2%}")

get_risk_metrics(df_bull, "情境 A: ETH 暴漲 (Risk ON)")
get_risk_metrics(df_norm, "情境 B: ETH 正常")


# === 分析 3: 個股影響力排行 (Ticker Breakdown) ===
print(f"\n=== 分析 3: 個股受影響程度排行 (Top 10) ===")
# 計算每檔股票在 ETH 暴漲時的平均績效
ticker_stats = df_bull.groupby('Ticker')['Strategy_Ret'].agg(['count', 'mean', 'min']).reset_index()
# 過濾掉樣本數太少的
ticker_stats = ticker_stats[ticker_stats['count'] >= 3]
ticker_stats.sort_values('mean', ascending=True, inplace=True) # 虧損最多的排前面

print(f"{'Ticker':<8} {'Count':<6} {'Avg Return':<12} {'Worst Trade':<12}")
print("-" * 45)
for _, row in ticker_stats.head(10).iterrows():
    print(f"{row['Ticker']:<8} {int(row['count']):<6} {row['mean']:>10.2%}   {row['min']:>10.2%}")

# 存檔
output_path = os.path.join(OUTPUT_DIR, 'exp_04_detailed_report.csv')
df_trades.to_csv(output_path, index=False)
print(f"\n詳細交易紀錄已儲存至: {output_path}")