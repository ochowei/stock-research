import os
import json
import yfinance as yf
import pandas as pd
import numpy as np

# === 設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 讀取標準資產池 (Asset Pool)
TARGET_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')

with open(TARGET_POOL_FILE, 'r') as f:
    raw_list = json.load(f)
    tickers = [t.split(':')[-1].strip().replace('.', '-') for t in raw_list]

CRYPTO_TICKER = "ETH-USD"

print(f"啟動標準池深度檢測：{len(tickers)} 檔股票 vs {CRYPTO_TICKER}")
print("目標：找出在 ETH 暴漲 (>5%) 時表現最差的標準股")

# 1. 下載數據
try:
    df_crypto = yf.download(CRYPTO_TICKER, interval="1h", period="730d", progress=False)
    df_stocks = yf.download(tickers, interval="1h", period="730d", group_by='ticker', progress=False)
except Exception as e:
    print(f"[Error] 下載失敗: {e}")
    exit()

# 時區處理
if df_crypto.index.tz is None:
    df_crypto.index = df_crypto.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df_crypto.index = df_crypto.index.tz_convert('America/New_York')

if df_stocks.index.tz is None:
    df_stocks.index = df_stocks.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df_stocks.index = df_stocks.index.tz_convert('America/New_York')

# 2. 建立交易日誌
trade_logs = []
mondays = [d for d in df_crypto.index.normalize().unique() if d.weekday() == 0]

for mon in mondays:
    # 定位時間點
    prev_fri = mon - pd.Timedelta(days=3)
    ts_fri_close = prev_fri + pd.Timedelta(hours=16)
    ts_mon_open = mon + pd.Timedelta(hours=9, minutes=30)
    
    try:
        # 計算 ETH 週末漲跌
        idx_start = df_crypto.index.get_indexer([ts_fri_close], method='nearest')[0]
        idx_end = df_crypto.index.get_indexer([ts_mon_open], method='nearest')[0]
        
        if abs(df_crypto.index[idx_end] - ts_mon_open) > pd.Timedelta(hours=4): continue

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
            
            gap_pct = (stock_open - stock_prev_close) / stock_prev_close
            
            # 策略：Gap > 0.5% 賣出
            if gap_pct > 0.005:
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
print(f"\n共蒐集到 {len(df_trades)} 筆潛在交易。")

if df_trades.empty:
    exit()

# === 分析核心：只看 ETH 暴漲 (>5%) 時的個股表現 ===
mask_bull = df_trades['ETH_Weekend_Ret'] > 0.05
df_bull = df_trades[mask_bull]

print(f"\n=== [重點分析] ETH 暴漲 (>5%) 期間，標準池受害清單 ===")
print(f"總交易次數: {len(df_bull)}")
print(f"平均報酬: {df_bull['Strategy_Ret'].mean():.2%}")

if not df_bull.empty:
    # 統計每檔股票在 ETH 暴漲時的表現
    ticker_stats = df_bull.groupby('Ticker')['Strategy_Ret'].agg(['count', 'mean', 'min']).reset_index()
    
    # 過濾：至少出現過 2 次以上交易才有統計意義
    # (如果只出現 1 次且大賠，可能是個案，但也值得注意)
    ticker_stats = ticker_stats[ticker_stats['count'] >= 2]
    
    # 依照平均虧損排序 (由小到大，負越多越前面)
    ticker_stats.sort_values('mean', ascending=True, inplace=True)
    
    print(f"\n{'Ticker':<8} {'Count':<6} {'Avg Return':<12} {'Worst Trade':<12}")
    print("-" * 45)
    
    # 列出表現最差的前 15 名
    for _, row in ticker_stats.head(15).iterrows():
        print(f"{row['Ticker']:<8} {int(row['count']):<6} {row['mean']:>10.2%}   {row['min']:>10.2%}")

    # 存檔
    output_path = os.path.join(OUTPUT_DIR, 'exp_04_control_detailed.csv')
    ticker_stats.to_csv(output_path, index=False)
    print(f"\n[Saved] 詳細個股清單已儲存: {output_path}")

    print("\n[決策建議]")
    print("1. 如果某股票在此清單中 Avg Return < -2.0%，建議將其移至 Toxic Pool。")
    print("2. 如果清單中的股票多為科技/半導體，說明市場連動性正在增強。")
else:
    print("恭喜！在 ETH 暴漲期間，標準池沒有產生足夠的交易樣本，或表現非常穩定。")