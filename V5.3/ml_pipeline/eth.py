import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定標的
stock_ticker = "TQQQ"
crypto_ticker = "ETH-USD"

print(f"正在分析 {stock_ticker} 與 {crypto_ticker} 的週末效應...")

# 1. 下載 1h 資料 (涵蓋過去 2 年)
# 1h 資料足夠精確定位到 週五 16:00 和 週一 09:30
df_stock = yf.download(stock_ticker, interval="1h", period="730d", progress=False)
df_crypto = yf.download(crypto_ticker, interval="1h", period="730d", progress=False)

# 2. 資料清理 & 時區校正
def clean_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # yfinance 下載的 1h 資料通常預設為 UTC，需轉為紐約時間
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    return df

df_stock = clean_data(df_stock)
df_crypto = clean_data(df_crypto)

results = []

# 3. 找出所有的 "週一"
# 我們要找的是：美股有開盤的週一 (排除國定假日)
# 邏輯：遍歷 Stock 的交易日，如果該日是週一 (weekday=0)，則進行分析
stock_days = df_stock.index.normalize().unique()

for day in stock_days:
    if day.weekday() != 0: continue # 只看週一
    
    current_monday = day
    # 尋找上一個週五 (通常是減 3 天，但也可能更遠如果週五放假)
    # 這裡簡化邏輯：往前找最近的一個交易日，且必須是週五
    prev_idx = stock_days.get_loc(current_monday) - 1
    if prev_idx < 0: continue
    prev_friday = stock_days[prev_idx]
    
    if prev_friday.weekday() != 4: continue # 如果上個交易日不是週五，跳過 (可能有長假)

    # === A. 計算美股跳空 (Stock Gap) ===
    # 上週五收盤價 (最後一根 K 的 Close)
    try:
        friday_data = df_stock[df_stock.index.normalize() == prev_friday]
        monday_data = df_stock[df_stock.index.normalize() == current_monday]
        
        if friday_data.empty or monday_data.empty: continue

        stock_close_fri = float(friday_data['Close'].iloc[-1])
        stock_open_mon = float(monday_data['Open'].iloc[0])
        
        stock_gap_pct = (stock_open_mon - stock_close_fri) / stock_close_fri
    except Exception as e:
        continue

    # === B. 計算 ETH 週末回報 (Crypto Weekend Return) ===
    # 時間點：週五 16:00 (美股收盤) -> 週一 09:30 (美股開盤)
    # 雖然 ETH 24h 交易，但我們要看的是 "對應美股休市期間" 的累積漲跌
    try:
        # 定位時間戳記
        ts_fri_close = prev_friday + pd.Timedelta(hours=16) 
        ts_mon_open = current_monday + pd.Timedelta(hours=9, minutes=30)
        
        # 在 Crypto 資料中找最接近的時間點 (method='nearest')
        # 注意：需容忍一點誤差，因為 crypto 1h K棒的時間標籤可能略有不同
        crypto_fri = df_crypto.iloc[df_crypto.index.get_indexer([ts_fri_close], method='nearest')[0]]
        crypto_mon = df_crypto.iloc[df_crypto.index.get_indexer([ts_mon_open], method='nearest')[0]]
        
        eth_price_start = float(crypto_fri['Close']) # 用 Close 比較準
        eth_price_end = float(crypto_mon['Open'])    # 用 Open 代表開盤瞬間
        
        eth_weekend_return = (eth_price_end - eth_price_start) / eth_price_start
    except Exception as e:
        continue

    results.append({
        'Date': current_monday.date(),
        'ETH_Weekend_Ret': eth_weekend_return,
        'Stock_Gap': stock_gap_pct,
        'Same_Direction': (eth_weekend_return * stock_gap_pct) > 0 # 同號為 True
    })

# 4. 分析結果
res_df = pd.DataFrame(results)

if not res_df.empty:
    print(f"=== {stock_ticker} vs {crypto_ticker} 週末預測效應 ({len(res_df)} 週) ===")
    
    # 計算相關係數
    corr = res_df['ETH_Weekend_Ret'].corr(res_df['Stock_Gap'])
    
    # 計算同向機率
    same_dir_prob = res_df['Same_Direction'].sum() / len(res_df)
    
    print(f"相關係數 (Correlation): {corr:.4f}")
    print(f"同向機率 (Direction Match): {same_dir_prob:.2%}")
    
    # 視覺化
    plt.figure(figsize=(10, 6))
    plt.scatter(res_df['ETH_Weekend_Ret'], res_df['Stock_Gap'], alpha=0.6)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(f'Weekend Return: ETH vs {stock_ticker} Gap')
    plt.xlabel('ETH Weekend Return')
    plt.ylabel(f'{stock_ticker} Monday Gap')
    
    # 加上趨勢線
    z = np.polyfit(res_df['ETH_Weekend_Ret'], res_df['Stock_Gap'], 1)
    p = np.poly1d(z)
    plt.plot(res_df['ETH_Weekend_Ret'], p(res_df['ETH_Weekend_Ret']), "r--")
    
    plt.show()
    
    # 策略解讀
    if corr > 0.5:
        print("\n[結論] 強烈正相關！週一開盤前請務必檢查 ETH 週末走勢。")
    elif corr > 0.3:
        print("\n[結論] 中度正相關，ETH 可作為參考濾網。")
    else:
        print("\n[結論] 兩者脫鉤，ETH 無法有效預測該股票開盤。")
else:
    print("資料不足。")