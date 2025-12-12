import yfinance as yf
import pandas as pd
import numpy as np

# 設定標的
ticker = "NVDA"

print(f"正在下載 {ticker} 的 1m 資料...")

# 1. 下載資料
df = yf.download(ticker, interval="1m", period="5d", progress=False)

# === 修正點開始：清理多層索引 (MultiIndex) ===
# 這一步會把 ('Open', 'NVDA') 這種雙層標籤，直接變成單純的 'Open'
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# === 修正點結束 ===

# 2. 時區處理 (轉為紐約時間)
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df.index = df.index.tz_convert('America/New_York')

# 3. 只保留正規交易時間
df = df.between_time('09:30', '16:00')
df = df.reset_index()
df['Date'] = df['Datetime'].dt.date
unique_dates = df['Date'].unique()

results = []

print(f"共取得 {len(unique_dates)} 個交易日資料，開始分析...\n")

for date in unique_dates:
    day_data = df[df['Date'] == date].copy()
    day_data = day_data.set_index('Datetime')
    
    if len(day_data) < 30: continue 
    
    # 定義開盤區間 (前 5 分鐘)
    start_time = day_data.index[0]
    orb_end_time = start_time + pd.Timedelta(minutes=5)
    orb_data = day_data[start_time:orb_end_time]
    
    if orb_data.empty: continue

    # 取得區間高低點 (轉為純浮點數，避免格式干擾)
    orb_high = float(orb_data['High'].max())
    orb_low = float(orb_data['Low'].min())
    orb_open = float(orb_data['Open'].iloc[0])
    
    # 觀察後續 1 小時
    monitor_end_time = orb_end_time + pd.Timedelta(minutes=60)
    future_data = day_data[orb_end_time:monitor_end_time]
    
    if len(future_data) == 0: continue

    future_high = float(future_data['High'].max())
    future_low = float(future_data['Low'].min())
    close_after_1h = float(future_data['Close'].iloc[-1])
    
    # 判斷突破
    breakout_up = future_high > orb_high
    breakout_down = future_low < orb_low
    
    results.append({
        'Date': date,
        'Breakout_Up': 1 if breakout_up else 0,   # 強制轉為數字 1 或 0
        'Breakout_Down': 1 if breakout_down else 0, # 強制轉為數字 1 或 0
        'Return_1h': (close_after_1h - orb_open) / orb_open
    })

# 4. 輸出結果
if not results:
    print("沒有足夠資料。")
else:
    res_df = pd.DataFrame(results)
    
    # 顯示每日詳情
    print("=== 每日突破詳情 ===")
    print(res_df[['Date', 'Breakout_Up', 'Breakout_Down']])
    
    # 統計總次數
    up_count = res_df['Breakout_Up'].sum()
    down_count = res_df['Breakout_Down'].sum()
    
    print("\n=== 統計結果 ===")
    print(f"總交易日: {len(res_df)}")
    print(f"向上突破次數: {up_count} 次")
    print(f"向下突破次數: {down_count} 次")
    
    # 簡單解讀
    if up_count > down_count:
        print(f"\n[解讀] {ticker} 近期開盤後「向上」動能較強。")
    elif down_count > up_count:
        print(f"\n[解讀] {ticker} 近期開盤後「向下」賣壓較重。")
    else:
        print(f"\n[解讀] {ticker} 近期開盤動能方向不明確 (震盪)。")