import yfinance as yf
import pandas as pd
import numpy as np

# 設定標的
ticker = "NVDA" 

print(f"正在分析 {ticker} 近 60 天 (5m) 的尾盤效應...")

# 1. 下載 5m 資料 (最大回溯 60天)
df = yf.download(ticker, interval="5m", period="60d", progress=False)

# 2. 資料清理
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.index.tz is None:
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
else:
    df.index = df.index.tz_convert('America/New_York')

# 只取正規交易時間，避免盤後資料干擾
df = df.between_time('09:30', '16:00')
df = df.reset_index()
df['Date'] = df['Datetime'].dt.date
unique_dates = sorted(df['Date'].unique())

print(f"有效交易日數: {len(unique_dates)} 天\n")

results = []

# 3. 迴圈比較 (T-1 vs T)
for i in range(len(unique_dates) - 1):
    date_prev = unique_dates[i]   # T-1
    date_curr = unique_dates[i+1] # T
    
    # 取得當日數據
    data_prev = df[df['Date'] == date_prev].set_index('Datetime')
    data_curr = df[df['Date'] == date_curr].set_index('Datetime')
    
    if data_prev.empty or data_curr.empty: continue

    # === 分析 T-1 尾盤 (15:30 - 16:00) ===
    # 5m K棒，最後 30 分鐘大約是最後 6 根
    # 使用時間切片比較準確，避免某些天K棒數不足
    tail_start_time = data_prev.index[-1].replace(hour=15, minute=30, second=0)
    tail_data = data_prev[tail_start_time:]
    
    if len(tail_data) == 0: continue 

    # 尾盤起點價 (15:30 的 Open)
    price_1530 = float(tail_data['Open'].iloc[0])
    # 尾盤收盤價 (16:00 的 Close)
    price_close = float(tail_data['Close'].iloc[-1])
    
    # 尾盤動能
    tail_momentum = (price_close - price_1530) / price_1530
    
    # === 分析 T 日 開盤表現 (隔夜跳空 Gap) ===
    price_open_curr = float(data_curr['Open'].iloc[0])
    
    # 隔夜跳空幅度
    gap_return = (price_open_curr - price_close) / price_close
    
    # 判斷是否延續 (同號為 Yes)
    is_continuation = (tail_momentum > 0 and gap_return > 0) or \
                      (tail_momentum < 0 and gap_return < 0)
    
    # 為了過濾雜訊，我們可以設定一個門檻
    # 例如：尾盤波動小於 0.1% 的視為盤整，不算動能，標記為 "-"
    if abs(tail_momentum) < 0.001: 
        continuation_str = "Flat"
    else:
        continuation_str = "Yes" if is_continuation else "No"

    results.append({
        'T-1_Date': date_prev,
        'T_Date': date_curr,
        'T-1_Tail_Momentum (%)': round(tail_momentum * 100, 3),
        'T_Gap_Return (%)': round(gap_return * 100, 3),
        'Continuation': continuation_str
    })

# 4. 輸出與統計
res_df = pd.DataFrame(results)

if not res_df.empty:
    # 顯示最近 10 筆就好，以免洗版
    print("=== 最近 10 筆交易對數據 ===")
    print(res_df.tail(10).to_string(index=False))
    
    # 過濾掉 Flat (盤整日)，只看有顯著尾盤動能的日子
    valid_days = res_df[res_df['Continuation'] != 'Flat']
    
    if len(valid_days) > 0:
        yes_count = len(valid_days[valid_days['Continuation'] == 'Yes'])
        total_valid = len(valid_days)
        success_rate = yes_count / total_valid
        
        print(f"\n=== {ticker} 60天統計結果 ===")
        print(f"總有效樣本數: {total_valid} 對 (扣除尾盤盤整日)")
        print(f"動能延續 (Yes): {yes_count} 次")
        print(f"動能反轉 (No) : {total_valid - yes_count} 次")
        print(f"延續機率: {success_rate:.2%}")
        
        # 簡易策略建議
        if success_rate > 0.55:
            print(">> 傾向順勢操作：尾盤強則留倉做多。")
        elif success_rate < 0.45:
            print(">> 傾向逆勢操作：尾盤強則隔日開盤放空 (反轉效應)。")
        else:
            print(">> 無顯著規律 (隨機漫步)。")
    else:
        print("樣本中無顯著波動日。")
else:
    print("無資料產出。")