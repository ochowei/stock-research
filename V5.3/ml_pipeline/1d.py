import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 獲取資料 ---
# 為了驗證長期的有效性，我們使用日線 (1d) 抓取 10 年資料
# 若想測試短線，可改為 interval="1h", period="730d"
ticker = "SPY"
df = yf.download(ticker, period="10y", interval="1d", progress=False)

# 清理資料 (MultiIndex 處理)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[['Close']].copy()

# --- 2. 計算指標 ---
# 計算 RSI (14)
window_length = 14
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- 3. 定義策略邏輯 ---
# 設定閾值
rsi_buy_threshold = 30   # 跌深
rsi_sell_threshold = 70  # 長多 (漲多)

# 建立信號 (Signal)
# 1 = 持有部位, 0 = 空手 (持有現金)
df['Signal'] = 0

# 向量化邏輯 (這是一個簡化的狀態機)
position = 0 # 初始空手
signals = []

for rsi in df['RSI']:
    if rsi < rsi_buy_threshold:
        position = 1 # 買入訊號
    elif rsi > rsi_sell_threshold:
        position = 0 # 賣出訊號
    signals.append(position)

df['Signal'] = signals

# --- 重要：Shift 1 ---
# 今天的訊號，只能用明天的開盤或收盤價成交。
# 若不 Shift，會產生「預知未來」的誤差 (Look-ahead Bias)。
df['Signal'] = df['Signal'].shift(1)

# --- 4. 計算績效 ---
# 計算每日漲跌幅
df['Market_Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Market_Returns'] * df['Signal']

# 計算累計報酬 (Cumulative Return)
df['Buy_Hold_Cumulative'] = (1 + df['Market_Returns']).cumprod()
df['Strategy_Cumulative'] = (1 + df['Strategy_Returns']).cumprod()

# --- 5. 視覺化結果 ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Buy_Hold_Cumulative'], label='Buy & Hold (SPY)', alpha=0.6)
plt.plot(df.index, df['Strategy_Cumulative'], label='RSI Strategy (Reversion)', color='orange')
plt.title(f'Experiment: RSI Reversion vs Buy & Hold ({ticker})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 6. 關鍵數據輸出 ---
strategy_total_return = df['Strategy_Cumulative'].iloc[-1] - 1
bnh_total_return = df['Buy_Hold_Cumulative'].iloc[-1] - 1

print(f"=== 實驗結果 ({ticker}) ===")
print(f"Buy & Hold 總報酬: {bnh_total_return:.2%}")
print(f"RSI 策略 總報酬:   {strategy_total_return:.2%}")

# 計算簡單夏普比率 (假設無風險利率為 0)
strategy_std = df['Strategy_Returns'].std() * np.sqrt(252)
strategy_sharpe = (df['Strategy_Returns'].mean() * 252) / strategy_std if strategy_std != 0 else 0

bnh_std = df['Market_Returns'].std() * np.sqrt(252)
bnh_sharpe = (df['Market_Returns'].mean() * 252) / bnh_std if bnh_std != 0 else 0

print(f"Buy & Hold 夏普比率: {bnh_sharpe:.2f}")
print(f"RSI 策略 夏普比率:   {strategy_sharpe:.2f}")