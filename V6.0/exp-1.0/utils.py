import json
import time
import re
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import config

def load_tickers_from_json(file_path):
    """讀取 JSON 並移除交易所前綴"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        cleaned_list = [re.sub(r'^.*:', '', ticker).strip() for ticker in raw_list]
        return list(set(cleaned_list))
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return []

def fetch_data(tickers):
    """下載 OHLC 資料並處理 MultiIndex"""
    data_dict = {}
    print(f"Fetching data for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers):
        try:
            time.sleep(config.REQUEST_DELAY)
            df = yf.download(
                ticker, 
                start=config.START_DATE, 
                end=config.END_DATE, 
                auto_adjust=True, 
                progress=False
            )
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty and 'Open' in df.columns and 'Close' in df.columns:
                data_dict[ticker] = df
            else:
                pass 
                # print(f"[Warning] {ticker} missing data.")
        except Exception as e:
            print(f"[Error] Failed to fetch {ticker}: {e}")
            
    return data_dict

def calculate_rsi(series, period=2):
    """計算 RSI 指標 (預設 period=2 for Mean Reversion)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 處理除以零的情況 (雖然 rolling mean 較少發生)
    rsi = rsi.fillna(50) 
    return rsi

def backtest_strategies(df, limit_buffer=0.98):
    """
    針對單一股票回測四種策略
    Signal: RSI(2) < 10 at T-1 Close
    """
    df = df.copy()
    
    # 1. 計算訊號 (基於收盤價)
    df['RSI'] = calculate_rsi(df['Close'], period=2)
    df['Signal'] = (df['RSI'] < 10).astype(int)
    
    # 2. 建立 Shift 後的價格欄位以方便計算
    # Signal 發生在 T-1 (row i)，交易發生在 T (row i+1) 或 T+1 (row i+2)
    # 為了向量化計算，我們將未來的價格 shift 到 T-1 這一行
    
    # T 日數據 (Next Day)
    df['Open_T'] = df['Open'].shift(-1)
    df['Close_T'] = df['Close'].shift(-1)
    df['Low_T'] = df['Low'].shift(-1)
    
    # T+1 日數據 (Next Next Day)
    df['Open_T1'] = df['Open'].shift(-2)
    
    # 3. 計算各策略報酬率 (只在 Signal == 1 的那天計算)
    
    # Strategy 1: Benchmark (MOO)
    # Buy Open T -> Sell Open T+1
    df['Ret_MOO'] = np.where(df['Signal'] == 1, (df['Open_T1'] - df['Open_T']) / df['Open_T'], 0)
    
    # Strategy 2: Delayed (MOC)
    # Buy Close T -> Sell Open T+1
    df['Ret_MOC'] = np.where(df['Signal'] == 1, (df['Open_T1'] - df['Close_T']) / df['Close_T'], 0)
    
    # Strategy 3: Ideal (Night T) - The Gap we missed
    # Buy Close T-1 (Today) -> Sell Open T
    df['Ret_Ideal'] = np.where(df['Signal'] == 1, (df['Open_T'] - df['Close']) / df['Close'], 0)
    
    # Strategy 4: Limit (Dip Buy)
    # Limit Price = Close T-1 * limit_buffer
    # If Low T < Limit Price, Buy at Limit, Sell Open T+1
    limit_price = df['Close'] * limit_buffer
    is_filled = df['Low_T'] < limit_price
    
    # 如果成交: (Open T+1 - Limit) / Limit
    # 如果沒成交: 0
    trade_ret = (df['Open_T1'] - limit_price) / limit_price
    df['Ret_Limit'] = np.where((df['Signal'] == 1) & is_filled, trade_ret, 0)
    
    # 記錄是否有訊號 (用於計算勝率等)
    df['Has_Signal'] = df['Signal']
    
    # 清除因為 Shift 產生的 NaN
    df.dropna(subset=['Ret_MOO', 'Ret_MOC', 'Ret_Ideal', 'Ret_Limit'], inplace=True)
    
    return df[['Ret_MOO', 'Ret_MOC', 'Ret_Ideal', 'Ret_Limit', 'Has_Signal']]

def calculate_max_drawdown(cumulative_returns):
    """計算最大回撤"""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_performance_summary(strategy_returns):
    """計算策略績效摘要"""
    if len(strategy_returns) == 0:
        return {}
        
    # 轉換為資金曲線 (累積報酬)
    equity_curve = (1 + strategy_returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    
    # 計算交易次數 (非零報酬的天數)
    n_trades = (strategy_returns != 0).sum()
    
    # 勝率
    win_rate = (strategy_returns > 0).sum() / n_trades if n_trades > 0 else 0
    
    # 平均每筆報酬
    avg_trade = strategy_returns[strategy_returns != 0].mean() if n_trades > 0 else 0
    
    # 最大回撤
    mdd = calculate_max_drawdown(equity_curve)
    
    return {
        'Total Return': total_return,
        'Win Rate': win_rate,
        'Avg Trade %': avg_trade,
        'Trades': n_trades,
        'Max Drawdown': mdd
    }