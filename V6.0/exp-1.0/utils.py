import json
import time
import re
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import config

def load_tickers_from_json(file_path):
    """讀取 JSON 並移除交易所前綴 (如 NASDAQ:TSLA -> TSLA)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        
        # 使用正則表達式移除冒號前的內容
        cleaned_list = [re.sub(r'^.*:', '', ticker).strip() for ticker in raw_list]
        # 移除重複項
        return list(set(cleaned_list))
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return []

def fetch_data(tickers):
    """
    下載 OHLC 資料。
    使用 auto_adjust=True 還原除權息價格，確保報酬率計算正確。
    並針對 yfinance 可能回傳 MultiIndex 的問題進行修復。
    """
    data_dict = {}
    print(f"Fetching data for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers):
        try:
            # 加入延遲避免 Rate Limit
            time.sleep(config.REQUEST_DELAY)
            
            df = yf.download(
                ticker, 
                start=config.START_DATE, 
                end=config.END_DATE, 
                auto_adjust=True,  # 關鍵：使用還原股價
                progress=False
            )
            
            # --- [Fix] 處理 yfinance 回傳 MultiIndex 欄位的問題 ---
            if isinstance(df.columns, pd.MultiIndex):
                # 如果欄位是 ('Open', 'TSLA') 這種格式，取第一層 ('Open')
                df.columns = df.columns.get_level_values(0)

            if not df.empty:
                # 簡單的資料檢查 (確保扁平化後有 Open/Close)
                if 'Open' in df.columns and 'Close' in df.columns:
                    data_dict[ticker] = df
                else:
                    print(f"[Warning] {ticker} missing Open/Close columns after download.")
        except Exception as e:
            print(f"[Error] Failed to fetch {ticker}: {e}")
            
    return data_dict

def calculate_decomposed_returns(df):
    """
    將每日報酬分解為：隔夜 (Night) 與 日內 (Day)
    
    Logic:
    - Night Return = (Open_t - Close_t-1) / Close_t-1
    - Day Return   = (Close_t - Open_t) / Open_t
    """
    # 建立副本以避免 SettingWithCopyWarning
    df = df.copy()
    
    # 取得前一日收盤價
    df['Prev_Close'] = df['Close'].shift(1)
    
    # 計算報酬率
    # 由於前面已經確保 columns 是單層索引，這裡取出的會是 Series，不會再報錯
    df['Night_Ret'] = (df['Open'] / df['Prev_Close']) - 1
    df['Day_Ret'] = (df['Close'] / df['Open']) - 1
    df['Total_Ret'] = (df['Close'] / df['Prev_Close']) - 1
    
    # 清除第一筆 NaN (因為沒有 Prev_Close)
    df.dropna(subset=['Night_Ret', 'Day_Ret'], inplace=True)
    
    return df[['Night_Ret', 'Day_Ret', 'Total_Ret']]

def calculate_max_drawdown(cumulative_returns):
    """計算最大回撤 (MDD)"""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min(), drawdown

def calculate_performance_metrics(returns_series, strategy_name):
    """
    計算績效指標: CAGR, Sharpe, Volatility, MDD
    returns_series: 每日報酬率序列 (Series)
    """
    # 移除 NaN
    returns = returns_series.dropna()
    if len(returns) == 0:
        return {}

    # 1. 累積報酬曲線
    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    
    # 2. CAGR (年化報酬率) - 假設一年 252 交易日
    n_years = len(returns) / 252
    cagr = (equity_curve.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
    
    # 3. 年化波動率 (Volatility)
    annual_vol = returns.std() * np.sqrt(252)
    
    # 4. 夏普比率 (Sharpe Ratio)
    # 簡化版：(Rp - Rf) / Sigma_p
    # 這裡將 Rf 轉換為日頻率
    daily_rf = (1 + config.RISK_FREE_RATE) ** (1/252) - 1
    excess_ret = returns - daily_rf
    sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # 5. 最大回撤 (MDD)
    mdd, _ = calculate_max_drawdown(equity_curve)
    
    return {
        'Strategy': strategy_name,
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility (Ann.)': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': mdd
    }