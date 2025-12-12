import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# --- 設定 ---
# 測試名單 (包含 Group A 動能股 與 Group B 有毒股)
TARGET_TICKERS = ['NVDA', 'TSLA', 'MARA', 'GME', 'PLTR', 'AAPL', 'VOO']

def get_premarket_analysis(tickers, days=5):
    """
    抓取包含盤前盤後的 5 分鐘線資料，分析盤前行為
    注意：yfinance 分鐘線限制最近 60 天
    """
    print(f"正在下載 {len(tickers)} 檔股票的 Intraday 數據 (含盤前盤後)...")
    
    # 下載數據 (Interval=5m, Prepost=True 是關鍵)
    # period='5d' 代表過去 5 個交易日
    data = yf.download(
        tickers, 
        period=f"{days}d", 
        interval="5m", 
        prepost=True, 
        group_by='ticker', 
        auto_adjust=True,
        progress=False
    )
    
    analysis_results = []
    
    # 設定時區 (美股是 US/Eastern)
    est = pytz.timezone('US/Eastern')
    
    for ticker in tickers:
        try:
            # 處理單一 ticker 或多 ticker 的 DataFrame 結構差異
            df = data[ticker] if len(tickers) > 1 else data
            
            if df.empty:
                continue
                
            # 轉換索引時區到 EST (方便判斷盤前盤後)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert(est)
            else:
                df.index = df.index.tz_convert(est)
            
            # 依日期分組
            grouped = df.groupby(df.index.date)
            
            for date, day_df in grouped:
                # 定義時段
                # Regular Market: 09:30 - 16:00
                # Pre-market: < 09:30
                # Post-market: > 16:00
                
                regular_df = day_df.between_time('09:30', '16:00')
                pre_df = day_df.between_time('04:00', '09:30')
                
                # 必須要有盤前資料且有前一日收盤價才算
                # 為了計算 Gap，我們需要「前一日的 Regular Close」
                # 這裡簡化邏輯：抓取當日 pre-market 的最後一筆價格 vs 前一日 close
                # 但在 groupby 迴圈中較難取得前一日，我們改用當日 Pre-market 的行為分析
                
                if pre_df.empty:
                    continue
                    
                # 1. 盤前第一筆與最後一筆 (模擬盤前 Open/Close)
                pre_open_price = pre_df.iloc[0]['Open']
                pre_close_price = pre_df.iloc[-1]['Close'] # 最接近 09:30 的價格
                
                # 2. 盤前波動 (High - Low)
                pre_high = pre_df['High'].max()
                pre_low = pre_df['Low'].min()
                pre_volatility = (pre_high - pre_low) / pre_low
                
                # 3. 盤前趨勢 (Drift)
                pre_drift = (pre_close_price - pre_open_price) / pre_open_price
                
                # 4. 取得正規盤開盤價 (Open)
                reg_open = regular_df.iloc[0]['Open'] if not regular_df.empty else np.nan
                
                # 5. Gap 分析 (Pre-Close vs Regular Open)
                # 這裡可以看出 09:29:59 到 09:30:00 之間是否有價差 (通常是造市商價差)
                gap_spread = (reg_open - pre_close_price) / pre_close_price if not np.isnan(reg_open) else np.nan

                analysis_results.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Pre_Open': round(pre_open_price, 2),
                    'Pre_Close': round(pre_close_price, 2),
                    'Reg_Open': round(reg_open, 2),
                    'Pre_Drift %': round(pre_drift * 100, 2),
                    'Pre_Vol %': round(pre_volatility * 100, 2),
                    'Open_Spread %': round(gap_spread * 100, 2)
                })
                
        except Exception as e:
            print(f"[Error] {ticker}: {e}")
            continue
            
    return pd.DataFrame(analysis_results)

def main():
    print(">>> 開始執行盤前價差與波動檢測...")
    df_results = get_premarket_analysis(TARGET_TICKERS)
    
    if not df_results.empty:
        # 排序：按波動率排序
        df_results.sort_values(by=['Ticker', 'Date'], ascending=[True, False], inplace=True)
        
        print("\n" + "="*80)
        print("【盤前行為分析報告】(Pre-Market Behavior)")
        print("Pre_Drift: 盤前時段內的漲跌幅")
        print("Open_Spread: 盤前最後一價 vs 正規開盤價 的差異 (流動性指標)")
        print("="*80)
        print(df_results.to_string(index=False))
        
        # 簡單統計：誰的盤前波動最大？
        print("\n【平均盤前波動率排名】")
        avg_vol = df_results.groupby('Ticker')['Pre_Vol %'].mean().sort_values(ascending=False)
        print(avg_vol)
    else:
        print("無法取得盤前數據，可能是今日尚未開盤或非交易日。")

if __name__ == '__main__':
    main()