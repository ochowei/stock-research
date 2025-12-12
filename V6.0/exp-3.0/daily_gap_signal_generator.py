import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# --- 1. 設定與匯入 ---
# 嘗試匯入 config 和 utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from exp_1_0 import config, utils 
except ImportError:
    import config
    import utils

# --- 2. 參數設定 ---
# 動能股黑名單 (不適合開盤賣出的股票)
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI'
]

# Gap 觸發門檻 (0.5%)
GAP_THRESHOLD_PCT = 0.005 

def get_market_data(tickers):
    """
    同時抓取「昨收價」、「最新盤前價」以及「過去1小時最高價」
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票的即時數據...")
    
    data_map = {}
    
    # 1. 抓取昨收 (Prev Close) - 使用日線
    try:
        # 下載過去 5 天確保有資料
        df_daily = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)
        
        # 處理 MultiIndex
        closes = df_daily['Close'] if len(tickers) > 1 else pd.DataFrame({tickers[0]: df_daily['Close']})
        
        # 取最後一筆非 NaN 的值作為昨收
        prev_closes = closes.iloc[-1]
        
    except Exception as e:
        print(f"[Error] 無法取得昨收價: {e}")
        return {}

    # 2. 抓取最新盤前價 (Current Price) & 過去1小時最高價 - 使用 1分K 含盤前
    try:
        # period="5d" 以確保有足夠的歷史資料來回溯 1 小時
        df_intraday = yf.download(tickers, period="5d", interval="1m", prepost=True, auto_adjust=True, progress=False)
        
        # 處理資料結構
        if len(tickers) == 1:
            # 單一股票
            if not df_intraday.empty:
                last_price = df_intraday['Close'].iloc[-1]
                last_time = df_intraday.index[-1]
                
                # --- 計算過去 1 小時最高價 ---
                # 嘗試使用 High，如果沒有則用 Close
                col_high = 'High' if 'High' in df_intraday.columns else 'Close'
                cutoff_time = last_time - timedelta(hours=1)
                mask = df_intraday.index >= cutoff_time
                highest_1h = df_intraday.loc[mask, col_high].max()
                
                data_map[tickers[0]] = {
                    'prev_close': prev_closes.iloc[0] if isinstance(prev_closes, pd.Series) else prev_closes,
                    'curr_price': last_price,
                    'last_time': last_time,
                    'highest_1h': highest_1h
                }
        else:
            # 多檔股票
            curr_prices = df_intraday['Close']
            
            # 嘗試取得 High 數據
            try:
                high_prices = df_intraday['High']
            except KeyError:
                high_prices = curr_prices

            for ticker in tickers:
                if ticker not in curr_prices.columns:
                    continue
                    
                # 取得該股票的最後一筆有效數據
                series_close = curr_prices[ticker].dropna()
                
                if not series_close.empty:
                    last_time = series_close.index[-1]
                    
                    # --- 計算過去 1 小時最高價 ---
                    if ticker in high_prices.columns:
                        series_high = high_prices[ticker].dropna()
                    else:
                        series_high = series_close
                    
                    if series_high.empty:
                         series_high = series_close

                    cutoff_time = last_time - timedelta(hours=1)
                    recent_highs = series_high[series_high.index >= cutoff_time]
                    highest_1h = recent_highs.max() if not recent_highs.empty else np.nan

                    data_map[ticker] = {
                        'prev_close': prev_closes[ticker],
                        'curr_price': series_close.iloc[-1],
                        'last_time': last_time,
                        'highest_1h': highest_1h
                    }
                else:
                    data_map[ticker] = {
                        'prev_close': prev_closes[ticker],
                        'curr_price': np.nan,
                        'last_time': None,
                        'highest_1h': np.nan
                    }
                    
    except Exception as e:
        print(f"[Error] 無法取得盤前價: {e}")

    return data_map

def generate_live_dashboard():
    print(f"\n>>> 啟動 Gap 策略即時儀表板 (Threshold: +{GAP_THRESHOLD_PCT*100}%)")
    
    # 1. 載入清單
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    pool_a_raw = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_a = [t for t in pool_a_raw if t not in MOMENTUM_BLACKLIST]
    
    all_tickers = list(set(pool_a))
    
    # 2. 取得數據
    market_data = get_market_data(all_tickers)
    
    report_data = []
    
    for ticker in all_tickers:
        if ticker not in market_data:
            continue
            
        data = market_data[ticker]
        prev_close = data['prev_close']
        curr_price = data['curr_price']
        highest_1h = data.get('highest_1h', np.nan)
        
        # 基本檢查
        if pd.isna(prev_close) or prev_close <= 0:
            continue
            
        # 計算觸發價 (Threshold)
        trigger_price = prev_close * (1 + GAP_THRESHOLD_PCT)
        
        # 計算目前狀態
        category = "Toxic (Priority)" if ticker in pool_b else "Standard"
        
        if pd.isna(curr_price):
            status = "NO DATA"
            gap_pct = 0.0
            dist_to_trigger = 0.0
            curr_price_display = "---"
            highest_1h_display = "---"
            hit_1h_pct = -999.0 # 排序用
            hit_1h_pct_display = "---"
        else:
            # 目前漲跌幅 (Gap %)
            gap_pct = (curr_price - prev_close) / prev_close
            
            # 距離觸發點還差多少 (Distance)
            dist_to_trigger = curr_price - trigger_price
            
            curr_price_display = f"{curr_price:.2f}"
            
            # --- 判斷過去 1 小時最高價距離觸發價的 % ---
            if pd.isna(highest_1h):
                 highest_1h_display = "---"
                 hit_1h_pct = -999.0
                 hit_1h_pct_display = "---"
            else:
                 highest_1h_display = f"{highest_1h:.2f}"
                 # 公式: (High_1h - Trigger) / Prev_Close
                 # 正值代表超過 Trigger 的幅度，負值代表距離 Trigger 還有多遠
                 hit_1h_val = (highest_1h - trigger_price) / prev_close
                 hit_1h_pct = hit_1h_val * 100
                 hit_1h_pct_display = f"{hit_1h_pct:+.2f}%"

            if gap_pct > GAP_THRESHOLD_PCT:
                status = "🔴 SELL SIGNAL"  # 目前價格已觸發
            else:
                status = "⚪ WAITING"      # 目前價格未觸發
        
        report_data.append({
            'Ticker': ticker,
            'Category': category,
            'Prev Close': round(prev_close, 2),
            'Trigger Price': round(trigger_price, 2),
            'Curr Price': curr_price_display,
            'High 1h': highest_1h_display,       
            'Hit 1h %': hit_1h_pct_display,      # 顯示用
            'Hit 1h Val': hit_1h_pct,            # 排序用 (數值)
            'Gap %': round(gap_pct * 100, 2) if not pd.isna(curr_price) else 0,
            'Dist to Trigger': round(dist_to_trigger, 2) if not pd.isna(curr_price) else 0,
            'Status': status
        })
    
    # 3. 轉為 DataFrame 並排序
    df = pd.DataFrame(report_data)
    
    # 排序邏輯：
    # 1. 依照 "Hit 1h Val" 由大到小排序 (衝過 Trigger 越多的排越前面，最接近 Trigger 的排其次)
    # 2. 若相同則看 Category
    df.sort_values(by=['Hit 1h Val', 'Category'], ascending=[False, False], inplace=True)
    
    # 4. 輸出美化報表
    print("\n" + "="*115)
    print(f"【盤前監控儀表板】 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"策略目標: 當 Gap > +0.5% 時賣出開盤")
    print("-" * 115)
    
    # 自定義顯示欄位寬度
    print(f"{'Ticker':<8} {'Category':<15} {'PrevClose':>10} {'Trigger':>10} {'CurrPrice':>10} {'High 1h':>10} {'Hit 1h %':>10} {'Gap %':>8} {'Dist':>8} {'Status':<12}")
    print("-" * 115)
    
    for _, row in df.iterrows():
        # 顏色標記 (在終端機顯示)
        mark = ">>" if "SELL" in row['Status'] else "  "
        # 如果曾經觸發 (Hit 1h % > 0) 但現在掉下來 (Status == WAITING)，給個不同標記
        try:
            val = row['Hit 1h Val']
            if "WAITING" in row['Status'] and val > 0:
                mark = "* " 
        except:
            pass
        
        print(f"{mark} {row['Ticker']:<5} {row['Category'][:15]:<15} "
              f"{row['Prev Close']:>10.2f} {row['Trigger Price']:>10.2f} "
              f"{str(row['Curr Price']):>10} "
              f"{str(row['High 1h']):>10} "
              f"{row['Hit 1h %']:>10} "
              f"{row['Gap %']:>7.2f}% "
              f"{row['Dist to Trigger']:>8.2f} {row['Status']:<12}")
              
    print("="*115)
    
    # 5. 儲存 (移除輔助排序的欄位後儲存)
    output_df = df.drop(columns=['Hit 1h Val'])
    output_file = os.path.join(config.OUTPUT_DIR, f'live_gap_dashboard_{datetime.now().strftime("%Y%m%d")}.csv')
    output_df.to_csv(output_file, index=False)
    print(f"[Saved] 詳細數據已儲存: {output_file}")

if __name__ == '__main__':

    try:
        generate_live_dashboard()
        # 可選擇是否要循環執行 (例如每分鐘更新一次)                
    
    except Exception as e:
        print(f"[Critical Error] {e}")
        