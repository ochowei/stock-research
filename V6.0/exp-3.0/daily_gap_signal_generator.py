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
    同時抓取「昨收價」與「最新盤前價」
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
        # 注意：如果是在盤中執行，iloc[-1]可能是今日目前的價格，所以要確保我們取的是「昨日」
        # 但為了簡化，我們假設這是在盤前執行，此时 iloc[-1] 通常是昨日收盤
        prev_closes = closes.iloc[-1]
        
    except Exception as e:
        print(f"[Error] 無法取得昨收價: {e}")
        return {}

    # 2. 抓取最新盤前價 (Current Price) - 使用 1分K 含盤前
    try:
        # period="1d" 包含今日盤前
        df_intraday = yf.download(tickers, period="5d", interval="1m", prepost=True, auto_adjust=True, progress=False)
        
        # 處理資料結構
        if len(tickers) == 1:
            # 單一股票
            if not df_intraday.empty:
                last_price = df_intraday['Close'].iloc[-1]
                last_time = df_intraday.index[-1]
                data_map[tickers[0]] = {
                    'prev_close': prev_closes.iloc[0] if isinstance(prev_closes, pd.Series) else prev_closes,
                    'curr_price': last_price,
                    'last_time': last_time
                }
        else:
            # 多檔股票
            # yfinance 的結構是 (Price, Ticker)
            curr_prices = df_intraday['Close']
            
            for ticker in tickers:
                if ticker not in curr_prices.columns:
                    continue
                    
                # 取得該股票的最後一筆有效數據
                series = curr_prices[ticker].dropna()
                if not series.empty:
                    data_map[ticker] = {
                        'prev_close': prev_closes[ticker],
                        'curr_price': series.iloc[-1],
                        'last_time': series.index[-1]
                    }
                else:
                    # 如果抓不到盤前 (可能沒成交)，就用昨收暫代或標記 NaN
                    data_map[ticker] = {
                        'prev_close': prev_closes[ticker],
                        'curr_price': np.nan,
                        'last_time': None
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
    
    all_tickers = list(set(pool_b + pool_a))
    
    # 2. 取得數據
    market_data = get_market_data(all_tickers)
    
    report_data = []
    
    for ticker in all_tickers:
        if ticker not in market_data:
            continue
            
        data = market_data[ticker]
        prev_close = data['prev_close']
        curr_price = data['curr_price']
        
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
        else:
            # 目前漲跌幅 (Gap %)
            gap_pct = (curr_price - prev_close) / prev_close
            
            # 距離觸發點還差多少 (Distance)
            # 負值代表還沒到，正值代表超過了 (要賣)
            dist_to_trigger = curr_price - trigger_price
            dist_pct = dist_to_trigger / prev_close
            
            curr_price_display = f"{curr_price:.2f}"
            
            if gap_pct > GAP_THRESHOLD_PCT:
                status = "🔴 SELL SIGNAL"  # 已觸發
            else:
                status = "⚪ WAITING"      # 未觸發
        
        report_data.append({
            'Ticker': ticker,
            'Category': category,
            'Prev Close': round(prev_close, 2),
            'Trigger Price': round(trigger_price, 2),
            'Curr Price': curr_price_display,
            'Gap %': round(gap_pct * 100, 2) if not pd.isna(curr_price) else 0,
            'Dist to Trigger': round(dist_to_trigger, 2) if not pd.isna(curr_price) else 0,
            'Status': status
        })
    
    # 3. 轉為 DataFrame 並排序
    df = pd.DataFrame(report_data)
    
    # 排序邏輯：
    # 1. 優先顯示 "SELL SIGNAL" (Gap % 大的排前面)
    # 2. 其次顯示 Category (Toxic 優先)
    df.sort_values(by=['Gap %', 'Category'], ascending=[False, False], inplace=True)
    
    # 4. 輸出美化報表
    print("\n" + "="*85)
    print(f"【盤前監控儀表板】 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"策略目標: 當 Gap > +0.5% 時賣出開盤")
    print("-" * 85)
    
    # 自定義顯示欄位寬度
    print(f"{'Ticker':<8} {'Category':<15} {'PrevClose':>10} {'Trigger':>10} {'CurrPrice':>10} {'Gap %':>8} {'Dist':>8} {'Status':<12}")
    print("-" * 85)
    
    for _, row in df.iterrows():
        # 顏色標記 (在終端機顯示)
        # 簡單版不加 ANSI Color code 以免亂碼，用符號區分
        mark = ">>" if "SELL" in row['Status'] else "  "
        
        print(f"{mark} {row['Ticker']:<5} {row['Category'][:15]:<15} "
              f"{row['Prev Close']:>10.2f} {row['Trigger Price']:>10.2f} "
              f"{str(row['Curr Price']):>10} {row['Gap %']:>7.2f}% "
              f"{row['Dist to Trigger']:>8.2f} {row['Status']:<12}")
              
    print("="*85)
    
    # 5. 儲存
    output_file = os.path.join(config.OUTPUT_DIR, f'live_gap_dashboard_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(output_file, index=False)
    print(f"[Saved] 詳細數據已儲存: {output_file}")

if __name__ == '__main__':
    while True:
        try:
            generate_live_dashboard()
            # 可選擇是否要循環執行 (例如每分鐘更新一次)
            user_input = input("\n按 Enter 重新整理，輸入 'q' 離開: ")
            if user_input.lower() == 'q':
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Critical Error] {e}")
            break