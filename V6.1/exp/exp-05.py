import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# ==========================================
# 0. 配置與路徑 (Configuration)
# ==========================================
# 取得當前腳本所在的目錄 (V6.1/exp/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 設定資源目錄 (V6.1/resource/)
RESOURCE_DIR = os.path.join(BASE_DIR, '../resource')
# 設定輸出目錄 (V6.1/exp/output/)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定義資產池檔案名稱
ASSET_POOLS = {
    'Final': '2025_final_asset_pool.json',
    'Toxic': '2025_final_toxic_asset_pool.json',
    'Crypto': '2025_final_crypto_sensitive_pool'
}

# ==========================================
# 1. Helper: 資料讀取 (Data Loading)
# ==========================================
def clean_ticker(ticker_raw):
    """
    清洗 Ticker 格式以符合 yfinance 需求
    Input: "NASDAQ:NVDA", "NYSE:BRK.B"
    Output: "NVDA", "BRK-B"
    """
    # 1. 移除交易所前綴 (取 ':' 後面的部分)
    if ':' in ticker_raw:
        ticker = ticker_raw.split(':')[-1]
    else:
        ticker = ticker_raw
        
    # 2. 修正特殊符號 (yfinance 使用 '-' 代替 '.')
    ticker = ticker.replace('.', '-')
    
    return ticker.strip()

def load_tickers_from_pool(pool_name):
    """從 JSON 檔案讀取並清洗 Ticker 列表"""
    filename = ASSET_POOLS.get(pool_name)
    if not filename:
        return []
    
    filepath = os.path.join(RESOURCE_DIR, filename)
    print(f"Loading pool from: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            raw_list = []
            if isinstance(data, list):
                raw_list = data
            elif isinstance(data, dict):
                raw_list = data.get('tickers', list(data.keys()))
            
            # 執行清洗
            clean_list = [clean_ticker(t) for t in raw_list]
            return clean_list
            
    except FileNotFoundError:
        print(f"[Error] Pool file not found: {filepath}")
        return []
    except Exception as e:
        print(f"[Error] Failed to load pool {filename}: {e}")
        return []

def fetch_data(ticker, start_date, end_date):
    """下載日線數據"""
    try:
        # 增加緩衝以確保計算 Gap
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if df.empty:
            print(f"Warning: No data for {ticker}")
            return None
            
        # 處理 MultiIndex columns (yfinance 新版特性)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# ==========================================
# 2. Helper: 日曆效應標記生成器 (Calendar Flags)
# ==========================================
def get_calendar_flags(start_date, end_date):
    """
    生成一個 DataFrame，Index 為交易日，包含 is_totm, is_pre_holiday 標記
    """
    print("Generating Calendar Flags...")
    
    # 建立美股交易日曆
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    df = pd.DataFrame(index=dates)
    df['is_totm'] = False
    df['is_pre_holiday'] = False
    
    # --- 標記 TOTM (Turn of the Month) ---
    # 定義：每個月的最後 1 個交易日 與 下個月的前 3 個交易日
    
    date_series = df.index.to_series()
    groups = date_series.groupby(date_series.dt.to_period('M'))
    
    totm_dates = []
    
    for period, dates_in_month in groups:
        days = dates_in_month.index
        if len(days) < 4: continue
        
        # 月底最後 1 天
        totm_dates.append(days[-1])
        # 月初前 3 天
        totm_dates.extend(days[:3])
        
    df.loc[df.index.isin(totm_dates), 'is_totm'] = True

    # --- 標記 Pre-Holiday ---
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    
    # 找出每個假期的前一個交易日
    for holiday in holidays:
        # 找 holiday 在交易日曆中的插入位置
        idx = dates.searchsorted(holiday)
        
        if idx > 0:
            prev_trade_day = dates[idx - 1]
            # 檢查兩者間隔 (避免跨年太久的情況)
            if (holiday - prev_trade_day).days <= 4:
                df.loc[prev_trade_day, 'is_pre_holiday'] = True

    return df[['is_totm', 'is_pre_holiday']]

# ==========================================
# 3. 實驗主邏輯 (Main Experiment)
# ==========================================
def run_exp_05_calendar_effects():
    print(f"Starting EXP-05: Calendar Effects Analysis...")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # 設定回測區間 (含 Lookback)
    START_DATE = '2020-01-01'
    END_DATE = '2025-12-31'
    
    # 1. 準備日曆標記 (這是全域共用的)
    calendar_df = get_calendar_flags(START_DATE, END_DATE)
    print(f"Calendar flags generated. Total days: {len(calendar_df)}")
    
    all_results = []
    
    # 2. 針對兩個資產池分別執行
    for pool_name in ['Final', 'Toxic']:
        tickers = load_tickers_from_pool(pool_name)
        print(f"\nProcessing {pool_name} Pool ({len(tickers)} tickers)...")
        
        for ticker in tickers:
            # 下載資料
            df = fetch_data(ticker, START_DATE, END_DATE)
            if df is None or len(df) < 50:
                continue
                
            # 時區校正：移除時區資訊以便合併
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 合併日曆特徵 (Inner Join 保證只留交易日)
            df = df.join(calendar_df, how='inner')
            
            # 計算策略訊號 (Gap Strategy A)
            df['prev_close'] = df['Close'].shift(1)
            # Gap %
            df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
            
            # 訊號：Gap > 0.5% (做空)
            df['signal_sell'] = df['gap_pct'] > 0.005
            
            # 計算單日報酬 (做空: Open - Close) / Open
            # 假設 MOC (以收盤價平倉)
            df['ret'] = (df['Open'] - df['Close']) / df['Open']
            
            # 只篩選出有交易的日子
            trade_days = df[df['signal_sell']].copy()
            
            if len(trade_days) == 0:
                continue
            
            # 標記分類
            # 優先級：Holiday > TOTM > Normal
            trade_days['day_type'] = 'Normal'
            trade_days.loc[trade_days['is_totm'], 'day_type'] = 'TOTM'
            trade_days.loc[trade_days['is_pre_holiday'], 'day_type'] = 'Holiday'
            
            # 統計每個類別的表現
            for d_type, group in trade_days.groupby('day_type'):
                all_results.append({
                    'Pool': pool_name,
                    'Ticker': ticker,
                    'Type': d_type,
                    'Trade_Count': len(group),
                    'Win_Count': (group['ret'] > 0).sum(),
                    'Total_Return': group['ret'].sum(), # 單利加總
                    'Avg_Return': group['ret'].mean(),
                    'Return_Std': group['ret'].std()
                })
    
    # 3. 彙整結果並輸出
    if not all_results:
        print("No trades generated.")
        return

    res_df = pd.DataFrame(all_results)
    
    # 計算單項 Win Rate
    res_df['Win_Rate'] = res_df['Win_Count'] / res_df['Trade_Count']
    
    # 輸出原始數據
    output_path = os.path.join(OUTPUT_DIR, 'exp_05_calendar_effect_raw.csv')
    res_df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to: {output_path}")
    
    # 輸出摘要報告 (Pivot Table)
    # 使用 Weighted Average 計算整體表現
    summary = res_df.groupby(['Pool', 'Type']).apply(
        lambda x: pd.Series({
            'Total_Trades': x['Trade_Count'].sum(),
            'Avg_Win_Rate': (x['Win_Count'].sum() / x['Trade_Count'].sum()),
            'Avg_Return_Per_Trade': (x['Total_Return'].sum() / x['Trade_Count'].sum())
        })
    )
    
    print("\n=== EXP-05 Summary Report ===")
    print(summary)
    
    summary_path = os.path.join(OUTPUT_DIR, 'exp_05_summary_report.csv')
    summary.to_csv(summary_path)
    print(f"Summary report saved to: {summary_path}")

# ==========================================
# 4. 程式進入點 (Entry Point)
# ==========================================
if __name__ == "__main__":
    run_exp_05_calendar_effects()