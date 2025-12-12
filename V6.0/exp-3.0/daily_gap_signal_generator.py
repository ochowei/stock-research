import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# --- 1. 路徑設定與匯入 ---
# 確保可以 import 同層級或上層的 config 與 utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    # 嘗試匯入 V6.0 結構下的模組
    from exp_1_0 import config, utils 
except ImportError:
    # 如果放在同一層目錄直接匯入
    import config
    import utils

# --- 2. 設定參數 ---
# 根據 EXP-03 結論，這些動能股不適合做 Gap Filter (賣飛風險高)
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI'
]

def get_latest_close_prices(tickers):
    """取得一籃子股票最新的收盤價"""
    print(f"正在下載 {len(tickers)} 檔股票的最新數據...")
    
    # 下載過去 5 天數據以確保包含最後一個交易日 (避開週末/假日問題)
    try:
        df = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        
        # 處理 MultiIndex (如果只有一檔股票，yf 格式會不同)
        if len(tickers) == 1:
            # 轉為 Series，Key 為 Ticker
            latest_close = pd.Series({tickers[0]: df['Close'].iloc[-1]})
        else:
            # 取 Close 欄位，並取最後一列 (最新的收盤價)
            latest_close = df['Close'].iloc[-1]
            
        return latest_close
    except Exception as e:
        print(f"[Error] 下載數據失敗: {e}")
        return pd.Series()

def generate_signals():
    print(f">>> 執行每日 Gap 策略訊號掃描 (Date: {datetime.now().strftime('%Y-%m-%d')})")
    
    # 1. 載入資產池
    # Group B: 優先獵殺名單 (Priority Harvest)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    
    # Group A: 一般名單 (需過濾黑名單)
    pool_a_raw = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_a = [t for t in pool_a_raw if t not in MOMENTUM_BLACKLIST]
    
    # 合併清單 (並標記來源)
    all_tickers = list(set(pool_b + pool_a))
    
    # 2. 取得昨收價
    closes = get_latest_close_prices(all_tickers)
    
    report_data = []
    
    for ticker in all_tickers:
        if ticker not in closes.index or np.isnan(closes[ticker]):
            continue
            
        prev_close = closes[ticker]
        
        # --- 核心邏輯: Gap > 0.5% ---
        # Trigger Price = Prev Close * (1 + 0.005)
        trigger_price = prev_close * 1.005
        
        # 分類標籤
        category = "Toxic/Meme (Priority)" if ticker in pool_b else "Standard"
        
        report_data.append({
            'Ticker': ticker,
            'Category': category,
            'Prev Close': round(prev_close, 2),
            'Gap Threshold (+0.5%)': round(trigger_price, 2),
            'Action': 'SELL OPEN'
        })
    
    # 3. 產生報表
    df_report = pd.DataFrame(report_data)
    
    # 排序：優先顯示 Toxic Group，其次按代號
    df_report.sort_values(by=['Category', 'Ticker'], ascending=[False, True], inplace=True)
    
    # 4. 輸出顯示
    print("\n" + "="*60)
    print(f"【每日開盤操作指引】 - 若開盤價 > Threshold 則執行賣出")
    print("="*60)
    
    # 格式化輸出到 Console
    # 使用 to_string 避免中間被省略
    print(df_report.to_string(index=False))
    
    # 5. 儲存 CSV (方便匯入券商或 Excel)
    output_file = os.path.join(config.OUTPUT_DIR, f'daily_gap_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    df_report.to_csv(output_file, index=False)
    print(f"\n[Saved] 訊號表已儲存至: {output_file}")
    
    # 特別提醒黑名單
    print("\n[Info] 以下動能股已排除 (Strategy A 不適用):")
    print(", ".join(MOMENTUM_BLACKLIST))

if __name__ == '__main__':
    generate_signals()