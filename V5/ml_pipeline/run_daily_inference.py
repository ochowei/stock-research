import yfinance as yf
import pandas as pd
import pandas_ta as ta
import joblib
import os
import json
import time
from datetime import datetime, timedelta

# --- 設定路徑 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
ASSET_POOL_PATH = os.path.join(SCRIPT_DIR, 'asset_pool.json')

def load_assets():
    with open(ASSET_POOL_PATH, 'r') as f:
        assets = json.load(f)
    # 清理 ticker 格式 (例如 BRK.B -> BRK-B)
    return [a.split(':')[-1].replace('.', '-') for a in assets]

def download_with_retry(tickers, start_date, retries=3):
    """帶重試機制的下載函數"""
    for i in range(retries):
        try:
            print(f"嘗試下載數據 (第 {i+1}/{retries} 次)...")
            df = yf.download(tickers, start=start_date, interval='1d', auto_adjust=False, progress=False, timeout=20)
            if not df.empty:
                return df
        except Exception as e:
            print(f"下載失敗: {e}")
        time.sleep(2) # 等待 2 秒重試
    return pd.DataFrame() # 失敗回傳空 DataFrame

def get_latest_data(tickers, lookback_days=365):
    """下載最近 N 天的數據 (確保足夠計算 SMA200)"""
    start_date = datetime.now() - timedelta(days=lookback_days)
    
    # 下載個股
    print(f"\n[1/2] 下載個股數據...")
    df_stocks = download_with_retry(tickers, start_date)
    
    if not df_stocks.empty:
        df_stocks = df_stocks.stack(future_stack=True).swaplevel(0, 1).sort_index()
        df_stocks.index.names = ['symbol', 'timestamp']
    
    # 下載 Macro (用於 L1)
    print(f"[2/2] 下載宏觀數據 (SPY, VIX)...")
    macro_tickers = ['SPY', 'IWO', '^VIX', '^TNX']
    df_macro = download_with_retry(macro_tickers, start_date)
    
    # --- 安全檢查 ---
    # 檢查關鍵宏觀數據是否下載成功
    if df_macro.empty:
        raise ValueError("錯誤：宏觀數據下載完全失敗。")
    
    # 檢查是否包含關鍵欄位 (如果部分下載失敗，yfinance 可能會缺欄位)
    # yfinance 多層索引檢查: ('Close', 'SPY') 或 'SPY'
    cols = df_macro.columns
    # 簡單檢查：確認 columns 中是否有 SPY
    has_spy = False
    if isinstance(cols, pd.MultiIndex):
        if 'SPY' in cols.get_level_values(1): has_spy = True # (Price, Ticker) structure
    else:
        if 'SPY' in cols: has_spy = True
            
    if not has_spy:
         raise ValueError("嚴重錯誤：關鍵數據 'SPY' 下載失敗，無法進行市場狀態判斷。請檢查網路或稍後再試。")

    return df_stocks, df_macro

def build_market_features_live(df_macro):
    """即時計算 L1 所需的市場特徵"""
    # 處理 yfinance 多層索引 (Price, Ticker) -> 取 Close
    try:
        closes = df_macro['Close']
    except KeyError:
        # 如果只有單一層 (例如只有一個 ticker 成功)，結構可能不同，這裡做個防呆
        if 'Close' in df_macro.columns:
            closes = df_macro['Close']
        else:
            closes = df_macro # 假設已經是 Close (較少見)

    feat = pd.DataFrame(index=closes.index)
    
    # 1. 計算所需的基礎指標
    # [Fix] 加入 fill_method=None 解決 FutureWarning
    feat['SPY_Ret'] = closes['SPY'].pct_change(fill_method=None)
    feat['IWO_Ret'] = closes['IWO'].pct_change(fill_method=None)
    
    feat['IWO_Vol_21d'] = feat['IWO_Ret'].rolling(21).std()
    feat['SPY_Vol_21d'] = feat['SPY_Ret'].rolling(21).std()
    
    feat['SPY_CumRet_21d'] = closes['SPY'].pct_change(21, fill_method=None)
    feat['IWO_CumRet_21d'] = closes['IWO'].pct_change(21, fill_method=None)
    feat['SPY_IWO_Div_21d'] = feat['SPY_CumRet_21d'] - feat['IWO_CumRet_21d']
    
    feat['VIX_Gap'] = closes['^VIX'] - closes['^VIX'].rolling(50).mean()
    feat['VIX_Change_1d'] = closes['^VIX'].diff()
    feat['TNX_Change_5d'] = closes['^TNX'].diff(5)
    
    return feat.iloc[-1:] # 只回傳最新的一筆 (Today)

def check_l1_defense(current_market_feat):
    """載入模型並進行推論"""
    print("\n--- L1 防禦系統檢查 ---")
    
    # 載入模型
    try:
        hmm_model = joblib.load(os.path.join(MODELS_DIR, 'hmm_model.joblib'))
        hmm_scaler = joblib.load(os.path.join(MODELS_DIR, 'hmm_scaler.joblib'))
        iso_model = joblib.load(os.path.join(MODELS_DIR, 'iso_forest.joblib'))
        state_map = joblib.load(os.path.join(MODELS_DIR, 'hmm_state_map.joblib'))
    except FileNotFoundError:
        print("錯誤：找不到模型文件 (models/*.joblib)。請確認是否已執行訓練步驟。")
        return False

    # 準備 HMM 輸入
    hmm_cols = ['SPY_Ret', 'IWO_Vol_21d', 'SPY_IWO_Div_21d']
    
    # 檢查是否有 NaN (如果數據不足 21 天，rolling 會是 NaN)
    if current_market_feat[hmm_cols].isnull().any().any():
        print("警告：今日市場特徵包含 NaN (可能數據不足 21 天或下載缺失)，無法準確判斷。")
        # 這裡可以選擇報錯或嘗試填補，為了安全起見，若數據缺失則視為風險
        return False
        
    X_hmm = current_market_feat[hmm_cols]
    X_hmm_scaled = hmm_scaler.transform(X_hmm)
    
    # ... 在 X_hmm_scaled 計算之後加入 ...
    print(f"\n[診斷數據] IWO Vol: {current_market_feat['IWO_Vol_21d'].iloc[0]:.4f}, Divergence: {current_market_feat['SPY_IWO_Div_21d'].iloc[0]:.4f}")    

    # HMM 預測
    hidden_state = hmm_model.predict(X_hmm_scaled)[0]
    mapped_state = state_map[hidden_state] # 轉換為 0=Bull, 1=Chop, 2=Crash
    
    # IsoForest 預測
    iso_cols = ['IWO_Vol_21d', 'SPY_Vol_21d', 'VIX_Change_1d', 'VIX_Gap', 'SPY_IWO_Div_21d', 'TNX_Change_5d']
    X_iso = current_market_feat[iso_cols].fillna(0)
    is_anomaly = iso_model.predict(X_iso)[0] # -1 is anomaly
    
    # 判斷結果
    is_safe = (mapped_state != 2) and (is_anomaly != -1)
    
    status_msg = "PASS (Safe)" if is_safe else "REJECT (High Risk)"
    print(f"Today's State: {mapped_state} (0=Bull, 1=Chop, 2=Crash)")
    print(f"Anomaly Check: {'Anomaly' if is_anomaly == -1 else 'Normal'}")
    print(f"System Decision: {status_msg}")
    
    return is_safe

def scan_candidates(df_stocks):
    """掃描 V5 基礎訊號"""
    print("\n--- 掃描 V5 候選標的 ---")
    if df_stocks.empty:
        print("錯誤：個股數據為空。")
        return pd.DataFrame()

    last_date = df_stocks.index.get_level_values('timestamp').max()
    print(f"數據日期: {last_date.date()}")
    
    # 重置 index 以方便計算
    df = df_stocks.reset_index(level='symbol')
    
    results = []
    # 使用 groupby 效率較低但邏輯清晰，適合盤後執行
    for symbol, group in df.groupby('symbol'):
        if len(group) < 200: continue
        
        close = group['Close']
        
        # 1. SMA 200
        sma200 = ta.sma(close, length=200)
        
        # 2. RSI 2
        rsi2 = ta.rsi(close, length=2)
        
        # 取得最新一筆
        curr_price = close.iloc[-1]
        curr_sma = sma200.iloc[-1]
        curr_rsi = rsi2.iloc[-1]
        
        # 檢查 NaN
        if pd.isna(curr_sma) or pd.isna(curr_rsi):
            continue

        # V5 邏輯: Price > SMA200 AND RSI2 < 10
        if (curr_price > curr_sma) and (curr_rsi < 10):
            results.append({
                'Symbol': symbol,
                'Close': curr_price,
                'RSI_2': curr_rsi,
                'Dist_SMA200': (curr_price / curr_sma) - 1
            })
            
    return pd.DataFrame(results)

def main():
    # 1. 準備數據
    tickers = load_assets()
    
    try:
        df_stocks, df_macro = get_latest_data(tickers)
    except ValueError as e:
        print(f"\n[執行終止] {e}")
        return
    
    # 2. L1 防禦
    market_feat = build_market_features_live(df_macro)
    is_safe = check_l1_defense(market_feat)
    
    if not is_safe:
        print("\n[系統熔斷] 今日市場風險過高，停止所有買入操作。")
        return
    
    # 3. L2 掃描
    buy_list = scan_candidates(df_stocks)
    
    if not buy_list.empty:
        # 排序：RSI 越低越好
        buy_list = buy_list.sort_values('RSI_2')
        print(f"\n[買入清單] 共發現 {len(buy_list)} 檔標的：")
        print(buy_list.to_string(index=False))
        
        # 儲存 CSV
        today = datetime.now().strftime('%Y-%m-%d')
        buy_list.to_csv(f'buy_list_{today}.csv', index=False)
    else:
        print("\n[無訊號] 今日無符合條件的標的。")

if __name__ == "__main__":
    main()
