import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

# --- 設定 ---
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "output"
START_DATE = "2020-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_ticker(ticker):
    """
    修正: 移除 yfinance 不支援的交易所前綴 (如 'NYSE:', 'NASDAQ:')
    """
    if ":" in ticker:
        return ticker.split(":")[-1]
    return ticker

def load_tickers():
    """從 V6.2 resource 載入資產清單並進行格式清洗"""
    tickers = []
    
    # 載入 Final Pool (績優股/權值股)
    final_pool_path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(final_pool_path):
        with open(final_pool_path, 'r') as f:
            data = json.load(f)
            # 處理 list 或 dict keys 兩種可能的 json 結構
            raw_tickers = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw_tickers)
            
    # 為了測試，我們也可以手動加入幾個代表性標的
    default_tickers = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'GME', 'AMC']
    tickers.extend(default_tickers)
    
    # 去重並清洗格式
    cleaned_tickers = list(set([clean_ticker(t) for t in tickers]))
    
    print(f"📋 Loaded {len(cleaned_tickers)} tickers (Cleaned).")
    return cleaned_tickers

def calculate_adx(df, window=14):
    """計算 ADX 指標 (簡易版)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/window).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx

def prepare_features_and_targets(df, lookahead=5):
    """
    核心邏輯：生成特徵 (X) 與 目標 (Y)
    
    Y (Target): 未來市場體制
      - 1 (Trend/Danger): 未來效率比率 (ER) 高，代表趨勢強烈，不適合反轉。
      - 0 (Chop/Safe): 未來效率比率低，代表震盪，適合 V6.1 反轉。
    """
    df = df.copy()
    
    # --- X: 特徵工程 (基於過去) ---
    
    # 1. BB Squeeze (布林通道擠壓)
    # 邏輯: 擠壓越小，變盤機率越大
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['Upper'] = df['MA20'] + 2 * df['STD20']
    df['Lower'] = df['MA20'] - 2 * df['STD20']
    df['BB_Width'] = (df['Upper'] - df['Lower']) / df['MA20']
    
    # 正規化 Squeeze (相對於過去 50 天)
    roll_min = df['BB_Width'].rolling(50).min()
    roll_max = df['BB_Width'].rolling(50).max()
    df['X_Squeeze_Norm'] = (df['BB_Width'] - roll_min) / (roll_max - roll_min + 1e-9)
    
    # 2. ADX (趨勢強度)
    df['X_ADX'] = calculate_adx(df)
    
    # 3. 日夜相關性 (Day-Night Correlation)
    # 計算過去 15 天日內回報與隔夜回報的相關係數
    df['R_day'] = (df['Close'] - df['Open']) / df['Open']
    df['R_night'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['X_Corr_15'] = df['R_day'].rolling(15).corr(df['R_night'])
    
    # 4. 相對波動率 (RVol)
    df['X_RVol'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # --- Y: 目標變數 (預測未來) ---
    
    # 計算未來 N 天的效率比率 (Efficiency Ratio)
    # ER = |Price_t+N - Price_t| / Sum(|Price_i - Price_i-1|)
    # ER 接近 1 -> 趨勢 (Trend)
    # ER 接近 0 -> 震盪 (Chop)
    future_net_change = (df['Close'].shift(-lookahead) - df['Close']).abs()
    future_sum_change = df['Close'].diff().abs().rolling(lookahead).sum().shift(-lookahead)
    df['Future_ER'] = future_net_change / (future_sum_change + 1e-9)
    
    # 定義 Label: 若 ER > 0.45 (經驗值)，視為強趨勢 (Danger Zone)
    # V6.1 策略在這種時候應該 SKIP
    threshold = 0.45
    df['Y_Is_Trend'] = (df['Future_ER'] > threshold).astype(int)
    
    # 清理 NaN
    df = df.dropna()
    return df

def run_experiment(tickers):
    all_X = []
    all_y = []
    
    print(f"📥 下載數據並生成特徵 (Period: {START_DATE} to {END_DATE})...")
    
    # 為了演示與快速驗證，這裡只取前 20 個 ticker
    # 正式跑實驗時請將切片 [:20] 移除 -> for ticker in tickers:
    for ticker in tickers[:20]: 
        try:
            print(f"   Processing {ticker}...")
            # 下載數據，遇到錯誤自動跳過
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            
            if len(df) < 200:
                print(f"   ⚠️ {ticker}: Data length insufficient ({len(df)}), skipping.")
                continue
            
            # 檢查是否有 NaN (yfinance 有時會下載空數據)
            if df.isnull().all().all():
                print(f"   ⚠️ {ticker}: Data is all NaN, skipping.")
                continue

            processed_df = prepare_features_and_targets(df)
            
            # 選取特徵欄位
            features = ['X_Squeeze_Norm', 'X_ADX', 'X_Corr_15', 'X_RVol']
            
            # 再次檢查處理後的數據是否為空
            if processed_df.empty:
                continue

            X = processed_df[features].values
            y = processed_df['Y_Is_Trend'].values
            
            all_X.append(X)
            all_y.append(y)
            
        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            
    if not all_X:
        print("❌ No data collected. Please check ticker list or internet connection.")
        return

    # 合併所有股票的數據
    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)
    
    print(f"📊 總樣本數: {len(y_full)}")
    print(f"   趨勢盤 (Trend) 比例: {np.mean(y_full):.2%}")
    print(f"   震盪盤 (Chop)  比例: {1 - np.mean(y_full):.2%}")
    
    # 分割訓練與測試集
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)
    
    # 訓練模型
    print("🤖 訓練 RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    print("\n----- Model Evaluation -----")
    # 注意: 這裡的 Target Names 對應 0 和 1
    print(classification_report(y_test, y_pred, target_names=['Safe (Chop)', 'Danger (Trend)']))
    
    # 特徵重要性
    importances = model.feature_importances_
    feature_names = ['BB Squeeze', 'ADX', 'Day-Night Corr', 'Relative Vol']
    indices = np.argsort(importances)[::-1]
    
    print("\n----- Feature Importance -----")
    for f in range(X_train.shape[1]):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # 繪製重要性圖表
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance for Detecting Trend Regime (Danger)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_regime_importance.png"))
    print(f"\n📈 圖表已儲存至 {OUTPUT_DIR}/exp_regime_importance.png")

if __name__ == "__main__":
    print("🔬 V6.2 Experiment: Regime-Alpha (Trend vs Chop Prediction)")
    tickers = load_tickers()
    run_experiment(tickers)