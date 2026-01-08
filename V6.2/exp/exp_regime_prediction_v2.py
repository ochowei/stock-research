import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
    """修正 yfinance 代號問題"""
    if "BRK.B" in ticker: return "BRK-B" # 修正波克夏
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    """從 V6.2 resource 載入資產清單"""
    tickers = []
    # 載入 Final Pool
    final_pool_path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(final_pool_path):
        with open(final_pool_path, 'r') as f:
            data = json.load(f)
            raw_tickers = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw_tickers)
            
    # 手動補強測試標的
    default_tickers = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'GME', 'AMC', 'COIN', 'MSTR']
    tickers.extend(default_tickers)
    
    cleaned_tickers = list(set([clean_ticker(t) for t in tickers]))
    print(f"📋 Loaded {len(cleaned_tickers)} tickers (Cleaned).")
    return cleaned_tickers

def calculate_efficiency_ratio(close_series, window=5):
    """
    計算考夫曼效率比率 (Efficiency Ratio)
    ER = Net Change / Sum of Absolute Changes
    """
    net_change = (close_series - close_series.shift(window)).abs()
    sum_abs_change = close_series.diff().abs().rolling(window).sum()
    return net_change / (sum_abs_change + 1e-9)

def prepare_features_and_targets_v2(df, lookahead=5):
    """
    V2 改進版特徵工程
    """
    df = df.copy()
    
    # --- X: 特徵工程 ---
    
    # 1. 體制慣性 (Regime Momentum) - NEW!
    # "過去的趨勢強度，是未來趨勢強度最好的預測指標"
    df['X_Prev_ER'] = calculate_efficiency_ratio(df['Close'], window=5)
    
    # 2. 缺口資訊 (Gap Info) - NEW!
    # 大缺口通常代表動能爆發
    df['Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['X_Gap_Abs'] = df['Gap_Pct'].abs() * 100
    
    # 3. BB Squeeze (擠壓度)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Width'] = (df['MA20'] + 2*df['STD20'] - (df['MA20'] - 2*df['STD20'])) / df['MA20']
    # 正規化 Squeeze (0~1)
    df['X_Squeeze_Norm'] = (df['BB_Width'] - df['BB_Width'].rolling(50).min()) / (df['BB_Width'].rolling(50).max() - df['BB_Width'].rolling(50).min() + 1e-9)
    
    # 4. 相對波動率 (RVol)
    df['X_RVol'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)

    # 5. Day-Night Correlation (15天)
    df['R_day'] = (df['Close'] - df['Open']) / df['Open']
    df['R_night'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['X_Corr_15'] = df['R_day'].rolling(15).corr(df['R_night']).fillna(0)

    # --- Y: 目標變數 (未來體制) ---
    
    # 計算未來 5 天的 ER
    future_net = (df['Close'].shift(-lookahead) - df['Close']).abs()
    future_sum = df['Close'].diff().abs().rolling(lookahead).sum().shift(-lookahead)
    df['Future_ER'] = future_net / (future_sum + 1e-9)
    
    # [關鍵修改] 使用分位數定義 Y，而非固定閾值
    # 我們只把 "最明顯的趨勢" (Top 25%) 標記為 Danger
    # 這會由外部函數統一計算閾值，這裡先保留數值
    
    df = df.dropna()
    return df

def run_experiment_v2(tickers):
    all_data = []
    
    print(f"📥 [V2] 下載數據並生成特徵 (Period: {START_DATE} to {END_DATE})...")
    
    # 遍歷所有 ticker 下載並處理
    for ticker in tickers: 
        try:
            # print(f"   Processing {ticker}...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            
            if len(df) < 200: continue
            if df.isnull().all().all(): continue

            processed_df = prepare_features_and_targets_v2(df)
            
            if not processed_df.empty:
                # 只保留需要的欄位以節省記憶體
                cols = ['X_Prev_ER', 'X_Gap_Abs', 'X_Squeeze_Norm', 'X_RVol', 'X_Corr_15', 'Future_ER']
                all_data.append(processed_df[cols])
            
        except Exception as e:
            pass # 忽略錯誤以保持輸出整潔
            
    if not all_data:
        print("❌ No data collected.")
        return

    # 合併所有數據
    full_df = pd.concat(all_data)
    
    # --- 動態定義標籤 (Labeling) ---
    # 定義 "Danger Zone" 為 Future_ER 的前 25% (強趨勢)
    # 這確保我們預測的是真正的極端狀況
    threshold = full_df['Future_ER'].quantile(0.75)
    print(f"📊 定義趨勢閾值 (Top 25% ER): {threshold:.4f}")
    
    full_df['Y_Is_Danger'] = (full_df['Future_ER'] > threshold).astype(int)
    
    # 準備 X, y
    feature_cols = ['X_Prev_ER', 'X_Gap_Abs', 'X_Squeeze_Norm', 'X_RVol', 'X_Corr_15']
    X = full_df[feature_cols].values
    y = full_df['Y_Is_Danger'].values
    
    print(f"📊 總樣本數: {len(y)}")
    print(f"   危險盤 (Danger/Trend) 比例: {np.mean(y):.2%}")
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 訓練模型
    print("🤖 訓練 RandomForest Classifier (V2)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=20, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n----- Model Evaluation (V2) -----")
    print(classification_report(y_test, y_pred, target_names=['Safe (Chop)', 'Danger (Trend)']))
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 特徵重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n----- Feature Importance (V2) -----")
    for f in range(X.shape[1]):
        print(f"{f+1}. {feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # 繪製圖表
    plt.figure(figsize=(10, 6))
    plt.title(f"Regime-Alpha V2 Importance (Top 25% Trend)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_cols[i] for i in indices])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_regime_v2_importance.png"))
    print(f"\n📈 圖表已儲存至 {OUTPUT_DIR}/exp_regime_v2_importance.png")

if __name__ == "__main__":
    print("🔬 V6.2 Experiment: Regime-Alpha V2 (Trend vs Chop)")
    tickers = load_tickers()
    run_experiment_v2(tickers)