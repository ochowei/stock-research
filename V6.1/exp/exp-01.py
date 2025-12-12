import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 1. 實驗配置 (擴充版) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', '..', 'V6.0', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 時間切分：訓練期拉長至 9 年
TRAIN_START = '2015-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# 新增標的 (大盤指數)
ADDITIONAL_TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']

def load_tickers():
    """讀取 V6.0 清單並加入大盤指數"""
    files = ['2025_final_asset_pool.json', '2025_final_toxic_asset_pool.json']
    tickers = []
    
    # 讀取原始 JSON
    for f in files:
        path = os.path.join(RESOURCE_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                raw = json.load(json_file)
                # 清洗 "NYSE:MP" -> "MP"
                clean = [t.split(':')[-1].strip() for t in raw]
                tickers.extend(clean)
    
    # 加入額外標的
    tickers.extend(ADDITIONAL_TICKERS)
    
    # 去重並清洗 (BRK.B -> BRK-B)
    clean_tickers = [t.replace('.', '-').strip() for t in tickers]
    return sorted(list(set(clean_tickers)))

def fetch_data(tickers):
    """
    下載長週期數據 (2015-2025)
    """
    print(f"Downloading data for {len(tickers)} tickers ({TRAIN_START} ~ {TEST_END})...")
    
    try:
        data = yf.download(
            tickers, 
            start=TRAIN_START, 
            end=TEST_END, 
            interval='1d', 
            auto_adjust=True, 
            progress=True,
            timeout=60,
            threads=True
        )
    except Exception as e:
        print(f"[Critical Error] Batch download failed: {e}")
        return {}
    
    # 處理 MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Pandas 2.x
            data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        except TypeError:
            # Pandas 1.x
            data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            
        data_dict = {}
        for ticker, group in data.groupby('Ticker'):
            df = group.set_index('Date').sort_index()
            
            # 簡單過濾無效數據
            if df['Close'].sum() == 0 or df.empty:
                print(f"  [Warning] {ticker} has no valid data. Dropping.")
                continue
            
            # 確保欄位
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_dict[ticker] = df
            
        print(f"Successfully processed {len(data_dict)} tickers.")
        return data_dict
        
    return {}

def calculate_features_and_labels(data_dict):
    """
    計算特徵與標籤 (基於擴充後的訓練期)
    """
    dataset = []
    
    print("Calculating Features & Labels (Train Period: 2015-2023)...")
    
    for ticker, df in data_dict.items():
        # --- 1. 切割 Train Data ---
        mask_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
        df_train = df.loc[mask_train].copy()
        
        # 由於拉長到 9 年，我們要求至少有 2 年 (500天) 的數據才納入訓練
        if len(df_train) < 500: 
            # print(f"  Skipping {ticker}: Insufficient history ({len(df_train)} days)")
            continue

        # --- 2. 標籤生成 (Y): Strategy A 表現 ---
        df_train['Prev_Close'] = df_train['Close'].shift(1)
        df_train['Ret_Hold'] = df_train['Close'].pct_change()
        df_train['Ret_Gap'] = (df_train['Open'] - df_train['Prev_Close']) / df_train['Prev_Close']
        
        mask_gap = df_train['Ret_Gap'] > 0.005
        strat_ret = np.where(mask_gap, df_train['Ret_Gap'], df_train['Ret_Hold'])
        
        # 績效指標
        cum_ret = (1 + pd.Series(strat_ret).fillna(0)).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        
        roll_max = cum_ret.cummax()
        drawdown = (cum_ret - roll_max) / roll_max
        max_dd = drawdown.min()
        
        calmar = total_ret / abs(max_dd) if max_dd < 0 else 0
        win_rate = (strat_ret > 0).mean()

        # Label 定義: 適合 (1) vs 不適合 (0)
        # 考慮到 9 年的長週期，Calmar > 0.5 已經相當不錯
        is_suitable = 1 if (calmar > 0.5 and win_rate > 0.52) else 0
        
        # --- 3. 特徵工程 (X) ---
        # 計算全訓練區間的統計特徵，代表該股票的「長期屬性」
        
        # Amihud (Liquidity)
        daily_illiq = df_train['Ret_Hold'].abs() / (df_train['Close'] * df_train['Volume'])
        feat_amihud = daily_illiq.mean() * 1e6
        if np.isinf(feat_amihud) or np.isnan(feat_amihud): feat_amihud = 0
        
        # Volatility (ATR %)
        df_train['ATR'] = ta.atr(df_train['High'], df_train['Low'], df_train['Close'], length=14)
        feat_volatility = (df_train['ATR'] / df_train['Close']).mean()
        
        # Gap Frequency
        feat_gap_freq = mask_gap.mean()
        
        # Dollar Volume (Log)
        feat_dollar_vol = (df_train['Close'] * df_train['Volume']).mean()
        
        dataset.append({
            'Ticker': ticker,
            'Feature_Amihud': feat_amihud,
            'Feature_Volatility': feat_volatility,
            'Feature_GapFreq': feat_gap_freq,
            'Feature_DollarVol': np.log1p(feat_dollar_vol),
            'Label': is_suitable,
            'Train_Calmar': calmar
        })
        
    return pd.DataFrame(dataset)

def backtest_oos(data_dict, suitable_tickers, output_prefix):
    """
    在 Test 期間 (2024-2025) 驗證
    """
    print(f"Running OOS Backtest on {len(suitable_tickers)} suitable tickers...")
    
    oos_equity = {}
    
    for ticker in suitable_tickers:
        if ticker not in data_dict: continue
        
        df = data_dict[ticker]
        mask_test = (df.index >= TEST_START) & (df.index <= TEST_END)
        df_test = df.loc[mask_test].copy()
        
        if df_test.empty: continue
        
        df_test['Prev_Close'] = df_test['Close'].shift(1)
        df_test['Ret_Hold'] = df_test['Close'].pct_change()
        df_test['Ret_Gap'] = (df_test['Open'] - df_test['Prev_Close']) / df_test['Prev_Close']
        
        mask_gap = df_test['Ret_Gap'] > 0.005
        strat_ret = np.where(mask_gap, df_test['Ret_Gap'], df_test['Ret_Hold'])
        
        oos_equity[ticker] = pd.Series(strat_ret, index=df_test.index).fillna(0)
        
    if not oos_equity:
        return None
        
    df_rets = pd.DataFrame(oos_equity)
    port_ret = df_rets.mean(axis=1)
    return (1 + port_ret).cumprod()

def main():
    # 1. 準備數據
    tickers = load_tickers()
    data_map = fetch_data(tickers)
    
    # 2. 產生訓練集
    df_dataset = calculate_features_and_labels(data_map)
    print(f"\nDataset size: {len(df_dataset)}")
    print("Label distribution:\n", df_dataset['Label'].value_counts())
    
    if len(df_dataset) < 10:
        print("Not enough data to train model.")
        return

    # 3. 訓練模型
    features = ['Feature_Amihud', 'Feature_Volatility', 'Feature_GapFreq', 'Feature_DollarVol']
    X = df_dataset[features]
    y = df_dataset['Label']
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    
    # 4. 特徵重要性
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(importances)
    importances.to_csv(os.path.join(OUTPUT_DIR, 'exp_01_feature_importance_expanded.csv'), index=False)
    
    # 5. OOS 預測與驗證
    probs = clf.predict_proba(X)[:, 1]
    df_dataset['Prob_Suitable'] = probs
    
    # 篩選 Smart vs Naive
    smart_tickers = df_dataset[df_dataset['Prob_Suitable'] > 0.55]['Ticker'].tolist()
    naive_tickers = df_dataset['Ticker'].tolist()
    
    print(f"\nSelected {len(smart_tickers)} tickers for Smart Portfolio (out of {len(naive_tickers)})")
    
    # 跑 2024-2025 回測
    smart_curve = backtest_oos(data_map, smart_tickers, "Smart")
    naive_curve = backtest_oos(data_map, naive_tickers, "Naive")
    
    # 繪圖
    plt.figure(figsize=(12, 6))
    if smart_curve is not None:
        plt.plot(smart_curve, label=f'Smart Portfolio (n={len(smart_tickers)})', linewidth=2)
    if naive_curve is not None:
        plt.plot(naive_curve, label=f'Naive Portfolio (n={len(naive_tickers)})', linestyle='--', alpha=0.7)
        
    plt.title(f'OOS Validation ({TEST_START}~{TEST_END}): Smart vs Naive (Expanded Data)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp_01_smart_vs_naive_expanded.png'))
    print(f"Chart saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()