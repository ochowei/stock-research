import pandas as pd
import numpy as np
import os
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- 設定隨機種子 ---
np.random.seed(42)

# --- 參數設定 (滾動視窗) ---
TRAIN_WINDOW = 252 * 2  # 訓練窗口：2 年 (約 504 交易日)
REFIT_STEP = 63         # 重訓練頻率：3 個月 (約 63 交易日)

def load_features(features_dir):
    """載入 L0 市場特徵數據"""
    path = os.path.join(features_dir, 'market_features_L0.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Market features not found at {path}")
    print(f"Loading market features from {path}...")
    df = pd.read_parquet(path)
    return df.sort_index()

def train_and_predict_fold(train_df, test_df, n_components=3):
    """
    單一 Fold 的訓練與預測流程
    """
    # ==========================
    # 1. 準備數據 (Fit Scaler on TRAIN only)
    # ==========================
    hmm_cols = ['SPY_Ret', 'IWO_Vol_21d', 'SPY_IWO_Div_21d']
    iso_cols = ['IWO_Vol_21d', 'SPY_Vol_21d', 'VIX_Change_1d', 'VIX_Gap', 'SPY_IWO_Div_21d', 'TNX_Change_5d']
    
    # 清理 NaN (僅針對訓練集，測試集若有 NaN 需填補或跳過)
    X_train_hmm = train_df[hmm_cols].dropna()
    
    # 若訓練數據過少，無法訓練
    if len(X_train_hmm) < 100:
        return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hmm)
    
    # 準備測試數據 (使用訓練集的 Scaler 轉換)
    # 注意：測試集可能有 NaN (如剛開盤)，這裡簡單用 0 填補或 forward fill，實際交易需更嚴謹
    X_test_hmm = test_df[hmm_cols].fillna(method='ffill').fillna(0)
    X_test_scaled = scaler.transform(X_test_hmm)
    
    # ==========================
    # 2. 訓練 HMM (Regime Detection)
    # ==========================
    hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42, verbose=False)
    hmm_model.fit(X_train_scaled)
    
    # --- 動態狀態映射 (Dynamic State Mapping) ---
    # 每次訓練後的 State 0,1,2 意義不同，需根據波動率 (IWO_Vol_21d) 重新定義
    # IWO_Vol_21d 是 hmm_cols 的第 2 個特徵 (Index 1)
    
    # 預測訓練集狀態以計算統計量
    train_states = hmm_model.predict(X_train_scaled)
    
    state_vol_means = []
    for i in range(n_components):
        mask = (train_states == i)
        if mask.sum() > 0:
            vol_mean = X_train_scaled[mask, 1].mean() # Index 1 is IWO_Vol_21d
        else:
            vol_mean = -999 # 該狀態未出現
        state_vol_means.append((i, vol_mean))
        
    # 排序：波動率低 -> 高 (0=Bull, 1=Chop, 2=Crash)
    sorted_states = sorted(state_vol_means, key=lambda x: x[1])
    state_map = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_states)}
    
    # --- 預測測試集 (OOS) ---
    hidden_states_oos = hmm_model.predict(X_test_scaled)
    mapped_states_oos = np.array([state_map[s] for s in hidden_states_oos])
    
    # ==========================
    # 3. 訓練 Isolation Forest (Anomaly Detection)
    # ==========================
    X_train_iso = train_df[iso_cols].fillna(0)
    X_test_iso = test_df[iso_cols].fillna(0)
    
    iso_model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_model.fit(X_train_iso)
    
    # 預測 OOS
    # predict: -1 = Anomaly, 1 = Normal
    is_anomaly_oos = iso_model.predict(X_test_iso)
    anomaly_scores_oos = -iso_model.decision_function(X_test_iso)
    
    # 轉換為 0/1 (1 = Anomaly)
    is_anomaly_oos = np.where(is_anomaly_oos == -1, 1, 0)
    
    # ==========================
    # 4. 打包結果
    # ==========================
    # 我們需要最後一個訓練好的模型用於 "Live Trading" (存檔用)
    artifacts = {
        'hmm_model': hmm_model,
        'hmm_scaler': scaler,
        'iso_model': iso_model,
        'state_map': state_map
    }
    
    oos_result_df = pd.DataFrame(index=test_df.index)
    oos_result_df['HMM_State'] = mapped_states_oos
    oos_result_df['Is_Anomaly'] = is_anomaly_oos
    oos_result_df['Anomaly_Score'] = anomaly_scores_oos
    
    return oos_result_df, artifacts

def main():
    # --- 路徑設定 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURES_DIR = os.path.join(SCRIPT_DIR, 'features')
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    SIGNALS_DIR = os.path.join(SCRIPT_DIR, 'signals')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    
    # 1. 載入數據
    market_df = load_features(FEATURES_DIR)
    print(f"Total Data Points: {len(market_df)}")
    
    # 2. 執行滾動視窗訓練 (Rolling Window Training)
    print(f"\n--- Starting Rolling Window Training ---")
    print(f"Window Size: {TRAIN_WINDOW} days, Step Size: {REFIT_STEP} days")
    
    oos_predictions = []
    latest_artifacts = None
    
    # 從足夠的數據開始滾動
    # Range: [Start of Test Data] -> End
    for i in range(TRAIN_WINDOW, len(market_df), REFIT_STEP):
        # 定義窗口
        # Train: [i - TRAIN_WINDOW : i]
        # Test:  [i : i + REFIT_STEP]
        train_idx = market_df.index[i - TRAIN_WINDOW : i]
        
        end_loc = min(i + REFIT_STEP, len(market_df))
        test_idx = market_df.index[i : end_loc]
        
        if len(test_idx) == 0:
            break
            
        train_df = market_df.loc[train_idx]
        test_df = market_df.loc[test_idx]
        
        test_start_date = test_df.index.min().date()
        test_end_date = test_df.index.max().date()
        
        print(f"Processing Fold: Train[{train_df.index.min().date()} ~ {train_df.index.max().date()}] -> Predict[{test_start_date} ~ {test_end_date}]")
        
        result_df, artifacts = train_and_predict_fold(train_df, test_df)
        
        if result_df is not None:
            oos_predictions.append(result_df)
            latest_artifacts = artifacts # 保存最後一折的模型作為「最新模型」
    
    # 3. 合併所有 OOS 預測
    if oos_predictions:
        full_oos_df = pd.concat(oos_predictions)
        full_oos_df = full_oos_df.sort_index()
        
        # 填補前面的 Warm-up 期 (用 0 填充或設為 NaN)
        # 為了與其他數據對齊，我們重新索引回原始 market_df 的索引
        final_signals = full_oos_df.reindex(market_df.index)
        
        # Warm-up 期設為安全狀態 (State 0, Normal) 避免回測報錯，但回測時應避開這段期間
        final_signals['HMM_State'] = final_signals['HMM_State'].fillna(0).astype(int)
        final_signals['Is_Anomaly'] = final_signals['Is_Anomaly'].fillna(0).astype(int)
        final_signals['Anomaly_Score'] = final_signals['Anomaly_Score'].fillna(0)
        
        # 4. 存檔
        print("\n--- Saving OOS Artifacts ---")
        
        # 保存信號
        signals_out_path = os.path.join(SIGNALS_DIR, 'regime_signals.parquet')
        final_signals.to_parquet(signals_out_path)
        print(f"OOS Signals saved to {signals_out_path}")
        
        # 保存最後一個模型 (用於實盤/明日推論)
        if latest_artifacts:
            joblib.dump(latest_artifacts['hmm_model'], os.path.join(MODELS_DIR, 'hmm_model.joblib'))
            joblib.dump(latest_artifacts['hmm_scaler'], os.path.join(MODELS_DIR, 'hmm_scaler.joblib'))
            joblib.dump(latest_artifacts['iso_model'], os.path.join(MODELS_DIR, 'iso_forest.joblib'))
            joblib.dump(latest_artifacts['state_map'], os.path.join(MODELS_DIR, 'hmm_state_map.joblib'))
            print(f"Latest Models saved to {MODELS_DIR} (Ready for Live Trading)")
            
        print("\nStep 2: L1 Rolling Regime Identification (100% OOS) Complete.")
        print(f"OOS Signal Count: {len(full_oos_df)} (Warm-up period: {TRAIN_WINDOW} days)")
        
    else:
        print("Error: No predictions generated. Check data length.")

if __name__ == "__main__":
    main()