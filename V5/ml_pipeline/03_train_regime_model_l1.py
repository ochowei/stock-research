import pandas as pd
import numpy as np
import os
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- 設定隨機種子以確保可重現性 ---
np.random.seed(42)

def load_features(features_dir):
    """Loads the L0 market features."""
    path = os.path.join(features_dir, 'market_features_L0.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Market features not found at {path}")
    print(f"Loading market features from {path}...")
    return pd.read_parquet(path)

def train_hmm_model(df, n_components=3):
    """
    Trains a Gaussian HMM to identify market regimes.
    Regimes are sorted by volatility to ensure: 0=Low Vol, 1=Med Vol, 2=High Vol (Crash).
    """
    print("\n--- Training Gaussian HMM ---")
    
    # Selecting features for HMM
    # We use features that distinctly characterize market regimes:
    # 1. SPY_Ret: Direction
    # 2. IWO_Vol_21d: Risk Appetite Volatility
    # 3. SPY_IWO_Div_21d: Market Breadth/Structure
    feature_cols = ['SPY_Ret', 'IWO_Vol_21d', 'SPY_IWO_Div_21d']
    X = df[feature_cols].copy()
    
    # Handle infinite/nan just in case
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train HMM
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)
    
    # Predict states
    hidden_states = model.predict(X_scaled)
    
    # --- State Interpretation & Sorting ---
    # HMM states (0, 1, 2) are arbitrary. We need to map them to logic.
    # Logic: The state with the highest Average Volatility (IWO_Vol_21d) is "Crash".
    # Logic: The state with lowest Volatility is likely "Bull" or "Quiet".
    
    state_vol_means = []
    for i in range(n_components):
        # Calculate mean normalized volatility for this state
        mask = (hidden_states == i)
        # using the 2nd column (index 1) which corresponds to IWO_Vol_21d
        vol_mean = X_scaled[mask, 1].mean() 
        state_vol_means.append((i, vol_mean))
    
    # Sort states by volatility: Low -> High
    sorted_states = sorted(state_vol_means, key=lambda x: x[1])
    
    # Create a mapping dictionary: Old_ID -> New_ID
    # New ID 0: Low Vol (Bull/Calm)
    # New ID 1: Med Vol (Chop/Correction)
    # New ID 2: High Vol (Crash/Panic)
    state_map = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_states)}
    
    print("State Mapping (based on IWO Volatility):")
    labels = ["Bull/Calm", "Chop/Correction", "Crash/Panic"]
    for old_id, mean_vol in sorted_states:
        new_id = state_map[old_id]
        print(f"  Original State {old_id} (Vol: {mean_vol:.4f}) -> New State {new_id} ({labels[new_id]})")
        
    # Remap the hidden states
    remapped_states = np.array([state_map[s] for s in hidden_states])
    
    # Save results to a Series aligning with original index (handling dropped NaNs)
    regime_series = pd.Series(remapped_states, index=X.index, name='HMM_State')
    
    return model, scaler, regime_series, state_map

def train_isolation_forest(df):
    """
    Trains an Isolation Forest to detect anomalies (unknown risks).
    """
    print("\n--- Training Isolation Forest ---")
    
    # Use a broader set of features for anomaly detection
    feature_cols = [
        'IWO_Vol_21d', 'SPY_Vol_21d', 
        'VIX_Change_1d', 'VIX_Gap', 
        'SPY_IWO_Div_21d', 'TNX_Change_5d'
    ]
    
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    
    # No need to standardize for Tree-based models, but good for consistency if we wanted
    # Isolation Forest works fine without scaling usually.
    
    # Contamination: Estimate of outlier proportion. 
    # V5 Logic: We want to flag the top 5% most "alien" market days.
    iso_model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_model.fit(X)
    
    # decision_function: Lower = More Abnormal. 
    # predict: -1 = Outlier, 1 = Inlier.
    # We want an "Anomaly Score" where Higher = More Anomalous.
    # Scikit-learn's score_samples or decision_function yields higher values for inliers.
    # So we take negative decision_function as our "Risk Score".
    
    anomaly_scores = -iso_model.decision_function(X)
    is_anomaly = iso_model.predict(X) # -1 for anomaly
    
    # Convert to binary flag (1 = Anomaly, 0 = Normal) for easier usage
    anomaly_flag = np.where(is_anomaly == -1, 1, 0)
    
    results = pd.DataFrame({
        'Anomaly_Score': anomaly_scores,
        'Is_Anomaly': anomaly_flag
    }, index=X.index)
    
    print(f"  - Detected {anomaly_flag.sum()} anomalies out of {len(X)} days.")
    return iso_model, results

def main():
    # --- Setup Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURES_DIR = os.path.join(SCRIPT_DIR, 'features')
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    SIGNALS_DIR = os.path.join(SCRIPT_DIR, 'signals')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    
    # 1. Load Data
    market_df = load_features(FEATURES_DIR)
    
    # 2. Train HMM
    hmm_model, hmm_scaler, hmm_states, state_map = train_hmm_model(market_df)
    
    # 3. Train Isolation Forest
    iso_model, iso_results = train_isolation_forest(market_df)
    
    # 4. Merge Signals
    # Use market_df index as base to ensure alignment
    signals = pd.DataFrame(index=market_df.index)
    signals['HMM_State'] = hmm_states
    signals['Anomaly_Score'] = iso_results['Anomaly_Score']
    signals['Is_Anomaly'] = iso_results['Is_Anomaly']
    
    # Fill NaNs (for the initial rolling window periods dropped during training)
    # Forward fill is risky for signals, better to fill with "Safe" defaults or drop.
    # Default: State 0 (Bull), Score 0 (Normal)
    signals['HMM_State'] = signals['HMM_State'].fillna(0).astype(int)
    signals['Anomaly_Score'] = signals['Anomaly_Score'].fillna(0)
    signals['Is_Anomaly'] = signals['Is_Anomaly'].fillna(0).astype(int)
    
    # 5. Save Artifacts
    print("\n--- Saving Artifacts ---")
    
    # Models
    joblib.dump(hmm_model, os.path.join(MODELS_DIR, 'hmm_model.joblib'))
    joblib.dump(hmm_scaler, os.path.join(MODELS_DIR, 'hmm_scaler.joblib')) # Important to save scaler!
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'iso_forest.joblib'))
    joblib.dump(state_map, os.path.join(MODELS_DIR, 'hmm_state_map.joblib'))
    
    # Signals
    signals_out_path = os.path.join(SIGNALS_DIR, 'regime_signals.parquet')
    signals.to_parquet(signals_out_path)
    
    print(f"Models saved to {MODELS_DIR}")
    print(f"Signals saved to {signals_out_path}")
    print("\nStep 2: L1 Regime Identification Complete.")
    
    # Simple Validation Print
    print("\n[Validation] Recent Signals:")
    print(signals.tail(10))

if __name__ == "__main__":
    main()