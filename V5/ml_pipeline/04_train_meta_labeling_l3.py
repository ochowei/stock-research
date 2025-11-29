import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

def load_data(base_dir):
    """Loads L0 features and L1 signals."""
    features_path = os.path.join(base_dir, 'features', 'stock_features_L0.parquet')
    signals_path = os.path.join(base_dir, 'signals', 'regime_signals.parquet')
    
    if not os.path.exists(features_path) or not os.path.exists(signals_path):
        raise FileNotFoundError("Input files not found. Run Step 1 & 2 first.")
        
    print("Loading features and signals...")
    stock_features = pd.read_parquet(features_path)
    regime_signals = pd.read_parquet(signals_path)
    return stock_features, regime_signals

def generate_l2_signals(df):
    """
    Generates Base Strategy (L2) candidates.
    Logic: Price > SMA200 (Trend) AND RSI(2) < 10 (Mean Reversion).
    """
    print("Generating L2 Base Strategy Signals...")
    
    # Filter conditions
    # Note: We use 'SMA_200' and 'RSI_2' created in Step 1
    # Dist_SMA_200 > 0 implies Close > SMA_200
    
    trend_condition = df['Dist_SMA_200'] > 0
    oversold_condition = df['RSI_2'] < 10
    
    # Create a boolean mask
    entries = trend_condition & oversold_condition
    
    # Extract signal rows
    signals = df[entries].copy()
    
    print(f"  - Found {len(signals)} candidate signals out of {len(df)} total rows.")
    return signals, df # Return full df for future return calculation

def create_meta_labels(signals, full_df, hold_period=5):
    """
    Creates Meta-Labels (Y) for supervised learning.
    Y = 1 if the trade was profitable after 'hold_period' days, else 0.
    """
    print(f"Creating Meta-Labels (Hold Period = {hold_period} days)...")
    
    # We need to look ahead to get the exit price.
    # Since full_df is MultiIndex (symbol, timestamp), we can pivot or use groupby shift.
    # Pivot is easier for vectorized lookahead.
    
    closes = full_df['Close'].unstack(level='symbol')
    
    # Calculate future returns for ALL days first (vectorized)
    # Ret_N = (Close_t+N / Close_t) - 1
    future_returns = closes.shift(-hold_period) / closes - 1
    
    # Stack back to match signals index structure
    future_returns_stacked = future_returns.stack().reorder_levels(['symbol', 'timestamp']).sort_index()
    
    # Join returns to our signals
    # Inner join will keep only the signal rows
    labeled_signals = signals.join(future_returns_stacked.rename('Future_Return'))
    
    # Define Label: 1 if Return > 0, else 0
    # (Optional: Can add transaction cost threshold here, e.g., > 0.002)
    labeled_signals['Meta_Label'] = (labeled_signals['Future_Return'] > 0).astype(int)
    
    # Drop signals where Future_Return is NaN (e.g., last 5 days of data)
    labeled_signals.dropna(subset=['Future_Return'], inplace=True)
    
    print(f"  - Labeled {len(labeled_signals)} signals. Win Rate: {labeled_signals['Meta_Label'].mean():.2%}")
    return labeled_signals

def train_l3_model(labeled_signals, regime_signals):
    """
    Trains the L3 Meta-Labeling Model (Random Forest).
    Inputs: L0 Stock Features + L1 Regime Features.
    Target: Meta_Label.
    """
    print("\n--- Training L3 Meta-Model ---")
    
    # 1. Merge L1 Regime Features into Signals
    # regime_signals index is timestamp. signals index is (symbol, timestamp).
    # We join on timestamp.
    
    # Reset index to join
    df_model = labeled_signals.reset_index()
    
    # Merge on timestamp
    df_model = df_model.merge(regime_signals, on='timestamp', how='left')
    
    # Set index back
    df_model.set_index(['symbol', 'timestamp'], inplace=True)
    
    # 2. Define Features & Target
    # L0 Features
    l0_cols = [
        'RSI_2', 'RSI_14', 
        'Dist_SMA_200', 'BB_PctB', 
        'ATR_Norm', 'Rel_Vol'
    ]
    # L1 Features
    l1_cols = ['HMM_State', 'Anomaly_Score'] # Is_Anomaly is a hard filter, maybe include score as feature
    
    X = df_model[l0_cols + l1_cols].copy()
    y = df_model['Meta_Label']
    
    # Handle NaNs (if any regime signals are missing)
    X.fillna(0, inplace=True)
    
    # 3. Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,       # Prevent overfitting
        min_samples_leaf=20, # Ensure generalization
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' # Handle potential imbalance
    )
    
    print("Performing Walk-Forward Validation...")
    
    fold_metrics = []
    
    # Sort by time for correct split
    X.sort_index(level='timestamp', inplace=True)
    y = y.reindex(X.index)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        
        print(f"  Fold {fold+1}: AUC={auc:.3f}, Acc={acc:.3f}, Precision={prec:.3f}")
        fold_metrics.append({'auc': auc, 'acc': acc, 'prec': prec})
        
    # 4. Final Training on All Data (up to T)
    print("Retraining final model on all data...")
    model.fit(X, y)
    
    # 5. Get Probabilities for the whole dataset (for analysis)
    # Note: In production, we only predict OOS. Here we just want to see the distribution.
    # To be rigorous, we should use OOS preds, but for 'l3_probabilities.csv' visualization it's okay.
    all_probs = model.predict_proba(X)[:, 1]
    df_model['L3_Prob'] = all_probs
    
    return model, df_model, fold_metrics

def main():
    # --- Setup Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PIPELINE_DIR = SCRIPT_DIR
    MODELS_DIR = os.path.join(PIPELINE_DIR, 'models')
    SIGNALS_DIR = os.path.join(PIPELINE_DIR, 'signals')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    
    # 1. Load Data
    stock_features, regime_signals = load_data(PIPELINE_DIR)
    
    # 2. Generate L2 Signals (Candidates)
    l2_signals, full_df = generate_l2_signals(stock_features)
    
    # Save Raw L2 Trades
    l2_out_path = os.path.join(SIGNALS_DIR, 'base_strategy_trades.parquet')
    l2_signals.to_parquet(l2_out_path)
    print(f"Saved Base Strategy Trades to {l2_out_path}")
    
    # 3. Create Labels (Meta-Labeling)
    labeled_data = create_meta_labels(l2_signals, full_df)
    
    # 4. Train L3 Model
    l3_model, results_df, metrics = train_l3_model(labeled_data, regime_signals)
    
    # 5. Save Artifacts
    model_path = os.path.join(MODELS_DIR, 'l3_meta_filter.joblib')
    joblib.dump(l3_model, model_path)
    
    probs_path = os.path.join(SIGNALS_DIR, 'l3_probabilities.csv')
    results_df[['Meta_Label', 'L3_Prob', 'Future_Return']].to_csv(probs_path)
    
    print(f"\nModels saved to {model_path}")
    print(f"Probabilities saved to {probs_path}")
    print("\nStep 3: L3 Meta-Labeling Complete.")
    
    # Summary
    avg_auc = np.mean([m['auc'] for m in metrics])
    print(f"\n[Summary] Average Test AUC: {avg_auc:.3f}")
    print("Interpretation: AUC > 0.5 means the model is adding value over random guessing.")

if __name__ == "__main__":
    main()