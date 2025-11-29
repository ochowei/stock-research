# V5/ml_pipeline/04_train_meta_labeling_l3.py (V5.1 Upgrade)

import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb  # Upgrade to LightGBM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

def load_data(base_dir):
    """Loads Stock Features, Regime Signals, AND Market Features."""
    features_path = os.path.join(base_dir, 'features', 'stock_features_L0.parquet')
    market_path = os.path.join(base_dir, 'features', 'market_features_L0.parquet') # New
    signals_path = os.path.join(base_dir, 'signals', 'regime_signals.parquet')
    
    if not os.path.exists(features_path) or not os.path.exists(market_path):
        raise FileNotFoundError("Input files not found. Run Step 1 & 2 first.")
        
    print("Loading features and signals...")
    stock_features = pd.read_parquet(features_path)
    market_features = pd.read_parquet(market_path) # New
    regime_signals = pd.read_parquet(signals_path)
    
    return stock_features, market_features, regime_signals

def generate_l2_signals(df):
    """Generates Base Strategy (L2) candidates."""
    print("Generating L2 Base Strategy Signals...")
    trend_condition = df['Dist_SMA_200'] > 0
    oversold_condition = df['RSI_2'] < 10
    entries = trend_condition & oversold_condition
    signals = df[entries].copy()
    print(f"  - Found {len(signals)} candidate signals.")
    return signals, df

def create_meta_labels(signals, full_df, hold_period=5):
    """
    Creates Meta-Labels using Triple Barrier Method (TBM).
    Label = 1 if price hits Take Profit (TP) before Stop Loss (SL) & Time Limit.
    TP = Entry + 1.0 * ATR
    SL = Entry - 1.0 * ATR (Symmetric Barrier)
    """
    print(f"Creating Meta-Labels (Triple Barrier Method, Width=1.0 ATR)...")
    
    # Need High/Low/Close for TBM
    # full_df is indexed by (symbol, timestamp)
    # Pivot for vectorized access (faster than loop)
    highs = full_df['High'].unstack(level='symbol')
    lows = full_df['Low'].unstack(level='symbol')
    closes = full_df['Close'].unstack(level='symbol')
    atrs = full_df['ATR_Norm'].unstack(level='symbol') * closes # Reconstruct absolute ATR
    
    # Align signals with price data
    # signals has columns: Dist_SMA_200, RSI_2, etc. We need its index.
    
    labels = []
    valid_signals_count = 0
    
    # Iterate through signals (vectorizing TBM is complex, looping signals is acceptable for ~8k rows)
    for (symbol, timestamp) in signals.index:
        try:
            # Get Entry Data
            entry_price = closes.loc[timestamp, symbol]
            vol = atrs.loc[timestamp, symbol]
            
            if pd.isna(vol) or vol == 0:
                continue
                
            # Define Barriers
            # V5 Strategy is Mean Reversion (Long), so we bet on up move.
            # Dynamic width: 1.0 ATR (Can be tuned to 0.5 or 2.0)
            tp_price = entry_price + (1.0 * vol)
            sl_price = entry_price - (1.0 * vol)
            
            # Get future path (next 5 days)
            # Find integer location of current timestamp
            idx = closes.index.get_loc(timestamp)
            
            # Slice next N days
            path_highs = highs[symbol].iloc[idx+1 : idx+1+hold_period]
            path_lows = lows[symbol].iloc[idx+1 : idx+1+hold_period]
            
            if path_highs.empty:
                continue
                
            # Check TBM Logic
            # 1. Did we hit SL?
            sl_hit_mask = path_lows < sl_price
            if sl_hit_mask.any():
                sl_date = sl_hit_mask.idxmax() # First date SL was hit
            else:
                sl_date = None
                
            # 2. Did we hit TP?
            tp_hit_mask = path_highs > tp_price
            if tp_hit_mask.any():
                tp_date = tp_hit_mask.idxmax()
            else:
                tp_date = None
            
            # Determine Outcome
            label = 0
            if tp_date:
                if sl_date:
                    # Both hit, check which was first
                    if tp_date <= sl_date:
                        label = 1 # Won before lost
                    else:
                        label = 0 # Lost before won
                else:
                    label = 1 # TP hit, SL never hit
            else:
                label = 0 # TP never hit (Time barrier or SL hit)
            
            # Store Result (We only need the label)
            labels.append(label)
            valid_signals_count += 1
            
        except KeyError:
            # Handle edge cases (e.g. delisted symbol)
            continue
        except Exception as e:
            print(f"Error processing {symbol} at {timestamp}: {e}")
            continue

    # Assign labels back to signals
    # Note: signals df might be larger than labels list if we skipped some. 
    # We should re-index or match carefully.
    # Simplest way: Re-filter signals to match the iterated valid ones.
    # But for simplicity in this snippet, let's assume we want to attach to original if possible.
    
    # Actually, simpler approach: Create a Series with MultiIndex
    # Since we looped over signals.index, labels correspond 1-to-1 IF we didn't skip.
    # We did skip NaNs/Empty paths.
    
    # Let's handle the list to Series conversion robustly
    # We need to filter the original signals to only those we processed?
    # No, let's just create a new DataFrame from the loop results.
    
    # Re-looping is inefficient but safe. 
    # Optimized approach:
    # Just store (index, label) tuples.
    
    # --- RERUNNING LOGIC FOR ROBUST STORAGE ---
    results = {}
    for (symbol, timestamp) in signals.index:
        try:
            entry_price = closes.loc[timestamp, symbol]
            vol = atrs.loc[timestamp, symbol]
            if pd.isna(vol) or vol == 0: continue
            
            tp_price = entry_price + (1.0 * vol)
            sl_price = entry_price - (1.0 * vol)
            
            idx = closes.index.get_loc(timestamp)
            path_highs = highs[symbol].iloc[idx+1 : idx+1+hold_period]
            path_lows = lows[symbol].iloc[idx+1 : idx+1+hold_period]
            
            if path_highs.empty: continue
            
            sl_hit = (path_lows < sl_price).any()
            sl_date = (path_lows < sl_price).idxmax() if sl_hit else pd.Timestamp.max
            
            tp_hit = (path_highs > tp_price).any()
            tp_date = (path_highs > tp_price).idxmax() if tp_hit else pd.Timestamp.max
            
            # Label = 1 only if TP hit AND TP <= SL
            if tp_hit and (tp_date <= sl_date):
                label = 1
            else:
                label = 0
            
            results[(symbol, timestamp)] = label
        except:
            continue
            
    # Create Series
    meta_labels = pd.Series(results, name='Meta_Label')
    meta_labels.index.names = ['symbol', 'timestamp']
    
    # Join back to signals
    labeled_signals = signals.join(meta_labels, how='inner') # Inner join drops failed calculations
    
    # Calculate "Future_Return" just for logging/analysis compatibility (use Close-to-Close)
    # Though Label is now path-dependent.
    ret_5d = (closes.shift(-hold_period) / closes - 1).stack()
    labeled_signals = labeled_signals.join(ret_5d.rename('Future_Return'), how='left')
    
    print(f"  - Labeled {len(labeled_signals)} signals. Win Rate (TP hit first): {labeled_signals['Meta_Label'].mean():.2%}")
    return labeled_signals

def train_l3_model(labeled_signals, market_features, regime_signals):
    """
    Trains the L3 Meta-Model using LightGBM.
    Inputs: Stock Features + Market Features + Regime Signals.
    """
    print("\n--- Training L3 Meta-Model (LightGBM) ---")
    
    # 1. Merge Features
    # Reset index to prepare for merge
    df_model = labeled_signals.reset_index()
    
    # Merge Regime Signals (on timestamp)
    df_model = df_model.merge(regime_signals, on='timestamp', how='left')
    
    # [New] Merge Market Features (on timestamp)
    # We select key macro indicators that might help the model
    market_cols_to_use = [
        'VIX_Close', 'VIX_Change_1d', 
        'SPY_Ret', 'SPY_Vol_21d', 
        'IWO_Vol_21d', 'SPY_IWO_Div_21d'
    ]
    # Ensure market features index is timestamp column, not index, for merging if needed
    # market_features index is timestamp.
    df_model = df_model.merge(market_features[market_cols_to_use], on='timestamp', how='left')
    
    df_model.set_index(['symbol', 'timestamp'], inplace=True)
    
    # 2. Define Features & Target
    # [Optimization] Drop 'RSI_2' (Noise)
    l0_cols = [
        'RSI_14', 'Dist_SMA_200', 'BB_PctB', 
        'ATR_Norm', 'Rel_Vol'
    ]
    l1_cols = ['Anomaly_Score'] # HMM_State is categorical, usually less helpful for Tree if Anomaly Score exists
    
    # Total Features = Stock + Macro + Anomaly
    feature_cols = l0_cols + market_cols_to_use + l1_cols
    
    X = df_model[feature_cols].copy()
    y = df_model['Meta_Label']
    
    X.fillna(0, inplace=True)
    
    # 3. LightGBM Parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'is_unbalance': True, # Important for handling class imbalance if any, or focusing on ranking
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    
    X.sort_index(level='timestamp', inplace=True)
    y = y.reindex(X.index)
    
    print("Performing Walk-Forward Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        
        print(f"  Fold {fold+1}: AUC={auc:.3f}, Acc={acc:.3f}")
        fold_metrics.append({'auc': auc})
        
    # 4. Final Training
    print("Retraining final model on all data...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    
    # Save Feature Importance for review
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\n[New Feature Importance Top 5]")
    print(importances.head(5))
    
    # 5. Get Probabilities
    all_probs = final_model.predict_proba(X)[:, 1]
    df_model['L3_Prob'] = all_probs
    
    return final_model, df_model, fold_metrics

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Adjust paths if needed
    PIPELINE_DIR = SCRIPT_DIR 
    MODELS_DIR = os.path.join(PIPELINE_DIR, 'models')
    SIGNALS_DIR = os.path.join(PIPELINE_DIR, 'signals')
    
    # 1. Load Data (Enhanced)
    stock_features, market_features, regime_signals = load_data(PIPELINE_DIR)
    
    # 2. Generate L2 Signals
    l2_signals, full_df = generate_l2_signals(stock_features)
    l2_signals.to_parquet(os.path.join(SIGNALS_DIR, 'base_strategy_trades.parquet'))
    
    # 3. Create Labels
    labeled_data = create_meta_labels(l2_signals, full_df)
    
    # 4. Train L3 Model (With Market Features)
    l3_model, results_df, metrics = train_l3_model(labeled_data, market_features, regime_signals)
    
    # 5. Save Artifacts
    joblib.dump(l3_model, os.path.join(MODELS_DIR, 'l3_meta_filter.joblib'))
    results_df[['Meta_Label', 'L3_Prob', 'Future_Return']].to_csv(os.path.join(SIGNALS_DIR, 'l3_probabilities.csv'))
    
    avg_auc = np.mean([m['auc'] for m in metrics])
    print(f"\n[Summary] Average Test AUC: {avg_auc:.3f}")
    if avg_auc > 0.54:
        print(">> Improvement detected! The new features are working.")
    else:
        print(">> AUC is still low. We might need deeper fundamental data or alternative labels.")

if __name__ == "__main__":
    main()