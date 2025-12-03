import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def load_data(base_dir):
    """
    Loads all features and signals constructed in previous steps.
    """
    features_dir = os.path.join(base_dir, 'features')
    signals_dir = os.path.join(base_dir, 'signals')
    
    # 1. Stock Features (Contains V5.1 Sector Features)
    stock_path = os.path.join(features_dir, 'stock_features_L0.parquet')
    stock_features = pd.read_parquet(stock_path)
    
    # 2. Market Features (Context)
    market_path = os.path.join(features_dir, 'market_features_L0.parquet')
    market_features = pd.read_parquet(market_path)
    
    # 3. Regime Signals (L1 Output)
    regime_path = os.path.join(signals_dir, 'regime_signals.parquet')
    regime_signals = pd.read_parquet(regime_path)
    
    return stock_features, market_features, regime_signals

def prepare_ranking_data(stock_df, market_df, regime_df):
    """
    Prepares the dataset for Learning to Rank (LGBMRanker).
    """
    print("Preparing data for Ranking...")
    
    # --- A. Generate Candidates (L2 Logic) ---
    # Filter: Price > SMA200 AND RSI(2) < 10
    candidates = stock_df[
        (stock_df['Dist_SMA_200'] > 0) & 
        (stock_df['RSI_2'] < 10)
    ].copy()
    
    print(f"  - Total L2 Candidates found: {len(candidates)}")
    
    # --- B. Calculate Target (Ranking Score) ---
    # Calculate 5-Day Forward Return
    # Close_t+5 / Close_t - 1
    full_closes = stock_df['Close'].unstack(level='symbol')
    future_ret = full_closes.shift(-5) / full_closes - 1
    future_ret = future_ret.stack().rename('Target_Return')
    
    # Join target back to candidates
    df_rank = candidates.join(future_ret)
    
    # Drop rows where Target is NaN (last 5 days of data)
    df_rank.dropna(subset=['Target_Return'], inplace=True)
    
    # --- C. Merge Context Features ---
    # Merge Regime (L1) & Market (L0-Macro) on timestamp
    df_rank = df_rank.reset_index().merge(regime_df, on='timestamp', how='left')
    df_rank = df_rank.merge(market_df[['IWO_Vol_21d', 'SPY_IWO_Div_21d', 'VIX_Change_1d']], on='timestamp', how='left')
    
    # Set index back
    df_rank.set_index(['symbol', 'timestamp'], inplace=True)
    
    return df_rank

def train_l3_ranker(df):
    """
    Trains LGBMRanker using Walk-Forward Validation.
    """
    print("\n--- Training L3 Ranker (LambdaRank) ---")
    
    # --- 1. Feature Selection (V5.1 Orthogonality) ---
    feature_cols = [
        # > Sector / Orthogonal (The Alpha Source)
        'Sector_RSI_14', 'RSI_Divergence', 'Rel_Strength_Daily',
        
        # > Microstructure
        'Down_Vol_Prop', 'Rel_Vol',
        
        # > Volatility / Risk
        'ATR_Norm',
        
        # > L1 Regime Context
        'HMM_State', 'Anomaly_Score',
        
        # > Macro Context
        'IWO_Vol_21d', 'SPY_IWO_Div_21d', 'VIX_Change_1d'
    ]
    
    # Ensure columns exist
    valid_features = [c for c in feature_cols if c in df.columns]
    print(f"  - Features used ({len(valid_features)}): {valid_features}")
    
    X = df[valid_features].copy()
    X = X.fillna(0)
    
    # --- [CRITICAL FIX] Discretize Target for LambdaRank ---
    # LGBMRanker requires INT labels (0, 1, 2...). 
    # We bin the returns by day into relevance grades.
    print("  - Discretizing returns into Relevance Grades (0-3)...")
    
    def get_daily_grades(group):
        # If too few samples, fallback to simple binary
        if len(group) < 4:
            return (group > 0).astype(int)
        
        # Rank within the day to ensure distribution is uniform
        # Use qcut on ranks to handle duplicate return values safely
        try:
            # Grades: 0 (Bottom 50%), 1 (Next 30%), 2 (Next 15%), 3 (Top 5%)
            return pd.qcut(group.rank(method='first'), q=[0, 0.5, 0.8, 0.95, 1.0], labels=[0, 1, 2, 3]).astype(int)
        except Exception:
            # Fallback
            return (group > 0).astype(int)

    # Apply transformation group by timestamp
    y_grades = df['Target_Return'].groupby(level='timestamp', group_keys=False).apply(get_daily_grades)
    
    # Use grades as the target for training
    y = y_grades
    
    # --- 2. Create Groups for Ranking ---
    X = X.sort_index(level='timestamp')
    y = y.reindex(X.index)
    
    timestamps = X.index.get_level_values('timestamp')
    unique_dates = timestamps.unique()
    
    print(f"  - Training over {len(unique_dates)} unique days.")
    
    # --- 3. Walk-Forward Validation ---
    n_splits = 5
    lgbm_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'label_gain': [0, 1, 3, 7], # Define gain for grades 0, 1, 2, 3
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRanker(**lgbm_params)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    
    for fold, (train_date_idx, test_date_idx) in enumerate(tscv.split(unique_dates)):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        
        # Filter Data
        train_mask = timestamps.isin(train_dates)
        test_mask = timestamps.isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Prepare Groups (count of rows per day)
        q_train = X_train.groupby(level='timestamp', sort=False).size().values
        q_test = X_test.groupby(level='timestamp', sort=False).size().values
        
        # Train
        model.fit(
            X_train, y_train, 
            group=q_train,
            eval_set=[(X_test, y_test)],
            eval_group=[q_test],
            eval_at=[1, 3, 5],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Log metric
        val_score = model.best_score_['valid_0']['ndcg@3']
        print(f"  Fold {fold+1}: NDCG@3 = {val_score:.4f} (Train Dates: {len(train_dates)}, Test Dates: {len(test_dates)})")
        metrics.append(val_score)

    # --- 4. Final Retrain ---
    print("Retraining final Ranker on all data...")
    q_all = X.groupby(level='timestamp', sort=False).size().values
    model.fit(X, y, group=q_all)
    
    # Feature Importance
    importances = pd.DataFrame({
        'Feature': valid_features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n[L3 Ranker Feature Importance]")
    print(importances.head(10))
    
    # --- 5. Generate OOS Scores ---
    # Save scores for backtest
    scores = model.predict(X)
    df['L3_Rank_Score'] = scores
    
    return model, df, np.mean(metrics)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Data
    stock_f, market_f, regime_s = load_data(SCRIPT_DIR)
    
    # 2. Prepare Data (Candidates & Target)
    rank_df = prepare_ranking_data(stock_f, market_f, regime_s)
    
    # 3. Train Ranker
    model, result_df, avg_ndcg = train_l3_ranker(rank_df)
    
    # 4. Save Artifacts
    models_dir = os.path.join(SCRIPT_DIR, 'models')
    signals_dir = os.path.join(SCRIPT_DIR, 'signals')
    
    joblib.dump(model, os.path.join(models_dir, 'l3_ranker.joblib'))
    
    # Save scores
    out_cols = ['Target_Return', 'L3_Rank_Score', 'HMM_State', 'RSI_2']
    result_df[out_cols].to_csv(os.path.join(signals_dir, 'l3_rank_scores.csv'))
    
    print(f"\n[Summary] Average NDCG@3: {avg_ndcg:.4f}")
    print(f"Models saved to {models_dir}")
    print(f"Scores saved to {signals_dir}/l3_rank_scores.csv")
    print("Step 3: L3 Learning-to-Rank Complete.")

if __name__ == "__main__":
    main()