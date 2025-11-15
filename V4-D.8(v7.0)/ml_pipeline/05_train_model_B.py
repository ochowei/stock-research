# V4-D.8(v7.0) - Step 5-2: Model Training (Scheme B: LightGBM Quantile Regression)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os

def train_model_B():
    """
    Trains three LightGBM quantile regression models using walk-forward validation.
    """
    # Load the dataset
    try:
        df = pd.read_parquet('model_ready_dataset.parquet')
    except FileNotFoundError:
        print("Error: model_ready_dataset.parquet not found. Please run Step 4 first.")
        return

    # --- Walk-Forward Validation Setup ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_oos_predictions = []

    # Create directory for models if it doesn't exist
    model_dir = 'models/scheme_B/'
    os.makedirs(model_dir, exist_ok=True)

    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"--- Processing Fold {fold+1}/{n_splits} ---")

        # --- Data Splitting ---
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        feature_cols = [col for col in train_df.columns if col not in ['Y', 'Fill_Status']]
        X_train = train_df[feature_cols]
        y_train = train_df['Y']
        X_test = test_df[feature_cols]
        y_test = test_df['Y']

        # --- Feature Standardization (SOP) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Model Training: Quantile Regression ---
        # Model for the lower quantile (10%)
        model_B_Lower = lgb.LGBMRegressor(objective='quantile', alpha=0.1)
        model_B_Lower.fit(X_train_scaled, y_train)
        y_pred_B_Lower = model_B_Lower.predict(X_test_scaled)

        # Model for the median (50%)
        model_B_Median = lgb.LGBMRegressor(objective='quantile', alpha=0.5)
        model_B_Median.fit(X_train_scaled, y_train)
        y_pred_B_Median = model_B_Median.predict(X_test_scaled)

        # Model for the upper quantile (90%)
        model_B_Upper = lgb.LGBMRegressor(objective='quantile', alpha=0.9)
        model_B_Upper.fit(X_train_scaled, y_train)
        y_pred_B_Upper = model_B_Upper.predict(X_test_scaled)

        # --- Store OOS Predictions ---
        fold_predictions = pd.DataFrame({
            'Y_true': y_test,
            'Y_pred_B_Lower': y_pred_B_Lower,
            'Y_pred_B_Median': y_pred_B_Median,
            'Y_pred_B_Upper': y_pred_B_Upper
        }, index=test_df.index)
        all_oos_predictions.append(fold_predictions)

        # --- Save Models for the Fold ---
        joblib.dump(model_B_Lower, os.path.join(model_dir, f'model_B_Lower_fold_{fold+1}.joblib'))
        joblib.dump(model_B_Median, os.path.join(model_dir, f'model_B_Median_fold_{fold+1}.joblib'))
        joblib.dump(model_B_Upper, os.path.join(model_dir, f'model_B_Upper_fold_{fold+1}.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, f'scaler_fold_{fold+1}.joblib'))

    # --- Combine and Save OOS Predictions ---
    oos_predictions_df = pd.concat(all_oos_predictions)
    oos_predictions_df.to_csv('predictions_oos_B.csv')

    print("--- Model Training (Scheme B) Complete ---")
    print(f"OOS predictions saved to predictions_oos_B.csv")
    print(f"Models saved in {model_dir}")

if __name__ == "__main__":
    train_model_B()
