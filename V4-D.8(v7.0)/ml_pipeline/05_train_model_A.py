# V4-D.8(v7.0) - Step 5-1: Model Training (Scheme A: Two-Stage Ridge Model)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import os

def train_model_A():
    """
    Trains a two-stage Ridge regression model using walk-forward validation.
    """
    # Load the dataset
    try:
        df = pd.read_parquet('V4-D.8(v7.0)/ml_pipeline/model_ready_dataset.parquet')
    except FileNotFoundError:
        print("Error: model_ready_dataset.parquet not found. Please run Step 4 first.")
        return

    # --- Walk-Forward Validation Setup ---
    n_splits = 5  # Or any other appropriate number of splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_oos_predictions = []

    # Create directory for models if it doesn't exist
    model_dir = 'V4-D.8(v7.0)/ml_pipeline/models/scheme_A/'
    os.makedirs(model_dir, exist_ok=True)

    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"--- Processing Fold {fold+1}/{n_splits} ---")

        # --- Data Splitting ---
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        X_train = train_df.drop('Y', axis=1)
        y_train = train_df['Y']
        X_test = test_df.drop('Y', axis=1)
        y_test = test_df['Y']

        # --- Feature Standardization (SOP) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Stage 1: Predict Y ---
        model_A_Y = Ridge(alpha=1.0)
        model_A_Y.fit(X_train_scaled, y_train)
        y_pred_A = model_A_Y.predict(X_test_scaled)

        # --- Stage 2: Predict Uncertainty ---
        y_error = np.abs(y_test - y_pred_A)
        model_A_Uncertainty = Ridge(alpha=1.0)
        model_A_Uncertainty.fit(X_train_scaled, y_error)
        y_uncertainty_A = model_A_Uncertainty.predict(X_test_scaled)

        # --- Store OOS Predictions ---
        fold_predictions = pd.DataFrame({
            'Y_true': y_test,
            'Y_pred_A': y_pred_A,
            'Y_uncertainty_A': y_uncertainty_A
        }, index=test_df.index)
        all_oos_predictions.append(fold_predictions)

        # --- Save Models for the Fold ---
        joblib.dump(model_A_Y, os.path.join(model_dir, f'model_A_Y_fold_{fold+1}.joblib'))
        joblib.dump(model_A_Uncertainty, os.path.join(model_dir, f'model_A_Uncertainty_fold_{fold+1}.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, f'scaler_fold_{fold+1}.joblib'))


    # --- Combine and Save OOS Predictions ---
    oos_predictions_df = pd.concat(all_oos_predictions)
    oos_predictions_df.to_csv('V4-D.8(v7.0)/ml_pipeline/predictions_oos_A.csv')

    print("--- Model Training (Scheme A) Complete ---")
    print(f"OOS predictions saved to V4-D.8(v7.0)/ml_pipeline/predictions_oos_A.csv")
    print(f"Models saved in {model_dir}")

if __name__ == "__main__":
    train_model_A()
