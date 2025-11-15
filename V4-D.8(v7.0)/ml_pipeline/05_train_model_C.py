# V4-D.8(v7.0) - Step 5-3: Model Training (Scheme C: Bayesian Regression)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
import joblib
import os

def train_model_C():
    """
    Trains a Bayesian Ridge regression model using walk-forward validation.
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
    model_dir = 'models/scheme_C/'
    os.makedirs(model_dir, exist_ok=True)

    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"--- Processing Fold {fold+1}/{n_splits} ---")

        # --- Data Splitting ---
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        # Define feature columns (all columns except 'Y' and 'Fill_Status')
        feature_cols = [col for col in train_df.columns if col not in ['Y', 'Fill_Status']]

        X_train = train_df[feature_cols]
        y_train = train_df['Y']
        X_test = test_df[feature_cols]
        y_test = test_df['Y']

        # --- Feature Standardization (SOP) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Model Training: BayesianRidge ---
        model_C_Bayesian = BayesianRidge()
        model_C_Bayesian.fit(X_train_scaled, y_train)

        # --- Prediction with Uncertainty ---
        Y_pred_C, Y_std_C = model_C_Bayesian.predict(X_test_scaled, return_std=True)

        # --- Store OOS Predictions ---
        fold_predictions = pd.DataFrame({
            'Y_true': y_test,
            'Y_pred_C': Y_pred_C,
            'Y_std_C': Y_std_C
        }, index=test_df.index)
        all_oos_predictions.append(fold_predictions)

        # --- Save Models for the Fold ---
        joblib.dump(model_C_Bayesian, os.path.join(model_dir, f'model_C_Bayesian_fold_{fold+1}.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, f'scaler_fold_{fold+1}.joblib'))


    # --- Combine and Save OOS Predictions ---
    oos_predictions_df = pd.concat(all_oos_predictions)
    oos_predictions_df.to_csv('predictions_oos_C.csv')

    print("--- Model Training (Scheme C) Complete ---")
    print(f"OOS predictions saved to predictions_oos_C.csv")
    print(f"Models saved in {model_dir}")

if __name__ == "__main__":
    # To be executed from the ml_pipeline directory
    # cd V4-D.8\(v7.0\)/ml_pipeline/
    # python 05_train_model_C.py
    train_model_C()
