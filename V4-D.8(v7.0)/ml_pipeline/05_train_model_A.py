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
        df = pd.read_parquet('model_ready_dataset.parquet')
    except FileNotFoundError:
        print("Error: model_ready_dataset.parquet not found. Please run Step 4 first.")
        return

    # --- Walk-Forward Validation Setup ---
    n_splits = 5  # Or any other appropriate number of splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_oos_predictions = []

    # Create directory for models if it doesn't exist
    model_dir = 'models/scheme_A/'
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

        # --- Stage 1: Predict Y ---
        model_A_Y = Ridge(alpha=1.0)
        model_A_Y.fit(X_train_scaled, y_train)

        # 在「測試集」上產生第一階段預測 (用於 OOS 儲存)
        y_pred_A = model_A_Y.predict(X_test_scaled)

        # --- Stage 2: Predict Uncertainty ---

        # (修正開始)
        # 1. 在「訓練集」上產生預測
        y_pred_train_A = model_A_Y.predict(X_train_scaled)

        # 2. 在「訓練集」上計算誤差
        y_error_train = np.abs(y_train - y_pred_train_A)

        # 3. 訓練不確定性模型 (使用訓練集的 X 和訓練集的 error)
        model_A_Uncertainty = Ridge(alpha=1.0)
        model_A_Uncertainty.fit(X_train_scaled, y_error_train)

        # 4. 在「測試集」上預測不確定性
        y_uncertainty_A = model_A_Uncertainty.predict(X_test_scaled)
        # (修正結束)

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
    oos_predictions_df.to_csv('predictions_oos_A.csv')

    print("--- Model Training (Scheme A) Complete ---")
    print(f"OOS predictions saved to predictions_oos_A.csv")
    print(f"Models saved in {model_dir}")

if __name__ == "__main__":
    train_model_A()
