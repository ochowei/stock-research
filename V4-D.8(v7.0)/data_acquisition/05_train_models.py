# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
import joblib
import os

# --- 0. Setup: Create models directory ---
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Directory '{MODELS_DIR}' created.")

# --- 1. Load Data ---
try:
    df = pd.read_parquet("model_ready_dataset.parquet")
except FileNotFoundError:
    print("Error: model_ready_dataset.parquet not found.")
    print("Please run the previous scripts (01-04) to generate it.")
    exit()

# --- 2. Define Features (X) and Target (Y) ---
Y_COLUMN = 'Y'
X_COLUMNS = [col for col in df.columns if col not in [Y_COLUMN, 'Fill_Status']]
# Sort by the second level of the index (timestamp)
# Using level=1 is more robust if the index level is unnamed
df.sort_index(level=1, inplace=True)

print("Data loaded and sorted successfully.")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(X_COLUMNS)}")

# --- 3. Walk-Forward Validation Framework ---
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
all_predictions = []

print(f"\nSetting up Walk-Forward Validation with {N_SPLITS} splits.")

for k, (train_index, test_index) in enumerate(tscv.split(df)):

    # --- 4. Get Train/Test Sets for fold k ---
    train_k, test_k = df.iloc[train_index], df.iloc[test_index]
    X_train_k, Y_train_k = train_k[X_COLUMNS], train_k[Y_COLUMN]
    X_test_k, Y_test_k = test_k[X_COLUMNS], test_k[Y_COLUMN]

    fold_num = k + 1
    print(f"\n--- Fold {fold_num}/{N_SPLITS} ---")
    print(f"Train: {len(train_k)} samples, Test: {len(test_k)} samples")

    # --- 5. Feature Standardization SOP ---
    scaler_k = StandardScaler()
    X_train_k_scaled = scaler_k.fit_transform(X_train_k)
    X_test_k_scaled = scaler_k.transform(X_test_k)
    print("Features standardized according to SOP.")

    # --- 6. Model Uncertainty Bake-off ---

    # Scheme A: Two-stage Ridge
    print("Training Scheme A (Two-stage Ridge)...")
    model_A_Y = Ridge()
    model_A_Y.fit(X_train_k_scaled, Y_train_k)
    Y_pred_A = model_A_Y.predict(X_test_k_scaled)

    Y_error_train = np.abs(Y_train_k - model_A_Y.predict(X_train_k_scaled))
    model_A_Uncertainty = Ridge()
    model_A_Uncertainty.fit(X_train_k_scaled, Y_error_train)
    Y_uncertainty_A = model_A_Uncertainty.predict(X_test_k_scaled)

    joblib.dump(model_A_Y, os.path.join(MODELS_DIR, f"model_A_Y_fold_{fold_num}.joblib"))
    joblib.dump(model_A_Uncertainty, os.path.join(MODELS_DIR, f"model_A_Uncertainty_fold_{fold_num}.joblib"))

    # Scheme B: Quantile LGBM
    print("Training Scheme B (Quantile LGBM)...")
    quantiles = {'Lower': 0.1, 'Median': 0.5, 'Upper': 0.9}
    predictions_B = {}
    for name, alpha in quantiles.items():
        model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, random_state=42)
        model.fit(X_train_k_scaled, Y_train_k)
        predictions_B[f'Y_pred_B_{name}'] = model.predict(X_test_k_scaled)
        joblib.dump(model, os.path.join(MODELS_DIR, f"model_B_{name}_fold_{fold_num}.joblib"))

    # Scheme C: Bayesian Ridge
    print("Training Scheme C (Bayesian Ridge)...")
    model_C_Bayesian = BayesianRidge()
    model_C_Bayesian.fit(X_train_k_scaled, Y_train_k)
    Y_pred_C, Y_std_C = model_C_Bayesian.predict(X_test_k_scaled, return_std=True)
    joblib.dump(model_C_Bayesian, os.path.join(MODELS_DIR, f"model_C_Bayesian_fold_{fold_num}.joblib"))

    print("All models for this fold trained and saved.")

    # --- 7. Store Predictions ---
    predictions_k = test_k.copy()
    predictions_k['Y_true'] = Y_test_k
    predictions_k['Y_pred_A'] = Y_pred_A
    predictions_k['Y_uncertainty_A'] = Y_uncertainty_A
    for name, preds in predictions_B.items():
        predictions_k[name] = preds
    predictions_k['Y_pred_C'] = Y_pred_C
    predictions_k['Y_std_C'] = Y_std_C

    all_predictions.append(predictions_k)

print("\nWalk-forward validation loop completed.")

# --- 8. Save All Predictions ---
if all_predictions:
    final_predictions_df = pd.concat(all_predictions)
    # Keep only the prediction and true value columns
    output_columns = [
        'Y_true', 'Y_pred_A', 'Y_uncertainty_A',
        'Y_pred_B_Lower', 'Y_pred_B_Median', 'Y_pred_B_Upper',
        'Y_pred_C', 'Y_std_C'
    ]
    # Ensure all expected columns are present, fill with NaN if not
    for col in output_columns:
        if col not in final_predictions_df.columns:
            final_predictions_df[col] = np.nan

    final_predictions_df = final_predictions_df[output_columns]

    predictions_output_path = "predictions_oos.csv"
    final_predictions_df.to_csv(predictions_output_path)
    print(f"\nAll out-of-sample predictions saved to {predictions_output_path}")
else:
    print("\nNo predictions were generated to save.")
