import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
import lightgbm as lgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the model training and uncertainty bake-off.
    """
    # --- Path Definition ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'model_ready_dataset.parquet')
    models_dir = os.path.join(script_dir, 'models')
    output_path = os.path.join(script_dir, 'predictions_oos.csv')

    os.makedirs(models_dir, exist_ok=True)

    # --- Data Loading ---
    print("Loading data...")
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    df = df.sort_index(level='timestamp')
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # --- Feature and Label Definition ---
    if 'Y' not in df.columns:
        print("Error: Target column 'Y' not found in the dataset.")
        return
    y = df['Y']
    X = df.drop(columns=['Y'])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median())
    feature_names = X.columns.tolist()
    if not feature_names:
        print("Error: No valid feature columns found.")
        return
    print(f"Number of features: {len(feature_names)}")

    # --- Walk-Forward Validation Setup ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_predictions = []

    print(f"Starting walk-forward validation with {n_splits} splits...")
    for k, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {k+1}/{n_splits} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Train period: {X_train.index.get_level_values('timestamp').min()} to {X_train.index.get_level_values('timestamp').max()}")
        print(f"Test period:  {X_test.index.get_level_values('timestamp').min()} to {X_test.index.get_level_values('timestamp').max()}")

        # --- Standardization SOP ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Model Training & Uncertainty Bake-off ---
        print(f"Training models for Fold {k+1}...")

        # --- Scheme A: Two-stage Ridge ---
        model_A_Y = Ridge()
        model_A_Y.fit(X_train_scaled, y_train)
        y_pred_A_train = model_A_Y.predict(X_train_scaled)
        y_error_train = np.abs(y_train - y_pred_A_train)

        model_A_Uncertainty = Ridge()
        model_A_Uncertainty.fit(X_train_scaled, y_error_train)

        y_pred_A = model_A_Y.predict(X_test_scaled)
        y_uncertainty_A = model_A_Uncertainty.predict(X_test_scaled)

        # --- Scheme B: Quantile LGBM ---
        params = {'objective': 'quantile', 'metric': 'quantile', 'n_estimators': 100, 'random_state': 42}
        model_B_Lower = lgb.LGBMRegressor(**params, alpha=0.1)
        model_B_Median = lgb.LGBMRegressor(**params, alpha=0.5)
        model_B_Upper = lgb.LGBMRegressor(**params, alpha=0.9)

        model_B_Lower.fit(X_train_scaled, y_train)
        model_B_Median.fit(X_train_scaled, y_train)
        model_B_Upper.fit(X_train_scaled, y_train)

        y_pred_B_Lower = model_B_Lower.predict(X_test_scaled)
        y_pred_B_Median = model_B_Median.predict(X_test_scaled)
        y_pred_B_Upper = model_B_Upper.predict(X_test_scaled)

        # --- Scheme C: Bayesian Ridge ---
        model_C_Bayesian = BayesianRidge()
        model_C_Bayesian.fit(X_train_scaled, y_train)
        y_pred_C, y_std_C = model_C_Bayesian.predict(X_test_scaled, return_std=True)

        # --- Save Models for the current fold ---
        print(f"Saving models for Fold {k+1}...")
        joblib.dump(scaler, os.path.join(models_dir, f'scaler_fold_{k+1}.joblib'))
        joblib.dump(model_A_Y, os.path.join(models_dir, f'model_A_Y_fold_{k+1}.joblib'))
        joblib.dump(model_A_Uncertainty, os.path.join(models_dir, f'model_A_Uncertainty_fold_{k+1}.joblib'))
        joblib.dump(model_B_Lower, os.path.join(models_dir, f'model_B_Lower_fold_{k+1}.joblib'))
        joblib.dump(model_B_Median, os.path.join(models_dir, f'model_B_Median_fold_{k+1}.joblib'))
        joblib.dump(model_B_Upper, os.path.join(models_dir, f'model_B_Upper_fold_{k+1}.joblib'))
        joblib.dump(model_C_Bayesian, os.path.join(models_dir, f'model_C_Bayesian_fold_{k+1}.joblib'))

        # --- Collect Predictions ---
        # Assemble a DataFrame for the current fold's out-of-sample predictions.
        predictions_k = pd.DataFrame(index=X_test.index)
        predictions_k['Y_true'] = y_test
        predictions_k['fold'] = k + 1
        predictions_k['Y_pred_A'] = y_pred_A
        predictions_k['Y_uncertainty_A'] = y_uncertainty_A
        predictions_k['Y_pred_B_Lower'] = y_pred_B_Lower
        predictions_k['Y_pred_B_Median'] = y_pred_B_Median
        predictions_k['Y_pred_B_Upper'] = y_pred_B_Upper
        predictions_k['Y_pred_C'] = y_pred_C
        predictions_k['Y_std_C'] = y_std_C

        all_predictions.append(predictions_k)
        print(f"Fold {k+1} complete.")

    # --- Save All Predictions ---
    # Concatenate predictions from all folds and save to a single CSV file.
    if all_predictions:
        print("\nSaving out-of-sample predictions...")
        final_predictions = pd.concat(all_predictions)
        final_predictions.to_csv(output_path)
        print(f"Predictions saved to {output_path}")
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()
