import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from lightgbm import LGBMRegressor
import joblib
from sklearn.metrics import mean_squared_error

# --- Configuration ---
# Get the absolute path to the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is one level up from the script directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input file
MODEL_READY_DATASET_PATH = os.path.join(SCRIPT_DIR, 'model_ready_dataset.parquet')

# Output files and directories
PREDICTIONS_OOS_PATH = os.path.join(PROJECT_ROOT, 'predictions_oos.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Main Logic ---
def load_data(file_path):
    """Loads the dataset from a parquet file."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please make sure you have run the previous steps of the pipeline.")
        sys.exit(1)
    df = pd.read_parquet(file_path)
    print("Data loaded successfully.")
    return df

def main():
    """Main function to run the model training and evaluation pipeline."""
    df = load_data(MODEL_READY_DATASET_PATH)

    # Check if the dataframe is empty
    if df.empty:
        print("Error: The loaded dataset is empty. Cannot proceed with model training.")
        print("This might be because the data processing pipeline did not produce any valid samples.")
        sys.exit(1)

    # Set the index names to ensure they are correct
    df.index.names = ['symbol', 'timestamp']

    # Make sure the dataframe is sorted by timestamp before splitting
    # The index is a MultiIndex ('symbol', 'timestamp'), so we sort by the 'timestamp' level
    df = df.sort_index(level='timestamp')

    # Identify feature columns (X) and target column (Y)
    feature_cols = [col for col in df.columns if col.startswith('X_')]
    target_col = 'Y'

    # Get the unique timestamps to perform the walk-forward split on
    unique_timestamps = df.index.get_level_values('timestamp').unique().sort_values()

    # --- Walk-Forward Validation Setup ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_predictions = []

    for i, (train_index, test_index) in enumerate(tscv.split(unique_timestamps)):
        fold_num = i + 1
        print(f"--- Fold {fold_num}/{n_splits} ---")

        train_timestamps = unique_timestamps[train_index]
        test_timestamps = unique_timestamps[test_index]

        train_df = df[df.index.get_level_values('timestamp').isin(train_timestamps)]
        test_df = df[df.index.get_level_values('timestamp').isin(test_timestamps)]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # --- Feature Standardization SOP ---
        print("Applying feature standardization SOP...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Feature standardization complete.")

        # --- Model Training and Uncertainty Bake-off ---
        print("Starting model training and uncertainty bake-off...")

        # Scheme A
        print("Training Scheme A...")
        model_A_Y = Ridge()
        model_A_Y.fit(X_train_scaled, y_train)
        y_pred_A = model_A_Y.predict(X_test_scaled)

        y_error = np.abs(y_train - model_A_Y.predict(X_train_scaled))
        model_A_Uncertainty = Ridge()
        model_A_Uncertainty.fit(X_train_scaled, y_error)
        y_uncertainty_A = model_A_Uncertainty.predict(X_test_scaled)

        # Scheme B
        print("Training Scheme B...")
        model_B_Lower = LGBMRegressor(objective='quantile', alpha=0.1)
        model_B_Median = LGBMRegressor(objective='quantile', alpha=0.5)
        model_B_Upper = LGBMRegressor(objective='quantile', alpha=0.9)
        model_B_Lower.fit(X_train_scaled, y_train)
        model_B_Median.fit(X_train_scaled, y_train)
        model_B_Upper.fit(X_train_scaled, y_train)
        y_pred_B_Lower = model_B_Lower.predict(X_test_scaled)
        y_pred_B_Median = model_B_Median.predict(X_test_scaled)
        y_pred_B_Upper = model_B_Upper.predict(X_test_scaled)

        # Scheme C
        print("Training Scheme C...")
        model_C_Bayesian = BayesianRidge()
        model_C_Bayesian.fit(X_train_scaled, y_train)
        y_pred_C, y_std_C = model_C_Bayesian.predict(X_test_scaled, return_std=True)

        print("Model training complete for this fold.")

        # --- Store Predictions ---
        predictions_k = test_df.copy()
        predictions_k['Y_true'] = y_test
        predictions_k['Y_pred_A'] = y_pred_A
        predictions_k['Y_uncertainty_A'] = y_uncertainty_A
        predictions_k['Y_pred_B_Lower'] = y_pred_B_Lower
        predictions_k['Y_pred_B_Median'] = y_pred_B_Median
        predictions_k['Y_pred_B_Upper'] = y_pred_B_Upper
        predictions_k['Y_pred_C'] = y_pred_C
        predictions_k['Y_std_C'] = y_std_C
        all_predictions.append(predictions_k)

        # --- Save Models ---
        print("Saving models for this fold...")
        joblib.dump(model_A_Y, os.path.join(MODELS_DIR, f'model_A_Y_fold_{fold_num}.joblib'))
        joblib.dump(model_A_Uncertainty, os.path.join(MODELS_DIR, f'model_A_Uncertainty_fold_{fold_num}.joblib'))
        joblib.dump(model_B_Lower, os.path.join(MODELS_DIR, f'model_B_Lower_fold_{fold_num}.joblib'))
        joblib.dump(model_B_Median, os.path.join(MODELS_DIR, f'model_B_Median_fold_{fold_num}.joblib'))
        joblib.dump(model_B_Upper, os.path.join(MODELS_DIR, f'model_B_Upper_fold_{fold_num}.joblib'))
        joblib.dump(model_C_Bayesian, os.path.join(MODELS_DIR, f'model_C_Bayesian_fold_{fold_num}.joblib'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_fold_{fold_num}.joblib'))
        print("Models saved.")

    # --- Finalize and Save Predictions ---
    print("Concatenating all out-of-sample predictions...")
    final_predictions_df = pd.concat(all_predictions)

    output_columns = [
        'Y_true', 'Y_pred_A', 'Y_uncertainty_A', 'Y_pred_B_Lower',
        'Y_pred_B_Median', 'Y_pred_B_Upper', 'Y_pred_C', 'Y_std_C'
    ]
    final_output_df = final_predictions_df[output_columns]

    print(f"Saving final predictions to {PREDICTIONS_OOS_PATH}...")
    final_output_df.to_csv(PREDICTIONS_OOS_PATH)
    print("Predictions saved.")
    print("\nScript finished successfully!")

if __name__ == '__main__':
    main()
