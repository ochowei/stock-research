"""
This script implements Step 5 of the execution plan: Model Training and Walk-Forward Validation.

It loads the cleaned and merged dataset, then performs walk-forward validation
to train a Ridge regression model and generate out-of-sample predictions.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Define file paths
# The script is in V4-D.8(v7.0)/data_acquisition/, so we need to go up one level for the root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_PATH = os.path.join(PROJECT_ROOT, 'data_acquisition', 'model_ready_dataset.parquet')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'predictions_oos.csv')

def run_training_and_validation():
    """
    Main function to run the walk-forward validation and model training.
    """
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load the dataset
    try:
        df = pd.read_parquet(INPUT_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {INPUT_DATA_PATH}")
        print("Please ensure you have run the previous steps of the execution plan.")
        return

    # Prepare data
    # The parquet file should have 'symbol' and 'timestamp' as the index.
    # If they are columns instead, set them as the index.
    if 'symbol' in df.columns and 'timestamp' in df.columns:
        df = df.set_index(['symbol', 'timestamp'])

    df = df.sort_index()

    # Extract year from timestamp for splitting
    df['year'] = df.index.get_level_values('timestamp').year

    # Identify feature columns (starting with 'X_') and the target column
    feature_columns = [col for col in df.columns if col.startswith('X_')]
    target_column = 'Y'

    all_oos_predictions = []

    # Walk-forward validation loop
    # We will use each year as a test set, and all prior years as the training set.
    # The first year of data cannot be a test set, so we start from the second year.
    unique_years = sorted(df['year'].unique())
    print(f"Unique years found in data: {unique_years}") # Debugging print

    if len(unique_years) < 2:
        print("Error: Not enough years in the dataset to perform walk-forward validation.")
        return

    for k, test_year in enumerate(unique_years[1:]):
        print(f"--- Processing Fold {k+1}: Training up to {test_year-1}, Testing on {test_year} ---")

        # 4. Split data into training and testing sets for the current fold
        # Explicitly cast test_year to the same type as the column to avoid issues
        year_col_type = df['year'].dtype
        train_df = df[df['year'] < year_col_type.type(test_year)]
        test_df = df[df['year'] == year_col_type.type(test_year)]

        print(f"Debug: Train df shape: {train_df.shape}") # Debug print
        print(f"Debug: Test df shape: {test_df.shape}") # Debug print

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        if X_train.empty or X_test.empty:
            print(f"Warning: Skipping fold {k+1} due to empty train or test set.")
            continue

        # 5. Feature Standardization SOP
        scaler_k = StandardScaler()
        scaler_k.fit(X_train) # Fit ONLY on training data

        X_train_scaled = scaler_k.transform(X_train)
        X_test_scaled = scaler_k.transform(X_test) # Transform test data using training stats

        # 6. Train and Predict
        model_k = Ridge()
        model_k.fit(X_train_scaled, y_train)

        predictions_k = model_k.predict(X_test_scaled)

        # Store results for this fold
        oos_predictions_k = test_df.copy()
        oos_predictions_k['predicted_Y'] = predictions_k
        all_oos_predictions.append(oos_predictions_k[[target_column, 'predicted_Y']])

        # 7. Save the model for the current fold
        model_filename = os.path.join(MODELS_DIR, f'model_fold_{k+1}.joblib')
        joblib.dump(model_k, model_filename)
        print(f"Saved model for fold {k+1} to {model_filename}")

    # Combine all out-of-sample predictions
    if all_oos_predictions:
        predictions_oos_df = pd.concat(all_oos_predictions)
        predictions_oos_df.to_csv(OUTPUT_PREDICTIONS_PATH)
        print(f"Successfully saved out-of-sample predictions to {OUTPUT_PREDICTIONS_PATH}")
    else:
        print("No predictions were generated.")

if __name__ == '__main__':
    run_training_and_validation()
