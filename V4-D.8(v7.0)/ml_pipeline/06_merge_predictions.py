import pandas as pd
import os

def merge_predictions():
    """
    Merges the out-of-sample predictions from the three modeling schemes (A, B, C)
    and the Z-Score feature into a single CSV file.
    """
    # Define file paths
    # Note: This script assumes it is run from the 'ml_pipeline' directory
    predictions_a_path = 'predictions_oos_A.csv'
    predictions_b_path = 'predictions_oos_B.csv'
    predictions_c_path = 'predictions_oos_C.csv'
    features_path = 'features_X_T-1.parquet'
    output_path = 'predictions_oos_merged.csv'

    # Check if all input files exist
    required_files = [
        predictions_a_path, predictions_b_path, predictions_c_path, features_path
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Input file not found at '{f}'.")
            print("Please ensure you have run the preceding pipeline scripts first.")
            return

    # --- Load and Process DataFrames ---

    # Load Scheme A predictions
    df_a = pd.read_csv(predictions_a_path, index_col=[0, 1])

    # Load Scheme B predictions
    df_b = pd.read_csv(predictions_b_path, index_col=[0, 1])
    df_b = df_b.drop(columns=['Y_true'])

    # Load Scheme C predictions
    df_c = pd.read_csv(predictions_c_path, index_col=[0, 1])
    df_c = df_c.drop(columns=['Y_true'])

    # --- Load Features to get Z-Score ---
    df_features = pd.read_parquet(features_path)

    # The feature parquet file should have 'asset' and 'T-1_timestamp' in the index
    if not isinstance(df_features.index, pd.MultiIndex):
        df_features = df_features.set_index(['asset', 'T-1_timestamp'])

    # Select and rename the Z-Score column
    z_score_col = 'X_T1_Z_Score_20_60m_ETH_Last_Partial'
    if z_score_col in df_features.columns:
        df_z_score = df_features[[z_score_col]].rename(columns={z_score_col: 'Z_Score_20'})
    else:
        print(f"Error: Z-Score column '{z_score_col}' not found in features file.")
        return

    # --- Merge DataFrames ---

    # Join the predictions and the Z-Score feature
    df_merged = df_a.join([df_b, df_c, df_z_score], how='inner')

    # --- Verification ---
    expected_columns = [
        'Y_true', 'Y_pred_A', 'Y_uncertainty_A',
        'Y_pred_B_Lower', 'Y_pred_B_Median', 'Y_pred_B_Upper',
        'Y_pred_C', 'Y_std_C', 'Z_Score_20'
    ]

    if all(col in df_merged.columns for col in expected_columns):
        print("All expected columns are present in the merged DataFrame.")
    else:
        print("Warning: Some expected columns are missing!")
        print(f"Expected: {expected_columns}")
        print(f"Found: {list(df_merged.columns)}")

    # --- Save the Merged DataFrame ---
    df_merged.to_csv(output_path)

    print(f"Successfully merged predictions from Schemes A, B, C and Z-Score.")
    print(f"Output saved to '{output_path}'")
    print(f"Final DataFrame shape: {df_merged.shape}")


if __name__ == "__main__":
    # To be executed from the ml_pipeline directory
    # cd V4-D.8\(v7.0\)/ml_pipeline/
    # python 06_merge_predictions.py
    merge_predictions()
