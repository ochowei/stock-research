import pandas as pd
import os

def merge_predictions():
    """
    Merges the out-of-sample predictions from the three modeling schemes (A, B, C)
    into a single CSV file.
    """
    # Define file paths
    # Note: This script assumes it is run from the 'ml_pipeline' directory
    predictions_a_path = 'predictions_oos_A.csv'
    predictions_b_path = 'predictions_oos_B.csv'
    predictions_c_path = 'predictions_oos_C.csv'
    output_path = 'predictions_oos_merged.csv'

    # Check if all input files exist
    for f in [predictions_a_path, predictions_b_path, predictions_c_path]:
        if not os.path.exists(f):
            print(f"Error: Input file not found at '{f}'.")
            print("Please ensure you have run the training scripts for all schemes (A, B, C) first.")
            return

    # --- Load and Process DataFrames ---

    # Load Scheme A predictions
    # The index is composed of 'asset' and 'T-1_timestamp', which are the first two columns.
    df_a = pd.read_csv(predictions_a_path, index_col=[0, 1])

    # Load Scheme B predictions
    df_b = pd.read_csv(predictions_b_path, index_col=[0, 1])
    # Drop the redundant Y_true column
    df_b = df_b.drop(columns=['Y_true'])

    # Load Scheme C predictions
    df_c = pd.read_csv(predictions_c_path, index_col=[0, 1])
    # Drop the redundant Y_true column
    df_c = df_c.drop(columns=['Y_true'])

    # --- Merge DataFrames ---

    # Join the three DataFrames along their common index
    # The index (`asset`, `T-1_timestamp`) ensures correct alignment.
    df_merged = df_a.join([df_b, df_c], how='inner')

    # --- Verification ---
    expected_columns = [
        'Y_true', 'Y_pred_A', 'Y_uncertainty_A',
        'Y_pred_B_Lower', 'Y_pred_B_Median', 'Y_pred_B_Upper',
        'Y_pred_C', 'Y_std_C'
    ]

    # Check if all expected columns are present
    if all(col in df_merged.columns for col in expected_columns):
        print("All expected columns are present in the merged DataFrame.")
    else:
        print("Warning: Some expected columns are missing!")
        print(f"Expected: {expected_columns}")
        print(f"Found: {list(df_merged.columns)}")


    # --- Save the Merged DataFrame ---
    df_merged.to_csv(output_path)

    print(f"Successfully merged predictions from Schemes A, B, and C.")
    print(f"Output saved to '{output_path}'")
    print(f"Final DataFrame shape: {df_merged.shape}")


if __name__ == "__main__":
    # To be executed from the ml_pipeline directory
    # cd V4-D.8\(v7.0\)/ml_pipeline/
    # python 06_merge_predictions.py
    merge_predictions()
