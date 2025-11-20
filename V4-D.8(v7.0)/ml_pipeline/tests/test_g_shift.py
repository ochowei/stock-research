
import pandas as pd
import pytest
import numpy as np
import os
import importlib.util

# Load the script as a module
def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("build_features_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Path to the script to be tested
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', '02_build_features.py')
build_features_module = load_module_from_path(SCRIPT_PATH)


def test_g_group_features_are_not_shifted(mocker):
    """
    Scenario: Verify G-group features use T-1 data for T-1 timestamp.
    """
    # Mock load_data to return a more complete 60m dataframe to avoid KeyErrors
    mock_60m = pd.DataFrame({
        'Open': [1, 1, 1, 1], 'High': [1, 1, 1, 1], 'Low': [1, 1, 1, 1], 'Close': [1, 1, 1, 1], 'Volume': [1, 1, 1, 1]
    }, index=pd.MultiIndex.from_tuples([
        ('TEST.A', pd.to_datetime('2023-01-02 09:30:00')),
        ('TEST.A', pd.to_datetime('2023-01-02 16:30:00')),
        ('TEST.A', pd.to_datetime('2023-01-02 17:30:00')),
        ('TEST.A', pd.to_datetime('2023-01-02 19:30:00'))
    ], names=['symbol', 'timestamp']))
    mocker.patch.object(build_features_module, 'load_data', return_value=(mock_60m, pd.DataFrame()))

    # Mock calculate_feature_group_g to return a known value with all G-group columns
    mock_g_features = pd.DataFrame({
        'X_34_Beta_6M': [1.0, 1.5],
        'X_35_Momentum_6_1M': [0.1, 0.2],
        'X_36_Z_Score_126_Daily': [-0.5, 0.5],
        'X_37_Liquidity_Amihud': [1e-6, 2e-6]
    }, index=pd.MultiIndex.from_tuples([
        ('TEST.A', pd.to_datetime('2023-01-01')),
        ('TEST.A', pd.to_datetime('2023-01-02'))
    ], names=['symbol', 'timestamp']))
    mocker.patch.object(build_features_module, 'calculate_feature_group_g', return_value=mock_g_features)

    # Mock the final to_parquet call to prevent file writing
    mocker.patch('pandas.DataFrame.to_parquet')

    # Run the function and get the returned dataframe
    saved_df = build_features_module.build_features()

    # Check the value of the beta feature for T-1
    t_minus_1_date = pd.to_datetime('2023-01-02').date()
    actual_beta = saved_df.loc[('TEST.A', t_minus_1_date), 'X_34_Beta_6M']

    # Assert that the Beta is the value from T-1 (1.5) and not T-2 (1.0)
    assert actual_beta == 1.5
