
import os
import pandas as pd
import numpy as np

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def generate_regime_signals(market_breadth_df, threshold=0.20):
    """
    Generates a binary 'Crash' signal based on market breadth.
    - Signal is 2 (Crash) if breadth < threshold.
    - Signal is 0 (Normal) otherwise.
    """
    market_breadth_df['signal'] = np.where(market_breadth_df['market_breadth'] < threshold, 2, 0)
    return market_breadth_df

def main():
    """
    Loads market breadth data and generates regime signals based on a threshold.
    """
    script_dir = get_script_dir()

    # Define paths relative to the script's directory
    features_dir = os.path.join(script_dir, 'features')
    signals_dir = os.path.join(script_dir, 'signals')
    os.makedirs(signals_dir, exist_ok=True)

    market_breadth_path = os.path.join(features_dir, 'market_breadth.parquet')
    regime_signals_output_path = os.path.join(signals_dir, 'regime_signals.parquet')

    # Load market breadth data
    if not os.path.exists(market_breadth_path):
        print(f"Error: Market breadth data not found at {market_breadth_path}")
        return

    print(f"Loading market breadth data from: {market_breadth_path}")
    market_breadth_df = pd.read_parquet(market_breadth_path)

    # Generate signals
    print("Generating regime signals...")
    regime_signals_df = generate_regime_signals(market_breadth_df)

    # Save signals
    print(f"Saving regime signals to: {regime_signals_output_path}")
    regime_signals_df.to_parquet(regime_signals_output_path)
    print("Regime signals saved successfully.")

if __name__ == '__main__':
    main()
