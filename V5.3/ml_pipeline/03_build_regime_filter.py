import os
import pandas as pd
import numpy as np

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def generate_hybrid_signals(breadth_df, macro_df):
    """
    Generates V5.3 Hybrid Regime Signals.
    Logic:
        Signal = 2 (Crash) if:
            (Breadth < 15%)  <-- Internal Market Structure Weakness
            OR
            (Junk_Bond_Stress < MA20 AND Risk_Off_Flow > MA20) <-- External Macro Stress
        Else:
            Signal = 0 (Normal)
    """
    # Merge dataframes on timestamp
    # Ensure indexes are datetime
    if not isinstance(breadth_df.index, pd.DatetimeIndex):
        breadth_df.index = pd.to_datetime(breadth_df.index)
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)

    # Combine (Inner join to ensure we have data for both, or Left join if Breadth is primary)
    # Using Left join based on Breadth (Universe) is safer
    df = breadth_df.join(macro_df, how='left')
    
    # --- Condition A: Internal Breadth Collapse ---
    # V5.3 Threshold: 15% (Stricter than V5.2's 20%)
    cond_breadth = df['market_breadth'] < 0.15
    
    # --- Condition B: External Macro Stress ---
    # Check if we have the macro features (Custom track has them, Index track might not)
    if 'Junk_Bond_Stress' in df.columns and 'Risk_Off_Flow' in df.columns:
        # Stress condition: High Yield underperforming (Ratio Down) AND Treasuries outperforming (Ratio Up)
        # Using Moving Average logic from Research Plan
        # Note: MA20 is already calculated in 02_build_features as *_MA20
        
        stress_down = df['Junk_Bond_Stress'] < df['Junk_Bond_Stress_MA20']
        fear_up = df['Risk_Off_Flow'] > df['Risk_Off_Flow_MA20']
        
        cond_macro = stress_down & fear_up
    else:
        # Fallback for Index track if HYG/IEF missing: relying only on breadth
        cond_macro = False
    
    # --- Final Decision ---
    # Signal: 2 = Crash (Liquidation), 0 = Normal
    is_crash = cond_breadth | cond_macro
    
    df['signal'] = np.where(is_crash, 2, 0)
    
    # Statistics for reporting
    total_days = len(df)
    crash_days = is_crash.sum()
    print(f"    - Total Days: {total_days}")
    print(f"    - Crash Days: {crash_days} ({crash_days/total_days:.1%})")
    print(f"      > Due to Breadth: {cond_breadth.sum()}")
    print(f"      > Due to Macro:   {cond_macro.sum() if isinstance(cond_macro, pd.Series) else 0}")
    
    return df[['signal']]

def main():
    print("=== V5.3 Step 2.3: Building L1 Hybrid Regime Filter ===")
    script_dir = get_script_dir()
    
    # Process both tracks
    data_tracks = ['custom', 'index']

    for track in data_tracks:
        print(f"\n--- Processing Track: {track} ---")
        
        # Paths
        base_data_dir = os.path.join(script_dir, 'data', track)
        features_dir = os.path.join(base_data_dir, 'features')
        signals_dir = os.path.join(base_data_dir, 'signals')
        
        os.makedirs(signals_dir, exist_ok=True)

        breadth_path = os.path.join(features_dir, 'market_breadth.parquet')
        macro_path = os.path.join(features_dir, 'macro_features.parquet')
        output_path = os.path.join(signals_dir, 'regime_signals.parquet')

        if not os.path.exists(breadth_path):
            print(f"Warning: Breadth data not found at {breadth_path}. Skipping.")
            continue
            
        print("Loading features...")
        breadth_df = pd.read_parquet(breadth_path)
        
        if os.path.exists(macro_path):
            macro_df = pd.read_parquet(macro_path)
        else:
            print("Warning: Macro features not found. Using empty DataFrame (Breadth-only mode).")
            macro_df = pd.DataFrame()

        # Generate Signals
        print("Generating Hybrid Signals...")
        regime_signals = generate_hybrid_signals(breadth_df, macro_df)

        # Save
        regime_signals.to_parquet(output_path)
        print(f"Saved L1 Signals to: {output_path}")

    print("\nL1 Regime Filter Construction Complete.")

if __name__ == '__main__':
    main()