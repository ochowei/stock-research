import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(pipeline_dir):
    """Loads all necessary artifacts for verification."""
    signals_dir = os.path.join(pipeline_dir, 'signals')
    features_dir = os.path.join(pipeline_dir, 'features')
    
    # 1. Trades with L3 Probabilities & Labels
    probs_path = os.path.join(signals_dir, 'l3_probabilities.csv')
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Missing {probs_path}")
    trades = pd.read_csv(probs_path)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    
    # 2. Regime Signals (L1)
    regime_path = os.path.join(signals_dir, 'regime_signals.parquet')
    regime = pd.read_parquet(regime_path)
    
    # 3. Market Features (for SPY price in Test A)
    market_path = os.path.join(features_dir, 'market_features_L0.parquet')
    market = pd.read_parquet(market_path)
    
    return trades, regime, market

def perform_test_a_defense_lag(regime, market):
    """
    Test A: Defense Lag Test.
    Checks how quickly HMM/Anomaly detection reacts to market peaks.
    Target: Lag < 5 days.
    """
    print("\n--- Test A: Defense Lag Verification ---")
    
    # Prepare data: Join SPY price with Regime signals
    # Use SPY_Close if available, or reconstruct from returns? 
    # market_features has 'SPY_Ret', let's reconstruct a proxy price or use market_indicators if needed.
    # Ideally we use market_indicators.parquet but let's try to work with what we loaded.
    # Note: market_features_L0 doesn't have raw price, only Ret. 
    # We will assume 'SPY_Ret' exists.
    
    df = pd.DataFrame(index=market.index)
    df['SPY_CumRet'] = (1 + market['SPY_Ret']).cumprod()
    df['HMM_State'] = regime['HMM_State']
    df['Is_Anomaly'] = regime['Is_Anomaly']
    
    # Define Major Peaks (Approximate dates for logic check)
    peaks = {
        'Covid_2020': '2020-02-19',
        'Bear_2022': '2022-01-03'
    }
    
    results = []
    
    for name, peak_date_str in peaks.items():
        peak_date = pd.to_datetime(peak_date_str)
        
        # Look at data starting from the peak
        post_peak = df[df.index >= peak_date].head(30) # Look 30 days ahead
        
        # Find first "Defense Trigger" (Crash State or Anomaly)
        # HMM State 2 = Crash
        triggers = post_peak[
            (post_peak['HMM_State'] == 2) | 
            (post_peak['Is_Anomaly'] == 1)
        ]
        
        if not triggers.empty:
            first_trigger_date = triggers.index[0]
            lag = (first_trigger_date - peak_date).days
            status = "PASS" if lag <= 5 else "FAIL"
            
            print(f"[{name}] Peak: {peak_date.date()} -> Trigger: {first_trigger_date.date()} | Lag: {lag} days ({status})")
            results.append({'Event': name, 'Lag': lag, 'Status': status})
        else:
            print(f"[{name}] Peak: {peak_date.date()} -> No trigger found in 30 days (FAIL)")
            results.append({'Event': name, 'Lag': np.nan, 'Status': 'FAIL'})
            
    return pd.DataFrame(results)

def perform_test_b_toxic_filter(trades):
    """
    Test B: Toxic Signal Filtering.
    Checks if L3 model filters out trades with > 5% loss.
    """
    print("\n--- Test B: Toxic Signal Filtering ---")
    
    # Definition of Toxic
    toxic_threshold = -0.05
    trades['Is_Toxic'] = trades['Future_Return'] < toxic_threshold
    
    toxic_trades = trades[trades['Is_Toxic']]
    normal_trades = trades[~trades['Is_Toxic']]
    
    # Filter Logic (L3 Prob > 0.5)
    l3_threshold = 0.5
    
    # How many toxic trades were BLOCKED? (Recall of Toxic class)
    # Blocked means L3_Prob <= threshold
    toxic_blocked = toxic_trades[toxic_trades['L3_Prob'] <= l3_threshold]
    block_rate = len(toxic_blocked) / len(toxic_trades) if len(toxic_trades) > 0 else 0
    
    # How many good trades were LOST? (False Negative Rate)
    # Lost means L3_Prob <= threshold but it was NOT toxic (or profitable)
    good_trades = trades[trades['Meta_Label'] == 1]
    good_lost = good_trades[good_trades['L3_Prob'] <= l3_threshold]
    lost_rate = len(good_lost) / len(good_trades) if len(good_trades) > 0 else 0
    
    print(f"Total Toxic Trades (< -5%): {len(toxic_trades)}")
    print(f"Toxic Trades Blocked: {len(toxic_blocked)} ({block_rate:.2%}) -> Target: > 60%")
    print(f"Profitable Trades Sacrificed: {len(good_lost)} ({lost_rate:.2%}) -> Target: < 20%")
    
    return {'Block_Rate': block_rate, 'Sacrifice_Rate': lost_rate}

def run_strategy_backtest(trades, regime):
    """
    Simulates Equity Curves for different strategy configurations.
    Assumption: Fixed capital allocation per trade (simple sum of returns).
    """
    print("\n--- Running Comprehensive Backtest ---")
    
    # Merge Regime info into Trades
    # trades has 'timestamp', regime has index 'timestamp'
    df = trades.merge(regime, on='timestamp', how='left')
    
    # Fill NA regimes (default to safe)
    # [Fix] Assign back instead of inplace=True
    df['HMM_State'] = df['HMM_State'].fillna(0)
    df['Is_Anomaly'] = df['Is_Anomaly'].fillna(0)
    
    # --- Define Strategies ---
    
    # 1. V5-Base: All L2 signals
    mask_base = pd.Series(True, index=df.index)
    
    # 2. V5 + L1 (Defense): Remove if Crash State (2) or Anomaly (1)
    mask_l1 = (df['HMM_State'] != 2) & (df['Is_Anomaly'] == 0)
    
    # 3. V5 + L3 (Filter): Keep if L3_Prob > 0.5
    mask_l3 = df['L3_Prob'] > 0.5
    
    # 4. V5-Full (L1 + L3)
    mask_full = mask_l1 & mask_l3
    
    strategies = {
        'V5_Base': mask_base,
        'V5_L1_Defense': mask_l1,
        'V5_L3_Filter': mask_l3,
        'V5_Full_System': mask_full
    }
    
    performance = {}
    
    plt.figure(figsize=(12, 6))
    
    for name, mask in strategies.items():
        # Filter trades
        selected_trades = df[mask].copy()
        
        # Calculate daily aggregated return (Portfolio View)
        # Sum of returns of all trades triggered on that day
        daily_rets = selected_trades.groupby('timestamp')['Future_Return'].mean().fillna(0)
        
        # Cumulative Return
        cum_ret = daily_rets.cumsum()
        
        # Metrics
        total_ret = daily_rets.sum()
        win_rate = (selected_trades['Future_Return'] > 0).mean()
        count = len(selected_trades)
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        
        performance[name] = {
            'Total_Return': total_ret,
            'Win_Rate': win_rate,
            'Trade_Count': count,
            'Sharpe': sharpe
        }
        
        # Plot
        cum_ret.plot(label=f"{name} (SR: {sharpe:.2f})")
        
    plt.title('V5 Strategy Performance Comparison (Cumulative Mean Returns)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return Points')
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    output_dir = os.path.dirname(os.path.abspath(__file__)) # V5/ml_pipeline/analysis
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    plot_path = os.path.join(analysis_dir, 'equity_curves_comparison.png')
    plt.savefig(plot_path)
    print(f"Equity curve plot saved to {plot_path}")
    
    # Print Metrics Table
    results_df = pd.DataFrame(performance).T
    print("\nStrategy Performance Metrics:")
    print(results_df[['Trade_Count', 'Win_Rate', 'Total_Return', 'Sharpe']])
    
    # Save Report
    report_path = os.path.join(analysis_dir, 'backtest_report.txt')
    with open(report_path, 'w') as f:
        f.write(results_df.to_string())
    print(f"Detailed report saved to {report_path}")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load
    trades, regime, market = load_data(SCRIPT_DIR)
    
    # 2. Test A
    perform_test_a_defense_lag(regime, market)
    
    # 3. Test B
    perform_test_b_toxic_filter(trades)
    
    # 4. Backtest
    run_strategy_backtest(trades, regime)
    
    print("\nStep 4: Backtest & Verification Complete.")

if __name__ == "__main__":
    main()