import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# --- Configuration ---
SLIPPAGE_RATES = [0.0000, 0.0005, 0.0010] # 0bps, 5bps, 10bps
ATR_MULTIPLIER = 2.0  # L4 Target = Entry + 2.0 * ATR
MAX_HOLD_DAYS = 5
TOP_K = 5 # L3 Select top 5 stocks per day

def load_data(base_dir):
    """Loads Rank Scores, Price Data (for simulation), and Regime Signals."""
    signals_dir = os.path.join(base_dir, 'signals')
    features_dir = os.path.join(base_dir, 'features')
    
    # 1. L3 Rank Scores (Candidates)
    scores_path = os.path.join(signals_dir, 'l3_rank_scores.csv')
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing {scores_path}")
    scores = pd.read_csv(scores_path)
    scores['timestamp'] = pd.to_datetime(scores['timestamp'])
    
    # 2. Stock Features (For OHLC & ATR to simulate L4)
    stock_path = os.path.join(features_dir, 'stock_features_L0.parquet')
    stock_df = pd.read_parquet(stock_path)
    
    # 3. Regime Signals
    regime_path = os.path.join(signals_dir, 'regime_signals.parquet')
    regime = pd.read_parquet(regime_path)
    
    # [Fix] Ensure timestamp is a column for merging
    if 'timestamp' not in regime.columns:
        regime = regime.reset_index()
    
    # Ensure timestamp type matches
    regime['timestamp'] = pd.to_datetime(regime['timestamp'])
    
    return scores, stock_df, regime

def run_simulation(trades_df, stock_df, slippage, use_l4=False):
    """
    Simulates trades with path-dependent exit (L4) and slippage.
    """
    results = []
    
    # Pre-fetch price data for faster lookup
    opens = stock_df['Open'].unstack(level='symbol')
    highs = stock_df['High'].unstack(level='symbol')
    closes = stock_df['Close'].unstack(level='symbol')
    atrs = stock_df['ATR_14'].unstack(level='symbol') # Raw ATR
    
    # Iterate through each trade
    for _, row in trades_df.iterrows():
        symbol = row['symbol']
        signal_date = row['timestamp']
        
        try:
            # 1. Determine Entry (T+1 Open)
            if signal_date not in opens.index:
                continue
            idx = opens.index.get_loc(signal_date)
            
            if idx + 1 >= len(opens):
                continue
                
            entry_date = opens.index[idx + 1]
            raw_entry_price = opens.iloc[idx + 1][symbol]
            entry_atr = atrs.iloc[idx][symbol] # ATR at Signal Day T
            
            if pd.isna(raw_entry_price) or pd.isna(entry_atr):
                continue
            
            # Apply Entry Slippage (Market Order)
            entry_price = raw_entry_price * (1 + slippage)
            
            exit_price = None
            exit_date = None
            exit_reason = ""
            
            # 2. Determine Exit
            if use_l4:
                # Dynamic TP
                target_price = entry_price + (ATR_MULTIPLIER * entry_atr)
                
                for i in range(1, MAX_HOLD_DAYS + 1):
                    if idx + i >= len(highs):
                        break
                        
                    curr_date = highs.index[idx + i]
                    curr_high = highs.iloc[idx + i][symbol]
                    
                    if pd.notna(curr_high) and curr_high >= target_price:
                        # TP Hit (Limit Order, no slippage assumed on TP exit)
                        exit_price = target_price
                        exit_date = curr_date
                        exit_reason = "TP"
                        break
                
                # If TP not hit, Time Exit
                if exit_price is None:
                    if idx + MAX_HOLD_DAYS < len(closes):
                        raw_exit = closes.iloc[idx + MAX_HOLD_DAYS][symbol]
                        exit_price = raw_exit * (1 - slippage) # Market Sell
                        exit_date = closes.index[idx + MAX_HOLD_DAYS]
                        exit_reason = "Time"
                    else:
                        continue
            
            else:
                # Fixed Time Exit (T+5 Close)
                if idx + MAX_HOLD_DAYS < len(closes):
                    raw_exit = closes.iloc[idx + MAX_HOLD_DAYS][symbol]
                    exit_price = raw_exit * (1 - slippage)
                    exit_date = closes.index[idx + MAX_HOLD_DAYS]
                    exit_reason = "Time"
                else:
                    continue
            
            # 3. Calculate Result
            if exit_price is not None:
                ret = (exit_price / entry_price) - 1
                results.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'return': ret,
                    'reason': exit_reason
                })
                
        except Exception:
            continue
            
    return pd.DataFrame(results)

def calculate_metrics(trade_results_df):
    if trade_results_df.empty:
        return {'Total_Return': 0, 'Sharpe': 0, 'Win_Rate': 0, 'Count': 0, 'Avg_Hold': 0}
        
    # Aggregate to Portfolio Level (Daily Returns)
    daily_rets = trade_results_df.groupby('entry_date')['return'].mean()
    
    # Fill missing days with 0 for Sharpe calc
    idx = pd.date_range(daily_rets.index.min(), daily_rets.index.max())
    daily_rets = daily_rets.reindex(idx, fill_value=0)
    
    total_ret = daily_rets.sum()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    win_rate = (trade_results_df['return'] > 0).mean()
    
    return {
        'Total_Return': total_ret,
        'Sharpe': sharpe,
        'Win_Rate': win_rate,
        'Count': len(trade_results_df),
        'Avg_Hold': (trade_results_df['exit_date'] - trade_results_df['entry_date']).dt.days.mean() if 'exit_date' in trade_results_df else 0
    }

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    scores, stock_df, regime = load_data(SCRIPT_DIR)
    
    # --- [Merge Fix] ---
    # scores already has 'HMM_State'. regime also has 'HMM_State'.
    # We only want 'Is_Anomaly' and 'Anomaly_Score' from regime to avoid duplicates.
    # However, to be safe, we merge carefully.
    
    regime_cols_to_use = ['timestamp', 'Is_Anomaly', 'Anomaly_Score']
    # If HMM_State is missing in scores for some reason, we might want it from regime.
    # But Step 3 output confirmed HMM_State is in scores.
    # Let's drop HMM_State from regime before merge to prevent collision
    
    df = scores.merge(regime[regime_cols_to_use], on='timestamp', how='left')
    
    # Fill NAs
    # Note: HMM_State is already in df (from scores)
    df['HMM_State'] = df['HMM_State'].fillna(0)
    df['Is_Anomaly'] = df['Is_Anomaly'].fillna(0)
    df['L3_Rank_Score'] = df['L3_Rank_Score'].fillna(-999)
    
    print(f"Data merged. Total Rows: {len(df)}")
    
    # --- Define Strategy Universes ---
    
    # A. V5 Base (L2 + L1 Defense)
    mask_l1 = (df['HMM_State'] != 2) & (df['Is_Anomaly'] == 0)
    candidates_v5 = df[mask_l1].copy()
    
    # B. V5.1 (L3 Ranking)
    print(f"Applying L3 Ranking (Top {TOP_K})...")
    candidates_l3 = candidates_v5.sort_values(
        ['timestamp', 'L3_Rank_Score'], ascending=[True, False]
    ).groupby('timestamp').head(TOP_K)
    
    # --- Run Simulations ---
    strategies = {
        'V5_Base (Fixed Exit)': (candidates_v5, False),
        'V5.1_L3 (Fixed Exit)': (candidates_l3, False),
        'V5.1_Full (Dynamic Exit)': (candidates_l3, True)
    }
    
    summary_metrics = []
    
    # Plotting setup
    plt.figure(figsize=(12, 7))
    baseline_slip = 0.0005 
    
    print("\nRunning Simulations...")
    
    for name, (input_df, use_l4) in strategies.items():
        print(f"  Simulating {name}...")
        res_df = run_simulation(input_df, stock_df, slippage=baseline_slip, use_l4=use_l4)
        met = calculate_metrics(res_df)
        
        # Plot Equity Curve
        if not res_df.empty:
            daily_avg = res_df.groupby('exit_date')['return'].mean()
            cum_ret = daily_avg.sort_index().cumsum()
            plt.plot(cum_ret.index, cum_ret.values, label=f"{name} (SR: {met['Sharpe']:.2f})")
            
        summary_metrics.append({
            'Strategy': name,
            'Slippage': '5bps',
            **met
        })
    
    # --- Stress Test for V5.1 Full ---
    print("\nRunning Stress Test on V5.1 Full...")
    for slip in SLIPPAGE_RATES:
        res_df = run_simulation(candidates_l3, stock_df, slippage=slip, use_l4=True)
        met = calculate_metrics(res_df)
        summary_metrics.append({
            'Strategy': f"V5.1_Full_Stress_{int(slip*10000)}bps",
            'Slippage': f"{int(slip*10000)}bps",
            **met
        })

    # --- Final Output ---
    plt.title(f'V5.1 Strategy Comparison (Slippage=5bps)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Points)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'v5.1_equity_curve.png'))
    
    # Save Text Report
    res_df = pd.DataFrame(summary_metrics)
    cols = ['Strategy', 'Slippage', 'Sharpe', 'Win_Rate', 'Total_Return', 'Count', 'Avg_Hold']
    print("\n" + res_df[cols].to_string())
    
    with open(os.path.join(ANALYSIS_DIR, 'v5.1_backtest_report.txt'), 'w') as f:
        f.write(res_df[cols].to_string())
        
    print(f"\nReport saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()