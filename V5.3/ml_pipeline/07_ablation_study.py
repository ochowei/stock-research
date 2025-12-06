import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import DataLoader
from risk_manager import RiskManager

# --- Configuration ---
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 5
SLIPPAGE = 0.0005 
TRANSACTION_COST = 0.0005 

class AblationBacktester:
    def __init__(self, stock_df, regime_df, rank_df, breadth_df, 
                 initial_capital=100000.0, max_positions=5,
                 # Ablation Flags
                 use_l1=True,
                 use_l3=True,
                 exit_mode='trailing', # 'trailing', 'fixed_5d'
                 force_equal_weight=False
                 ):
        
        self.stock_df = stock_df.sort_index()
        self.regime_df = regime_df.sort_index()
        self.rank_df = rank_df.sort_index()
        self.breadth_df = breadth_df.sort_index()
        
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        
        # Flags
        self.use_l1 = use_l1
        self.use_l3 = use_l3
        self.exit_mode = exit_mode
        self.force_equal_weight = force_equal_weight
        
        self.rm = RiskManager(target_risk=0.01, max_position_pct=0.2)
        
        self._precalculate_signals()
        
        self.cash = initial_capital
        self.positions = {} 
        self.equity_curve = []
        self.trade_log = []
        
        self.daily_data = self.stock_df.reorder_levels(['timestamp', 'symbol']).sort_index()
        self.all_dates = self.daily_data.index.get_level_values('timestamp').unique().sort_values()

    def _precalculate_signals(self):
        # Shift logic (Anti-Lookahead)
        cols_to_shift = ['RSI_2', 'SMA_200', 'close', 'ATR_14']
        col_map = {c: c for c in self.stock_df.columns}
        for target in cols_to_shift:
            for c in self.stock_df.columns:
                if c.lower() == target.lower():
                    col_map[target] = c
                    break
        
        self.stock_df['prev_RSI_2'] = self.stock_df.groupby('symbol')[col_map.get('RSI_2', 'RSI_2')].shift(1)
        self.stock_df['prev_SMA_200'] = self.stock_df.groupby('symbol')[col_map.get('SMA_200', 'SMA_200')].shift(1)
        self.stock_df['prev_close'] = self.stock_df.groupby('symbol')[col_map.get('close', 'close')].shift(1)
        self.stock_df['prev_ATR_14'] = self.stock_df.groupby('symbol')[col_map.get('ATR_14', 'ATR_14')].shift(1)

        if not self.rank_df.empty:
            self.rank_df['prev_L3_Rank_Score'] = self.rank_df.groupby('symbol')['L3_Rank_Score'].shift(1)

        self.regime_df['prev_signal'] = self.regime_df['signal'].shift(1)
        self.breadth_df['prev_market_breadth'] = self.breadth_df['market_breadth'].shift(1)

    def get_market_context(self, date):
        regime = 0
        breadth = 0.5 
        
        if date in self.regime_df.index:
            val = self.regime_df.loc[date, 'prev_signal']
            regime = val.iloc[-1] if isinstance(val, pd.Series) else val
            
        if date in self.breadth_df.index:
            val = self.breadth_df.loc[date, 'prev_market_breadth']
            breadth = val.iloc[-1] if isinstance(val, pd.Series) else val
            
        if pd.isna(regime): regime = 0
        if pd.isna(breadth): breadth = 0.5
            
        # Ablation: Force regime 0 if L1 disabled
        if not self.use_l1:
            regime = 0
            
        return regime, breadth

    def run(self):
        for date in self.all_dates:
            regime, breadth = self.get_market_context(date)
            trailing_k = 1.5 if breadth < 0.30 else 3.0
            
            try:
                today_bar = self.daily_data.loc[date]
            except KeyError:
                continue

            # --- A. L1 Liquidation ---
            if regime == 2: 
                symbols_to_sell = list(self.positions.keys())
                for sym in symbols_to_sell:
                    self._execute_sell(sym, date, today_bar, reason="L1_Liquidation")
            else:
                # --- B. Exits (L4 or Fixed) ---
                symbols_to_check = list(self.positions.keys())
                for sym in symbols_to_check:
                    if sym not in today_bar.index: continue
                    pos = self.positions[sym]
                    row = today_bar.loc[sym]
                    
                    pos['days_held'] += 1
                    
                    # 1. Trailing Stop (L4)
                    if self.exit_mode == 'trailing':
                        current_high = row['high']
                        if current_high > pos['highest_high']:
                            pos['highest_high'] = current_high
                        
                        stop_price = pos['highest_high'] - (trailing_k * pos['entry_atr'])
                        if row['low'] < stop_price:
                            exec_price = min(row['open'], stop_price)
                            self._execute_sell(sym, date, today_bar, reason="L4_Trailing", override_price=exec_price)
                    
                    # 2. Fixed 5-Day (V5.2 Logic)
                    elif self.exit_mode == 'fixed_5d':
                        if pos['days_held'] >= 5:
                            self._execute_sell(sym, date, today_bar, reason="Fixed_5D")

            # --- C. Entries ---
            if regime != 2:
                self._process_entries(date, today_bar)

            self._update_equity(date, today_bar)

        return pd.DataFrame(self.equity_curve).set_index('timestamp'), pd.DataFrame(self.trade_log)

    def _process_entries(self, date, today_bar):
        open_slots = self.max_positions - len(self.positions)
        if open_slots <= 0 or self.cash < 1000:
            return

        # L2 Signal
        candidates = today_bar[
            (today_bar['prev_RSI_2'] < 10) & 
            (today_bar['prev_close'] > today_bar['prev_SMA_200']) 
        ]
        
        if candidates.empty: return

        # L3 Sorting (Ablation)
        if self.use_l3 and (date in self.rank_df.index.get_level_values('timestamp')):
            ranks = self.rank_df.xs(date, level='timestamp')
            candidates = candidates.join(ranks[['prev_L3_Rank_Score']], how='inner')
            candidates = candidates.sort_values('prev_L3_Rank_Score', ascending=False)
        else:
            # Fallback to RSI (V5.1/V5.2 Logic)
            candidates = candidates.sort_values('prev_RSI_2', ascending=True)

        targets = candidates.head(open_slots)

        for sym, row in targets.iterrows():
            if sym in self.positions: continue
            
            price = row['open']
            atr = row['prev_ATR_14']
            
            # --- Sizing Logic ---
            if self.force_equal_weight:
                # V5.1 Aggressive Logic
                total_equity = self._get_current_equity()
                alloc_per_trade = total_equity / self.max_positions
                shares = int(alloc_per_trade / price)
            else:
                # V5.2/V5.3 Risk-Aware Logic
                shares = self.rm.calculate_position_size(self._get_current_equity(), price, atr)
            
            if shares > 0:
                cost = shares * price * (1 + SLIPPAGE + TRANSACTION_COST)
                if self.cash >= cost:
                    self.cash -= cost
                    self.positions[sym] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_atr': atr if pd.notna(atr) else price*0.02,
                        'highest_high': price, 
                        'entry_date': date,
                        'days_held': 0
                    }

    def _execute_sell(self, sym, date, today_bar, reason, override_price=None):
        if sym not in self.positions: return
        pos = self.positions.pop(sym)
        shares = pos['shares']
        
        if override_price:
            price = override_price
        elif sym in today_bar.index:
            price = today_bar.loc[sym]['open'] 
        else:
            price = pos['entry_price']

        proceeds = shares * price * (1 - SLIPPAGE - TRANSACTION_COST)
        self.cash += proceeds
        
        ret = (price / pos['entry_price']) - 1
        self.trade_log.append({
            'symbol': sym, 'entry_date': pos['entry_date'], 'exit_date': date,
            'return': ret, 'reason': reason, 'hold_days': pos['days_held']
        })

    def _get_current_equity(self):
        eq = self.cash
        for p in self.positions.values(): eq += p['shares'] * p['highest_high'] 
        return eq

    def _update_equity(self, date, today_bar):
        curr_eq = self.cash
        for sym, pos in self.positions.items():
            if sym in today_bar.index:
                price = today_bar.loc[sym]['close'] 
                curr_eq += pos['shares'] * price
            else:
                curr_eq += pos['shares'] * pos['entry_price']
        self.equity_curve.append({'timestamp': date, 'equity': curr_eq})

# --- Main ---
def load_data(base_dir, track='custom'):
    track_dir = os.path.join(base_dir, 'data', track)
    feat_path = os.path.join(track_dir, 'features', 'stock_features.parquet')
    regime_path = os.path.join(track_dir, 'signals', 'regime_signals.parquet')
    rank_path = os.path.join(track_dir, 'signals', 'l3_rank_scores.csv')
    breadth_path = os.path.join(track_dir, 'features', 'market_breadth.parquet')
    
    if not os.path.exists(feat_path): return None, None, None, None
    stock = pd.read_parquet(feat_path)
    regime = pd.read_parquet(regime_path)
    breadth = pd.read_parquet(breadth_path)
    rank = pd.DataFrame()
    if os.path.exists(rank_path):
        rank = pd.read_csv(rank_path)
        rank['timestamp'] = pd.to_datetime(rank['timestamp'])
        rank = rank.set_index(['timestamp', 'symbol'])
    return stock, regime, rank, breadth

def load_spy_benchmark(base_dir, track='custom'):
    market_path = os.path.join(base_dir, 'data', track, 'market_indicators.parquet')
    if not os.path.exists(market_path): return pd.Series()
    df = pd.read_parquet(market_path).reset_index()
    
    if 'symbol' in df.columns:
        spy_df = df[df['symbol'].str.upper() == 'SPY'].copy()
    else:
        return pd.Series()

    if spy_df.empty: return pd.Series()
    spy_df = spy_df.set_index('timestamp').sort_index()
    price = spy_df['close']
    return (price / price.iloc[0]) * INITIAL_CAPITAL

def filter_tickers(df, tickers):
    return df[df.index.get_level_values('symbol').isin(tickers)]

def calculate_metrics(curve):
    if curve.empty: return {}
    ret = curve.iloc[-1]/curve.iloc[0] - 1
    dd = (curve/curve.cummax() - 1).min()
    daily = curve.pct_change().fillna(0)
    sharpe = daily.mean()/daily.std() * np.sqrt(252) if daily.std()!=0 else 0
    return {'Total Return': f"{ret:.2%}", 'MaxDD': f"{dd:.2%}", 'Sharpe': f"{sharpe:.2f}"}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'analysis')
    
    loader = DataLoader(script_dir, normal_file='final_asset_pool.json', toxic_file='final_toxic_asset_pool.json')
    merged_tickers = loader.get_all_tickers()
    
    # Load Data (Custom Track)
    stock, regime, rank, breadth = load_data(script_dir, 'custom')
    stock = filter_tickers(stock, merged_tickers)
    
    # --- Defined Scenarios ---
    scenarios = [
        # --- Ablation Set ---
        {'name': 'V5.3 Full (Trailing)',        'l1': True,  'l3': True,  'exit': 'trailing', 'eqwt': False},
        {'name': 'V5.3 Fixed 5D (No Trailing)', 'l1': True,  'l3': True,  'exit': 'fixed_5d', 'eqwt': False},
        {'name': 'No L1 Defense (Trailing)',    'l1': False, 'l3': True,  'exit': 'trailing', 'eqwt': False},
        {'name': 'No L3 Rank (RSI Sort)',       'l1': True,  'l3': False, 'exit': 'trailing', 'eqwt': False},
        {'name': 'V5.2-Like (Fixed 5D, No L3)', 'l1': True,  'l3': False, 'exit': 'fixed_5d', 'eqwt': False},
        
        # --- Historical Benchmarks (Re-run on same data) ---
        {'name': 'V5.1 Aggressive',             'l1': False, 'l3': False, 'exit': 'fixed_5d', 'eqwt': True},
        {'name': 'V5.2 Risk-Aware',             'l1': True,  'l3': False, 'exit': 'fixed_5d', 'eqwt': False},
    ]
    
    results = []
    equity_curves = {} 
    
    print("=== V5.3 Comprehensive Ablation & Benchmark Study ===")
    
    # Run Strategies
    for s in scenarios:
        print(f"Running: {s['name']}...")
        bt = AblationBacktester(stock, regime, rank, breadth, 
                                use_l1=s['l1'], use_l3=s['l3'], 
                                exit_mode=s['exit'], force_equal_weight=s['eqwt'])
        eq, _ = bt.run()
        
        if not eq.empty:
            met = calculate_metrics(eq['equity'])
            met['Scenario'] = s['name']
            results.append(met)
            equity_curves[s['name']] = eq['equity']

    # Get SPY Benchmark
    print("Loading SPY Benchmark...")
    spy_curve = load_spy_benchmark(script_dir, 'custom')
    if not spy_curve.empty:
        common_idx = equity_curves[list(equity_curves.keys())[0]].index
        spy_curve = spy_curve.reindex(common_idx, method='ffill').fillna(method='bfill')
        spy_curve = spy_curve / spy_curve.iloc[0] * INITIAL_CAPITAL
        met = calculate_metrics(spy_curve)
        met['Scenario'] = 'SPY (Buy & Hold)'
        results.append(met)
        equity_curves['SPY (Buy & Hold)'] = spy_curve

    # Report
    df = pd.DataFrame(results)
    cols = ['Scenario', 'Total Return', 'Sharpe', 'MaxDD']
    print("\n" + df[cols].to_string(index=False))
    df.to_csv(os.path.join(output_dir, 'v5.3_full_ablation.csv'), index=False)
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    styles = {
        'V5.3 Fixed 5D (No Trailing)': {'color': '#2ca02c', 'lw': 3, 'ls': '-'}, # Green, Bold (Winner)
        'V5.3 Full (Trailing)':        {'color': '#9467bd', 'lw': 1.5, 'ls': ':'}, 
        'V5.2 Risk-Aware':             {'color': '#1f77b4', 'lw': 2, 'ls': '--'}, 
        'V5.1 Aggressive':             {'color': '#ff7f0e', 'lw': 2, 'ls': '--'}, 
        'SPY (Buy & Hold)':            {'color': 'gray', 'lw': 1.5, 'ls': '-.', 'alpha': 0.7}
    }
    
    for name, curve in equity_curves.items():
        norm = curve / curve.iloc[0]
        s = styles.get(name, {'color': 'black', 'lw': 1, 'alpha': 0.5}) # Default style for others
        plt.plot(norm.index, norm, label=name, **s)
        
    plt.title('V5.3 Full Comparison: Evolution & Ablation')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'v5.3_full_ablation_chart.png'))
    print(f"\nFull analysis saved to {output_dir}")

if __name__ == "__main__":
    main()