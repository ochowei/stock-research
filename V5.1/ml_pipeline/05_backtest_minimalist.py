import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Configuration ---
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 5
SLIPPAGE = 0.0005
HOLD_DAYS = 5
RSI_THRESHOLD = 10

def load_data(base_dir):
    features_dir = os.path.join(base_dir, 'features')
    stock_path = os.path.join(features_dir, 'stock_features_L0.parquet')
    if not os.path.exists(stock_path):
        raise FileNotFoundError(f"Missing {stock_path}")
    print(f"Loading stock features from {stock_path}...")
    stock_df = pd.read_parquet(stock_path)
    return stock_df

class MinimalistBacktester:
    def __init__(self, stock_df, initial_capital=100000.0):
        self.stock_df = stock_df
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {} 
        self.equity_curve = []
        self.trade_log = []
        self.daily_data = self.stock_df.reorder_levels(['timestamp', 'symbol']).sort_index()
        self.all_dates = self.daily_data.index.get_level_values('timestamp').unique().sort_values()
        
    def run(self):
        print(f"Running Final Minimalist Backtest ({self.all_dates[0].date()} to {self.all_dates[-1].date()})...")
        print("Logic: RSI(2) < 10 & Price > SMA(200) | No Pyramiding | Hold 5 Days")
        
        for i, current_date in enumerate(self.all_dates):
            try:
                today_bar = self.daily_data.loc[current_date]
            except KeyError:
                continue
                
            # 1. Update Equity
            current_equity = self.cash
            for sym, pos in self.positions.items():
                if sym in today_bar.index:
                    price = today_bar.loc[sym]['Close']
                    if pd.isna(price): price = pos['entry_price']
                    current_equity += pos['shares'] * price
                else:
                    current_equity += pos['shares'] * pos['entry_price']
            
            # 2. Exit Logic
            to_sell = []
            for symbol, pos in self.positions.items():
                pos['days_held'] += 1
                if pos['days_held'] >= HOLD_DAYS:
                    if symbol in today_bar.index:
                        exit_price = today_bar.loc[symbol]['Close'] * (1 - SLIPPAGE)
                        if pd.isna(exit_price): continue
                        
                        revenue = pos['shares'] * exit_price
                        self.cash += revenue
                        
                        ret = (exit_price / pos['entry_price']) - 1
                        self.trade_log.append({
                            'symbol': symbol,
                            'entry_date': pos['entry_date'],
                            'exit_date': current_date,
                            'return': ret
                        })
                        to_sell.append(symbol)
            for sym in to_sell:
                del self.positions[sym]
                
            # 3. Entry Logic
            open_slots = MAX_POSITIONS - len(self.positions)
            if open_slots > 0 and self.cash > 0:
                candidates = today_bar[
                    (today_bar['RSI_2'] < RSI_THRESHOLD) & 
                    (today_bar['Dist_SMA_200'] > 0)
                ]
                
                if not candidates.empty:
                    top_picks = candidates.sort_values('RSI_2', ascending=True)
                    
                    target_per_pos = current_equity / MAX_POSITIONS
                    
                    for symbol, row in top_picks.iterrows():
                        if open_slots <= 0: break
                        
                        # [FIX] 防止重複買入導致覆蓋持倉
                        if symbol in self.positions:
                            continue
                            
                        price = row['Close']
                        if pd.isna(price) or price <= 0: continue
                        
                        buy_price = price * (1 + SLIPPAGE)
                        buy_amt = min(self.cash, target_per_pos)
                        shares = int(buy_amt / buy_price)
                        
                        if shares > 0:
                            self.cash -= shares * buy_price
                            self.positions[symbol] = {
                                'shares': shares,
                                'entry_price': buy_price,
                                'entry_date': current_date,
                                'days_held': 0
                            }
                            open_slots -= 1
                            
            self.equity_curve.append({'timestamp': current_date, 'equity': current_equity})
            
        return pd.DataFrame(self.equity_curve).set_index('timestamp'), pd.DataFrame(self.trade_log)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    stock_df = load_data(SCRIPT_DIR)
    backtester = MinimalistBacktester(stock_df, initial_capital=INITIAL_CAPITAL)
    equity, trades = backtester.run()
    
    if equity.empty:
        print("No trades generated.")
        return
        
    total_ret = (equity.iloc[-1]['equity'] / INITIAL_CAPITAL) - 1
    daily_ret = equity['equity'].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd = (equity['equity'] / equity['equity'].cummax() - 1).min()
    win_rate = (trades['return'] > 0).mean() if not trades.empty else 0
    
    print("\n=== Final Minimalist Baseline Result ===")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Final Equity: ${equity.iloc[-1]['equity']:,.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity['equity'], label=f'Baseline (SR: {sharpe:.2f})')
    plt.title('V5 Baseline Equity Curve (Bug Fixed)')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'baseline_equity_curve.png'))

if __name__ == "__main__":
    main()