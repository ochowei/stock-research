import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import DataLoader
from risk_manager import RiskManager

# --- Configuration ---
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 5
SLIPPAGE = 0.0005 # 5 bps
TRANSACTION_COST = 0.0005 # 5 bps

class V5_3_Backtester:
    def __init__(self, stock_df, regime_df, rank_df, breadth_df, 
                 initial_capital=100000.0, max_positions=5):
        
        # 1. 數據排序
        self.stock_df = stock_df.sort_index()
        self.regime_df = regime_df.sort_index()
        self.rank_df = rank_df.sort_index()
        self.breadth_df = breadth_df.sort_index()
        
        # 2. [CRITICAL FIX] 預先計算 T-1 訊號 (Shift Logic)
        # 我們在 T 日開盤交易，只能看到 T-1 的收盤資訊
        print("Pre-calculating T-1 signals to avoid Look-Ahead Bias...")
        
        # Stock Features: Group by symbol and shift
        # 這裡假設欄位都是小寫 (由 01_format_data 保證)
        # 為了安全，先檢查欄位是否存在
        cols_to_shift = ['RSI_2', 'SMA_200', 'close', 'ATR_14']
        # 處理大小寫相容性 (若 01_format 沒轉好)
        col_map = {c: c for c in self.stock_df.columns}
        # 嘗試找對應的欄位名 (Case insensitive search)
        for target in cols_to_shift:
            for c in self.stock_df.columns:
                if c.lower() == target.lower():
                    col_map[target] = c
                    break
        
        # 建立 prev_ 欄位
        self.stock_df['prev_RSI_2'] = self.stock_df.groupby('symbol')[col_map.get('RSI_2', 'RSI_2')].shift(1)
        self.stock_df['prev_SMA_200'] = self.stock_df.groupby('symbol')[col_map.get('SMA_200', 'SMA_200')].shift(1)
        self.stock_df['prev_close'] = self.stock_df.groupby('symbol')[col_map.get('close', 'close')].shift(1)
        self.stock_df['prev_ATR_14'] = self.stock_df.groupby('symbol')[col_map.get('ATR_14', 'ATR_14')].shift(1)

        # Rank Scores: Group by symbol and shift
        if not self.rank_df.empty:
            self.rank_df['prev_L3_Rank_Score'] = self.rank_df.groupby('symbol')['L3_Rank_Score'].shift(1)

        # Regime & Breadth: 直接 shift (Time Series)
        self.regime_df['prev_signal'] = self.regime_df['signal'].shift(1)
        self.breadth_df['prev_market_breadth'] = self.breadth_df['market_breadth'].shift(1)

        # 3. 初始化回測變數
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rm = RiskManager(target_risk=0.01, max_position_pct=0.2)
        
        self.cash = initial_capital
        self.positions = {} 
        self.equity_curve = []
        self.trade_log = []
        
        # 4. 準備每日迭代數據
        self.daily_data = self.stock_df.reorder_levels(['timestamp', 'symbol']).sort_index()
        self.all_dates = self.daily_data.index.get_level_values('timestamp').unique().sort_values()

    def get_market_context(self, date):
        """取得 T-1 的 L1 狀態與市場寬度 (用於 T 日決策)"""
        regime = 0
        breadth = 0.5 
        
        if date in self.regime_df.index:
            val = self.regime_df.loc[date, 'prev_signal'] # Use Shifted
            regime = val.iloc[-1] if isinstance(val, pd.Series) else val
            
        if date in self.breadth_df.index:
            val = self.breadth_df.loc[date, 'prev_market_breadth'] # Use Shifted
            breadth = val.iloc[-1] if isinstance(val, pd.Series) else val
            
        # Handle NaN at start
        if pd.isna(regime): regime = 0
        if pd.isna(breadth): breadth = 0.5
            
        return regime, breadth

    def run(self):
        print(f"Running Backtest on {len(self.all_dates)} days...")
        
        for date in self.all_dates:
            # 1. 取得環境資訊 (基於 T-1)
            regime, breadth = self.get_market_context(date)
            
            # L4 動態 K 值
            trailing_k = 1.5 if breadth < 0.30 else 3.0
            
            # 2. 取得今日 (T) 數據 (用於執行)
            try:
                today_bar = self.daily_data.loc[date]
            except KeyError:
                continue

            # --- A. L1 混合防禦 (Liquidation) ---
            if regime == 2: # Crash State (detected at T-1 Close)
                symbols_to_sell = list(self.positions.keys())
                for sym in symbols_to_sell:
                    # T 日開盤執行清倉
                    self._execute_sell(sym, date, today_bar, reason="L1_Liquidation")
            else:
                # --- B. L4 動態出場 (Trailing Stop) ---
                symbols_to_check = list(self.positions.keys())
                for sym in symbols_to_check:
                    if sym not in today_bar.index: continue
                    
                    pos = self.positions[sym]
                    row = today_bar.loc[sym]
                    
                    # 更新最高價 (用 T 日 High)
                    current_high = row['high'] # Lowercase
                    if current_high > pos['highest_high']:
                        pos['highest_high'] = current_high
                    
                    # 計算止損價
                    stop_price = pos['highest_high'] - (trailing_k * pos['entry_atr'])
                    
                    # 檢查觸發 (T 日 Low 穿價)
                    if row['low'] < stop_price: # Lowercase
                        # 模擬成交：若 Open < Stop (跳空跌停)，用 Open；否則用 Stop
                        exec_price = min(row['open'], stop_price) # Lowercase
                        self._execute_sell(sym, date, today_bar, reason="L4_Trailing", override_price=exec_price)

            # --- C. L2 & L3 進場邏輯 ---
            if regime != 2:
                self._process_entries(date, today_bar)

            # --- D. 結算 ---
            self._update_equity(date, today_bar)

        return pd.DataFrame(self.equity_curve).set_index('timestamp'), pd.DataFrame(self.trade_log)

    def _process_entries(self, date, today_bar):
        open_slots = self.max_positions - len(self.positions)
        if open_slots <= 0 or self.cash < 1000:
            return

        # 篩選 L2 訊號 (使用 prev_ T-1 欄位)
        candidates = today_bar[
            (today_bar['prev_RSI_2'] < 10) & 
            (today_bar['prev_close'] > today_bar['prev_SMA_200']) 
        ]
        
        if candidates.empty:
            return

        # L3 排序 (使用 prev_L3_Rank_Score)
        if date in self.rank_df.index.get_level_values('timestamp'):
            ranks = self.rank_df.xs(date, level='timestamp')
            # 這裡我們需要 ranks 裡的 prev_L3_Rank_Score
            # 因為 ranks 是從 self.rank_df (含 prev) 切出來的
            
            # 安全合併
            candidates = candidates.join(ranks[['prev_L3_Rank_Score']], how='inner')
            candidates = candidates.sort_values('prev_L3_Rank_Score', ascending=False)
        else:
            # Fallback
            candidates = candidates.sort_values('prev_RSI_2', ascending=True)

        targets = candidates.head(open_slots)

        for sym, row in targets.iterrows():
            if sym in self.positions: continue
            
            # T 日開盤買入
            price = row['open'] # Lowercase
            atr = row['prev_ATR_14']
            
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
                        'entry_date': date
                    }

    def _execute_sell(self, sym, date, today_bar, reason, override_price=None):
        if sym not in self.positions: return
        pos = self.positions.pop(sym)
        shares = pos['shares']
        
        if override_price:
            price = override_price
        elif sym in today_bar.index:
            price = today_bar.loc[sym]['open'] # Lowercase
        else:
            price = pos['entry_price']

        proceeds = shares * price * (1 - SLIPPAGE - TRANSACTION_COST)
        self.cash += proceeds
        
        ret = (price / pos['entry_price']) - 1
        self.trade_log.append({
            'symbol': sym,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'return': ret,
            'reason': reason,
            'hold_days': (date - pos['entry_date']).days
        })

    def _get_current_equity(self):
        eq = self.cash
        for p in self.positions.values():
            eq += p['shares'] * p['highest_high'] 
        return eq

    def _update_equity(self, date, today_bar):
        curr_eq = self.cash
        for sym, pos in self.positions.items():
            if sym in today_bar.index:
                price = today_bar.loc[sym]['close'] # Lowercase
                curr_eq += pos['shares'] * price
            else:
                curr_eq += pos['shares'] * pos['entry_price']
        self.equity_curve.append({'timestamp': date, 'equity': curr_eq})

# --- Main Execution Flow ---
def load_track_data(base_dir, track):
    print(f"Loading data for track: {track}...")
    track_dir = os.path.join(base_dir, 'data', track)
    
    # Paths (Corrected for V5.3 structure)
    feat_path = os.path.join(track_dir, 'features', 'stock_features.parquet')
    # Signal paths are in data/{track}/signals
    regime_path = os.path.join(track_dir, 'signals', 'regime_signals.parquet')
    rank_path = os.path.join(track_dir, 'signals', 'l3_rank_scores.csv')
    breadth_path = os.path.join(track_dir, 'features', 'market_breadth.parquet')
    
    if not os.path.exists(feat_path): return None, None, None, None
    
    stock_df = pd.read_parquet(feat_path)
    regime_df = pd.read_parquet(regime_path)
    breadth_df = pd.read_parquet(breadth_path)
    
    if os.path.exists(rank_path):
        rank_df = pd.read_csv(rank_path)
        rank_df['timestamp'] = pd.to_datetime(rank_df['timestamp'])
        rank_df = rank_df.set_index(['timestamp', 'symbol'])
    else:
        rank_df = pd.DataFrame()
        
    return stock_df, regime_df, rank_df, breadth_df

def filter_tickers(df, tickers):
    return df[df.index.get_level_values('symbol').isin(tickers)]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== V5.3 Final Backtest: Full System Validation (Fixed Look-Ahead) ===")
    
    loader = DataLoader(script_dir, normal_file='final_asset_pool.json', toxic_file='final_toxic_asset_pool.json')
    
    scenarios = [
        ('V5.3 Custom (Merged)', 'custom', loader.get_all_tickers),
        ('V5.3 Custom (Toxic)', 'custom', loader.get_toxic_tickers),
        ('V5.3 Index (S&P100)', 'index', None)
    ]
    
    results = []
    
    for name, track, get_tickers in scenarios:
        print(f"\n>> Simulating Scenario: {name}")
        
        stock, regime, rank, breadth = load_track_data(script_dir, track)
        if stock is None:
            print(f"Skipping {name}: Data not found.")
            continue
            
        if get_tickers:
            target_list = get_tickers()
            stock = filter_tickers(stock, target_list)
            
        if stock.empty:
            print(f"Skipping {name}: No stock data after filter.")
            continue
            
        backtester = V5_3_Backtester(stock, regime, rank, breadth)
        equity, trades = backtester.run()
        
        if not equity.empty:
            total_ret = (equity.iloc[-1]['equity'] / INITIAL_CAPITAL) - 1
            daily_ret = equity['equity'].pct_change().fillna(0)
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else 0
            max_dd = (equity['equity'] / equity['equity'].cummax() - 1).min()
            
            results.append({
                'Scenario': name,
                'Total Return': f"{total_ret:.2%}",
                'Sharpe': f"{sharpe:.2f}",
                'MaxDD': f"{max_dd:.2%}",
                'Trades': len(trades),
                'Final Equity': f"${equity.iloc[-1]['equity']:,.0f}"
            })
            
            plt.figure(figsize=(10, 6))
            plt.plot(equity.index, equity['equity'])
            plt.title(f'{name} Equity Curve')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'v5.3_{track}_equity.png'))
            plt.close()

    if results:
        df_res = pd.DataFrame(results)
        print("\n=== V5.3 Final Results ===")
        print(df_res.to_string(index=False))
        df_res.to_csv(os.path.join(output_dir, 'v5.3_final_report.csv'), index=False)
        print(f"\nReport saved to {output_dir}")

if __name__ == "__main__":
    main()