import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# --- Configuration (回測參數設定) ---
INITIAL_CAPITAL = 100_000.0  # 初始本金
MAX_POSITIONS = 5            # 最大持倉檔數
SLIPPAGE = 0.0005            # 滑價 5bps
ATR_MULTIPLIER = 2.0         # L4 止盈目標 (2.0 * ATR)

def load_data(base_dir):
    """載入回測所需的 L3 分數、股價特徵與市場狀態"""
    signals_dir = os.path.join(base_dir, 'signals')
    features_dir = os.path.join(base_dir, 'features')
    
    # 1. L3 Rank Scores (交易訊號來源)
    scores_path = os.path.join(signals_dir, 'l3_rank_scores.csv')
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing {scores_path}. Run step 04 first.")
    
    scores = pd.read_csv(scores_path)
    scores['timestamp'] = pd.to_datetime(scores['timestamp'])
    # 設定索引為 (timestamp, symbol) 以便每日快速查詢
    scores = scores.set_index(['timestamp', 'symbol']).sort_index()
    
    # 2. Stock Features (價格數據 OHLCV + ATR)
    stock_path = os.path.join(features_dir, 'stock_features_L0.parquet')
    stock_df = pd.read_parquet(stock_path)
    
    # 3. Regime Signals (L1 防禦訊號)
    regime_path = os.path.join(signals_dir, 'regime_signals.parquet')
    regime = pd.read_parquet(regime_path)
    
    # 確保 regime 有 timestamp 索引
    if 'timestamp' in regime.columns:
        regime = regime.set_index('timestamp')
        
    return scores, stock_df, regime

class CapitalPoolBacktester:
    def __init__(self, stock_df, regime_df, scores_df, 
                 initial_capital=100000.0, 
                 max_positions=5, 
                 slippage=0.0005,
                 ranking_col='L3_Rank_Score', 
                 ranking_ascending=False,
                 use_dynamic_exit=True):
        
        self.stock_df = stock_df
        self.regime_df = regime_df.sort_index()
        self.scores_df = scores_df
        
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.slippage = slippage
        
        # 策略配置
        self.ranking_col = ranking_col        # 排序欄位 (L3_Rank_Score 或 RSI_2)
        self.ranking_ascending = ranking_ascending # 排序方向 (L3 為 False, RSI 為 True)
        self.use_dynamic_exit = use_dynamic_exit   # 是否啟用 L4 動態出場
        
        # 帳戶狀態
        self.cash = initial_capital
        self.positions = {} 
        self.equity_curve = []
        self.trade_log = []
        self.pending_orders = [] 
        
        # 預處理：將 stock_df 轉為以 timestamp 為主索引
        print(f"Initializing Backtester ({ranking_col}, DynamicExit={use_dynamic_exit})...")
        self.daily_data = self.stock_df.reorder_levels(['timestamp', 'symbol']).sort_index()
        self.all_dates = self.daily_data.index.get_level_values('timestamp').unique().sort_values()
        
    def run(self):
        # print(f"Starting Backtest from {self.all_dates[0].date()} to {self.all_dates[-1].date()}...")
        
        for i, current_date in enumerate(self.all_dates):
            if i == 0: continue 
            
            # 1. 取得今日行情
            try:
                today_bar = self.daily_data.loc[current_date]
            except KeyError:
                continue

            # 計算權益 (Mark-to-Market)
            current_equity = self.cash
            for sym, pos in self.positions.items():
                if sym in today_bar.index:
                    current_equity += pos['shares'] * today_bar.loc[sym]['Close']
                else:
                    current_equity += pos['shares'] * pos['entry_price']
            
            # --- A. 執行掛單 ---
            for symbol in self.pending_orders:
                if symbol in self.positions: continue
                if self.cash <= 0: break
                
                if symbol in today_bar.index:
                    open_price = today_bar.loc[symbol]['Open']
                    buy_price = open_price * (1 + self.slippage)
                    
                    target_amt = current_equity * (1.0 / self.max_positions)
                    buy_amt = min(self.cash, target_amt)
                    shares = int(buy_amt / buy_price)
                    
                    if shares > 0:
                        self.cash -= shares * buy_price
                        atr = today_bar.loc[symbol].get('ATR_14', buy_price * 0.02)
                        if pd.isna(atr): atr = buy_price * 0.02
                        
                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_price': buy_price,
                            'entry_date': current_date,
                            'days_held': 0,
                            'atr_at_entry': atr
                        }
            self.pending_orders = []
            
            # --- B. 處理出場 ---
            to_sell = []
            for symbol, pos in self.positions.items():
                pos['days_held'] += 1
                if symbol not in today_bar.index: continue
                
                bar = today_bar.loc[symbol]
                high_price = bar['High']
                close_price = bar['Close']
                
                exit_price = None
                exit_reason = ""
                
                # 1. L4 Dynamic Exit (Conditional)
                if self.use_dynamic_exit:
                    target_price = pos['entry_price'] + (ATR_MULTIPLIER * pos['atr_at_entry'])
                    if high_price >= target_price:
                        exit_price = target_price
                        exit_reason = "L4_TP"
                
                # 2. Fixed Time Exit (Fallback or Primary)
                if exit_price is None and pos['days_held'] >= 5:
                    exit_price = close_price * (1 - self.slippage)
                    exit_reason = "Time_Exit"
                    
                if exit_price is not None:
                    revenue = pos['shares'] * exit_price
                    self.cash += revenue
                    ret = (exit_price / pos['entry_price']) - 1
                    self.trade_log.append({
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'return': ret,
                        'reason': exit_reason
                    })
                    to_sell.append(symbol)
            
            for sym in to_sell:
                del self.positions[sym]
                
            # --- C. 更新權益 ---
            updated_equity = self.cash
            for sym, pos in self.positions.items():
                if sym in today_bar.index:
                    updated_equity += pos['shares'] * today_bar.loc[sym]['Close']
                else:
                    updated_equity += pos['shares'] * pos['entry_price']
            
            self.equity_curve.append({
                'timestamp': current_date,
                'equity': updated_equity
            })
            
            # --- D. 生成明日訊號 ---
            is_safe = True
            if current_date in self.regime_df.index:
                r_row = self.regime_df.loc[current_date]
                if isinstance(r_row, pd.DataFrame): r_row = r_row.iloc[-1]
                if (r_row['HMM_State'] == 2) or (r_row['Is_Anomaly'] == 1):
                    is_safe = False
            
            if is_safe:
                open_slots = self.max_positions - len(self.positions)
                if open_slots > 0:
                    try:
                        if current_date in self.scores_df.index.get_level_values('timestamp'):
                            candidates = self.scores_df.loc[current_date]
                            candidates = candidates[~candidates.index.isin(self.positions.keys())]
                            
                            if not candidates.empty:
                                # [修改] 根據策略參數進行排序
                                if self.ranking_col in candidates.columns:
                                    top_buys = candidates.sort_values(
                                        self.ranking_col, 
                                        ascending=self.ranking_ascending
                                    ).head(open_slots)
                                    self.pending_orders = top_buys.index.tolist()
                    except KeyError:
                        pass

        return pd.DataFrame(self.equity_curve).set_index('timestamp'), pd.DataFrame(self.trade_log)

def run_voo_benchmark(stock_df, start_date, end_date, initial_capital=100000.0):
    print("Simulating VOO Benchmark...")
    target_symbol = 'VOO' if 'VOO' in stock_df.index.get_level_values('symbol') else 'SPY'
    if target_symbol not in stock_df.index.get_level_values('symbol'):
        return pd.DataFrame()
        
    df = stock_df.xs(target_symbol, level='symbol').sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty: return pd.DataFrame()
    
    shares = initial_capital / df.iloc[0]['Open']
    equity = df['Close'] * shares
    return equity.to_frame(name='equity')

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    print("=== V5.1 Multi-Strategy Capital Pool Backtest ===")
    scores, stock_df, regime = load_data(SCRIPT_DIR)
    
    # 定義要比較的策略
    strategies = {
        'V5_Base (Fixed)': {
            'ranking_col': 'RSI_2', 
            'ranking_ascending': True, 
            'use_dynamic_exit': False
        },
        'V5.1_L3 (Fixed)': {
            'ranking_col': 'L3_Rank_Score', 
            'ranking_ascending': False, 
            'use_dynamic_exit': False
        },
        'V5.1_Full (Dynamic)': {
            'ranking_col': 'L3_Rank_Score', 
            'ranking_ascending': False, 
            'use_dynamic_exit': True
        }
    }
    
    equity_curves = {}
    
    # 執行所有策略
    for name, config in strategies.items():
        print(f"\n--- Running {name} ---")
        backtester = CapitalPoolBacktester(
            stock_df, regime, scores,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            slippage=SLIPPAGE,
            **config
        )
        eq, _ = backtester.run()
        if not eq.empty:
            equity_curves[name] = eq['equity']
            
    # 執行 VOO 基準
    if equity_curves:
        # 取所有策略的共同時間區間 (通常是一樣的)
        first_strategy = list(equity_curves.keys())[0]
        start_date = equity_curves[first_strategy].index.min()
        end_date = equity_curves[first_strategy].index.max()
        
        voo_eq = run_voo_benchmark(stock_df, start_date, end_date, INITIAL_CAPITAL)
        if not voo_eq.empty:
            equity_curves['VOO (Buy&Hold)'] = voo_eq['equity']
            
        # 合併與比較
        combined = pd.DataFrame(equity_curves).dropna()
        
        # 計算指標
        stats = []
        for col in combined.columns:
            series = combined[col]
            total_ret = (series.iloc[-1] / series.iloc[0]) - 1
            daily_ret = series.pct_change().dropna()
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
            
            roll_max = series.cummax()
            dd = (series / roll_max) - 1
            max_dd = dd.min()
            
            stats.append({
                'Strategy': col,
                'Total_Return': f"{total_ret:.2%}",
                'Sharpe': f"{sharpe:.2f}",
                'Max_DD': f"{max_dd:.2%}",
                'Final_Equity': f"${series.iloc[-1]:,.0f}"
            })
            
        stats_df = pd.DataFrame(stats)
        print("\n=== Performance Comparison Table ===")
        print(stats_df.to_string(index=False))
        
        stats_df.to_csv(os.path.join(ANALYSIS_DIR, 'multi_strategy_comparison.csv'), index=False)
        
        # 繪圖
        plt.figure(figsize=(14, 8))
        for col in combined.columns:
            # 調整線條樣式以區分
            linewidth = 2.0 if 'V5.1_L3' in col else 1.5
            alpha = 0.8 if 'VOO' in col else 1.0
            plt.plot(combined.index, combined[col], label=col, linewidth=linewidth, alpha=alpha)
            
        plt.title('V5.1 Capital Pool Comparison (Base vs L3 vs Dynamic vs VOO)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(ANALYSIS_DIR, 'v5_1_multi_strategy_comparison.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()