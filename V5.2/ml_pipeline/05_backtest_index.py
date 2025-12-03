import pandas as pd
import numpy as np
import os
from backtesting_utils import run_backtest, analyze_performance

def load_data(features_path, regime_signals_path):
    """Loads and merges feature and regime data for the full index."""
    try:
        features_df = pd.read_parquet(features_path)
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')

    # Corrected column name handling
    df.rename(columns={'signal': 'regime_signal'}, inplace=True)

    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.set_index('timestamp')
    return df

def run_benchmark_backtest(all_data, initial_capital=100000.0):
    """Calculates an 'Index Equal Weight Buy & Hold' equity curve."""
    all_symbols = all_data['symbol'].unique()
    start_date = all_data.index.min()

    investment_per_stock = initial_capital / len(all_symbols)
    positions = {}

    initial_prices = all_data.loc[start_date]
    for symbol in all_symbols:
        symbol_data = initial_prices[initial_prices['symbol'] == symbol]
        if not symbol_data.empty:
            open_price = symbol_data['Open'].iloc[0]
            if open_price > 0:
                positions[symbol] = investment_per_stock / open_price

    equity = pd.Series(index=all_data.index.unique().sort_values())

    close_prices_pivot = all_data.pivot(columns='symbol', values='Close')

    for date in equity.index:
        portfolio_value = 0
        prices_for_day = close_prices_pivot.loc[date].dropna()
        for symbol, shares in positions.items():
            if symbol in prices_for_day.index:
                portfolio_value += shares * prices_for_day[symbol]
        equity[date] = portfolio_value

    return equity.dropna()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    FEATURES_PATH = os.path.join(project_root, 'features', 'stock_features_index.parquet')
    REGIME_SIGNALS_PATH = os.path.join(project_root, 'signals', 'regime_signals.parquet')
    OUTPUT_DIR = os.path.join(script_dir, 'analysis')

    all_data = load_data(FEATURES_PATH, REGIME_SIGNALS_PATH)
    if all_data is not None and not all_data.empty:
        strategy_equity = run_backtest(all_data)
        benchmark_equity = run_benchmark_backtest(all_data)

        if not strategy_equity.empty and not benchmark_equity.empty:
            common_index = strategy_equity.index.intersection(benchmark_equity.index)
            analyze_performance(
                equity_curve=strategy_equity.loc[common_index],
                benchmark_curve=benchmark_equity.loc[common_index],
                output_dir=OUTPUT_DIR,
                filename_prefix='index',
                title='Index Portfolio Equity Curve vs. Benchmark',
                benchmark_label='Benchmark (Buy & Hold Equal Weight)'
            )
    else:
        print("No data loaded for index backtest. Aborting.")

if __name__ == "__main__":
    main()
