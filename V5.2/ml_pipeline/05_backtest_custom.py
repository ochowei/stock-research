import pandas as pd
import os
import json
from backtesting_utils import run_backtest, analyze_performance

def load_data(features_path, regime_signals_path, asset_pool):
    """Loads and merges feature and regime data."""
    try:
        features_df = pd.read_parquet(features_path)
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    asset_pool_cleaned = [a.replace('.', '-') for a in asset_pool]
    features_df = features_df[features_df.index.get_level_values('symbol').isin(asset_pool_cleaned)]

    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')

    df.rename(columns={'signal': 'regime_signal'}, inplace=True)

    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.set_index('timestamp')
    return df

def load_v5_1_benchmark(benchmark_path, all_dates, initial_capital=100000.0):
    """
    Loads V5.1 trades and correctly builds a daily equity curve using the pre-calculated return.
    """
    try:
        trades = pd.read_csv(benchmark_path, parse_dates=['entry_date', 'exit_date'])
    except FileNotFoundError:
        print(f"Benchmark file not found at {benchmark_path}. Skipping comparison.")
        return None

    # Corrected logic: Use the 'return' column directly
    daily_returns = trades.groupby('exit_date')['return'].mean()

    daily_returns = daily_returns.reindex(all_dates, fill_value=0)

    equity_curve = pd.Series(index=all_dates, dtype=float)
    equity_curve.iloc[0] = initial_capital

    for i in range(1, len(equity_curve)):
        equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + daily_returns.iloc[i])

    return equity_curve.dropna()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    FEATURES_PATH = os.path.join(project_root, 'features', 'stock_features.parquet')
    REGIME_SIGNALS_PATH = os.path.join(project_root, 'signals', 'regime_signals.parquet')
    ASSET_POOL_PATH = os.path.join(script_dir, 'asset_pool.json')
    OUTPUT_DIR = os.path.join(script_dir, 'analysis')
    repo_root = os.path.abspath(os.path.join(project_root, '..'))
    BENCHMARK_PATH = os.path.join(repo_root, 'V5.1', 'ml_pipeline', 'analysis', 'minimalist_trades_fixed.csv')

    with open(ASSET_POOL_PATH, 'r') as f:
        asset_pool_raw = json.load(f)
    asset_pool = [ticker.split(':')[1] for ticker in asset_pool_raw]

    all_data = load_data(FEATURES_PATH, REGIME_SIGNALS_PATH, asset_pool)
    if all_data is not None and not all_data.empty:
        strategy_equity = run_backtest(all_data)

        all_dates = all_data.index.unique().sort_values()
        benchmark_equity = load_v5_1_benchmark(BENCHMARK_PATH, all_dates)

        if not strategy_equity.empty:
            if benchmark_equity is not None:
                common_index = strategy_equity.index.intersection(benchmark_equity.index)
                strategy_equity = strategy_equity.loc[common_index]
                benchmark_equity = benchmark_equity.loc[common_index]

            analyze_performance(
                equity_curve=strategy_equity,
                output_dir=OUTPUT_DIR,
                filename_prefix='custom',
                title='Custom Portfolio Equity Curve vs. V5.1 Benchmark',
                benchmark_curve=benchmark_equity,
                benchmark_label='V5.1 Benchmark'
            )
    else:
        print("No data loaded for custom backtest. Aborting.")

if __name__ == "__main__":
    main()
