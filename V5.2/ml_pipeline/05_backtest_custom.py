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

    features_df.index = features_df.index.set_names(['timestamp', 'symbol'])

    df = features_df.join(regime_signals_df, on='timestamp', how='left')
    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.reset_index().set_index('timestamp')
    return df

def load_v5_1_benchmark(benchmark_path, all_dates, initial_capital=100000.0):
    """
    Loads V5.1 trades, calculates returns, and correctly builds a daily equity curve.
    """
    try:
        trades = pd.read_csv(benchmark_path, parse_dates=['entry_date', 'exit_date'])
    except FileNotFoundError:
        print(f"Benchmark file not found at {benchmark_path}. Skipping comparison.")
        return None

    trades['return'] = (trades['exit_price'] / trades['entry_price']) - 1
    daily_returns = trades.groupby('exit_date')['return'].mean()
    daily_returns = daily_returns.reindex(all_dates, fill_value=0)

    equity_curve = pd.Series(index=all_dates, dtype=float)
    equity_curve.iloc[0] = initial_capital

    for i in range(1, len(equity_curve)):
        equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + daily_returns.iloc[i])

    return equity_curve.dropna()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The project root is one level up from the ml_pipeline directory
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    FEATURES_PATH = os.path.join(project_root, 'features', 'stock_features.parquet')
    REGIME_SIGNALS_PATH = os.path.join(project_root, 'signals', 'regime_signals.parquet')
    ASSET_POOL_PATH = os.path.join(script_dir, 'asset_pool.json')
    OUTPUT_DIR = os.path.join(script_dir, 'analysis')
    # The V5.1 path needs to go up two levels from the script dir to get to the repo root
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
