
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

def calculate_performance_metrics(returns):
    """Calculates performance metrics for a given returns series."""
    cumulative_return = (1 + returns).cumprod()
    total_return = cumulative_return.iloc[-1] - 1
    if returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Assuming daily returns

    roll_max = cumulative_return.cummax()
    drawdown = cumulative_return / roll_max - 1.0
    max_drawdown = drawdown.min()

    return total_return, sharpe_ratio, max_drawdown, cumulative_return, drawdown

def main():
    # Define file paths
    input_file = 'predictions_oos_merged.csv'
    report_file = 'backtest_report.txt'
    cumulative_return_plot = 'cumulative_return.png'
    drawdown_plot = 'drawdown.png'
    uncertainty_plot = 'uncertainty_calibration.png'

    # a. Load the data
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)

    with open(report_file, 'w') as f:
        f.write("Backtest Analysis Report\n")
        f.write("========================\n\n")

        # b. Prediction Accuracy Comparison
        f.write("1. Prediction Accuracy Comparison\n")
        f.write("---------------------------------\n")

        models = {
            'A': 'Y_pred_A',
            'B': 'Y_pred_B_Median',
            'C': 'Y_pred_C'
        }

        for model_name, pred_col in models.items():
            rmse = np.sqrt(mean_squared_error(df['Y_true'], df[pred_col]))
            mae = mean_absolute_error(df['Y_true'], df[pred_col])
            spearman_corr, _ = spearmanr(df['Y_true'], df[pred_col])

            f.write(f"Model {model_name}:\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  MAE: {mae:.4f}\n")
            f.write(f"  Spearman Correlation: {spearman_corr:.4f}\n\n")

        # c. Uncertainty Calibration
        f.write("2. Uncertainty Calibration Comparison\n")
        f.write("-------------------------------------\n")

        df['Y_error_A'] = (df['Y_true'] - df['Y_pred_A']).abs()
        df['Y_error_B'] = (df['Y_true'] - df['Y_pred_B_Median']).abs()
        df['Y_error_C'] = (df['Y_true'] - df['Y_pred_C']).abs()

        df['Uncertainty_A'] = df['Y_uncertainty_A'].clip(lower=0)
        df['Uncertainty_B'] = df['Y_pred_B_Upper'] - df['Y_pred_B_Lower']
        df['Uncertainty_C'] = df['Y_std_C']

        uncertainty_metrics = {
            'A': ('Y_error_A', 'Uncertainty_A'),
            'B': ('Y_error_B', 'Uncertainty_B'),
            'C': ('Y_error_C', 'Uncertainty_C')
        }

        plt.figure(figsize=(12, 8))

        for model_name, (error_col, uncertainty_col) in uncertainty_metrics.items():
            corr, _ = spearmanr(df[error_col], df[uncertainty_col])
            f.write(f"Model {model_name} (Error vs. Uncertainty) Spearman Correlation: {corr:.4f}\n")
            plt.scatter(df[uncertainty_col], df[error_col], alpha=0.5, label=f'Model {model_name} (Corr: {corr:.2f})')

        f.write("\n")

        plt.title('Uncertainty Calibration')
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Actual Prediction Error')
        plt.legend()
        plt.savefig(uncertainty_plot)
        plt.close()

        # d. Trading Simulation
        f.write("3. Trading Simulation Comparison\n")
        f.write("---------------------------------\n")

        # --- Strategy Parameters ---
        PRED_THRESHOLD = 0.2   # For ML models
        UNCERT_THRESHOLD = 1.0 # For ML models
        BB_LONG_THRESHOLD = -2.0 # For Bollinger Band Strategy
        BB_SHORT_THRESHOLD = 2.0 # For Bollinger Band Strategy
        # --- End Parameters ---

        # Dictionary to hold performance results for each strategy
        performance_results = {}

        # --- Strategy Definitions ---
        strategies = {
            "Model_A": {
                "type": "ml", "pred": "Y_pred_A", "uncert": "Uncertainty_A"
            },
            "Model_B": {
                "type": "ml", "pred": "Y_pred_B_Median", "uncert": "Uncertainty_B"
            },
            "Model_C": {
                "type": "ml", "pred": "Y_pred_C", "uncert": "Uncertainty_C"
            },
            "Bollinger_Band": {
                "type": "bb"
            }
        }

        # --- Run Simulations ---
        for name, params in strategies.items():
            f.write(f"\n--- Strategy: {name} ---\n")

            if params["type"] == "ml":
                long_signals = (df[params["pred"]] > PRED_THRESHOLD) & (df[params["uncert"]] < UNCERT_THRESHOLD)
                short_signals = pd.Series(False, index=df.index) # ML models are long-only for now
                f.write(f"Parameters: PRED_THRESHOLD > {PRED_THRESHOLD}, UNCERT_THRESHOLD < {UNCERT_THRESHOLD}\n")

            elif params["type"] == "bb":
                if 'Z_Score_20' not in df.columns:
                    f.write("Z_Score_20 column not found, skipping Bollinger Band strategy.\n")
                    continue
                long_signals = df['Z_Score_20'] < BB_LONG_THRESHOLD
                short_signals = df['Z_Score_20'] > BB_SHORT_THRESHOLD
                f.write(f"Parameters: LONG_THRESHOLD < {BB_LONG_THRESHOLD}, SHORT_THRESHOLD > {BB_SHORT_THRESHOLD}\n")

            total_signals = long_signals.sum() + short_signals.sum()
            f.write(f"Total Signals Generated: {total_signals} (Long: {long_signals.sum()}, Short: {short_signals.sum()})\n")

            # Calculate returns based on signals
            # Long signal: profit = Y_true
            # Short signal: profit = -Y_true
            returns = np.select(
                [long_signals, short_signals],
                [df['Y_true'], -df['Y_true']],
                default=0
            )
            returns = pd.Series(returns, index=df.index)

            # Calculate and store metrics
            total_return, sharpe, max_dd, cum_ret, drawdown = calculate_performance_metrics(returns)
            performance_results[name] = {'cum_ret': cum_ret, 'drawdown': drawdown}

            f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
            f.write(f"Cumulative Return: {total_return:.4f}\n")
            f.write(f"Max Drawdown: {max_dd:.4f}\n")

        # --- Plotting ---

        # Plot Cumulative Returns for all strategies
        plt.figure(figsize=(12, 6))
        for name, results in performance_results.items():
            results['cum_ret'].plot(label=name)
        plt.title('Strategy Cumulative Returns Comparison')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(cumulative_return_plot)
        plt.close()

        # Plot Drawdowns for all strategies
        plt.figure(figsize=(12, 6))
        for name, results in performance_results.items():
            results['drawdown'].plot(label=name, alpha=0.7)
        plt.title('Strategy Drawdown Comparison')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.savefig(drawdown_plot)
        plt.close()

if __name__ == '__main__':
    main()
