
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

        # d. Trading Simulation (using Model C as the best model)
        f.write("3. Trading Simulation (Model C)\n")
        f.write("---------------------------------\n")

        # Strategy: Long when Y_pred > 0.5 and Uncertainty < 0.8
        long_signals = (df['Y_pred_C'] > 0.5) & (df['Uncertainty_C'] < 0.8)

        # Assuming Y_true represents the return for the period if we enter a trade
        returns = pd.Series(np.where(long_signals, df['Y_true'], 0), index=df.index)

        total_return, sharpe_ratio, max_drawdown, cumulative_return, drawdown = calculate_performance_metrics(returns)

        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Cumulative Return: {total_return:.4f}\n")
        f.write(f"Max Drawdown: {max_drawdown:.4f}\n")

        # Plot Cumulative Return
        plt.figure(figsize=(12, 6))
        cumulative_return.plot()
        plt.title('Cumulative Return')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.savefig(cumulative_return_plot)
        plt.close()

        # Plot Drawdown
        plt.figure(figsize=(12, 6))
        drawdown.plot(color='red')
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.savefig(drawdown_plot)
        plt.close()

if __name__ == '__main__':
    main()
