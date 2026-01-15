import pandas as pd

class LabMetrics:
    """
    Centralized definition of success metrics for the Sell Model Lab.
    """
    # North Star Metrics
    TARGET_WIN_RATE = 0.55      # 55%
    TARGET_AVG_RETURN = 0.0020  # 0.20%

    @staticmethod
    def evaluate_experiment(df_results: pd.DataFrame) -> dict:
        """
        Evaluates the experiment results against the lab's targets.

        Args:
            df_results (pd.DataFrame): DataFrame containing at least 'is_profit' (bool/int) and 'return' (float) columns.

        Returns:
            dict: A status report containing PASS/FAIL status and calculated metrics.
        """
        if df_results.empty:
            return {"status": "FAIL", "reason": "No trades generated"}

        # Calculate core metrics
        win_rate = df_results['is_profit'].mean()
        avg_ret = df_results['return'].mean()
        count = len(df_results)

        # Determine Status
        status = "FAIL"
        if win_rate >= LabMetrics.TARGET_WIN_RATE and avg_ret >= LabMetrics.TARGET_AVG_RETURN:
            status = "PASS"
        elif win_rate >= LabMetrics.TARGET_WIN_RATE:
            status = "MIXED (High WinRate, Low Return)"
        elif avg_ret >= LabMetrics.TARGET_AVG_RETURN:
            status = "MIXED (High Return, Low WinRate)"

        return {
            "status": status,
            "metrics": {
                "Signal Count": count,
                "Win Rate": f"{win_rate:.2%}",
                "Avg Return": f"{avg_ret:.4f}"
            },
            "targets": {
                "Target Win Rate": f"{LabMetrics.TARGET_WIN_RATE:.2%}",
                "Target Avg Return": f"{LabMetrics.TARGET_AVG_RETURN:.4f}"
            }
        }
