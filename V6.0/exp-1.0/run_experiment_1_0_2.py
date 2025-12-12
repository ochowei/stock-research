import os
import pandas as pd
import config
import utils

# 顯示設定
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def run_individual_analysis(group_name, tickers):
    print(f"\n=== Analyzing {group_name} (Individual Ticker Breakdown) ===")
    
    data_map = utils.fetch_data(tickers)
    if not data_map:
        return None

    all_trades = []
    
    print("Backtesting individual tickers...")
    for ticker, df in data_map.items():
        res = utils.backtest_strategies(df, limit_buffer=0.98)
        # --- [修正版補丁] 計算 MOC No Delay ---
        # 原因：res 可能沒有 'Close' 欄位，所以我們改用原始資料 df 來算
        
        # 1. 確保欄位名稱正確 (有的資料源是 'Close', 有的是小寫 'close')
        c_col = 'Close' if 'Close' in df.columns else 'close'
        
        # 2. 計算無延遲回報: (明日收盤 / 今日收盤) - 1
        # 3. 使用 Pandas 的索引對齊功能，將計算結果塞回 res
        #    即使 res 的行數比 df 少 (例如被過濾過)，Pandas 會依照 Date Index 正確對應
        res['Ret_MOC_0'] = df[c_col].shift(-1) / df[c_col] - 1
        # -----------------------------------------------------------
        active_trades = res[res['Has_Signal'] == 1].copy()
        
        if not active_trades.empty:
            active_trades['Ticker'] = ticker
            active_trades['Group'] = group_name
            all_trades.append(active_trades)
            
    if not all_trades:
        print("No signals found.")
        return None
        
    combined_trades = pd.concat(all_trades)
    
    individual_stats = []
    
    # --- [關鍵修改] 定義要比較的策略 ---
    # 請確保您的 utils.backtest_strategies 有計算 'Ret_MOC_0' (或是您對應的無延遲欄位)
    strategies = {
        'MOC (No Delay)': 'Ret_MOC_0',  # 假設：訊號日當天收盤進場 (Benchmark)
        'MOC (Delayed)':  'Ret_MOC',    # 假設：隔日收盤進場 (Experiment)
    }
    
    unique_tickers = combined_trades['Ticker'].unique()
    
    for ticker in unique_tickers:
        ticker_trades = combined_trades[combined_trades['Ticker'] == ticker]
        
        for strategy_name, col_name in strategies.items():
            # 防呆：確保欄位存在才計算
            if col_name in ticker_trades.columns:
                perf = utils.calculate_performance_summary(ticker_trades[col_name])
                
                perf['Ticker'] = ticker
                perf['Group'] = group_name
                perf['Strategy'] = strategy_name
                
                individual_stats.append(perf)
            else:
                print(f"Warning: Column '{col_name}' not found for ticker {ticker}")
            
    return pd.DataFrame(individual_stats)

def main():
    print(">>> Starting Experiment EXP-1.0.2: MOC Delayed vs No-Delay Analysis")
    
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    
    df_a = run_individual_analysis("Group A (Final Pool)", pool_a)
    df_b = run_individual_analysis("Group B (Toxic Pool)", pool_b)
    
    if df_a is not None and df_b is not None:
        final_df = pd.concat([df_a, df_b], ignore_index=True)
        
        # 整理基本報表
        cols = ['Group', 'Ticker', 'Strategy', 'Win Rate', 'Avg Trade %', 'Trades', 'Total Return', 'Max Drawdown']
        final_df = final_df[cols]
        final_df.sort_values(by=['Group', 'Strategy', 'Win Rate'], ascending=[True, True, False], inplace=True)
        
        # 輸出 Raw Data
        csv_path = os.path.join(config.OUTPUT_DIR, 'exp_1_0_2_individual_report.csv')
        final_df.to_csv(csv_path, index=False)
        print(f"\n>>> Individual Report saved to: {csv_path}")
        
        # --- [關鍵修改] 比較 Delayed vs No Delay ---
        print("\n>>> Calculating Differences (Delayed - No Delay)...")

        # 定義策略名稱變數 (需與上面 strategies 字典一致)
        strat_delayed = 'MOC (Delayed)'
        strat_nodelay = 'MOC (No Delay)'

        # 建立樞紐分析表 (Pivot)
        pivot_df = final_df.pivot_table(
            index=['Group', 'Ticker'], 
            columns='Strategy', 
            values=['Win Rate', 'Total Return', 'Max Drawdown']
        )

        # 攤平欄位名稱
        pivot_df.columns = [f'{val}_{strat}' for val, strat in pivot_df.columns]
        
        # 檢查是否兩個策略都有資料
        col_ret_delayed = f'Total Return_{strat_delayed}'
        col_ret_nodelay = f'Total Return_{strat_nodelay}'

        if col_ret_delayed in pivot_df.columns and col_ret_nodelay in pivot_df.columns:
            # 1. 計算 Total Return 差異 (正值 = Delayed 比較好)
            pivot_df['Diff_Total_Return'] = pivot_df[col_ret_delayed] - pivot_df[col_ret_nodelay]

            # 2. 計算 Win Rate 差異
            col_wr_delayed = f'Win Rate_{strat_delayed}'
            col_wr_nodelay = f'Win Rate_{strat_nodelay}'
            pivot_df['Diff_Win_Rate'] = pivot_df[col_wr_delayed] - pivot_df[col_wr_nodelay]

            # 3. 計算 Max Drawdown 差異 (正值 = Delayed 回檔較小/較安全)
            col_mdd_delayed = f'Max Drawdown_{strat_delayed}'
            col_mdd_nodelay = f'Max Drawdown_{strat_nodelay}'
            pivot_df['Diff_Max_Drawdown'] = pivot_df[col_mdd_delayed] - pivot_df[col_mdd_nodelay]

            # 排序：看「延遲」是否帶來更好的總回報
            pivot_df.sort_values('Diff_Total_Return', ascending=False, inplace=True)
            
            # 輸出比較報表
            # 檔名改為 moc_delayed_vs_nodelay_diff 以符合語意
            diff_csv_path = os.path.join(config.OUTPUT_DIR, 'exp_1_0_2_moc_delayed_vs_nodelay_diff.csv')
            pivot_df.to_csv(diff_csv_path)
            print(f"\n>>> Comparison saved to: {diff_csv_path}")

            # --- Terminal 摘要 ---
            print(f"\n[Top 5: Where {strat_delayed} BEATS {strat_nodelay}]")
            display_cols = ['Diff_Total_Return', col_ret_delayed, col_ret_nodelay, 'Diff_Max_Drawdown']
            print(pivot_df[display_cols].head(5))

            print(f"\n[Bottom 5: Where {strat_delayed} LOSES to {strat_nodelay}]")
            print(pivot_df[display_cols].tail(5))
        else:
            print("\nError: Could not find both strategies in the result. Check your column names in 'strategies' dict.")

if __name__ == '__main__':
    main()