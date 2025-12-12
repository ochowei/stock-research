import os
import pandas as pd
import numpy as np
import config
import utils

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def get_detailed_stats(df, start_date, end_date):
    """計算詳細的交易統計數據"""
    mask = (df.index >= start_date) & (df.index <= end_date)
    sub_df = df.loc[mask].copy()
    
    if len(sub_df) < 10: return None
    
    # 1. 基準回報 (BnH)
    start_price = sub_df['Close'].iloc[0]
    end_price = sub_df['Close'].iloc[-1]
    ret_bnh = (end_price / start_price) - 1
    
    # 2. 策略回報 (MOC Delayed)
    # 假設 Ret_MOC 已經計算好
    ret_strat = (1 + sub_df['Ret_MOC'].fillna(0)).prod() - 1
    
    # 3. 交易統計
    # 找出有訊號的日子 (假設 Has_Signal == 1 且持有到隔天產生了 Ret_MOC)
    # 注意: Ret_MOC 非 0 或非 NaN 的日子代表有持倉
    trade_days = sub_df[sub_df['Ret_MOC'] != 0].copy()
    num_trades = len(trade_days)
    
    if num_trades > 0:
        win_rate = len(trade_days[trade_days['Ret_MOC'] > 0]) / num_trades
        avg_trade_ret = trade_days['Ret_MOC'].mean()
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0

    return {
        'BnH_Return': ret_bnh,
        'Strat_Return': ret_strat,
        'Alpha': ret_strat - ret_bnh,
        'Win_Rate': win_rate,
        'Avg_Trade_Ret': avg_trade_ret,
        'Trade_Count': num_trades
    }

def main():
    print(">>> Starting Experiment EXP-1.0.4: Failure Analysis (Post-Mortem)")
    
    # 1. 載入資料
    tickers = utils.load_tickers_from_json(config.ASSET_POOL_PATH) + \
              utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    data_map = utils.fetch_data(tickers)
    
    analysis_rows = []
    
    # 定義時期
    TRAIN_START = '2023-01-01'
    TRAIN_END   = '2024-12-31'
    TEST_START  = '2025-01-01'
    TEST_END    = '2025-12-31'

    for ticker, df in data_map.items():
        # 執行回測與補丁
        res = utils.backtest_strategies(df, limit_buffer=0.98)
        c_col = 'Close' if 'Close' in df.columns else 'close'
        res['Close'] = df[c_col] # 補上 Close 以計算 BnH
        
        # 取得兩個時期的數據
        stats_train = get_detailed_stats(res, TRAIN_START, TRAIN_END)
        stats_test = get_detailed_stats(res, TEST_START, TEST_END)
        
        if not stats_train or not stats_test: continue
        
        # 判斷是否為「變節者」(Train Win -> Test Loss)
        train_win = stats_train['Alpha'] > 0
        test_win = stats_test['Alpha'] > 0
        
        if train_win and not test_win:
            # 這是我們要分析的目標：過去的好學生，現在變壞了
            row = {
                'Ticker': ticker,
                # 績效衰退幅度
                'Alpha_Decay': stats_test['Alpha'] - stats_train['Alpha'],
                
                # 診斷因子 1: 勝率是否崩盤?
                'Train_WinRate': stats_train['Win_Rate'],
                'Test_WinRate': stats_test['Win_Rate'],
                'Diff_WinRate': stats_test['Win_Rate'] - stats_train['Win_Rate'],
                
                # 診斷因子 2: 是否踏空? (BnH 太強)
                'Test_BnH': stats_test['BnH_Return'],
                'Test_Strat': stats_test['Strat_Return'],
                
                # 診斷因子 3: 交易品質
                'Diff_Avg_Trade': stats_test['Avg_Trade_Ret'] - stats_train['Avg_Trade_Ret'],
                'Test_Count': stats_test['Trade_Count']
            }
            
            # 自動判斷死因 (簡易規則)
            reason = []
            if stats_test['Win_Rate'] < 0.45 and row['Diff_WinRate'] < -0.1:
                reason.append("WinRate_Crash(勝率崩盤)")
            
            if stats_test['BnH_Return'] > 0.5 and stats_test['Strat_Return'] < 0.2:
                 reason.append("Missed_Rally(嚴重踏空)")
            
            if stats_test['Strat_Return'] < -0.2:
                reason.append("Heavy_Loss(虧損嚴重)")
            
            if not reason:
                reason.append("Alpha_Decay(單純超額報酬消失)")
                
            row['Diagnosis'] = ", ".join(reason)
            analysis_rows.append(row)

    df_res = pd.DataFrame(analysis_rows)
    
    if df_res.empty:
        print("No failed tickers found to analyze.")
        return

    # 排序：依照 Alpha 衰退程度 (誰變爛最多)
    df_res.sort_values('Alpha_Decay', ascending=True, inplace=True)
    
    print(f"\nAnalyzed {len(df_res)} tickers that failed in 2025.")
    
    # 輸出重點欄位
    cols = ['Ticker', 'Diagnosis', 'Diff_WinRate', 'Test_BnH', 'Test_Strat', 'Alpha_Decay']
    print("\n[Top 10 Most Dramatic Failures]")
    print(df_res[cols].head(10))
    
    # 輸出完整報告
    out_path = os.path.join(config.OUTPUT_DIR, 'exp_1_0_4_failure_analysis.csv')
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved diagnosis to: {out_path}")

if __name__ == '__main__':
    main()