import os
import pandas as pd
import numpy as np
import config
import utils

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def calculate_period_stats(df, start_date_limit, end_date_limit, min_days=60):
    """
    計算特定區間內的 BnH 與 MOC 績效 (支援動態起始日)
    """
    # 1. 切割時間
    mask = (df.index >= start_date_limit) & (df.index <= end_date_limit)
    sub_df = df.loc[mask].copy()
    
    # 2. 檢查數據量是否足夠
    if len(sub_df) < min_days:
        return None  # 資料太少

    # 3. 計算 Buy and Hold (BnH) 回報
    # [修正] 這裡需要 'Close' 欄位，稍後會在 main 裡面補上
    if 'Close' not in sub_df.columns:
        return None
        
    start_price = sub_df['Close'].iloc[0]
    end_price = sub_df['Close'].iloc[-1]
    ret_bnh = (end_price / start_price) - 1

    # 4. 計算策略回報 (Compound Return)
    def get_cum_ret(col_name):
        if col_name not in sub_df.columns:
            return 0.0
        # 填補 NaN 為 0，計算累積報酬率
        return (1 + sub_df[col_name].fillna(0)).prod() - 1

    ret_moc_delayed = get_cum_ret('Ret_MOC')
    ret_moc_nodelay = get_cum_ret('Ret_MOC_0')

    return {
        'Start_Date': sub_df.index[0].strftime('%Y-%m-%d'),
        'End_Date': sub_df.index[-1].strftime('%Y-%m-%d'),
        'Days': len(sub_df),
        'BnH': ret_bnh,
        'MOC_Delayed': ret_moc_delayed,
        'MOC_NoDelay': ret_moc_nodelay
    }

def main():
    print(">>> Starting Experiment EXP-1.0.3: Alpha Persistence (Fixed Version)")
    
    # 1. 載入所有 Tickers
    pool_a = utils.load_tickers_from_json(config.ASSET_POOL_PATH)
    pool_b = utils.load_tickers_from_json(config.TOXIC_POOL_PATH)
    all_tickers = pool_a + pool_b
    
    data_map = utils.fetch_data(all_tickers)
    
    results = []

    # 定義時間視窗 (Global)
    TRAIN_START_LIMIT = '2023-01-01'
    TRAIN_END_LIMIT   = '2024-12-31'
    TEST_START_LIMIT  = '2025-01-01'
    TEST_END_LIMIT    = '2025-12-31'

    print(f"Target Training Window: {TRAIN_START_LIMIT} ~ {TRAIN_END_LIMIT}")
    print(f"Target Testing Window:  {TEST_START_LIMIT} ~ {TEST_END_LIMIT}")
    print("-" * 60)

    for ticker, df in data_map.items():
        # 先跑策略回測取得每日報表
        res = utils.backtest_strategies(df, limit_buffer=0.98)
        
        # --- [關鍵修正 1] 確保 res 裡面有 'Close' 欄位 ---
        c_col = 'Close' if 'Close' in df.columns else 'close'
        
        # 使用 Pandas 索引對齊，將原始資料的 Close 塞進回測結果 res
        res['Close'] = df[c_col]
        # -----------------------------------------------

        # --- [關鍵修正 2] 計算 MOC No Delay ---
        if len(df) > 1:
            res['Ret_MOC_0'] = df[c_col].shift(-1) / df[c_col] - 1
            res.loc[res['Has_Signal'] == 0, 'Ret_MOC_0'] = 0.0
        else:
            res['Ret_MOC_0'] = 0.0

        # --- 計算 Training 期間績效 ---
        train_stats = calculate_period_stats(res, TRAIN_START_LIMIT, TRAIN_END_LIMIT, min_days=60)
        
        if not train_stats: 
            continue

        # --- 計算 Testing 期間績效 ---
        test_stats = calculate_period_stats(res, TEST_START_LIMIT, TEST_END_LIMIT, min_days=10)
        
        if not test_stats:
            continue
        
        # 紀錄數據
        row = {
            'Ticker': ticker,
            'Train_Start': train_stats['Start_Date'], 
            'Train_Days': train_stats['Days']
        }
        
        # 計算 Alpha
        row['Train_Alpha_Delayed'] = train_stats['MOC_Delayed'] - train_stats['BnH']
        row['Test_Alpha_Delayed'] = test_stats['MOC_Delayed'] - test_stats['BnH']
        
        row['Train_Alpha_NoDelay'] = train_stats['MOC_NoDelay'] - train_stats['BnH']
        row['Test_Alpha_NoDelay'] = test_stats['MOC_NoDelay'] - test_stats['BnH']

        # 判定勝負
        row['Win_Train_Delayed'] = row['Train_Alpha_Delayed'] > 0
        row['Win_Test_Delayed'] = row['Test_Alpha_Delayed'] > 0
        
        row['Win_Train_NoDelay'] = row['Train_Alpha_NoDelay'] > 0
        row['Win_Test_NoDelay'] = row['Test_Alpha_NoDelay'] > 0
        
        results.append(row)

    if not results:
        print("No valid tickers found with sufficient data.")
        return

    df_res = pd.DataFrame(results)
    
    print(f"\nAnalyzed {len(df_res)} valid tickers.")

    def print_strategy_stats(strategy_name, col_train, col_test):
        print(f"\n[Strategy Analysis: {strategy_name}]")
        good_students = df_res[df_res[col_train] == True]
        
        if good_students.empty:
            print("No tickers beat BnH in the training period.")
            return

        continued_success = good_students[good_students[col_test] == True]
        
        accuracy = len(continued_success) / len(good_students)
        
        print(f"Tickers that beat BnH in Training: {len(good_students)}")
        print(f"Tickers that CONTINUED to beat BnH in 2025: {len(continued_success)}")
        print(f">>> Persistence Rate: {accuracy:.2%}")
        
        fresh_stars = continued_success[continued_success['Train_Days'] < 250]
        if not fresh_stars.empty:
            print(f"\n[Fresh Stars] (New IPOs):")
            print(fresh_stars[['Ticker', 'Train_Days', col_train, col_test]].head(5))

    print_strategy_stats("MOC Delayed", 'Win_Train_Delayed', 'Win_Test_Delayed')
    print_strategy_stats("MOC No Delay", 'Win_Train_NoDelay', 'Win_Test_NoDelay')
    
    out_path = os.path.join(config.OUTPUT_DIR, 'exp_1_0_3_persistence_robust.csv')
    df_res.to_csv(out_path, index=False)
    print(f"\nDetailed report saved to: {out_path}")

if __name__ == '__main__':
    main()