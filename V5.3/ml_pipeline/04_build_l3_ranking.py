import pandas as pd
import os
import numpy as np

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def main():
    print("=== V5.3 Step 2.4: L3 Microstructure Ranking (Rule-Based) ===")
    script_dir = get_script_dir()

    # 處理兩個軌道
    tracks = ['custom', 'index']

    for track in tracks:
        print(f"\n--- Processing Track: {track} ---")
        
        # 定義路徑 (V5.3 結構)
        base_dir = os.path.join(script_dir, 'data', track)
        features_path = os.path.join(base_dir, 'features', 'stock_features.parquet')
        output_dir = os.path.join(base_dir, 'signals')
        output_path = os.path.join(output_dir, 'l3_rank_scores.csv')

        if not os.path.exists(features_path):
            print(f"Warning: Features not found at {features_path}. Skipping.")
            continue

        print("Loading features...")
        df = pd.read_parquet(features_path)

        # 檢查必要欄位 (來自 Step 2.2)
        req_cols = ['RSI_2', 'Amihud_Illiquidity', 'Down_Vol_Prop']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Error: Missing columns {missing}. Please re-run 02_build_features.py.")
            continue

        print("Calculating Ranking Scores...")
        
        # 為了避免 SettingWithCopyWarning，先拷貝需要的欄位
        score_df = df[req_cols].copy()

        # --- 計算每日截面排名 (Cross-Sectional Rank) ---
        # 使用 pct=True 將排名標準化到 0~1 之間
        # Rank 越小代表數值越小
        
        # 1. RSI Rank: 我們喜歡低 RSI
        # Rank 低 (e.g. 0.01) -> (1 - 0.01) = 0.99 分 -> 高分
        rsi_rank = score_df.groupby(level='timestamp')['RSI_2'].rank(pct=True, ascending=True)

        # 2. Amihud Rank: 我們喜歡低衝擊 (流動性好)
        # Rank 低 -> 高分
        amihud_rank = score_df.groupby(level='timestamp')['Amihud_Illiquidity'].rank(pct=True, ascending=True)

        # 3. DownVol Rank: 我們喜歡低拋壓
        # Rank 低 -> 高分
        downvol_rank = score_df.groupby(level='timestamp')['Down_Vol_Prop'].rank(pct=True, ascending=True)

        # --- 綜合評分公式 ---
        # 權重可調整，目前設定：RSI (1.0) + 微結構 (各 0.5)
        # 總分越高越好
        score_df['L3_Rank_Score'] = (1 - rsi_rank) * 1.0 + \
                                    (1 - amihud_rank) * 0.5 + \
                                    (1 - downvol_rank) * 0.5

        # 儲存結果
        os.makedirs(output_dir, exist_ok=True)
        
        # 只保留分數與 RSI (供回測參考)
        out_df = score_df[['L3_Rank_Score', 'RSI_2']]
        out_df.to_csv(output_path)
        print(f"Saved ranking scores to: {output_path}")
        
        # 驗證：顯示最新日期的 Top 5
        try:
            latest_date = df.index.get_level_values('timestamp').max()
            top_picks = out_df.xs(latest_date, level='timestamp').sort_values('L3_Rank_Score', ascending=False).head(5)
            print(f"\n[Preview] Top 5 Picks for {latest_date.date()}:")
            print(top_picks)
        except Exception as e:
            print(f"Could not print preview: {e}")

    print("\nL3 Ranking Calculation Complete.")

if __name__ == '__main__':
    main()