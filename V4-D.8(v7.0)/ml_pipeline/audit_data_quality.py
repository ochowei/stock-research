# 檔名: ml_pipeline/audit_data_quality.py
import pandas as pd
import pytz

def audit_volume():
    print("正在讀取 raw_60m.parquet...")
    try:
        df = pd.read_parquet('raw_60m.parquet')
    except FileNotFoundError:
        print("找不到 raw_60m.parquet，請確認檔案位置。")
        return

    # 確保索引正確 (假設是 MultiIndex: symbol, timestamp 或 timestamp, symbol)
    # 這裡根據您的 schema 調整，假設 reset_index 後 timestamp 是欄位
    df = df.reset_index()
    
    # 轉換時間戳記為美東時間 (US/Eastern) 以便正確過濾 RTH
    # 假設原始 timestamp 是 UTC
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') if df['timestamp'].dtype == 'int64' else pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
             # 假設是 UTC
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # 轉為美東時間
        df['timestamp_et'] = df['timestamp'].dt.tz_convert('US/Eastern')
    else:
        print("錯誤：找不到 timestamp 欄位")
        return

    # 定義 RTH (正常交易時段): 09:30 - 16:00
    # 過濾出 RTH 的資料
    rth_mask = (df['timestamp_et'].dt.time >= pd.to_datetime('09:30').time()) & \
               (df['timestamp_et'].dt.time < pd.to_datetime('16:00').time())
    
    df_rth = df[rth_mask].copy()

    print(f"RTH 總樣本數: {len(df_rth)}")

    # 檢查 RTH 時段的異常零成交量
    zero_vol_rth = df_rth[df_rth['Volume'] <= 0]
    
    print(f"RTH 時段零成交量樣本數: {len(zero_vol_rth)}")
    
    if len(zero_vol_rth) > 0:
        print("\n=== RTH 零成交量異常排行榜 (Top 20) ===")
        # 統計每個 Symbol 在 RTH 時段有多少根 K 棒是 0 成交量
        anomaly_counts = zero_vol_rth['symbol'].value_counts().head(20)
        
        for symbol, count in anomaly_counts.items():
            total_bars = len(df_rth[df_rth['symbol'] == symbol])
            ratio = (count / total_bars) * 100 if total_bars > 0 else 0
            print(f"Symbol: {symbol:<10} | 異常 K 棒數: {count:<5} | 異常比例: {ratio:.2f}%")
            
        print("\n=== 異常樣本範例 (時間為美東時間) ===")
        print(zero_vol_rth[['symbol', 'timestamp_et', 'Open', 'Close', 'Volume']].head(10))
    else:
        print("\n恭喜！在 RTH 時段沒有發現零成交量的異常數據。")

if __name__ == "__main__":
    audit_volume()