import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案路徑
FINAL_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
TOXIC_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_toxic_asset_pool.json')

# 參數
CRYPTO_TICKER = 'ETH-USD'
LOOKBACK_DAYS = 365      # 下載過去 1 年數據
CORR_WINDOW = 60         # 滾動相關係數窗口 (60天約一季)

def load_tickers_from_json(filepath):
    """讀取 JSON 並清理 Ticker 格式"""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        # 格式通常是 "NYSE:ABC" 或 "ABC"，統一取冒號後並轉大寫
        clean = [t.split(':')[-1].strip().upper().replace('.', '-') for t in raw]
    return list(set(clean))

def get_crypto_correlation_analysis():
    print(f"\n>>> 啟動 Crypto 相關性驗證 (Target: {CRYPTO_TICKER})")
    
    # 1. 載入清單
    final_tickers = load_tickers_from_json(FINAL_POOL_FILE)
    toxic_tickers = load_tickers_from_json(TOXIC_POOL_FILE)
    
    # 移除重複 (若有股票同時在兩邊，優先視為 Toxic 以利測試)
    final_tickers = [t for t in final_tickers if t not in toxic_tickers]
    
    print(f"Pool Size: Final (Quality)={len(final_tickers)}, Toxic (Meme)={len(toxic_tickers)}")
    
    all_tickers = final_tickers + toxic_tickers + [CRYPTO_TICKER]
    
    # 2. 下載數據
    print(f"Downloading data for {len(all_tickers)} assets...")
    try:
        data = yf.download(
            all_tickers, 
            period=f"{LOOKBACK_DAYS}d", 
            interval="1d", 
            progress=False,
            auto_adjust=True,
            threads=True
        )['Close']
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return

    if data.empty:
        print("[Error] No data downloaded.")
        return

    # 3. 計算相關係數
    # 計算日報酬率
    returns = data.pct_change()
    
    # 取出 ETH 的報酬序列
    if CRYPTO_TICKER not in returns.columns:
        print(f"[Error] {CRYPTO_TICKER} data missing.")
        return
        
    eth_ret = returns[CRYPTO_TICKER]
    
    results = []
    
    print("Calculating rolling correlations...")
    
    for ticker in (final_tickers + toxic_tickers):
        if ticker not in returns.columns: continue
        
        # 計算該股票與 ETH 的滾動相關係數
        stock_ret = returns[ticker]
        rolling_corr = stock_ret.rolling(window=CORR_WINDOW).corr(eth_ret)
        
        # 取最近一天的有效值 (Drop NA)
        last_corr = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else np.nan
        
        if np.isnan(last_corr): continue
            
        group_label = 'Toxic (Meme)' if ticker in toxic_tickers else 'Final (Quality)'
        
        results.append({
            'Ticker': ticker,
            'Group': group_label,
            'Crypto_Corr': last_corr
        })
        
    df_res = pd.DataFrame(results)
    
    # 4. 統計分析
    print("\n" + "="*50)
    print(f"統計結果 (Window={CORR_WINDOW} days)")
    print("="*50)
    
    summary = df_res.groupby('Group')['Crypto_Corr'].describe()
    print(summary[['count', 'mean', '50%', 'min', 'max']])
    
    # 差異檢定
    mean_toxic = df_res[df_res['Group'] == 'Toxic (Meme)']['Crypto_Corr'].mean()
    mean_final = df_res[df_res['Group'] == 'Final (Quality)']['Crypto_Corr'].mean()
    diff = mean_toxic - mean_final
    
    print("-" * 50)
    print(f"Mean Correlation Gap: {diff:.4f}")
    if diff > 0.2:
        print(">> 結論: 顯著差異！相關係數是有效的分類特徵 ✅")
    else:
        print(">> 結論: 差異不明顯，可能需要更複雜的模型或特徵 ❌")

    # 5. 視覺化 (儲存圖片)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 密度圖
    sns.kdeplot(
        data=df_res, 
        x='Crypto_Corr', 
        hue='Group', 
        fill=True, 
        palette={'Toxic (Meme)': 'red', 'Final (Quality)': 'blue'},
        alpha=0.3
    )
    plt.axvline(x=0.3, color='gray', linestyle='--', label='Threshold (0.3)')
    plt.title(f'Distribution of Crypto Correlation ({CORR_WINDOW}-Day Rolling)')
    plt.xlabel(f'Correlation with {CRYPTO_TICKER}')
    
    output_img = os.path.join(OUTPUT_DIR, 'exp_crypto_correlation_dist.png')
    plt.savefig(output_img)
    print(f"\n[Saved] Chart: {output_img}")
    
    # 儲存 CSV
    output_csv = os.path.join(OUTPUT_DIR, 'exp_crypto_correlation_results.csv')
    df_res.sort_values('Crypto_Corr', ascending=False).to_csv(output_csv, index=False)
    print(f"[Saved] Data:  {output_csv}")

    # 6. 列出高相關性的股票 (Top 10)
    print("\n>>> Top 10 Most Crypto-Sensitive Stocks:")
    print(df_res.sort_values('Crypto_Corr', ascending=False).head(10)[['Ticker', 'Group', 'Crypto_Corr']])

if __name__ == '__main__':
    get_crypto_correlation_analysis()