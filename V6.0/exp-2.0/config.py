import os

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 來源檔案路徑
ASSET_POOL_PATH = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
TOXIC_POOL_PATH = os.path.join(RESOURCE_DIR, '2025_final_toxic_asset_pool.json')

# --- 實驗參數 ---
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'  # yfinance 會自動抓取到最新可用的日期
RISK_FREE_RATE = 0.04    # 用於計算夏普比率 (4%)

# --- 實驗分組 ---
# Group A: 優質/成長資產池
# Group B: 有毒/迷因資產池
# Group C: 市場基準 (使用 SPY ETF 作為 S&P 500 的代理)
BENCHMARK_TICKER = ['SPY'] 

# --- 下載設定 ---
REQUEST_DELAY = 1.0  # 避免觸發 Rate Limit 的延遲秒數 (1秒)
