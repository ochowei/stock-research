import json
import os

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')

ASSET_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
# [NEW] 建立新的敏感池檔案
SENSITIVE_POOL_FILE = os.path.join(RESOURCE_DIR, '2025_final_crypto_sensitive_pool.json')

# 定義要遷移的高連動名單 (來自 EXP-04 Control Detailed)
MIGRATION_LIST = [
    'SOUN', 'CIFR', 'ONDS', 'POWI', 
    'HIMS', 'TMDX', 'RXRX', 'BROS', 
    'TSLA', 'NVTS', 'TTD', 'SOFI'
]

def migrate_to_new_list():
    # 1. 讀取 Asset Pool
    with open(ASSET_POOL_FILE, 'r') as f:
        asset_pool = json.load(f)
    
    # 2. 準備新的清單
    sensitive_pool = []
    
    # 如果檔案已存在，先讀取舊內容以免覆蓋
    if os.path.exists(SENSITIVE_POOL_FILE):
        with open(SENSITIVE_POOL_FILE, 'r') as f:
            sensitive_pool = json.load(f)

    new_asset_pool = []
    transferred_count = 0
    
    # 3. 執行篩選與遷移
    for item in asset_pool:
        ticker_name = item.split(':')[-1].strip()
        
        if ticker_name in MIGRATION_LIST:
            # 移入敏感池
            if item not in sensitive_pool:
                sensitive_pool.append(item)
                transferred_count += 1
                print(f"[Move] {ticker_name} -> Crypto Sensitive Pool")
        else:
            # 保留在標準池
            new_asset_pool.append(item)

    # 4. 存檔
    if transferred_count > 0:
        new_asset_pool.sort()
        sensitive_pool.sort()
        
        with open(ASSET_POOL_FILE, 'w') as f:
            json.dump(new_asset_pool, f, indent=2)
            
        with open(SENSITIVE_POOL_FILE, 'w') as f:
            json.dump(sensitive_pool, f, indent=2)
            
        print(f"\n成功遷移 {transferred_count} 檔股票至新清單！")
        print(f"Asset Pool 剩餘: {len(new_asset_pool)}")
        print(f"Sensitive Pool 目前有: {len(sensitive_pool)}")
        print(f"新檔案位置: {SENSITIVE_POOL_FILE}")
    else:
        print("\n沒有發現需要遷移的股票 (可能已經移動過了)。")

if __name__ == "__main__":
    migrate_to_new_list()