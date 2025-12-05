import json
import os

class DataLoader:
    """
    V5.3 資料載入器 (Data Loader) - Refactored for Dual-Track
    負責管理與讀取資產清單，支援動態指定檔案來源。
    """
    def __init__(self, base_dir=None, normal_file='asset_pool.json', toxic_file='toxic_asset_pool.json'):
        # 若未指定目錄，預設為此腳本所在目錄 (ml_pipeline)
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
            
        # 動態定義檔案路徑
        self.normal_pool_path = os.path.join(self.base_dir, normal_file)
        self.toxic_pool_path = os.path.join(self.base_dir, toxic_file)

    def get_normal_tickers(self):
        """讀取標準自選股清單"""
        print(f"[DataLoader] Loading Normal Pool from {os.path.basename(self.normal_pool_path)}...")
        return self._load_and_clean(self.normal_pool_path)

    def get_toxic_tickers(self):
        """讀取毒性壓力測試清單"""
        print(f"[DataLoader] Loading Toxic Pool from {os.path.basename(self.toxic_pool_path)}...")
        return self._load_and_clean(self.toxic_pool_path)
    
    def get_all_tickers(self):
        """讀取並合併所有清單 (用於一次性下載所有數據)"""
        normal = self.get_normal_tickers()
        toxic = self.get_toxic_tickers()
        combined = sorted(list(set(normal + toxic)))
        print(f"[DataLoader] Loaded Combined Pool: {len(combined)} tickers.")
        return combined

    def _load_and_clean(self, path):
        """內部函數：讀取 JSON 並清理 ticker 格式"""
        if not os.path.exists(path):
            print(f"Warning: Pool file not found at {path}")
            return []
            
        with open(path, 'r') as f:
            raw_list = json.load(f)
            
        # 統一清洗邏輯：
        # 1. 去除交易所前綴 (NYSE:MP -> MP)
        # 2. 將 '.' 轉為 '-' (BRK.B -> BRK-B) 以符合 yfinance 格式
        cleaned_list = []
        for t in raw_list:
            ticker = t.split(':')[-1].replace('.', '-')
            cleaned_list.append(ticker)
            
        return cleaned_list

# 簡單測試用
if __name__ == "__main__":
    loader = DataLoader()
    print("Normal:", loader.get_normal_tickers()[:5])
    print("Toxic:", loader.get_toxic_tickers()[:5])