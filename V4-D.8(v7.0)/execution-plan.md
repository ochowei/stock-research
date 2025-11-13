### **📈 專案 V4-D.8 (v7.0) 執行計畫 (修訂版 - 忽略財報過濾)**

#### **步驟 0：專案初始化 (Setup)**

此步驟負責建立專案環境並獲取最新的資產池。

* **目標：** 從 Gist 獲取標的列表，為數據下載做準備。  
* **執行：**  
  1. 撰寫一個腳本 (e.g., 00\_setup.py)。  
  2. 從 Gist URL (https://gist.github.com/ochowei/9b24d188882ab92e2cc5a336e9675f17) 下載資產列表。  
  3. 解析列表（看起來像 JSON 格式）並將其保存為本地的資產池文件，供後續步驟使用。  
* **產出檔案：**  
  * asset\_pool.json: 本地保存的標的列表。

---

#### **步驟 1：原始數據獲取 (Data Acquisition)**

根據 v7.0 規格，我們需要 60m 數據（特徵）和 Daily 數據（標籤與部分特徵）。

* **目標：** 下載 asset\_pool.json 中所有資產的 yfinance 原始數據。  
* **執行：**  
  1. 讀取 asset\_pool.json。  
  2. **60m 數據：** 遍歷所有標的，使用 yfinance 下載 60m 頻率的 K 線數據（包含 RTH 和 ETH）。2222

  3. **Daily 數據：** 遍歷所有標的，下載 1d 頻率的 K 線數據，用於計算 G 組上下文特徵 3和 Y 標籤的 vol 單位 (T-1 ATR) 4。

* **產出檔案：**  
  * raw\_60m.parquet: 包含所有資產 60 分鐘 K 線的 Panel Data。  
  * raw\_daily.parquet: 包含所有資產日 K 線的 Panel Data。  
  * ~earnings_dates.parquet: (根據您的決定，此檔案不再需要)~。

---

#### **步驟 2：特徵工程 (Feature Engineering \- X)**

此為 v7.0 的核心。我們將建立規格書 2.1.1 節中定義的 67 個特徵 6。

* **目標：** 根據 T-1 日收盤後的可用資訊 7，計算 67 個特徵 (X)。

* **執行：**  
  1. 載入 raw\_60m.parquet 和 raw\_daily.parquet。  
  2. **Base Metrics:** 建立 9 個基礎指標 (RSI, ATR, MFI, Vol\_Ratio 等) 的計算函數 8。

  3. **方案 C (Full/Partial)：**  
     * 嚴格區分 RTH vs ETH 9。

     * 嚴格分離 \_Full (完整 60m K 棒) 和 \_Partial (不完整 K 棒) 10。

  4. **A/B/C 組 (63 特徵)：**  
     * 計算 A 組：RTH 完整 K 棒匯總 (9 metrics \* 3 agg) \= 27 特徵 11。

     * 計算 B 組：ETH 完整 K 棒匯總 (9 metrics \* 3 agg) \= 27 特徵 12。

     * 計算 C 組：ETH 不完整 K 棒 (9 metrics \* 1 agg) \= 9 特徵 13。

  5. **G 組 (4 特徵)：**  
     * 使用 raw\_daily.parquet 計算 X\_34 (Beta), X\_35 (Momentum), X\_36 (Z-Score), X\_37 (Amihud) 14141414。

  6. 將所有 67 個特徵合併為一個以 (asset, T-1 timestamp) 為索引的特徵矩陣。  
* **產出檔案：**  
  * features\_X\_T-1.parquet: 包含 67 個特徵的最終特徵集。

---

#### **步驟 3：標籤工程 (Label Engineering \- Y)**

此步驟建立 v7.0 的「動態風險標準化回報」回歸標籤 15。

* **目標：** 計算 Y 標籤（回歸值）。  
* **執行：**  
  1. 載入 raw\_60m.parquet 和 raw\_daily.parquet。  
  2. **計算輸入參數 (T-1 日)：**  
     * p (進場價格) \= T-1 Close 16。

     * vol (波動率單位) \= T-1 日的 14 天 ATR (來自 Daily 數據) 17。

  3. **計算結算價格 (T+1 日)：**  
     * p\_exit (結算價格) \= T+1 日的開盤價 18。

  4. **情境判斷：**  
     * **情境 B (未成交)：** 檢查 T 日的 60m Low 是否曾觸及 p 19。若無，標記為 NO\_FILL。

     * **情境 A (成交)：** 若 T 日 60m Low 觸及 p 20，則計算標籤。

  5. **Y 標籤計算 (情境 A)：**  
     * Y \= (p\_exit \- p) / vol 21。

  6. **Y 標籤處理 (情境 B)：**  
     * Y \= 0 (或 Null) 22。

* **產出檔案：**  
  * labels\_Y.parquet: 包含 (asset, T-1 timestamp) 索引及對應的 Y 值和 Fill\_Status (e.g., 'FILLED', 'NO\_FILL')。

---

#### **步驟 4：數據清理與合併 (Data Cleaning & Merging)**

根據 3.1 節的規則 23，建立最終的建模數據集。

* **目標：** 合併 X 和 Y，並應用嚴格的清理規則。  
* **執行：**  
  1. 載入 features\_X\_T-1.parquet 和 labels\_Y.parquet。  
  2. 載入 earnings\_dates.parquet。  
  3. **合併：** 將 X 和 Y 矩陣對齊 (asset, timestamp)。  
  4. ~**規則一 (財報)：** 刪除 T 日或 T+1 日區間內有財報的樣本 24。~

  5. **規則二 (未成交)：** 嚴格刪除所有 Fill\_Status 為 NO\_FILL (即 Y=0 或 Null) 的樣本 25。

* **產出檔案：**  
  * model\_ready\_dataset.parquet: 最終、乾淨的、可用於訓練的面板數據。

---

#### **步驟 5：模型訓練與滾動驗證 (Training & Walk-Forward Validation)**

此步驟是 3.3 節和 3.4 節的核心實作 26262626。

* **目標：** 執行「滾動窗口驗證」並遵守「特徵標準化 SOP」。  
* **執行：**  
  1. 載入 model\_ready\_dataset.parquet。  
  2. **定義滾動窗口 (k)：**  
     * 設定 Train\_k 和 Test\_k 的時間範圍 27。

  3. **在窗口 k 中循環：**  
     * **擬合標量 (Fit Scaler)：** \* scaler\_k \= StandardScaler()  
       * scaler\_k.fit(Train\_k\[X\_features\]) (僅在 Train\_k 上 fit！) 28

     * **轉換數據 (Transform Data)：**  
       * X\_train\_scaled \= scaler\_k.transform(Train\_k\[X\_features\]) 29

       * X\_test\_scaled \= scaler\_k.transform(Test\_k\[X\_features\]) (使用 scaler\_k 轉換 Test\_k) 30

     * **訓練與預測 (Train & Predict)：**  
       * model\_k.fit(X\_train\_scaled, Train\_k\[Y\]) (訓練通用模型) 31

       * predictions\_k \= model\_k.predict(X\_test\_scaled) 32

     * 儲存 predictions\_k（包含真實 Y 值和預測 Y 值）和 model\_k。  
  4. **滾動 (Roll Forward)：** 移動窗口，重複步驟 3。  
* **產出檔案：**  
  * predictions\_oos.csv: 合併所有 predictions\_k 的帶外 (Out-of-Sample) 預測結果。  
  * models/model\_fold\_{k}.joblib: 每個滾動窗口訓練出的模型檔案。

---

#### **步驟 6：回測分析 (Backtest Analysis)**

* **目標：** 分析 predictions\_oos.csv 的預測表現。  
* **執行：**  
  1. 載入 predictions\_oos.csv。  
  2. 由於 Y 是「風險標準化回報」，您需要分析預測值 (Predicted Y) 與真實 Y 之間的相關性（例如 Spearman 相關性）。  
  3. **模擬交易：** 根據預測的 Y 值（例如 Predicted Y \> 0.5）決定是否執行 v7.0 假設的交易（T-1 收盤價進場 33，T+1 開盤價出場 34）。

  4. 計算投資組合的夏普比率 (Sharpe Ratio)、累積回報等績效指標。  
* **產出檔案：**  
  * backtest\_report.ipynb (或 .html): 包含績效圖表和分析結果的報告。
