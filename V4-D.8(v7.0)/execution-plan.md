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
  2. **60m 數據：** 遍歷所有標的，使用 yfinance 下載 60m 頻率的 K 線數據（包含 RTH 和 ETH）。
  3. **Daily 數據：** 遍歷所有標的，下載 1d 頻率的 K 線數據，用於計算 G 組上下文特徵 3和 Y 標籤的 vol 單位 (T-1 ATR) 。

* **產出檔案：**  
  * raw\_60m.parquet: 包含所有資產 60 分鐘 K 線的 Panel Data。
    * 索引 (Index): ['symbol', 'timestamp']
    * 欄位 (Columns):    
      * Open (用於計算 K 棒形狀)    
      * High (用於計算 ATR, MFI)    
      * Low (用於計算 ATR, MFI, T 日進場判斷)    
      * Close (用於計算 RSI, Z-Score)    
      * Volume (用於計算 MFI, Vol_Ratio)
  * raw\_daily.parquet: 包含所有資產日 K 線的 Panel Data。
    * 索引 (Index): ['symbol', 'timestamp']
    * 欄位 (Columns):   
      * Open (用於 T+1 開盤價出場 p_exit)
      * High (用於 T-1 Daily ATR)
      * Low (用於 T-1 Daily ATR)
      * Close (用於 T-1 收盤價進場 p, T-1 Daily ATR)
      * Volume (用於 G 組 Amihud 流動性)
      * Adj Close (G 組 Momentum/Z-Score 計算用)
  * ~earnings_dates.parquet: (根據您的決定，此檔案不再需要)~。

---

#### **步驟 2：特徵工程 (Feature Engineering \- X)**

此為 v7.0 的核心。我們將建立規格書 2.1.1 節中定義的 67 個特徵 。

* **目標：** 根據 T-1 日收盤後的可用資訊 ，計算 67 個特徵 (X)。

* **執行：**  
  1. 載入 raw\_60m.parquet 和 raw\_daily.parquet。  
  2. **Base Metrics:** 建立 9 個基礎指標 (RSI, ATR, MFI, Vol\_Ratio 等) 的計算函數。

  3. **方案 C (Full/Partial)：**  
     * 嚴格區分 RTH vs ETH。

     * 嚴格分離 \_Full (完整 60m K 棒) 和 \_Partial (不完整 K 棒) 。

  4. **A/B/C 組 (63 特徵)：**  
     * 計算 A 組：RTH 完整 K 棒匯總 (9 metrics \* 3 agg) \= 27 特徵 。

     * 計算 B 組：ETH 完整 K 棒匯總 (9 metrics \* 3 agg) \= 27 特徵 。

     * 計算 C 組：ETH 不完整 K 棒 (9 metrics \* 1 agg) \= 9 特徵 。

  5. **G 組 (4 特徵)：**  
     * 使用 raw\_daily.parquet 計算 X\_34 (Beta - 126天), X\_35 (Momentum), X\_36 (Z-Score - 126天), X\_37 (Amihud) 。

  6. 將所有 67 個特徵合併為一個以 (asset, T-1 timestamp) 為索引的特徵矩陣。  
* **產出檔案：**  
  * features\_X\_T-1.parquet: 包含 67 個特徵的最終特徵集。

---

#### **步驟 3：標籤工程 (Label Engineering \- Y)**

此步驟建立 v7.0 的「動態風險標準化回報」回歸標籤，並模擬「限價單」進場，這可能導致成交失敗 (NO\_FILL)。

* **目標：** 計算 Y 標籤（回歸值）及成交狀態。
* **執行：**
  1. 載入 `raw_daily.parquet`。
  2. **計算輸入參數：**
     * `p` (進場價格) = **T-1 Close**。
     * `vol` (波動率單位) = T-1 日的 14 天 ATR。
     * `T_Low` (T 日最低價) = **T-day Low** (使用 `.shift(-1)`)。
  3. **計算出場價格：**
     * `p_exit` (出場價格) = **T+1 Open** (使用 `.shift(-2)`)。
  4. **Y 標籤與成交狀態計算：**
     * **成交條件判斷：** `if T_Low <= p:`
       * `Fill_Status = 'FILLED'`
       * `Y = (p_exit - p) / vol`
     * **失敗條件：** `else:`
       * `Fill_Status = 'NO_FILL'`
       * `Y = NaN`
* **產出檔案：**
  * `labels_Y.parquet`: 包含 (asset, T-1 timestamp) 索引及對應的 `Y` 值和 `Fill_Status`。

---

#### **步驟 4：數據清理與合併 (Data Cleaning & Merging)**

根據 3.1 節的規則，建立最終的建模數據集。

* **目標：** 合併 X 和 Y，並應用嚴格的清理規則。  
* **執行：**  
  1. 載入 features\_X\_T-1.parquet 和 labels\_Y.parquet。  
  2. 載入 earnings\_dates.parquet。  
  3. **合併：** 將 X 和 Y 矩陣對齊 (asset, timestamp)。  
  4. ~**規則一 (財報)：** 刪除 T 日或 T+1 日區間內有財報的樣本。~

  5. **規則二 (未成交)：** 嚴格刪除所有 Fill\_Status 為 NO\_FILL (即 Y=0 或 Null) 的樣本。

* **產出檔案：**  
  * model\_ready\_dataset.parquet: 最終、乾淨的、可用於訓練的面板數據。

---

#### **步驟 5-1：模型訓練 (方案 A：兩階段 Ridge 模型)**

* **目標：** 使用兩階段的 Ridge 回歸，同時預測目標值 (Y) 和預測誤差（作為不確定性代理）。
* **依賴 (Dependencies)：**
  * `model_ready_dataset.parquet`: 清理過的、用於模型訓練的數據集。
* **執行：**
  1. 載入 `model_ready_dataset.parquet`。
  2. 設計滾動驗證循環 (Walk-forward Validation)。
  3. 在每個 Fold 中：
     * 標準化特徵 (Standard Scaler)。
     * **第一階段：** 訓練 Ridge 模型 (`model_A_Y`) 來預測 `Y`，得到 `Y_pred_A`。
     * **計算誤差：** 計算 `Y_error = |Y_true - Y_pred_A|`。
     * **第二階段：** 訓練另一個 Ridge 模型 (`model_A_Uncertainty`) 來預測 `Y_error`，得到 `Y_uncertainty_A`。
     * 儲存該 Fold 的預測結果。
* **產出檔案 (Output Files)：**
  * `predictions_oos_A.csv`: 包含 `Y_true`, `Y_pred_A`, `Y_uncertainty_A` 的帶外預測結果。
  * `models/scheme_A/`: 儲存每個 Fold 訓練好的 `model_A_Y` 和 `model_A_Uncertainty` 模型物件。

---

#### **步驟 5-2：模型訓練 (方案 B：分位數回歸)**

* **目標：** 使用 LightGBM 的分位數回歸功能，直接預測目標值的特定分位數（例如 10%、50%、90%），以構建預測區間。
* **依賴 (Dependencies)：**
  * `model_ready_dataset.parquet`: 清理過的數據集。
* **執行：**
  1. 載入 `model_ready_dataset.parquet`。
  2. 設計滾動驗證循環。
  3. 在每個 Fold 中：
     * 標準化特徵。
     * **模型訓練：** 分別訓練三個 LGBMRegressor 模型：
       * `model_B_Lower`: 設定 `objective='quantile'` 和 `alpha=0.1`。
       * `model_B_Median`: 設定 `objective='quantile'` 和 `alpha=0.5` (作為點預測)。
       * `model_B_Upper`: 設定 `objective='quantile'` 和 `alpha=0.9`。
     * **儲存預測：** 儲存三個模型對應的預測結果。
* **產出檔案 (Output Files)：**
  * `predictions_oos_B.csv`: 包含 `Y_true`, `Y_pred_B_Lower`, `Y_pred_B_Median`, `Y_pred_B_Upper` 的預測結果。
  * `models/scheme_B/`: 儲存每個 Fold 訓練好的三個分位數模型物件。

---

#### **步驟 5-3：模型訓練 (方案 C：貝氏回歸)**

* **目標：** 使用貝氏回歸模型，在預測時同時產出預測值的平均值 (mean) 和標準差 (standard deviation)，後者可作為不確定性的度量。
* **依賴 (Dependencies)：**
  * `model_ready_dataset.parquet`: 清理過的數據集。
* **執行：**
  1. 載入 `model_ready_dataset.parquet`。
  2. 設計滾動驗證循環。
  3. 在每個 Fold 中：
     * 標準化特徵。
     * **模型訓練：** 訓練一個 `BayesianRidge` 模型 (`model_C_Bayesian`)。
     * **預測與不確定性：** 使用 `.predict()` 方法時設定 `return_std=True`，直接得到預測平均值 `Y_pred_C` 和標準差 `Y_std_C`。
     * **儲存預測：** 儲存預測結果。
* **產出檔案 (Output Files)：**
  * `predictions_oos_C.csv`: 包含 `Y_true`, `Y_pred_C`, `Y_std_C` 的預測結果。
  * `models/scheme_C/`: 儲存每個 Fold 訓練好的貝氏回歸模型物件。

---

#### **步驟 6-1：預測結果合併**

* **目標：** 將來自三個不同模型方案的帶外預測結果合併成一個統一的檔案，以供後續的綜合分析使用。
* **依賴 (Dependencies)：**
  * `predictions_oos_A.csv`: 方案 A 的預測結果。
  * `predictions_oos_B.csv`: 方案 B 的預測結果。
  * `predictions_oos_C.csv`: 方案 C 的預測結果。
* **執行：**
  1. 分別載入 `predictions_oos_A.csv`, `predictions_oos_B.csv`, `predictions_oos_C.csv`。
  2. 確保三個檔案的索引 (asset, timestamp) 對齊。
  3. 將三個檔案的欄位合併成一個寬表格 (wide-format DataFrame)。
* **產出檔案 (Output Files)：**
  * `predictions_oos_merged.csv`: 包含所有方案預測結果的單一檔案。欄位應包含：
    * `Y_true`
    * `Y_pred_A`, `Y_uncertainty_A`
    * `Y_pred_B_Lower`, `Y_pred_B_Median`, `Y_pred_B_Upper`
    * `Y_pred_C`, `Y_std_C`

---

#### **步驟 6-2：回測分析與模型比較**

* **目標：** 基於合併後的預測結果，全面評估三個方案的預測準確性和不確定性校準度，並進行模擬交易回測。
* **依賴 (Dependencies)：**
  * `predictions_oos_merged.csv`: 包含所有模型預測結果的合併檔案。
* **執行：**
  1. 載入 `predictions_oos_merged.csv`。
  2. **比較 (1) - 預測準確性：**
     * 比較 `Y_pred_A`, `Y_pred_B_Median`, `Y_pred_C` 的點預測準確度 (RMSE, MAE, Spearman)。
  3. **比較 (2) - 不確定性校準：**
     * 分析各方案的不確定性指標 (`Y_uncertainty_A`, `Y_pred_B_Upper - Y_pred_B_Lower`, `Y_std_C`) 與實際預測誤差 `|Y_true - Y_pred|` 的相關性。
  4. **模擬交易：**
     * 選擇綜合表現最佳的模型。
     * 設計交易策略（例如，結合點預測和不確定性進行過濾）。
     * 計算夏普比率、累積回報、最大回撤等績效指標。
* **產出檔案 (Output Files)：**
  * `backtest_report.txt`: 包含模型比較結果和回測績效指標的文字報告。
  * `cumulative_return.png`: 策略累積回報圖。
  * `drawdown.png`: 策略回撤圖。
  * `uncertainty_calibration.png`: 不確定性校準度比較圖。
