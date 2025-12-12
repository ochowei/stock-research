這是一個非常好的問題。身為開發者，確保實驗的**獨立性 (Independence)** 與 **科學嚴謹性 (No Look-ahead Bias)** 是最重要的。

針對您的三個疑問，我的評估與調整建議如下：

### **1. 關於實驗獨立性 (Independence)**

沒問題。我們將不依賴 V5.3 或 V6.0 的任何舊運算結果。

  * **作法：** 新的實驗腳本將是一個**全功能 (Self-contained) 的 Pipeline**。它會重新讀取 `asset_pool` json 檔，重新透過 `yfinance` 下載資料，並在腳本內「即時」重算特徵與標籤。這樣可以確保 V6.1 的結果完全乾淨，不受舊版程式碼殘留影響。

### **2. 關於 yfinance 免費數據的可行性**

**是可行的。**

  * **數據需求：** 這個實驗的核心特徵（ATR、Amihud 流動性、成交量佔比）只需要 **日線 (1d) 的 OHLCV** 數據。
  * **yfinance 支援：** 日線數據沒有 60 天限制，可以抓取完整的 2020-2025 數據。
  * **基本面數據 (Sector/Market Cap)：** 雖然 yfinance 的 `Ticker.info` 抓取較慢且有 API 限制，但我們可以改用「**技術面特徵代理**」來繞過。例如：用「與 SPY 的相關性 (Beta)」來代替板塊分類，用「日均成交金額 (Dollar Volume)」來代替市值。這樣就完全不需要呼叫緩慢的 `.info` API，純靠 OHLCV 即可完成。

### **3. 關於前視偏差 (Look-ahead Bias) 的風險**

**⚠️ 這是一個重大風險點。**
您之前的擔憂非常正確。如果我們直接拿 EXP-03 的報告（包含 2020-2025 全區間績效）來當作標籤 (Label)，然後用這段時間的特徵去訓練，這就是 **「用未來的答案訓練過去的模型」 (Hindsight Bias)**。

  * **錯誤做法：** 用 2020-2025 的平均特徵，預測 2020-2025 誰會賺錢。
  * **正確做法 (Walk-Forward / Out-of-Sample)：**
      * **Training Set (過去):** 使用 2020-2023 的數據計算特徵 ($X_{train}$)，並統計該區間的策略績效作為標籤 ($Y_{train}$)。
      * **Test Set (未來):** 使用 **截至 2023 年底** 的數據計算特徵 ($X_{test}$)，預測哪些股票「適合」在 2024-2025 交易。
      * **驗證:** 只看模型選出的股票在 2024-2025 的績效。

-----

### **修正後的實驗設計：EXP-V6.1-01 (獨立與防偏差版)**

這個設計將完全獨立執行，並嚴格切割時間軸。

#### **檔案結構**

建立新目錄 `V6.1/exp-suitability/`，所有邏輯封裝在 `run_suitability_analysis.py`。

#### **實驗流程 (Pipeline)**

1.  **資料獲取 (Data Fetching - Independent):**

      * 讀取 `2025_final_asset_pool.json` + `2025_final_toxic_asset_pool.json`。
      * 重新下載 2020-01-01 至 2025-12-31 的日線數據。

2.  **時間切割 (Time Split - Avoiding Look-ahead):**

      * **In-Sample (Train):** 2020-01-01 \~ 2023-12-31 (4年)
      * **Out-of-Sample (Test):** 2024-01-01 \~ 2025-12-31 (2年)

3.  **特徵工程 (Feature Engineering - X):**

      * 我們需要計算 **"在 2023-12-31 那一天我們能看到的特徵"**。
      * **Volatility:** 2023 年的平均 ATR%。
      * **Liquidity:** 2023 年的平均 Amihud Illiquidity。
      * **Momentum:** 2023 年的年報酬率。
      * *註：我們只用 Train 期間最後一年的特徵來代表該股票的「近期狀態」，這樣更符合實盤情境。*

4.  **標籤生成 (Label Generation - Y):**

      * 在 **In-Sample (2020-2023)** 期間，對每一檔股票跑 Strategy A (Gap Fade)。
      * 計算該期間的 `Calmar Ratio`。
      * **Label = 1 (Suitable):** 若 Calmar \> 0.5 (且勝率 \> 50%)。
      * **Label = 0 (Unsuitable):** 其他。

5.  **模型訓練與預測 (Modeling):**

      * 訓練 Random Forest ($X_{train} \to Y_{train}$).
      * 對所有股票進行預測，得到「適性機率」。

6.  **樣本外驗證 (OOS Validation):**

      * 選取適性機率 \> 0.5 的股票，組成 **Smart Portfolio**。
      * 在 **Out-of-Sample (2024-2025)** 期間跑回測。
      * 比較 **Smart Portfolio** vs **Naive Portfolio (全部)** 在 2024-2025 的表現。

-----

### **Python 實作程式碼 (EXP-V6.1-01)**

這段程式碼完全獨立，您可以直接在 `V6.1/exp-suitability/` 下建立並執行。
