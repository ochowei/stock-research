# **📈 專案 V5.1-ML 執行計畫：正交資訊與動態出場實作**

Date: 2025-12-01 21:00 PM

Topics: \#quant-trading \#system-design \#workflow \#backtesting \#data-analysis

Status: \#draft \#execution-plan

本計畫旨在將 V5.1 的研究構想轉化為具體的程式碼實作，重點解決 L3 模型特徵重疊問題，並引入 L4 動態出場機制以提升資本效率。

## **步驟 1：數據工程與正交特徵構建 (Data & Orthogonal Features)**

此步驟負責擴充數據源，並計算與價格技術面低相關的「正交特徵」，為 L3 模型提供新的資訊維度。

* **目標：** 獲取板塊數據，計算板塊相對強弱與微結構特徵。  
* **執行細節：**  
  1. **擴充下載清單 (00\_download\_data\_v5.py)：**  
     * 新增 Sector ETFs 下載列表：XLK (科技), XLF (金融), XLV (醫療), XLY (非必需消費), XLP (必需消費), XLE (能源), XLI (工業), XLB (原物料), XLU (公用事業)。  
     * 確保板塊數據與個股數據的時間區間對齊。  
  2. **實作板塊特徵 (02\_build\_features\_l0\_v5.py)：**  
     * **映射邏輯：** 建立個股與 ETF 的對照表 (Map Stock to Sector ETF)。  
     * **Sector RSI:** 計算該板塊的 RSI(14)。  
     * **Rel\_Strength:** 計算個股相對於板塊的 RS (Relative Strength) 分數。  
     * **Divergence:** 計算 Stock\_RSI \- Sector\_RSI (尋找「錯殺」訊號：個股崩但板塊穩)。  
  3. **實作微結構特徵 (02\_build\_features\_l0\_v5.py)：**  
     * **Volume Structure:** 計算下跌段的量能特徵 (e.g., Down\_Vol / Avg\_Vol)，區分無量下跌與爆量下跌。  
* **產出檔案：**  
  * data/temp\_raw/raw\_sector\_data.pkl: 原始板塊數據。  
  * features/sector\_features\_L0.parquet: 板塊與正交特徵集。

## **步驟 2：L3 模型重構 \- Learning to Rank (L3 Upgrade)**

此步驟將 L3 從「二元分類器」重構為「排序器」，並嚴格執行特徵篩選以避免資訊重疊。

* **目標：** 訓練 LGBMRanker，在 RSI \< 10 的候選名單中，排序出預期反彈幅度最大的標的。  
* **執行細節：**  
  1. **特徵篩選 (04\_train\_meta\_labeling\_l3.py)：**  
     * **移除 (Blacklist):** 所有 L2 用過的特徵 (RSI\_2, RSI\_14, Dist\_SMA\_200, BB\_PctB)。  
     * **保留 (Whitelist):** L1 Context (HMM\_State, Anomaly\_Score), L0 Orthogonal (Sector Features, Volatility Structure, ATR\_Norm)。  
  2. **標籤重定義 (Re-labeling):**  
     * 從 Class {0, 1} 改為 Rank Score。  
     * Target \= Future\_5D\_Return (未來 5 日漲幅)。  
     * 使用 Group 參數 (每日為一個 Query Group) 進行訓練。  
  3. **模型訓練:**  
     * 切換模型為 lightgbm.LGBMRanker。  
     * Objective: lambdarank。  
     * Metric: ndcg (Normalized Discounted Cumulative Gain)。  
* **產出檔案：**  
  * models/l3\_ranker.joblib: 訓練好的排序模型。  
  * signals/l3\_rank\_scores.csv: 每日候選名單的排序分數。

## **步驟 3：L4 動態出場機制實作 (L4 Dynamic Exit)**

此步驟實作基於波動率的動態止盈機制，取代固定的持有天數。

* **目標：** 提升資金周轉率，在獲利達標時提前釋放資金。  
* **執行細節 (05\_backtest\_and\_verify.py 模擬邏輯)：**  
  1. **實作波動率止盈 (Volatility-Adjusted TP):**  
     * 設定基礎邏輯：Target\_Price \= Entry\_Price \+ (K \* ATR\_Entry)。  
     * 參數 $K$ 設定：  
       * 初始測試：$K \= 2.0$ (捕捉 2 倍 ATR 的反彈)。  
       * 進階 (Optional)：根據 L1 狀態調整 (Bull: $K=3$, Chop: $K=1.5$)。  
  2. **出場優先級邏輯：**  
     * If High \> Target\_Price: 以 Target\_Price 出場 (止盈)。  
     * Else If Days\_Held \>= 5: 以 Close 出場 (時間止損)。  
* **產出檔案：**  
  * (整合於回測報告中)

## **步驟 4：整合回測與壓力測試 (Integration & Verification)**

此步驟將 L1 防禦、L2 篩選、L3 排序、L4 出場串聯，並加入嚴格的滑價模擬。

* **目標：** 驗證 V5.1 完整體系的績效，確認是否超越 V5 (Sharpe \> 2.6)。  
* **執行細節 (05\_backtest\_and\_verify.py)：**  
  1. **全流程模擬：**  
     * **Filter 1 (L1):** 剔除 Crash State / Anomaly 日期。  
     * **Filter 2 (L2):** 篩選 RSI \< 10 & Price \> SMA200。  
     * **Selection (L3):** 根據 Rank Score 選取 Top N (e.g., Top 5)。  
     * **Execution (L4):** 模擬動態出場路徑。  
  2. **滑價壓力測試 (Slippage Stress Test):**  
     * 在進場與出場價格上，分別加上 Slippage Penalty。  
     * 測試級距：0.00% (Baseline), 0.05% (Normal), 0.10% (Stress)。  
  3. **L1 滾動訓練模擬 (Rolling Window):**  
     * (若資源允許) 模擬每半年重新訓練一次 HMM 模型，觀察績效穩定性。  
* **驗證項目 (No UI Testing):**  
  * **Sharpe Ratio:** 目標 \> 3.0。  
  * **Max Drawdown:** 目標 \< 10%。  
  * **Avg Holding Period:** 觀察 L4 是否有效降低持倉天數。  
* **產出檔案：**  
  * analysis/v5.1\_backtest\_report.txt: 含滑價測試的詳細數據。  
  * analysis/v5.1\_equity\_curve.png: V5.1 vs V5 資金曲線比較。

## **檔案清單 (Files Summary)**

本計畫的檔案異動狀態如下：

### **需修改或新增 (Modified/New)**

1. **ml\_pipeline/00\_download\_data\_v5.py** (Modified): 新增板塊 ETF 下載。  
2. **ml\_pipeline/02\_build\_features\_l0\_v5.py** (Modified): 新增板塊 RSI 與相對強弱特徵計算。  
3. **ml\_pipeline/04\_train\_meta\_labeling\_l3.py** (Major Refactor): 重寫為 LGBMRanker，並更改特徵輸入。  
4. **ml\_pipeline/05\_backtest\_and\_verify.py** (Modified): 加入 L3 排序邏輯、L4 動態出場邏輯與滑價模擬。  
5. **ml\_pipeline/run\_daily\_inference.py** (Modified): 更新實盤推論邏輯以匹配 V5.1。

### **直接沿用 V5 (Unchanged)**

1. **ml\_pipeline/01\_format\_data\_v5.py**: 數據格式化邏輯通用，無需修改。  
2. **ml\_pipeline/03\_train\_regime\_model\_l1.py**: L1 防禦模型 (HMM+IsoForest) 在 V5 已驗證有效，直接沿用。
