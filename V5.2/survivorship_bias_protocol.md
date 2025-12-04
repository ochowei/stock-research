這是一份針對 V5.2 專案如何處理與驗證「倖存者偏差」的技術文件草稿。

---

# **V5.2 研究文件：倖存者偏差處理與毒性壓力測試協議**

**Date:** 2025-12-04
**Project:** V5.2-Risk
**Status:** #documentation #risk-management

## **1. 問題陳述 (Problem Statement)**

在 V5.2 的開發過程中，我們識別出兩個主要影響回測真實性的偏差來源：

1.  **倖存者偏差 (Survivorship Bias):**
    * `00_download_index.py` 使用當前 (2025年) 的 Wikipedia 成分股清單去回測 2015 年的市場。
    * 這導致回測自動排除了過去 10 年間因經營失敗、破產而被剔除出指數的公司（如能源、傳統零售股），人為高估了策略的期望回報。
2.  **上市時間偏差 (Listing Time Bias):**
    * 自選池 (`asset_pool.json`) 包含大量 2020 年後上市的科技股 (如 PLTR, COIN)。
    * 這導致 2015-2019 年的回測績效僅由少數幾檔長期存活的菁英股 (如 NVDA, AMZN) 貢獻，無法代表真實的選股能力。

## **2. 解決方案：毒性壓力測試 (Toxicity Stress Test)**

由於免費數據源 (yfinance) 缺乏歷史成分股清單 (Point-in-Time Data) 且會移除下市股票數據，我們採用 **「主動注入毒藥 (Inject Toxicity)」** 的方式來驗證風控系統的強健性。

### **2.1 核心邏輯**
不依賴完美的歷史清單，而是**刻意**將已知失敗、暴跌、破產的標的加入回測池，觀察 `RiskManager` 在面對這些必死無疑的股票時，能否有效執行止損，守住本金。

### **2.2 實作架構**

我們引入了全新的資料載入與分類機制：

1.  **資料分流 (Data Segmentation):**
    * **Normal Pool (菁英池):** 原有的 `asset_pool.json`，代表理想選股。
    * **Toxic Pool (毒藥池):** 新增 `toxic_asset_pool.json`，包含 SIVB, BBBY, NKLA, HTZ 等歷史失敗案例。
2.  **模組化載入器 (`DataLoader`):**
    * 建立 `ml_pipeline/data_loader.py`，負責統一讀取與合併上述兩份清單，確保回測時可以靈活切換場景。
3.  **多場景回測 (`Comparative Backtest`):**
    * 修改 `05_backtest_custom.py` 與 `06_comprehensive_comparison.py`，在同一張圖表上同時繪製三條曲線：
        * **Scenario A (Normal):** 理想情境（存活者）。
        * **Scenario B (Toxic):** 極端情境（全選到地雷股）。
        * **Scenario C (Merged):** 真實情境（好壞參半，接近實盤期望值）。

## **3. 驗證目標 (Verification Objectives)**

在毒性池 (Toxic Pool) 的回測中，我們**不追求獲利**，而是驗證 **V5.2 風控模組** 的以下能力：

1.  **ATR 部位管理 (Volatility Sizing):** 當股價波動率 (ATR) 在崩盤前夕急劇放大時，系統是否能自動縮小開倉部位？
2.  **技術止損 (RSI Exit):** 當股價呈現「陰跌」（RSI 長期鈍化在低檔）時，策略是否能避免連續接刀？
3.  **系統性熔斷 (Regime Filter):** 在 2020/2022 等系統性崩盤期間，L1 模型是否能阻止對個別毒性資產的抄底行為？

## **4. 數據源限制與應對**

在執行下載時，我們確認了 yfinance 的限制：
* **限制:** 已完全清算下市的公司 (如 `SIVB`, `FRC`, `BBBYQ`) 數據已被移除，無法下載。
* **應對:**
    * 保留尚在 OTC 市場交易 (`LKNCY`, `HTZ`) 或股價極低但未下市 (`GOEV`, `NKLA`) 的標的作為測試樣本。
    * 承認回測無法模擬「瞬間歸零」的極端流動性風險，這部分需透過 **實盤模擬 (Paper Trading)** 來補足。

## **5. 結論**

透過引入 **Toxic Asset Pool** 與 **DataLoader**，V5.2 成功將「倖存者偏差」從一個隱形的威脅，轉化為一個可見、可測量的風險指標。

* 若 **Merged Pool** 的績效顯著低於 **Normal Pool**，代表策略過度依賴選股運氣。
* 若 **Toxic Pool** 的最大回撤 (MaxDD) 能被控制在一定範圍內（非 -100%），則證明 V5.2 的風控機制具有真實的防禦力。