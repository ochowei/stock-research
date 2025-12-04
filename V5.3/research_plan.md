# **研究計畫：V5.3-Dynamic 智能攻防與動態出場系統 (Final Revised)**

**Date:** 2025-12-04
**Based on:** V5.2 (Survival) & V5.1 (Lessons Learned)
**Topics:** \#quant-trading \#hybrid-defense \#dynamic-exit \#ablation-study \#slippage-stress
**Status:** \#execution-plan

## **1. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5.2 成功與侷限**
V5.2 確立了「生存基石」，透過 ATR 控倉與市場寬度濾網，成功將回撤控制在 -25% 內。然而，**剝離研究 (Ablation Study)** 顯示，「5天時間止損」是績效的最大拖油瓶，截斷了妖股的主升段利潤。

### **1.2 V5.3 核心目標**
在 **繼承 V5.2 所有風控底層** 的前提下，透過 **「混合防禦」** 與 **「動態追蹤」**，解決 V5.2 「太早賣」與「太晚跑」的問題。

* **目標：** 顯著擊敗 V5.2 Merged Pool (Sharpe 0.87, Return 481%)。
* **約束：** 在 Toxic Pool 壓力測試中，最大回撤不得超過 -30%。

## **2. 系統架構：混合與動態 (Hybrid & Dynamic Architecture)**

V5.3 不是全盤推翻，而是在 V5.2 的鋼骨上加裝動態感應器。

### **2.1 L1 防禦層：混合式崩盤預警 (Hybrid L1)**
結合「模型的敏銳度」與「規則的剛性」。

* **A. 預測模組 (XGBoost Classifier):**
    * **Target:** 預測未來 10 日 `MaxDD > 5%` 或 `VIX > 30` 的機率。
    * **Action:** 若風險機率高，**收緊** ATR 止損倍數 (e.g., 3.0x -> 1.5x) 並停止開新倉。
* **B. 硬性熔斷 (Hard Liquidation - V5.2 Foundation):**
    * **Rule:** 若 `Market Breadth (S&P 100 > SMA200)` < **15%**。
    * **Action:** **強制市價清倉**。此規則具有**最高優先級 (Override)**，無視任何模型預測。這是對抗「模型失靈」的最後一道防線。

### **2.2 L3 排序層：基本面增強 (Fundamental-Enhanced Ranking)**
* **特徵工程:** 引入 P/S Ratio, Revenue Growth (QoQ)。
* **頻率對齊 (Frequency Check):** 必須計算基本面因子與短線回報的 IC 值。若 IC 不顯著，則回退至 V5.2 的純 RSI 排序，避免引入雜訊。

### **2.3 L4 出場層：動態追蹤止盈 (ATR Trailing Stop)**
取代固定的 5 天持有期。

* **機制:** `Stop_Price = Highest_High - (K * ATR)`。
* **動態 K 值:** 由 L1 模型決定。市場安全時 $K=3$ (放寬讓利潤跑)，市場危險時 $K=1.5$ (快速鎖利)。
* **滑價懲罰分級 (Slippage Penalty Grading):**
    * **Entry (Limit Order):** 設為 **5bps** (與 V5.2 相同)。
    * **Trailing Exit (Stop Market Order):** 設為 **10bps**。因為動態止損通常觸發於價格快速下跌時，滑價成本必然高於限價單。

### **2.4 底層風控 (Risk Foundation - Inherited)**
* **Position Cap:** 單一標的權重上限 (20%)。防止基本面數據錯誤 (如市值計算錯誤) 導致 ATR Sizing 失控。
* **ATR Sizing:** 維持 V5.2 的波動率倒數加權邏輯。

## **3. 執行階段規劃 (Execution Phases)**

* **Phase 1: 數據與基礎設施** (繼承 `data_loader`, 擴充基本面數據)。
* **Phase 2: 模型開發** (訓練 L1 XGBoost 崩盤預警, L3 Learning-to-Rank)。
* **Phase 3: 機制實作** (修改 `backtesting_utils` 支援 Trailing Stop 與 分級滑價)。
* **Phase 4: 驗證與剝離研究** (執行下列關鍵驗證協議)。

## **4. 驗證協議：剝離研究 (Ablation Study)**

為了證明 V5.3 的複雜度是值得的，必須執行以下 **「減法測試」**。基準 (Baseline) 為 **V5.3 Full System**。

我們將測試分為三組：**動態效益驗證**、**防禦底層驗證**、**成本壓力驗證**。

| 測試組別 | 測試場景 (Scenario) | 移除/修改組件 | 測試目的 (Hypothesis to Validate) |
| :--- | :--- | :--- | :--- |
| **A. 動態效益** | **No Trailing Stop** | 回退至 **固定 5 天出場** | 證明 L4 動態出場真的能抓到肥尾利潤 (Fat Tail)，而不僅是增加交易次數。 |
| | **No Predictive Defense** | 回退至 **純 V5.2 寬度濾網** | 證明 L1 模型能比硬性規則「更早」偵測風險，減少回撤幅度。 |
| | **No Fundamental Sort** | 回退至 **純 RSI 排序** | 證明加入基本面數據能提升選股勝率 (Win Rate)，而非僅是噪聲。 |
| **B. 防禦底層** | **No Hard Liquidation** | 移除 **強制清倉** (僅靠模型) | **(關鍵)** 證明當模型預測失敗 (False Negative) 時，V5.2 的硬性規則是救命稻草。 |
| | **No Position Cap** | 移除 **20% 上限** | 測試在基本面數據異常時，ATR Sizing 是否會導致單一標的過度曝險。 |
| **C. 成本壓力** | **No Slippage Penalty** | 全程使用 **統一 5bps** | **(真實性檢查)** 移除對 Stop Order 的 10bps 懲罰。若此場景績效遠高於 Baseline，代表策略利潤可能被滑價吃光，需重新評估可行性。 |

## **5. 成功指標 (Success Metrics)**

| 指標 | V5.2 Merged (基準) | V5.3 目標 |
| :--- | :--- | :--- |
| **Total Return** | 481% | **> 600%** (由 L4 貢獻) |
| **Max Drawdown** | -25.8% | **< -25%** (由 L1 貢獻) |
| **Sharpe Ratio** | 0.87 | **> 1.0** |
| **Ablation Validity** | N/A | 所有剝離場景的績效皆應 **低於** Full System (代表每個組件都有貢獻)。 |

---
**結語:**
V5.3 是一次「防守反擊」的升級。我們在 V5.2 的 **防禦底層 (Liquidation, Position Cap)** 之上，疊加了 **滑價分級** 的現實考驗，確保策略不僅在理論上可行，在包含交易摩擦的真實市場中依然具備超額獲利能力。