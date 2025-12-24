好的，這是將您的新想法 **「方向六：隔夜持有分析 (Overnight Holding Analysis)」** 整合進去後的 **V6.1 完整結案報告**。

這份報告現在涵蓋了從「選股」、「進場」、「濾網」到「持倉週期」的完整優化邏輯。

---

# **V6.1 策略優化結案報告 (Final Strategy Optimization Report)**

**版本:** V6.1-Final  
**核心結論:** **「簡化預測，精細過濾，驗證隔夜」 (Simplify Prediction, Refine Filtering, Verify Overnight)**

本報告基於 Exp-01 至 Exp-05 的實驗數據，並納入最新的「隔夜持有」假說，為 V6.1 實盤策略定調。

---

### **第一部分：已驗證的優化結論 (Proven Optimizations)**

#### **1. 資產池選擇 (Direction 1: Asset Suitability)**
* **結論：** **維持廣泛撒網 (Naive Portfolio)**。
* **數據支撐：** Exp-01 證明 AI 預測模型的「聰明組合」因過度篩選而踏空，淨值 (2.3) 遠輸給「全市場組合」 (3.4)。
* **執行：** 放棄靜態黑名單，擁抱市場隨機性。

#### **2. 進場門檻 (Direction 2: Thresholding)**
* **結論：** **維持固定 0.5% (Fixed 0.5%)**。
* **數據支撐：** Exp-02 顯示固定門檻的夏普比率 (2.59) 優於 ATR 動態門檻。0.5% 是市場微結構的關鍵心理價位。
* **執行：** 主策略不做改動。若需降低交易頻率，可選用 ATR k=0.2。

#### **3. 微結構濾網 (Direction 3: Microstructure)**
* **結論：** **作為高勝率確認指標**。
* **分析：** 「T-1 尾盤急拉」與「盤前價格衰竭 (Fade)」能有效識別假突破。
* **執行：** 在儀表板中標記符合 `Pre-Fade > 1.0%` 的標的為「高信心 setup」。

#### **4. 跨資產連動 (Direction 4: Cross-Asset)**
* **結論：** **針對垃圾股 (Toxic Pool) 的週一特效藥**。
* **分析：** 幣圈週末大漲是 Meme 股週一軋空的先行指標。
* **執行：** 若 ETH 週末漲幅 > 5%，週一暫停 Toxic Pool 的做空交易。

#### **5. 日曆效應 (Direction 5: Calendar Effects)**
* **結論：** **分池治理 (Differential Treatment)**。
* **數據支撐：** Exp-05 發現驚人分歧：
    * **Final Pool (績優股):** 避開月初 (TOTM) 與節前 (Holiday)。(機構買盤強)
    * **Toxic Pool (垃圾股):** 月初 (TOTM) 反而要加碼。(資金撤出垃圾股)
* **執行：** 實施分眾日曆規則。

---

### **第二部分：新增探索方向 (New Exploration)**

#### **6. 隔夜風險溢酬分析 (Direction 6: Overnight Risk Premium) [NEW]**

**核心問題：**
目前的策略是「日內結清 (Intraday Only)」。我們是否應該持有空單過夜，以博取趨勢的延續？還是應該嚴格避開隔夜風險？

**理論假設：**
* **美股特性 (Night Effect):** 學術研究指出 SPY 的長期上漲主要來自「夜間跳空」。持有空單過夜通常是在對抗「正期望值的隔夜漂移」，長期來看是 **負期望值 (Negative EV)**。
* **垃圾股特性 (Toxic Exception):** 垃圾股可能存在「動能崩潰」，即日內大跌後，夜間恐慌蔓延導致隔日開盤續跌。

**實驗設計 (EXP-V6.1-06):**
* **變數：** 比較 $R_{day}$ (Open-Close) 與 $R_{night}$ (Close-NextOpen) 的損益分佈。
* **預期動作：**
    * 若 Final Pool 的 $R_{night}$ 為負（預期中）：**確認 MOC 出場是最佳解**。
    * 若 Toxic Pool 的 $R_{night}$ 為正（驚喜）：**考慮對特定崩盤股持有過夜 (Swing Short)**。

---

### **第三部分：V6.1 最終執行路線圖 (Execution Roadmap)**

基於上述六點，V6.1 的實盤腳本 (`daily_gap_signal_generator.py`) 將進行以下邏輯更新：

1.  **資料源：** 擴大監控 `2025_final_asset_pool.json` 全體標的。
2.  **基本訊號：** `Gap > 0.5%` (Fixed)。
3.  **環境檢查 (Global Filters):**
    * **Calendar:** 檢查是否為 TOTM 或 Pre-Holiday。
    * **Crypto:** 檢查 ETH 週末漲幅 (僅週一)。
4.  **分眾決策 (Decision Logic):**
    * **Case A (Final Pool):**
        * 若遇 TOTM/Holiday -> **SKIP** (避開機構買盤)。
        * 若正常日 -> **GO** (MOC 出場)。
    * **Case B (Toxic Pool):**
        * 若遇 TOTM -> **DOUBLE** (加碼/放寬)。
        * 若 ETH 暴漲 -> **SKIP** (避開軋空)。
        * 若正常日 -> **GO**。
5.  **AI 決策 (AI Overlay):**
    * 載入 `exp_07_model.joblib`。
    * 若 AI 預測機率 > 50% -> **✅ GO** (標記為 Actionable)。
    * 若 AI 預測機率 < 50% -> **❌ SKIP** (標記為 Low Confidence)。
6.  **待驗證 (Pending):**
    * 執行 Exp-06。若發現顯著隔夜利潤，將在 V6.2 引入「隔夜留倉模組」。

### **關鍵突破：AI 適性濾網 (The AI Breakthrough)**

基於 **Exp-07** 的驚人成果，V6.1 策略迎來了最後一塊拼圖：**「機器學習適性濾網 (Next-Day Suitability Classifier)」**。

* **邏輯轉變：** 從「被動過濾 (規則)」轉向「主動預測 (AI)」。
* **實戰效果：** * 勝率從 **51%** 提升至 **60%**。
    * 單筆期望值從 **0.05%** 提升至 **0.98%**。
* **運作機制：** 每日早晨針對所有 Gap 訊號，利用 XGBoost 模型分析其 RSI、VIX 與量能結構，給出 `GO` (信心 > 50%) 或 `SKIP` 的建議。


---

### **總結**

您的 V6.1 策略已經從單純的技術指標策略，進化為一個 **「融合宏觀資金流 (日曆/Crypto) 與微觀結構 (Fade/Gap)」** 的多因子系統。

加入 **方向六 (隔夜分析)** 是非常嚴謹的最後一步，它將回答「我們是否浪費了晚上的利潤」，確保策略在時間維度上也是最優的。