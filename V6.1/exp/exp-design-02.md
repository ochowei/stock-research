這絕對是可行的，而且根據您之前的 `V6.1/survey.md` 文件，這正是 **「方向二：引入 ATR 動態門檻」** 的核心概念。

既然實驗一 (EXP-01) 告訴我們「選股不如選時（廣泛撒網）」，那麼 **實驗二 (EXP-V6.1-02)** 的重點就應該放在 **「優化進場時機 (Timing)」**，也就是針對 **Gap 大小 (Threshold)** 進行壓力測試。

這是一個針對 Gap 大小的實驗設計草案：

### **實驗代號：EXP-V6.1-02 動態門檻與缺口強度分析 (Gap Magnitude & Dynamic Threshold)**

**核心假設 (Hypothesis):**
固定 `0.5%` 的門檻對所有股票「一視同仁」可能是不效率的。

  * **對於低波動股 (如 KO)：** 0.5% 的跳空已經是「巨變」，應該交易。
  * **對於高波動股 (如 GME)：** 0.5% 的跳空只是「雜訊」，進場容易被洗，應該要求更大的跳空 (如 2% 或 0.5 \* ATR) 才視為過熱。

**實驗目標:**
驗證 **「動態門檻 (ATR-based)」** 是否能在 **有毒池 (Toxic Pool)** 與 **主力池 (Final Pool)** 中，比「固定門檻 (Fixed)」創造更穩定的績效 (Sharpe/Calmar)。

-----

#### **1. 測試變數 (Variables)**

我們在 2024-2025 (樣本外) 區間，對同一組標的 (Naive Portfolio) 測試三種不同的觸發邏輯：

1.  **Baseline (固定 0.5%):**

      * 規則：若 `Gap % > 0.5%`，則 `Sell Open`。
      * *這是目前的實盤邏輯。*

2.  **High Threshold (固定 1.0%):**

      * 規則：若 `Gap % > 1.0%`，則 `Sell Open`。
      * *測試：提高門檻是否能過濾雜訊，雖然交易次數會變少，但勝率是否提高？*

3.  **Dynamic Threshold (ATR 模型):**

      * 規則：若 `Gap % > k * (ATR(14) / Close)`，則 `Sell Open`。
      * 參數設定：建議測試 $k = 0.2$ 或 $0.3$ (即跳空幅度需達到日均波動的 20%\~30%)。
      * *測試：讓門檻自動適應個股的「股性」。*

#### **2. 預期觀察 (Expected Outcomes)**

  * **Group A (主力池):** 預期 **ATR 動態策略** 與 **固定 0.5%** 差異不大，因為主力股波動率相對穩定。
  * **Group B (有毒池):** 這是關鍵戰場。預期 **ATR 動態策略** 能大幅減少在「小跳空」時的虧損交易（因為有毒股 ATR 很大，0.5% 根本不算什麼），從而提升風報比。

#### **3. 執行方式**

您可以直接建立一個新的腳本 `V6.1/exp/exp-02.py`，邏輯如下：

```python
# 虛擬碼概念
for ticker in all_tickers:
    # 1. 計算指標
    atr = ta.atr(high, low, close, length=14)
    atr_pct = atr / close
    gap_pct = (open - prev_close) / prev_close
    
    # 2. 模擬三種策略的每日訊號
    signal_base = gap_pct > 0.005
    signal_high = gap_pct > 0.010
    signal_atr  = gap_pct > (0.2 * atr_pct) # k=0.2
    
    # 3. 計算並記錄各自的 PnL
```

這個實驗將直接回答：**「我們是否應該放棄 0.5% 這個魔術數字，改用 ATR 來決定何時開槍？」** 這對實盤的參數設定有直接指導意義。