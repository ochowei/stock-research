
### **1. 實驗設計概念 (EXP-08 Design)**

**核心假設：**
Gap Fade 策略是賭「開盤情緒過熱後的回調」。

* **MOO (市價開盤):** 雖然保證成交，但往往賣在「情緒還沒到頂」的位置，且容易被瞬間的波動掃出場。
* **Blind Limit (盲限價):** 掛在 **開盤價上方 N%**。我們賭開盤後會有一個「慣性衝高 (Morning Spike)」，正好撞到我們的限價單，然後才開始下跌。這樣我們能**空在更高點**。

**實驗變數 (Variables)：**

1. **Control (對照組):** **Market on Open (MOO)**
* 進場價 = `Open`
* 成交率 = 100%


2. **Test Group A (固定百分比):** **Limit @ Open + k%**
* 進場價 = `Open * (1 + k)`
* 成交條件：`High >= Limit_Price`
* 測試參數 : 0.3%, 0.5%, 1.0%


3. **Test Group B (動態 ATR):** **Limit @ Open + k * ATR**
* 進場價 = `Open + k * ATR`
* 成交條件：`High >= Limit_Price`
* 測試參數 : 0.1, 0.2, 0.3



**評估指標 (Metrics)：**

* **Fill Rate (成交率):** 我們犧牲了多少交易機會？ (例如從 100% 降到 60%)
* **Win Rate (勝率):** 是否因為進場點變好，勝率顯著提升？
* **Profit Factor (獲利因子):** 賺賠比是否優化？
* **Total Return (總回報):** 雖然交易變少，但總賺的錢有變多嗎？

