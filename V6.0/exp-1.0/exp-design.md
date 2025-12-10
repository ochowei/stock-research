### **實驗代號：EXP-V6.0-01 隔夜效應驗證 (修訂版 Rev 1\)**

更新日期： 2025-12-10  
實驗目標： 驗證不同資產池（優質 vs 有毒）的報酬結構與風險特徵（隔夜 vs 日內），並產出包含最大回撤的完整績效報表。

#### **1\. 實驗對象 (Asset Universe)**

將實驗對象分為三組進行獨立與交叉對比：

1. **Group A: Final Asset Pool (主力池)**  
   * 來源：2025\_final\_asset\_pool.json  
   * 特徵：包含 TSLA, NVDA 等高動能與體質較佳的成長股。  
2. **Group B: Toxic Asset Pool (對照池)**  
   * 來源：2025\_final\_toxic\_asset\_pool.json  
   * 特徵：包含 MEME 股（GME, AMC）、高負債或虧損股。預期此組的「日內下跌」效應應最強。  
3. **Group C: VOO Constituents (市場基準池) \[Optional\]**  
   * 來源：使用者提供的 voo\_constituents.json (若無則跳過，改用 SPY ETF 單一標的作為基準)。  
   * *備註：* 根據 Survey，S\&P 500 是隔夜效應最顯著的群體。

#### **2\. 數據策略 (Data Strategy)**

* **來源：** yfinance  
* **頻率：** 1d (Daily)  
* **資料行：** Open, High, Low, Close (必須包含 High/Low 以進行進階波動率分析)。  
* **調整：** 使用 auto\_adjust=True 以還原股息與拆股影響。

#### **3\. 評估指標 (Metrics)**

除了累積報酬，新增以下風險指標（針對 Night 與 Day 分別計算）：

1. **CAGR (年化報酬率):** 衡量成長速度。  
2. **Max Drawdown (MDD, 最大回撤):**  
   * 計算歷史上資產價值從最高點滑落的最大幅度。  
   * *關鍵觀察：* 只持有隔夜是否能顯著降低 MDD？  
3. **Sharpe Ratio (夏普比率):** 衡量承擔單位風險的超額報酬（無風險利率設為 0% 簡化）。  
4. **Win Rate (勝率):** 每日 $R \> 0$ 的天數佔比。  
5. **Volatility (年化波動率):**  
   * **Day Vol:** 基於 $\\ln(Close/Open)$  
   * **Night Vol:** 基於 $\\ln(Open/PreClose)$

#### **4\. 實驗流程與邏輯**

**Step 1: 資料獲取 (ETL)**

* 讀取兩個 JSON 檔案。  
* (Optional) 嘗試讀取 VOO 清單。  
* 下載 OHLCV 數據，並處理 Missing Data。

Step 2: 每日報酬分解  
對於每一檔股票 $i$ 在第 $t$ 天：

* $R\_{night} \= (O\_t \- C\_{t-1}) / C\_{t-1}$  
* $R\_{day} \= (C\_t \- O\_t) / O\_t$  
* $R\_{hold} \= (C\_t \- C\_{t-1}) / C\_{t-1}$ (基準)

Step 3: 投資組合模擬 (Portfolio Simulation)  
假設「等權重 (Equal Weight)」配置：

* 計算整個 Asset Pool 每天的平均 $R\_{night}$ 與 $R\_{day}$。  
* 生成三個權益曲線 (Equity Curves)：Night\_Only, Day\_Only, Buy\_Hold。

**Step 4: 波動率與風險分析**

* 計算上述三個曲線的 MDD 與 Volatility。

#### **5\. 預期產出報表 (Output Deliverables)**

程式執行後將產生以下檔案：

1. **exp\_01\_performance\_summary.csv (文字報表):**  
   * Columns: Pool, Strategy (Night/Day/Hold), Total Return, CAGR, MDD, Sharpe, Daily Vol  
   * Rows: Group A, Group B, Group C 的各策略數據。  
2. **exp\_01\_equity\_curves.png (視覺化):**  
   * Subplot 1: Group A (Final Pool) 的 Night vs Day 走勢。  
   * Subplot 2: Group B (Toxic Pool) 的 Night vs Day 走勢。  
   * *預期：* Toxic Pool 的 Day 曲線應該是顯著向下的（一路殺盤）。  
3. **exp\_01\_drawdown\_curves.png (視覺化):**  
   * 比較 Night 策略與 Day 策略的水下曲線 (Underwater Plot)，直觀展示「誰讓你睡得安穩」。  
4. **exp\_01\_volatility\_analysis.png (視覺化):**  
   * Bar Chart: 比較各組的 Night Volatility vs Day Volatility。

---

這份實驗設計將能夠回答：**「我們是否應該避開日間交易？」** 以及 **「有毒資產是否在開盤後特別危險？」** 這兩個核心問題。