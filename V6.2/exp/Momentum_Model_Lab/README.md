# **V6.2 Momentum Model Lab (動能模型實驗室)**

這是一個專門用於迭代與優化 **Momentum Model (動能模型)** 的實驗室環境。目標是透過系統化的實驗流程，提升模型對於強勢股的篩選能力，並找出能持續貢獻正 Alpha 的動能因子。

## **📂 檔案結構 (File Structure)**

本實驗室採用 **Self-Contained Experiment (獨立實驗封裝)** 架構，確保每個實驗的 Context 完整且互不干擾。

`V6.2/exp/Momentum_Model_Lab/`

├── `00_Legacy_Context/`               # [知識庫] 歷史版本的對照組 (唯讀)

│   ├── `V6.1_Baseline/`               # 基準版本 (現有的 momentum_model.joblib)

│   └── `initial_diagnosis.md`         # [AI產出] 現有動能模型在不同市況下的失效診斷

│

├── `01_Backlog_and_Insights/`         # [大腦] 實驗規劃與總結

│   ├── `experiment_queue.md`          # 待執行的實驗清單 (如：窗口優化、成交量濾網)

│   └── `global_learning_log.md`       # 跨實驗的 Insight 總結 (例如：哪些板塊動能最穩定)

│

├── `02_Lab_Utils/`                    # [工具箱] 實驗共用的程式碼

│   ├── `common_paths.py`              # 設定 sys.path 以引用 V6.2 根目錄模組

│   ├── `metrics.py`                   # 動能專用績效標準 (IC值、分組報酬分析)

│   └── `reporting.py`                 # 自動產生實驗報告的工具

│

└── `03_Experiments/`                  # [實驗區] 每個實驗一個獨立資料夾

└── `EXP_Template/`                # (模板) 包含 Design, Implementation, Output, Review

## **🎯 Success Metrics (驗收標準)**

動能模型的優化目標是產出具備強大續航力且風險可控的策略。評估標準如下：

1. **Average Return (平均回報)**: 每筆交易 `> 0.25%` (扣除成本前)。
2. **Win Rate (勝率)**: `> 55%`。
3. **Information Coefficient (IC 值)**: 模型預測分數與未來報酬的相關性應顯著為正。
4. **Signal Count (訊號數量)**: 確保樣本數充足 (e.g., `> 800` trades/year)。

## **🤖 AI Agent Workflow (工作流程)**

我們使用三個主要的 AI Agent 角色來推進實驗：

### **Phase 1: 診斷與回顧 (Analyst Agent)**

* **任務**: 比對不同時間窗口（如 5、10、20日）的動能衰竭速度，找出過擬合或特徵失效的原因。
* **輸出**: `04_Review.md` 或診斷報告。

### **Phase 2: 設計與規劃 (Architect Agent)**

* **任務**: 提出優化假設（例如：加入「板塊相對強度」作為過濾條件），設計實驗步驟與變數控制。
* **輸出**: `01_Design.md`。

### **Phase 3: 實作與執行 (Engineer Agent)**

* **任務**: 撰寫 Python 代碼進行模型訓練、特徵重要性分析及回測。
* **輸出**: `02_Implementation.py` 與 `03_Output/`。

## **🚀 如何開始新的實驗**

1. **Check Backlog**: 確認 `experiment_queue.md` 中的優先順序。
2. **Create Folder**: 複製 `EXP_Template` 並命名為 `EXP_XX_Description`。
3. **Write Design**: 在 `01_Design.md` 定義實驗假設。
4. **Code & Run**: 撰寫實作腳本並產出結果。
5. **Review**: 完成檢討報告並更新 `global_learning_log.md`。

## **📏 版本命名規範**

所有成品均視為 **V6.2 架構下的迭代**：

* **Production Candidate**: 使用 `.RC` 後綴（例如 `V6.2.x_momentum_RC.joblib`）。
* **禁止**：在實驗階段隨意跳動主版本號至 `V6.3`。
