# **V6.3 Sell Model Lab (賣出模型實驗室)**

這是一個專門用於迭代優化 **Sell Model (賣出模型)** 的實驗室環境。 目標是透過系統化的實驗流程，利用 AI Agent (Gemini/Jules) 找出舊版本 (V6.2) 的問題，並探索新的優化方向，不斷產出更好的模型。

## **📂 檔案結構 (File Structure)**

本實驗室採用 **Self-Contained Experiment (獨立實驗封裝)** 的架構，確保每個實驗的 Context 完整且互不干擾。

V6.2/exp/Sell\_Model\_Lab/  
├── 00\_Legacy\_Context/               \# \[知識庫\] 歷史版本的對照組 (唯讀)  
│   ├── V6.1\_Baseline/               \# 基準版本 (成功的 exp\_07)  
│   │   ├── exp\_07.py  
│   │   └── exp-07-report.md  
│   ├── V6.2.2\_Attempt/              \# 嘗試優化但失敗的版本  
│   │   ├── exp\_07\_v2\_training.py  
│   │   ├── exp\_07\_v2\_report.md  
│   │   └── exp\_07\_repro\_baseline.py  
│   └── initial\_diagnosis.md         \# \[AI產出\] V6.1 vs V6.2.2 的差異診斷報告  
│  
├── 01\_Backlog\_and\_Insights/         \# \[大腦\] 實驗規劃與總結  
│   ├── experiment\_queue.md          \# 待執行的實驗清單與優先順序  
│   └── global\_learning\_log.md       \# 跨實驗的 Insight 總結 (什麼有效，什麼無效)  
│  
├── 02\_Lab\_Utils/                    \# \[工具箱\] 實驗共用的程式碼  
│   ├── common\_paths.py              \# 設定 sys.path 讓實驗能 import 上層 V6.2 的模組  
│   ├── metrics.py                   \# 統一的績效計算標準  
│   └── reporting.py                 \# 自動產生 Markdown 報告的工具  
│  
└── 03\_Experiments/                  \# \[實驗區\] 每個實驗一個獨立資料夾  
    ├── EXP\_Template/                \# (模板) 複製這個來開新實驗  
    │   ├── 01\_Design.md             \# \[設計\] 假設、實驗步驟、預期結果  
    │   ├── 02\_Implementation.py     \# \[實作\] 訓練與回測代碼  
    │   ├── 03\_Output/               \# \[結果\] 產出的 Model, CSV, PNG  
    │   └── 04\_Review.md             \# \[檢討\] 結果解讀、結論、下一步  
    │  
    ├── EXP\_01\_Crypto\_Ablation/      \# (範例) 實驗 1  
    └── EXP\_02\_Simple\_XGB/           \# (範例) 實驗 2

## **🎯 Success Metrics (驗收標準)**

本實驗室的優化目標是產出高勝率、正期望值的賣出策略。所有實驗結果應根據以下優先順序進行評估：

1.  **Average Return (平均回報)**: `> 0.20%` per trade (扣除成本前)。
    * *Rationale*: 必須顯著高於交易成本與滑價，確保實際獲利能力。
2.  **Win Rate (勝率/準確率)**: `> 55%` (理想目標 `60%`)。
    * *Rationale*: 作為 Filter Model，減少 False Positive 是核心任務。
3.  **Signal Count (訊號數量)**: 確保測試集樣本數充足 (e.g., `> 1000` trades/year)，具有統計意義。
    * *Note*: 不需為了追求數量而犧牲上述兩項指標。

## **🤖 AI Agent Workflow (工作流程)**

我們使用三個主要的 AI Agent 角色來推進實驗：

### **Phase 1: 診斷與回顧 (Diagnosis & Review)**

* **Role**: **Analyst Agent**  
* **Input**: 00\_Legacy\_Context 或 03\_Experiments/EXP\_XX/03\_Output/  
* **Task**:  
  1. 比對新舊代碼與結果。  
  2. 找出過擬合、特徵失效或邏輯錯誤的根本原因。  
  3. 解讀回測報表，判斷實驗是否成功。  
* **Output**: initial\_diagnosis.md 或 04\_Review.md

### **Phase 2: 設計與規劃 (Design & Architecture)**

* **Role**: **Architect Agent**  
* **Input**: experiment\_queue.md 和 Analyst 的診斷報告。  
* **Task**:  
  1. 根據診斷提出具體的優化假設 (Hypothesis)。  
  2. 設計實驗步驟、變數控制與驗證指標。  
  3. 建立新的實驗資料夾結構。  
* **Output**: 03\_Experiments/EXP\_XX/01\_Design.md

### **Phase 3: 實作與執行 (Implementation)**

* **Role**: **Engineer Agent**  
* **Input**: 01\_Design.md 與 02\_Lab\_Utils/  
* **Task**:  
  1. 撰寫 Python 訓練與回測代碼。  
  2. 確保引用正確的路徑與共用模組。  
  3. 執行程式並產出結果。  
* **Output**: 02\_Implementation.py 和 03\_Output/ (Models, Plots)

## **🚀 如何開始新的實驗 (How to Start)**

1. **Check Backlog**: 查看 01\_Backlog\_and\_Insights/experiment\_queue.md 確認下一個優先級最高的實驗。  
2. **Create Folder**: 複製 EXP\_Template 並重新命名 (例如 EXP\_03\_New\_Feature)。  
3. **Write Design**: 編輯 01\_Design.md，定義這個實驗要做什麼。  
4. **Code**: 根據 Design 撰寫 02\_Implementation.py。  
5. **Run**: 執行程式碼並檢查 03\_Output。  
6. **Review**: 分析結果並填寫 04\_Review.md，最後更新 global\_learning\_log.md。

## **🛠️ 環境設定 (Environment)**

此實驗室依賴於上層目錄 V6.2/exp/ 的資源。  
在每個實驗腳本中，請確保加入以下 Path 設定以引用共用模組：  
import sys  
import os

\# 自動抓取專案根目錄路徑  
current\_dir \= os.path.dirname(os.path.abspath(\_\_file\_\_))  
\# 假設位於 V6.2/exp/Sell\_Model\_Lab/03\_Experiments/EXP\_XX/  
project\_root \= os.path.abspath(os.path.join(current\_dir, "../../../"))   
sys.path.append(project\_root)

\# Import V6.2 modules  
try:  
    from daily\_gap\_signal\_generator import ...  
    print("Successfully imported V6.2 modules.")  
except ImportError:  
    print("Error importing V6.2 modules. Check sys.path.")

## **📏 Versioning & Naming Convention (版本命名規範)**

為保持專案結構一致性，本實驗室內的所有實驗均視為 **V6.2 架構下的迭代**。

1.  **RC (Release Candidate) Naming**:
    * 當實驗產生可用的 Production Candidate 時，請使用 `.RC` 後綴。
    * ❌ **禁止** 使用 `V6.3`, `V6.4` 等未來主版本號。
    * ✅ **正確**: `V6.2.3.RC`, `V6.2.4.RC`。

2.  **Mapping (實驗對照)**:
    * `EXP-08` -> `V6.2.3.RC` (Heterogeneous Ensemble Integration)
    * `EXP-13` -> `V6.2.4.RC` (Full Deployment with SPY Context)

3.  **File Naming (檔名)**:
    * Script: `production_daily_plan_v6_2_x_rc.py`
    * Model: `v6.2.x_rc_model.joblib`
