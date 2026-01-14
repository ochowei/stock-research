V6.3 Sell Model Lab (賣出模型實驗室)這是一個專門用於迭代優化 Sell Model (賣出模型) 的實驗室環境。 目標是透過系統化的實驗流程，利用 AI Agent (Gemini/Jules) 找出舊版本 (V6.2) 的問題，並探索新的優化方向，不斷產出更好的模型。📂 檔案結構 (File Structure)本實驗室採用 Self-Contained Experiment (獨立實驗封裝) 的架構，確保每個實驗的 Context 完整且互不干擾。V6.2/exp/Sell_Model_Lab/
├── 00_Legacy_Context/               # [知識庫] 歷史版本的對照組 (唯讀)
│   ├── V6.1_Baseline/               # 基準版本 (成功的 exp_07)
│   │   ├── exp_07.py
│   │   └── exp-07-report.md
│   ├── V6.2.2_Attempt/              # 嘗試優化但失敗的版本
│   │   ├── exp_07_v2_training.py
│   │   ├── exp_07_v2_report.md
│   │   └── exp_07_repro_baseline.py
│   └── initial_diagnosis.md         # [AI產出] V6.1 vs V6.2.2 的差異診斷報告
│
├── 01_Backlog_and_Insights/         # [大腦] 實驗規劃與總結
│   ├── experiment_queue.md          # 待執行的實驗清單與優先順序
│   └── global_learning_log.md       # 跨實驗的 Insight 總結 (什麼有效，什麼無效)
│
├── 02_Lab_Utils/                    # [工具箱] 實驗共用的程式碼
│   ├── common_paths.py              # 設定 sys.path 讓實驗能 import 上層 V6.2 的模組
│   ├── metrics.py                   # 統一的績效計算標準
│   └── reporting.py                 # 自動產生 Markdown 報告的工具
│
└── 03_Experiments/                  # [實驗區] 每個實驗一個獨立資料夾
    ├── EXP_Template/                # (模板) 複製這個來開新實驗
    │   ├── 01_Design.md             # [設計] 假設、實驗步驟、預期結果
    │   ├── 02_Implementation.py     # [實作] 訓練與回測代碼
    │   ├── 03_Output/               # [結果] 產出的 Model, CSV, PNG
    │   └── 04_Review.md             # [檢討] 結果解讀、結論、下一步
    │
    ├── EXP_01_Crypto_Ablation/      # (範例) 實驗 1
    └── EXP_02_Simple_XGB/           # (範例) 實驗 2
🤖 AI Agent Workflow (工作流程)我們使用三個主要的 AI Agent 角色來推進實驗：Phase 1: 診斷與回顧 (Diagnosis & Review)Role: Analyst AgentInput: 00_Legacy_Context 或 03_Experiments/EXP_XX/03_Output/Task:比對新舊代碼與結果。找出過擬合、特徵失效或邏輯錯誤的根本原因。解讀回測報表，判斷實驗是否成功。Output: initial_diagnosis.md 或 04_Review.mdPhase 2: 設計與規劃 (Design & Architecture)Role: Architect AgentInput: experiment_queue.md 和 Analyst 的診斷報告。Task:根據診斷提出具體的優化假設 (Hypothesis)。設計實驗步驟、變數控制與驗證指標。建立新的實驗資料夾結構。Output: 03_Experiments/EXP_XX/01_Design.mdPhase 3: 實作與執行 (Implementation)Role: Engineer AgentInput: 01_Design.md 與 02_Lab_Utils/Task:撰寫 Python 訓練與回測代碼。確保引用正確的路徑與共用模組。執行程式並產出結果。Output: 02_Implementation.py 和 03_Output/ (Models, Plots)🚀 如何開始新的實驗 (How to Start)Check Backlog: 查看 01_Backlog_and_Insights/experiment_queue.md 確認下一個優先級最高的實驗。Create Folder: 複製 EXP_Template 並重新命名 (例如 EXP_03_New_Feature)。Write Design: 編輯 01_Design.md，定義這個實驗要做什麼。Code: 根據 Design 撰寫 02_Implementation.py。Run: 執行程式碼並檢查 03_Output。Review: 分析結果並填寫 04_Review.md，最後更新 global_learning_log.md。🛠️ 環境設定 (Environment)此實驗室依賴於上層目錄 V6.2/exp/ 的資源。在每個實驗腳本中，請確保加入以下 Path 設定以引用共用模組：import sys
import os

# 自動抓取專案根目錄路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假設位於 V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_XX/
project_root = os.path.abspath(os.path.join(current_dir, "../../../")) 
sys.path.append(project_root)

# Import V6.2 modules
try:
    from daily_gap_signal_generator import ...
    print("Successfully imported V6.2 modules.")
except ImportError:
    print("Error importing V6.2 modules. Check sys.path.")
