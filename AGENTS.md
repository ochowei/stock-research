# AI Agents Protocol (操作守則)

此文件專為協助開發的 AI Agent (如 Jules, Copilot, Gemini 等) 設計。在執行任何檔案操作（新增、修改、讀取、移動）前，請務必遵守以下「專案目錄隔離」原則。

## 🔴 最高指導原則：專案目錄隔離 (Strict Project Isolation)

不同的專案版本（如 `V4`, `V5`, `V5.1`）必須視為完全獨立的實體。

1.  **鎖定工作目錄 (Lock Working Directory):**
    * 在開始任務前，確認 User 指定的目標版本（例如 `V5.1`）。
    * 你的所有操作範圍僅限於該版本的根目錄內（例如 `./V5.1/`）。
    * **嚴禁**將當前任務的檔案寫入其他版本（如 `V4-D.8`）的資料夾中。

2.  **路徑明確化 (Explicit File Paths):**
    * 生成或讀取檔案時，路徑**必須**包含該版本的根目錄名稱。
    * **✅ 正確:** `V5.1/ml_pipeline/risk_manager.py`
    * **❌ 錯誤:** `ml_pipeline/risk_manager.py` (路徑不明確，容易存錯位置)
    * **❌ 錯誤:** `V5/ml_pipeline/risk_manager.py` (跑到舊版本去了)

3.  **環境自給自足 (Self-Contained Environment):**
    * 每個版本資料夾應具備獨立的完整性。
    * **禁止跨版本 Import:** 除非 User 明確指示進行遷移，否則不要在 `V5` 的程式碼中 `import` 來自 `V4` 資料夾的模組。
    * **獨立配置:** 每個版本應有自己的 `requirements.txt`、`README.md` 和數據子目錄（如 `data/`、`features/`），不應依賴外部共用檔案。

4.  **新專案建立 (New Project Creation):**
    * 若 User 要求建立新版本（例如 V6）：
        1.  先在根目錄建立 `V6/` 資料夾。
        2.  在 `V6/` 內建立所需的子目錄結構。
        3.  將所有新生成的程式碼與文件放入 `V6/` 中。

## 📝 操作前自我檢查 (Pre-Action Checklist)

* [ ] **Target Check:** 我是否清楚知道現在是針對哪個版本資料夾 (Folder) 進行操作？
* [ ] **Path Check:** 我輸出的檔案路徑是否以該版本資料夾名稱（如 `V5.2/`）作為開頭？
* [ ] **Isolation Check:** 我是否確保沒有修改到其他平行版本（如 `V4` 或 `V5`）的檔案？
