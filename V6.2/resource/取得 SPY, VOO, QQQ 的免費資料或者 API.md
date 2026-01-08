取得 2020-2025 年的 SPY (S\&P 500)、VOO (S\&P 500\) 和 QQQ (Nasdaq 100\) 的**歷史成份股 (Constituents)** 資料，如果是要「免費」且「完整每日清單」，在官方渠道通常是付費服務。

不過，開源社群有整理好的資料庫，通常採用 **「目前名單 \+ 歷史變動紀錄 (Changes Log)」** 的方式來推算。以下是目前最可行的免費資源與方法：

### **1\. S\&P 500 (適用於 SPY, VOO)**

SPY 和 VOO 都是追蹤 S\&P 500 指數，因此它們的成份股是一樣的。

#### **最佳免費來源：GitHub 開源資料庫**

這是目前最推薦的免費獲取方式，有開發者專門維護歷史變動清單。

* **Repository:** fja05680/sp500  
  * **資料內容:** 這個 Repo 維護了非常完整的 S\&P 500 歷史成份股清單。  
  * **關鍵檔案:**  
    * S\&P 500 Historical Components & Changes(MM-DD-YYYY).csv: 包含從 1996 年至今的歷史成份股。  
    * sp500\_changes\_since\_2019.csv: 專門記錄 2019 年以後的「新增 (Added)」與「移除 (Removed)」的股票代碼與日期。  
  * **如何使用:** 您可以下載 CSV 檔，利用 Python Pandas 讀取。如果只有變動表，您可以拿今天的 S\&P 500 清單，根據變動表的日期「倒推」回去，就能還原 2020-2025 任一天的成份股。  
* **維基百科 (Wikipedia)**  
  * **頁面:** [List of S\&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)  
  * **用途:** 頁面中有一個 **"Selected changes to the list of S\&P 500 components"** 的表格。這份表格非常準確，記錄了每次調整的日期、新增代碼和移除代碼。許多 GitHub 的爬蟲其實就是爬這個表格。

### **2\. Nasdaq 100 (適用於 QQQ)**

QQQ 追蹤的是 Nasdaq 100 指數。

#### **最佳免費來源：GitHub 開源工具**

* **Repository:** jmccarrell/n100tickers  
  * **資料內容:** 這是一個專門設計來取得 Nasdaq 100 歷史成份股的 Python 工具/資料集。  
  * **覆蓋範圍:** 宣稱包含從 2015 年至今的準確資料。  
  * **特點:** 它解決了 Nasdaq 100 每年 12 月會進行一次大重組 (Reconstitution) 的複雜問題。您可以直接查看其 data 資料夾中的 JSON 或 CSV 檔案，或者使用它提供的 Python script 來查詢特定日期的成份股。

### **3\. 使用 Python yfinance 抓取 (替代方案)**

如果您不想下載 CSV，也可以用 Python 寫一個簡單的腳本，結合維基百科來獲取。

**範例程式碼概念 (S\&P 500):**

Python

import pandas as pd

\# 1\. 讀取維基百科上的 S\&P 500 現有成份股  
table \= pd.read\_html('https://en.wikipedia.org/wiki/List\_of\_S%26P\_500\_companies')  
current\_df \= table\[0\]  
current\_tickers \= current\_df\['Symbol'\].tolist()

\# 2\. 讀取歷史變動表 (Changes)  
changes\_df \= table\[1\] \# 這通常是變動表  
\# 注意：維基百科表格格式較複雜，需要資料清理 (Data Cleaning)  
\# 邏輯：  
\# 如果您要找 2023-01-01 的名單：  
\# 從今天開始，往回遍歷變動表：  
\# \- 遇到 "Added" 的股票，表示它在當時還沒加入，所以從清單中移除。  
\# \- 遇到 "Removed" 的股票，表示它在當時還在，所以加回清單中。

### **總結建議**

| ETF / 指數 | 資料來源建議 | 備註 |
| :---- | :---- | :---- |
| **SPY, VOO** | **GitHub (fja05680/sp500)** | 資料最齊全，直接有 CSV 可下載。 |
| **QQQ** | **GitHub (jmccarrell/n100tickers)** | 專注於 Nasdaq 100，處理了年度重組的問題。 |

重要提醒：  
免費資料通常只有「股票代碼 (Tickers)」清單，不包含權重 (Weights)。如果您需要知道當時某支股票佔 ETF 的權重（例如 Apple 在 2021 年佔 SPY 多少 %），這通常需要付費購買專業數據 (如 Bloomberg, FactSet, 或 Financial Modeling Prep 的付費方案)。但如果只是做回測需要「選股池 (Universe)」，上述免費資源已經足夠。