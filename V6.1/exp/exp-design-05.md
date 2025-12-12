這是一個非常切中要害的實驗方向。根據華爾街的量化研究（例如 *Liquidity Cascades* 或 *Passive Flows* 理論），每個月初與長假前，退休基金（401k）與被動型基金（ETF）會有結構性的「自動買入」資金流入。

這種 **「只買不賣」** 的資金流，對於我們 **「做空 (Short Open)」** 的 Gap Strategy 來說是極大的逆風。如果在這些日子硬要做空，很容易遇到「怎麼跌不下去」甚至「尾盤被買盤軋空」的情況。

以下是為您設計的 **EXP-V6.1-05 實驗設計文件**，您可以直接將其存為 `V6.1/exp/exp-design-05.md` 並開始實作。

-----

### **EXP-V6.1-05：日曆效應與結構性資金流驗證 (Calendar Effects & Structural Flows)**

#### **1. 實驗背景與假設 (Hypothesis)**

  * **背景：** `V6.1/survey.md` 指出「方向五：日曆效應」可能影響策略表現。現有策略 (Strategy A) 是無差別的做空跳空缺口。
  * **假設 (Hypothesis)：** 在 **「月初 (Turn of the Month, TOTM)」** 與 **「節假日前夕 (Pre-Holiday)」** 這些特定窗口期，市場存在結構性的多頭資金流（Structural Long Flow）。
      * **推論：** 在這些日子執行 Gap Fade (做空) 策略，其 **勝率 (Win Rate)** 與 **盈虧比 (P/L Ratio)** 應顯著低於「普通日子」。
      * **目標：** 若能證實此假設，我們應在這些日子 **「暫停交易」** 或 **「提高進場門檻」**，以避開法人買盤的輾壓。

#### **2. 實驗變數定義 (Variables)**

我們將時間軸上的每一天劃分為兩類標籤：

1.  **TOTM (月初效應期)：**

      * **定義：** 每個月的 **最後 1 個交易日** 與 **下個月的前 3 個交易日** (T-1 to T+3)。
      * *邏輯：這是 401k 自動提撥與基金再平衡 (Rebalancing) 的主要發生期。*

2.  **Pre-Holiday (長假前夕)：**

      * **定義：** 美股休市日 (如聖誕節、勞動節、感恩節等) 的 **前 1 個交易日**。
      * *邏輯：節前交易量縮，且空頭傾向回補過節，賣壓減輕，容易緩漲。*

3.  **Normal Days (普通日)：**

      * 除去上述兩者之外的所有交易日。

#### **3. 實驗流程與評估指標**

**資料範圍：** 2020-01-01 至 2025-12-31 (完整 6 年，涵蓋多空循環)。
**測試對象：** 針對 **2025 Final Asset Pool** (主力池) 與 **Toxic Asset Pool** (有毒池) 分別統計。

**步驟：**

1.  **標記日期：** 為每一天標記 `is_totm` 與 `is_pre_holiday`。
2.  **模擬交易：** 對每一天執行標準策略 (Gap \> 0.5% Sell Open, MOC Close)。
3.  **分組統計：** 比較以下三組的績效：
      * **Group TOTM:** 僅在月初窗口交易的績效。
      * **Group Holiday:** 僅在節前交易的績效。
      * **Group Normal:** 在普通日子交易的績效。

**預期結果 (Success Metric)：**
若 **Group Normal** 的 Sharpe Ratio 或 Win Rate **顯著高於** TOTM/Holiday 組，則證明「避開這些日子」能提升整體策略績效。

-----

### **4. Python 實作邏輯 (EXP-V6.1-05)**

建議建立 `V6.1/exp/exp-05.py`。為了確保實驗獨立性，我們使用 `pandas.tseries.holiday` 來精確處理美股假期。

```python
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# ==========================================
# 1. Helper: 日曆效應標記生成器
# ==========================================
def get_calendar_flags(start_date, end_date):
    """
    生成一個 DataFrame，Index 為日期，包含 is_totm, is_pre_holiday 標記
    """
    # 建立美股交易日曆
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    df = pd.DataFrame(index=dates)
    df['is_totm'] = False
    df['is_pre_holiday'] = False
    
    # --- 標記 TOTM (Turn of the Month) ---
    # 邏輯：月底最後1天 + 月初前3天
    # 技巧：將日期分組為 Month，然後找該組的 iloc
    df['year_month'] = df.index.to_period('M')
    
    for ym, group in df.groupby('year_month'):
        # 該月交易日列表
        days = group.index
        if len(days) < 4: continue # 防禦性檢查
        
        # 月底最後 1 天
        df.loc[days[-1], 'is_totm'] = True
        # 下個月初前 3 天 (這裡需要跨月處理，簡化法：直接對所有交易日 iterate 判斷 rank)
    
    # 更穩健的 TOTM 算法：計算每個交易日在當月是第幾天，或倒數第幾天
    # 重算 month_rank
    df['day_in_month'] = df.groupby('year_month').cumcount() + 1
    df['days_in_month_total'] = df.groupby('year_month')['day_in_month'].transform('max')
    df['day_reverse_rank'] = df['days_in_month_total'] - df['day_in_month']
    
    # TOTM Rule: Month End (last 1 day) or Month Start (1st, 2nd, 3rd day)
    # day_in_month 1, 2, 3 OR day_reverse_rank 0 (last day)
    mask_totm = (df['day_in_month'].isin([1, 2, 3])) | (df['day_reverse_rank'] == 0)
    df['is_totm'] = mask_totm

    # --- 標記 Pre-Holiday ---
    # 取得所有假期
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    
    # 假期的前一個交易日
    # 邏輯：若明天是假期，今天就是 Pre-Holiday
    # 但因為我們只有交易日 Index，比較簡單的做法是：看兩個交易日之間是否間隔 > 1 天 (週末除外)
    # 或是直接用 shift
    # 這裡採用精確做法：Shift index，若 next trading day > current day + 1 (且非週末)，通常難判斷
    # 簡單做法：直接對比 holidays
    for holiday in holidays:
        # 找 holiday 前一天的那個交易日
        # 透過 searchsorted 找插入點
        loc = dates.searchsorted(holiday)
        if loc > 0:
            # loc 是 holiday 之後的第一個交易日或是 holiday 本身(如果沒休市)
            # 但 holiday 通常休市，所以 dates 裡面沒有 holiday
            # dates[loc-1] 就是假期前的最後一個交易日
            prev_trade_day = dates[loc-1]
            
            # 檢查兩者是否接近 (例如差距小於 5 天，避免聖誕節跨年太久)
            if (holiday - prev_trade_day).days <= 4: 
                df.loc[prev_trade_day, 'is_pre_holiday'] = True

    return df[['is_totm', 'is_pre_holiday']]

# ==========================================
# 2. 實驗主程式
# ==========================================
def run_exp_05_calendar_effects():
    print("Starting EXP-05: Calendar Effects Analysis...")
    
    # A. 準備資料 (假設已有 OHLCV，或是重新下載)
    # 為了簡化，這裡假設我們已經有一個函數 get_historical_data() 
    # 或是讀取先前的 csv
    # ...
    
    # B. 準備日期標籤
    calendar_df = get_calendar_flags('2020-01-01', '2025-12-31')
    
    # C. 模擬策略與統計
    results = []
    
    # 遍歷股票 (從 asset_pool 讀取)
    tickers = load_asset_pool() # 自定義函數
    
    for ticker in tickers:
        df = fetch_data(ticker) # 自定義函數，取得日線
        
        # 合併日曆特徵
        df = df.join(calendar_df, how='inner')
        
        # 計算策略訊號 (Gap Fade)
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        
        # 訊號：Gap > 0.5%
        df['signal'] = df['gap_pct'] > 0.005
        
        # 計算單日報酬 (做空: Open - Close)
        # 假設 MOC 離場
        df['ret'] = (df['Open'] - df['Close']) / df['Open']
        
        # 只看有訊號的日子
        trade_days = df[df['signal']].copy()
        
        if len(trade_days) == 0: continue
            
        # D. 分組標記
        trade_days['type'] = 'Normal'
        trade_days.loc[trade_days['is_totm'], 'type'] = 'TOTM'
        trade_days.loc[trade_days['is_pre_holiday'], 'type'] = 'Holiday'
        # 注意：TOTM 和 Holiday 可能重疊，優先級可自行定義，這裡假設 Holiday 影響更大
        
        for t_type, group in trade_days.groupby('type'):
            results.append({
                'Ticker': ticker,
                'Type': t_type,
                'Trade_Count': len(group),
                'Win_Rate': (group['ret'] > 0).mean(),
                'Avg_Return': group['ret'].mean(),
                'Total_Return': group['ret'].sum()
            })
            
    # E. 輸出報告
    res_df = pd.DataFrame(results)
    print(res_df.groupby('Type')[['Win_Rate', 'Avg_Return']].mean())
    # 預期：Normal 的 Win_Rate 應該 > TOTM 和 Holiday
    
    # 儲存
    res_df.to_csv('V6.1/exp/output/exp_05_calendar_effect.csv')

```

#### **5. 下一步行動**

1.  執行此實驗，觀察 **TOTM** 與 **Holiday** 組別的 `Win Rate` 與 `Avg Return` 是否顯著低於 **Normal** 組。
2.  **若驗證為真 (Positive Result)：**
      * 在 `V6.1/resource/trading_calendar_filters.json` (擬建) 中建立過濾規則。
      * 在實盤腳本 `daily_gap_signal_generator.py` 中加入檢查：`if is_totm or is_pre_holiday: skip_trading = True` (或提高 Gap 門檻)。

這個實驗將幫助您避開那些「被機構資金輾壓」的無謂虧損日。