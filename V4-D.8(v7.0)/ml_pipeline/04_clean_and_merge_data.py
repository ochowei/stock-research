import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
features_path = os.path.join(script_dir, 'features_X_T-1.parquet')
labels_path = os.path.join(script_dir, 'labels_Y.parquet')
output_path = os.path.join(script_dir, 'model_ready_dataset.parquet')
uncleaned_output_path = os.path.join(script_dir, 'model_merged_uncleaned.parquet')

# --- Logika ---

# 1. 讀取特徵和標籤數據
print("讀取特徵數據...")
features_df = pd.read_parquet(features_path)
print(f"特徵數據讀取完畢，共 {len(features_df)} 筆。")

# --- 特徵 (X) NaN 診斷報告 ---
print("\n--- 特徵 (X) NaN 診斷報告 ---")
total_rows_x = len(features_df)
print(f"總行數: {total_rows_x}")
nan_in_features = features_df.isnull().sum()
nan_in_features = nan_in_features[nan_in_features > 0]
if not nan_in_features.empty:
    print("發現 NaN 的特徵 (僅列出 > 0%):")
    nan_in_features.sort_values(ascending=False, inplace=True)
    for col, count in nan_in_features.items():
        percentage = (count / total_rows_x) * 100
        print(f"- {col}: {count} NaN ({percentage:.2f}%)")
else:
    print("特徵數據中無 NaN。")
print("--- 報告結束 ---\n")


print("讀取標籤數據...")
labels_df = pd.read_parquet(labels_path)
print(f"標籤數據讀取完畢，共 {len(labels_df)} 筆。")

# --- 標籤 (Y) NaN 診斷報告 ---
print("\n--- 標籤 (Y) NaN 診斷報告 ---")
total_rows_y = len(labels_df)
print(f"總行數: {total_rows_y}")
nan_in_labels = labels_df['Y'].isnull().sum()
if nan_in_labels > 0:
    percentage = (nan_in_labels / total_rows_y) * 100
    print(f"- Y: {nan_in_labels} NaN ({percentage:.2f}%)")
else:
    print("標籤數據中無 NaN。")
print("--- 報告結束 ---\n")


# 2. 合併數據 (Inner Join)
print("合併特徵與標籤數據...")
# The execution plan specifies that both files are indexed by ('asset', 'T-1_timestamp')
merged_df = pd.merge(features_df, labels_df, left_index=True, right_index=True, how='inner')
print(f"數據合併完畢，共 {len(merged_df)} 筆。")

# 儲存未清理的合併數據
print(f"儲存未清理的合併數據至 {uncleaned_output_path}...")
merged_df.to_parquet(uncleaned_output_path)
print("數據儲存完畢。")

# --- 合併後 (Merged) NaN 診斷報告 ---
print("\n--- 合併後 (Merged) NaN 診斷報告 ---")
total_rows_merged = len(merged_df)
print(f"合併後總行數: {total_rows_merged}")
rows_with_nan = merged_df.isnull().any(axis=1).sum()
if total_rows_merged > 0:
    percentage = (rows_with_nan / total_rows_merged) * 100
    print(f"即將因 NaN 刪除的行數: {rows_with_nan} ({percentage:.2f}%)")
else:
    print("合併後的數據集為空。")
print("--- 報告結束 ---\n")


# 3. 應用清理規則 (規則二：未成交)
print("應用清理規則：刪除 'NO_FILL' 樣本...")
# Check if the 'Fill_Status' column exists
if 'Fill_Status' in merged_df.columns:
    initial_rows = len(merged_df)
    cleaned_df = merged_df[merged_df['Fill_Status'] != 'NO_FILL']
    rows_removed = initial_rows - len(cleaned_df)
    print(f"清理完畢，共刪除 {rows_removed} 筆 'NO_FILL' 樣本。")
else:
    print("警告：'Fill_Status' 欄位不存在，無法應用清理規則。")
    cleaned_df = merged_df


print(f"清理 'NaN' 特徵值... (目前筆數: {len(cleaned_df)})")
# 移除任何欄位 (包含 X 或 Y) 中含有 NaN 的整筆樣本
# 你的標籤 Y 也可能因為 vol=0 而產生 NaN
initial_rows = len(cleaned_df)
cleaned_df.dropna(inplace=True) 
rows_removed = initial_rows - len(cleaned_df)
print(f"清理完畢，共刪除 {rows_removed} 筆含有 'NaN' 的樣本。")


# 4. 儲存產出
print(f"儲存清理後的數據至 {output_path}...")
cleaned_df.to_parquet(output_path)
print("數據儲存完畢。")

print("\n腳本執行成功！")
