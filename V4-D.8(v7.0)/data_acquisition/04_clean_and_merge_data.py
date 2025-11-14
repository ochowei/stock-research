import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
features_path = os.path.join(script_dir, 'features_X_T-1.parquet')
labels_path = os.path.join(script_dir, 'labels_Y.parquet')
output_path = os.path.join(script_dir, 'model_ready_dataset.parquet')

# --- Logika ---

# 1. 讀取特徵和標籤數據
print("讀取特徵數據...")
features_df = pd.read_parquet(features_path)
print(f"特徵數據讀取完畢，共 {len(features_df)} 筆。")

print("讀取標籤數據...")
labels_df = pd.read_parquet(labels_path)
print(f"標籤數據讀取完畢，共 {len(labels_df)} 筆。")

# 2. 合併數據 (Inner Join)
print("合併特徵與標籤數據...")
# The execution plan specifies that both files are indexed by ('asset', 'T-1_timestamp')
merged_df = pd.merge(features_df, labels_df, left_index=True, right_index=True, how='inner')
print(f"數據合併完畢，共 {len(merged_df)} 筆。")


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

# 4. 儲存產出
print(f"儲存清理後的數據至 {output_path}...")
cleaned_df.to_parquet(output_path)
print("數據儲存完畢。")

print("\n腳本執行成功！")
