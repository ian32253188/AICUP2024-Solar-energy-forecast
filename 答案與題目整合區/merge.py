import pandas as pd

# 載入兩個檔案
file1_path = r"答案與題目整合區\output_Regression+LSTM_Small_20241123.csv"  # 第一個檔案的路徑
file2_path = r"答案與題目整合區\upload(no answer).csv"  # 第二個檔案的路徑

import chardet

# 檢測檔案的編碼
with open(file1_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']

# 用檢測出的編碼讀取檔案
file1 = pd.read_csv(file1_path, encoding=encoding, header=None)

# 讀取第一個檔案，移除標題行並保留有效數值
file1_cleaned = file1.iloc[1:].reset_index(drop=True)  # 移除第一列
file1_cleaned.columns = ["Value"]

# 將數據轉換為數字類型
file1_cleaned["Value"] = pd.to_numeric(file1_cleaned["Value"], errors='coerce')

# 將負值替換為零
file1_cleaned["Value"] = file1_cleaned["Value"].apply(lambda x: max(x, 0))

# 讀取第二個檔案，移除標題行並保留序號
file2 = pd.read_csv(file2_path, encoding="utf-8", header=None)
file2_cleaned = file2.iloc[1:].reset_index(drop=True)  # 移除第一列
file2_cleaned.columns = ["Timestamp", "Answer"]

# 合併兩個檔案，將第一個檔案的數值填入第二個檔案的 "Answer" 欄位
file2_cleaned["Answer"] = file1_cleaned["Value"]

# 將負值替換為零
file2_cleaned["Answer"] = file2_cleaned["Answer"].apply(lambda x: max(x, 0))

# 最終格式化，僅保留序號與答案
final_output = file2_cleaned[["Timestamp", "Answer"]]

# 將合併結果另存為 UTF-8 無 BOM 格式並使用 Unix 換行符號
output_path = "merged_output01.csv"  # 輸出的檔案路徑
final_output.to_csv(output_path, index=False, header=False, encoding="utf-8", lineterminator="\n")

print(f"合併完成，結果已儲存於: {output_path}")