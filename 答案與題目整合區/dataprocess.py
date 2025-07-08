import pandas as pd
import numpy as np
from scipy.stats import zscore

# 讀取數據
file_path = 'AvgDATA_00.csv'
data = pd.read_csv(file_path)

# 定義Z分數檢測異常值
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = zscore(df[column])
    outliers = np.abs(z_scores) > threshold
    return outliers

# 定義IQR方法檢測異常值
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

# 定義異常值處理函數（使用鄰近值填補）
def handle_outliers_with_neighbors(df, column, method="zscore", threshold=3):
    if method == "zscore":
        outliers = detect_outliers_zscore(df, column, threshold)
    elif method == "iqr":
        outliers = detect_outliers_iqr(df, column)
    else:
        raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")

    # 複製數據以進行操作
    fixed_data = df[column].copy()
    
    # 對異常值進行鄰近值填補
    for idx in fixed_data[outliers].index:
        # 如果是第一行或最後一行，用最鄰近的非異常值填補
        if idx == 0:
            fixed_data[idx] = fixed_data[~outliers].iloc[1]
        elif idx == len(fixed_data) - 1:
            fixed_data[idx] = fixed_data[~outliers].iloc[-2]
        else:
            # 嘗試用前後值的平均進行填補
            prev_value = fixed_data.iloc[idx - 1]
            next_value = fixed_data.iloc[idx + 1]
            fixed_data[idx] = (prev_value + next_value) / 2

    return fixed_data

# 針對數值特徵執行異常值檢測和處理
numeric_columns = ['WindSpeed(m/s)', 'Sunlight(Lux)']
processed_data = data.copy()

for col in numeric_columns:
    print(f"Processing {col} with Z-score method:")
    processed_data[col] = handle_outliers_with_neighbors(data, col, method="zscore", threshold=3)
    print(f"Finished handling {col}.")

    print(f"Processing {col} with IQR method:")
    processed_data[col] = handle_outliers_with_neighbors(data, col, method="iqr")
    print(f"Finished handling {col}.")

# 儲存處理後的數據
processed_file_path = 'AvgDATA_00.csv'
processed_data.to_csv(processed_file_path, index=False)
print(f"Processed data saved to: {processed_file_path}")





# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 讀取數據
# file_path = 'AvgDATA_00.csv'
# data = pd.read_csv(file_path)

# # 定義檢測離群值的函數（使用IQR方法）
# def detect_outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)  # 第25百分位
#     Q3 = df[column].quantile(0.75)  # 第75百分位
#     IQR = Q3 - Q1  # 四分位距
#     lower_bound = Q1 - 1.5 * IQR  # 下界
#     upper_bound = Q3 + 1.5 * IQR  # 上界
#     outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
#     return outliers, lower_bound, upper_bound

# # 連續變數列表
# continuous_columns = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']

# # 建立圖表文件夾
# output_folder = '圖表'
# import os
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 繪製每個欄位的單獨圖表
# for col in continuous_columns:
#     outliers, lower, upper = detect_outliers_iqr(data, col)
    
#     # 打印邊界與離群值數量
#     print(f"Feature: {col}")
#     print(f"  Number of outliers: {len(outliers)}")
#     print(f"  Lower Bound: {lower:.2f}, Upper Bound: {upper:.2f}")
#     print("")

#     # 繪製圖表
#     plt.figure(figsize=(8, 6))
    
#     # Boxplot
#     sns.boxplot(x=data[col])
#     plt.title(f"Boxplot of {col}")
#     plt.xlabel(col)
#     plt.tight_layout()
    
#     # 儲存箱線圖
#     boxplot_file_path = os.path.join(output_folder, f"{col.replace('/', '_')}_boxplot.png")
#     plt.savefig(boxplot_file_path)
#     print(f"Saved boxplot for {col} to {boxplot_file_path}")
#     plt.close()
    
#     # Histogram
#     plt.figure(figsize=(8, 6))
#     sns.histplot(data[col], bins=30, kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.xlabel(col)
#     plt.tight_layout()
    
#     # 儲存直方圖
#     histplot_file_path = os.path.join(output_folder, f"{col.replace('/', '_')}_histogram.png")
#     plt.savefig(histplot_file_path)
#     print(f"Saved histogram for {col} to {histplot_file_path}")
#     plt.close()

# print("All individual plots have been saved.")
