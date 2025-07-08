import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 讀取資料
data = pd.read_csv('ExampleTrainData(AVG)\AvgDATA_00_光照度與發電量關係修正.csv')

# 1. 檢查並處理缺失值
data.fillna(method='ffill', inplace=True)

# 2. 處理異常值
for column in data.columns[1:]:
    mean = np.mean(data[column])
    std = np.std(data[column])
    data = data[(data[column] > mean - 3 * std) & (data[column] < mean + 3 * std)]


# 4. 特徵工程
data['Temp_Sunlight'] = data['Temperature(°C)'] * data['Sunlight(Lux)']

# 保存處理後的數據
data.to_csv('processed_AvgDATA_00.csv', index=False)