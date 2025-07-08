import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 讀取CSV文件
data = pd.read_csv(r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00.csv')

# 特徵列表
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target = 'Power(mW)'

# 準備數據
X = data[features]
y = data[target]

# 設置光照度感測計的最大值
max_lux = 117758.2

# 過濾出光照度未達到最大值的數據
filtered_data = data[data['Sunlight(Lux)'] < max_lux]

# 準備迴歸分析的數據
X_filtered = filtered_data[['Sunlight(Lux)']]
y_filtered = filtered_data[target]

# 訓練線性迴歸模型
reg = LinearRegression()
reg.fit(X_filtered, y_filtered)

# 使用迴歸模型預測光照度達到最大值時的發電量
predicted_power_at_max_lux = reg.predict(np.array([[max_lux]]))[0]

# 修正光照度達到最大值時的數據
data['Sunlight(Lux)'] = data.apply(
    lambda row: row['Sunlight(Lux)'] if row['Sunlight(Lux)'] < max_lux else max_lux * (row[target] / predicted_power_at_max_lux),
    axis=1
)

# 保存修正後的數據到新的CSV文件
output_file_path = r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00_corrected.csv'
data.to_csv(output_file_path, index=False)

print(f'修正後的數據已保存到 {output_file_path}')