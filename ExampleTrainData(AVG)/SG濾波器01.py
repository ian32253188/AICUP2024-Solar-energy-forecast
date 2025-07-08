import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 讀取CSV文件
data = pd.read_csv(r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00.csv')

# 特徵列表
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']

# 設置窗口大小和多項式階數
window_length = 11  # 必須是奇數
polyorder = 2

# 繪製每個特徵的原始數據和平滑後的數據
for feature in features:
    y = data[feature].values
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    
    plt.figure()
    plt.plot(y, label='原始數據')
    plt.plot(y_smooth, label='平滑後數據', color='red')
    plt.title(feature)
    plt.legend()
    plt.show()