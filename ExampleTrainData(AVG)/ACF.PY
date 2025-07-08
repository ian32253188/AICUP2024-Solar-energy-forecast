import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 設置字體
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 讀取CSV文件
data = pd.read_csv(r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00.csv')

# 特徵列表
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']

# 繪製每個特徵的自相關函數（ACF）圖
for feature in features:
    plot_acf(data[feature], lags=50)
    plt.xlabel('滯後數')
    plt.ylabel('自相關')
    plt.title(f'{feature} 的自相關函數（ACF）')
    plt.show()