import pandas as pd
import numpy as np
from scipy import stats

# 讀取CSV檔案
df = pd.read_csv(r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00_光照度與發電量關係修正.csv')

# 定義需要處理的欄位
columns_to_check = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']

# 使用Z-score方法檢測異常值，並用前後值插值替換異常值
for column in columns_to_check:
    z_scores = np.abs(stats.zscore(df[column]))
    df.loc[z_scores > 3, column] = np.nan

# 使用插值方法填補NaN值
df.interpolate(method='linear', inplace=True)

# 新增裝置號欄位
df['Device_ID'] = df['Serial'].astype(str).str[-2:].astype(int)

# 新增日期和時間特徵欄位
df['Date'] = df['Serial'].astype(str).str[4:8]
df['Time'] = df['Serial'].astype(str).str[8:12]

# 新增Sunlight(Lux)與Temperature(°C)相乘的新特徵欄位
df['Sunlight_Temperature'] = df['Sunlight(Lux)'] * df['Temperature(°C)']

# 保持指定欄位的小數點位數
df[columns_to_check] = df[columns_to_check].round(2)
df['Sunlight_Temperature'] = df['Sunlight_Temperature'].round(2)

# 檢查結果
print(df.head())

# 將修改後的資料儲存回CSV檔案
df.to_csv('AVGDATA_00_光照度與發電量關係修正_更新.CSV', index=False)