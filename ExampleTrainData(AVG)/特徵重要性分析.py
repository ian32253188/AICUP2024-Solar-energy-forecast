import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 讀取CSV文件
data = pd.read_csv(r'C:\Users\ian32\Downloads\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_00.csv')

# 特徵列表
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target = 'Power(mW)'

# 準備數據
X = data[features]
y = data[target]

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 獲取特徵重要性
importances = model.feature_importances_
feature_names = X.columns

# 繪製特徵重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('特徵重要性')
plt.title('特徵重要性分析')
plt.show()

# 選擇重要特徵（例如，選擇重要性前3的特徵）
important_features = [feature for feature, importance in zip(feature_names, importances) if importance > 0.1]

# 使用重要特徵重新準備數據
X_important = data[important_features]

# 分割數據集為訓練集和測試集
X_train_important, X_test_important, y_train, y_test = train_test_split(X_important, y, test_size=0.2, random_state=42)

# 重新訓練模型
model_important = RandomForestRegressor(n_estimators=100, random_state=42)
model_important.fit(X_train_important, y_train)

# 評估模型性能
y_pred = model_important.predict(X_test_important)
mse = mean_squared_error(y_test, y_pred)
print(f'均方誤差: {mse}')