#%%

import sys
import tensorflow as tf

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

import numpy as np
import pandas as pd
import os

#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數

#載入訓練資料
DataName = r'C:\Users\ian32\OneDrive\文件\研究所課程-碩一\週一＿ai\期末報告\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)\AvgDATA_000.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

#迴歸分析 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
#(發電量)
Regression_X_train = SourceData[['Temperature(°C)','Sunlight(Lux)']].values
Regression_y_train = SourceData[['Power(mW)']].values

#LSTM 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
AllOutPut = SourceData[['Temperature(°C)','Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

#設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum,len(AllOutPut_MinMax)):
  X_train.append(AllOutPut_MinMax[i-LookBackNum:i, :])
  y_train.append(AllOutPut_MinMax[i, :])


X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
#(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
X_train = np.reshape(X_train,(X_train.shape [0], X_train.shape [1], 2))


#%%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))







#%%
#============================建置&訓練「LSTM模型」============================
#建置LSTM模型
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# ...existing code...

# 定義學習率調整函數
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# 建立 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 2)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=2))

model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型並保存歷史記錄
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[LearningRateScheduler(lr_schedule)])

# 繪製損失曲線
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 繪製學習率曲線
plt.figure(figsize=(12, 6))
plt.plot(history.history['lr'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.show()

#保存模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save('WheatherLSTM_'+NowDateTime+'.h5')
print('Model Saved')




#%%
#============================建置&訓練「回歸模型」========================

#開始迴歸分析(對發電量做迴歸)
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

#儲存回歸模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
joblib.dump(RegressionModel, 'WheatherRegression_'+NowDateTime)

#取得截距
print('截距: ',RegressionModel.intercept_)

#取得係數
print('係數 : ', RegressionModel.coef_)

#取得R平方
print('R squared: ',RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))







#%%
#============================預測數據============================
#載入模型
regressor = load_model('WheatherLSTM_2024-11-23T19_26_55Z.h5')
Regression = joblib.load('WheatherRegression_2024-11-23T19_26_55Z')


#載入測試資料
DataName = os.getcwd()+r'\ExampleTestData\upload(no answer).csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

inputs = [] #存放參考資料
PredictOutput = [] #存放預測值(天氣參數)
PredictPower = [] #存放預測值(發電量) 

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode

  DataName = os.getcwd()+'\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[['Temperature(°C)','Sunlight(Lux)']].values
  
  inputs = []#重置存放參考資料

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      TempData = ReferData[DaysCount].reshape(1,-1)
      TempData = LSTM_MinMaxModel.transform(TempData)
      inputs.append(TempData)

  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs.append(PredictOutput[i-1].reshape(1,2))

    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
    #Reshaping
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 2))
    
    predicted = regressor.predict(NewTest)
    PredictOutput.append(predicted)
    PredictPower.append(np.round(Regression.predict(predicted),2).flatten())
  
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48

#寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictPower, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv('output_Regression+LSTM_Small_20241123.csv', index=False) 
print('Output CSV File Saved')

# %%
