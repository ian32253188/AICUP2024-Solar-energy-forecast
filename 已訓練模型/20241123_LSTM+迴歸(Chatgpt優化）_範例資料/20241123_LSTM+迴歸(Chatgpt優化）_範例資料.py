#%%
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set parameters
LookBackNum = 12  # Number of previous data points to look back
ForecastNum = 48  # Number of forecast steps

# Load training data
DataName = os.getcwd() + '\ExampleTrainData(AVG)\AvgDATA_00.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

# Prepare regression and LSTM datasets
Regression_X_train = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
Regression_y_train = SourceData[['Power(mW)']].values
AllOutPut = Regression_X_train

# Normalize data
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train, y_train = [], []
for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i - LookBackNum:i, :])
    y_train.append(AllOutPut_MinMax[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

#%%
# ============================ Improved LSTM Model ============================
regressor = Sequential()

# Add LSTM layers with Dropout and Batch Normalization
regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 5)))
regressor.add(Dropout(0.3))
regressor.add(BatchNormalization())

regressor.add(LSTM(units=64, return_sequences=False))
regressor.add(Dropout(0.3))
regressor.add(BatchNormalization())

# Output layer
regressor.add(Dense(units=5))
regressor.compile(optimizer='adam', loss='huber_loss')  # Changed to Huber Loss

# Train the model with validation split
history = regressor.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Save LSTM model
current_datetime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save(f'WeatherLSTM_{current_datetime}.h5')
print('LSTM Model Saved')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'LSTM_Loss_{current_datetime}.png')
plt.show()

#%%
# ============================ Linear Regression Model ============================
# Fit regression model
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

# Save regression model
joblib.dump(RegressionModel, f'WeatherRegression_{current_datetime}.joblib')

# Print regression statistics
print('Intercept:', RegressionModel.intercept_)
print('Coefficients:', RegressionModel.coef_)
print('R-squared:', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

# Plot feature importance
coefficients = RegressionModel.coef_.flatten()
features = ['WindSpeed', 'Pressure', 'Temperature', 'Humidity', 'Sunlight']
plt.figure(figsize=(8, 5))
plt.bar(features, coefficients, color='blue')
plt.title('Linear Regression Feature Importance')
plt.ylabel('Coefficient Value')
plt.savefig(f'Regression_FeatureImportance_{current_datetime}.png')
plt.show()

#%%
# ============================ Prediction and Evaluation ============================
# Load models
regressor = load_model(f'WeatherLSTM_{current_datetime}.h5')
Regression = joblib.load(f'WeatherRegression_{current_datetime}.joblib')

# Load test data
DataName = os.getcwd() + r'\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
EXquestion = SourceData[['序號']].values

PredictOutput = []  # Store weather parameter predictions
PredictPower = []  # Store power predictions

count = 0
while count < len(EXquestion):
    print('Count:', count)
    LocationCode = int(EXquestion[count])
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = '0' + strLocationCode

    DataName = os.getcwd() + '\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_' + strLocationCode + '.csv'
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
    inputs = []

    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = LSTM_MinMaxModel.transform(TempData)
            inputs.append(TempData)

    for i in range(ForecastNum):
        if i > 0:
            inputs.append(PredictOutput[i - 1].reshape(1, 5))
        X_test = np.array(inputs[i:i + LookBackNum])
        X_test = np.reshape(X_test, (1, X_test.shape[0], 5))

        predicted = regressor.predict(X_test)
        PredictOutput.append(predicted)
        PredictPower.append(np.round(Regression.predict(predicted), 2).flatten())

    count += 48

# Save predictions to CSV
df = pd.DataFrame(PredictPower, columns=['Predicted Power (mW)'])
output_csv_name = f'PredictionOutput_{current_datetime}.csv'
df.to_csv(output_csv_name, index=False)
print('Output CSV File Saved:', output_csv_name)

# Evaluate predictions
y_true = Regression_y_train[:len(PredictPower)]  # Dummy true values for demonstration
mse = mean_squared_error(y_true, PredictPower)
r2 = r2_score(y_true, PredictPower)
print(f'Mean Squared Error: {mse}')
print(f'R-Squared: {r2}')

# %%
