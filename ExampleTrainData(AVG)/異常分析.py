import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('AvgDATA_00.csv')

# Extract year and month from Serial
df['YearMonth'] = df['Serial'].astype(str).str[:6]

# Function to detect anomalies
def detect_anomalies(df):
    anomalies = {}
    for column in df.columns[1:-1]:  # Skip the Serial and YearMonth columns
        mean = df[column].mean()
        std = df[column].std()
        anomalies[column] = df[(df[column] < mean - 3 * std) | (df[column] > mean + 3 * std)]
    return anomalies

anomalies = detect_anomalies(df)

# Visualize anomalies
for column, anomaly_df in anomalies.items():
    if not anomaly_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df['YearMonth'], df[column], label='Normal values')
        plt.scatter(anomaly_df['YearMonth'], anomaly_df[column], color='red', label='Anomalies')
        plt.xlabel('YearMonth (YYYYMM)')
        plt.ylabel(column)
        plt.title(f'{column} Anomaly Detection')
        plt.legend()
        plt.show()