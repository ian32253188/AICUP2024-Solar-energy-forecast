import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files
df_original = pd.read_csv('AvgDATA_000.csv')
df_new = pd.read_csv('9600筆.csv', header=None, names=['Serial', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)'])

# Extract year and month from Serial
df_original['YearMonth'] = df_original['Serial'].astype(str).str[:6]
df_new['YearMonth'] = df_new['Serial'].astype(str).str[:6]

# Function to detect anomalies
def detect_anomalies(df):
    anomalies = {}
    for column in df.columns[1:-1]:  # Skip the Serial and YearMonth columns
        mean = df[column].mean()
        std = df[column].std()
        anomalies[column] = df[(df[column] < mean - 3 * std) | (df[column] > mean + 3 * std)]
    return anomalies

# Detect anomalies in both datasets
anomalies_original = detect_anomalies(df_original)
anomalies_new = detect_anomalies(df_new)

# Calculate anomaly proportions
proportions_original = {column: len(anomaly_df) / len(df_original) for column, anomaly_df in anomalies_original.items()}
proportions_new = {column: len(anomaly_df) / len(df_new) for column, anomaly_df in anomalies_new.items()}

# Calculate the number of anomalies
num_anomalies_original = {column: len(anomaly_df) for column, anomaly_df in anomalies_original.items()}
num_anomalies_new = {column: len(anomaly_df) for column, anomaly_df in anomalies_new.items()}

# Print the number of anomalies
print("Number of anomalies in Original Data:")
for column, count in num_anomalies_original.items():
    print(f"{column}: {count}")

print("\nNumber of anomalies in Data 9600:")
for column, count in num_anomalies_new.items():
    print(f"{column}: {count}")

# Plot anomaly proportions
columns = list(proportions_original.keys())
proportions_original_values = list(proportions_original.values())
proportions_new_values = list(proportions_new.values())

x = range(len(columns))

plt.figure(figsize=(14, 6))
plt.bar(x, proportions_original_values, width=0.4, label='Original Data', color='green', align='center')
plt.bar(x, proportions_new_values, width=0.4, label='Data 9600', color='red', align='edge')
plt.xlabel('Columns')
plt.ylabel('Anomaly Proportion')
plt.title('Anomaly Proportion Comparison')
plt.xticks(x, columns, rotation=0)  # Set rotation to 0 for horizontal text
plt.legend()

# Annotate the number of anomalies
for i, (original, new) in enumerate(zip(num_anomalies_original.values(), num_anomalies_new.values())):
    plt.text(i - 0.2, proportions_original_values[i] + 0.005, str(original), color='green', ha='center')
    plt.text(i + 0.2, proportions_new_values[i] + 0.005, str(new), color='red', ha='center')

plt.show()