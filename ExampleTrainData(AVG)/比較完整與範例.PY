import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files
df_updated = pd.read_csv('AvgDATA_00.csv')
df_original = pd.read_csv('AvgDATA_000.csv')

# Compare the number of rows
num_rows_updated = len(df_updated)
num_rows_original = len(df_original)
num_rows_difference = num_rows_updated - num_rows_original

print(f'Number of rows in AvgDATA_00.csv: {num_rows_updated}')
print(f'Number of rows in AvgDATA_000.csv: {num_rows_original}')
print(f'Difference in number of rows: {num_rows_difference}')

# Plot distribution of Sunlight
plt.figure(figsize=(14, 6))
plt.hist(df_updated['Sunlight(Lux)'], bins=50, alpha=0.5, label='Updated Sunlight')
plt.hist(df_original['Sunlight(Lux)'], bins=50, alpha=0.5, label='Original Sunlight')
plt.xlabel('Sunlight (Lux)')
plt.ylabel('Frequency')
plt.title('Distribution of Sunlight (Lux)')
plt.legend()
plt.show()

# Plot distribution of Power
plt.figure(figsize=(14, 6))
plt.hist(df_updated['Power(mW)'], bins=50, alpha=0.5, label='Updated Power')
plt.hist(df_original['Power(mW)'], bins=50, alpha=0.5, label='Original Power')
plt.xlabel('Power (mW)')
plt.ylabel('Frequency')
plt.title('Distribution of Power (mW)')
plt.legend()
plt.show()