import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files
df_original = pd.read_csv('AvgDATA_00.csv')
df_updated = pd.read_csv('AVGDATA_00_光照度與發電量關係修正_更新.CSV')

# Merge dataframes on Serial
df_merged = pd.merge(df_original, df_updated, on='Serial', suffixes=('_original', '_updated'))

# Calculate changes in Sunlight and Power
df_merged['Sunlight_change'] = df_merged['Sunlight(Lux)_updated'] - df_merged['Sunlight(Lux)_original']
df_merged['Power_change'] = df_merged['Power(mW)_updated'] - df_merged['Power(mW)_original']

# Plot changes in Sunlight
plt.figure(figsize=(14, 6))
plt.plot(df_merged['Serial'], df_merged['Sunlight_change'], label='Sunlight Change')
plt.xlabel('Serial')
plt.ylabel('Change in Sunlight (Lux)')
plt.title('Change in Sunlight (Lux) between Original and Updated Data')
plt.legend()
plt.show()

# Plot changes in Power
plt.figure(figsize=(14, 6))
plt.plot(df_merged['Serial'], df_merged['Power_change'], label='Power Change', color='orange')
plt.xlabel('Serial')
plt.ylabel('Change in Power (mW)')
plt.title('Change in Power (mW) between Original and Updated Data')
plt.legend()
plt.show()