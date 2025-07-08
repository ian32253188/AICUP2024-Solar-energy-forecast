import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('AvgDATA_00.csv')

# Calculate Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(14, 12))  # Increase the figure size
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix')

# Rotate the y-axis labels to horizontal
plt.yticks(rotation=0)

plt.show()