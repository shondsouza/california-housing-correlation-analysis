import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing_data = fetch_california_housing(as_frame=True)
data = housing_data.frame

# Show first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Compute the correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Create a pair plot for selected features (to keep it readable)
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'MedHouseVal']
sns.pairplot(data[selected_features], diag_kind='kde', corner=True)
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()
