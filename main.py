# main.py - Regression Model to predict house pricing based on features which is trained using California Housing dataset.

# importing required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# fetching dataset and storing it in df

df_pre_bunch = datasets.fetch_california_housing(as_frame=True)
df = df_pre_bunch.frame

# Uncomment one next line if you want to create a csv file of dataset in same directory as this file
# df.to_csv('california_housing.csv')

# initial inspection of dataset

print('California Housing Dataset loaded successfully.\nDataset Head - ')
print(df.head(),
      '\nDataset Tail -',
      df.tail(),
      '\nDataset Info -',
      df.info())

# slicing the dataframe using iloc x will have everything except MedHouseVal While y will have MedHouseVal only.
# X is the data/features we will use to predict y

# x = df.iloc[:, :-1] # This line can also be used instead of next one but this checks for last column in dataframe and deletes it So the next one is better as it specifically looks for MedHouseVal
x = df.drop(columns=['MedHouseVal'])
print('Features - ', x.columns)
y = df['MedHouseVal']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2s = r2_score(y_test,y_pred)
print(f'Mean Squared Error - {mse}',
      f'R2 Score - {r2s}')

# setting up the plot style for better visualization
plt.style.use('default')
sns.set_palette("husl")

# creating a figure with subplots to visualize the dataset features
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle('California Housing Dataset - Feature Distributions', fontsize=16)

# plotting histograms for each feature in the dataset
for idx, col in enumerate(x.columns):
    ax = axes[idx // 4, idx % 4]
    ax.hist(x[col], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# creating a correlation heatmap to understand feature relationships
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap - California Housing Dataset')
plt.show()

# visualizing the target variable distribution
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.title('Distribution of Median House Values (Target Variable)')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

# creating scatter plots to visualize actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Median House Values')
plt.ylabel('Predicted Median House Values')
plt.title('Actual vs Predicted House Values - Linear Regression Model')
plt.text(0.05, 0.95, f'R² Score: {r2s:.3f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()

# creating residual plot to check model performance
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Checking Model Assumptions')
plt.show()

# creating a comparison plot showing test vs predicted values side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# plotting first 50 samples for better visibility
sample_size = 50
indices = np.arange(sample_size)

ax1.bar(indices, y_test.iloc[:sample_size].values, alpha=0.7, label='Actual Values')
ax1.bar(indices, y_pred[:sample_size], alpha=0.7, label='Predicted Values')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('House Value')
ax1.set_title('Actual vs Predicted Values - First 50 Samples')
ax1.legend()

# plotting prediction errors for the same samples
ax2.bar(indices, (y_test.iloc[:sample_size].values - y_pred[:sample_size]),
        color='red', alpha=0.7)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Prediction Error')
ax2.set_title('Prediction Errors - First 50 Samples')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()