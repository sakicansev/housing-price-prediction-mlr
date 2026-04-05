# Multiple Linear Regression Analysis - California Housing Dataset
# Saki Cansev | MSc Data Analytics
# University for the Creative Arts - Berlin School of Business & Innovation

# ─────────────────────────────────────────
# 1. Importing Libraries and Loading the Dataset
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Load California Housing dataset
california = datasets.fetch_california_housing()

# ─────────────────────────────────────────
# 2. Data Preparation
# ─────────────────────────────────────────

# Create a DataFrame from the features
california_df = pd.DataFrame(california.data, columns=california.feature_names)

# Append target as a new column
california_df['MedHouseVal'] = california.target

# Check for missing values
print("Missing values:\n", california_df.isnull().sum())

# Check summary statistics
print("\nSummary statistics:\n", california_df.describe())

# Define features and target
features = california_df.drop('MedHouseVal', axis=1)
target = california_df[['MedHouseVal']]

# ─────────────────────────────────────────
# 3. Data Visualization
# ─────────────────────────────────────────

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(california_df['MedHouseVal'], kde=True, color='#9370DB')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Distribution of Median House Value')
plt.show()

# Pairplot: target vs all features
sns.pairplot(california_df,
             y_vars=["MedHouseVal"],
             x_vars=california_df.columns[:-1],
             plot_kws={'color': '#9370DB'})
plt.show()

# Individual scatter plots for each feature
sns.set(style="ticks")
for feature in california_df.columns[:-1]:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=california_df, x=feature, y="MedHouseVal", color='#9370DB')
    plt.xlabel(feature)
    plt.ylabel("Median House Value")
    plt.title("Scatter Plot: {} vs. Median House Value".format(feature))
    plt.show()

# ─────────────────────────────────────────
# 4. Splitting the Dataset
# ─────────────────────────────────────────

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=1
)

# ─────────────────────────────────────────
# 5. Implementing the MLR Model
# ─────────────────────────────────────────

mlr = LinearRegression()
mlr.fit(features_train, target_train)
target_pred = mlr.predict(features_test)

# ─────────────────────────────────────────
# 6. Visualizing Actual vs Predicted
# ─────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.scatter(target_test, target_pred, color='#9370DB', label='Predicted', alpha=0.7)
plt.plot(
    [target_test.min(), target_test.max()],
    [target_test.min(), target_test.max()],
    '--', lw=2, color='#4B0082', label='Perfect Prediction'
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
plt.legend()
plt.show()

# ─────────────────────────────────────────
# 7. Model Evaluation
# ─────────────────────────────────────────

# RMSE
rmse = np.sqrt(metrics.mean_squared_error(target_test, target_pred))

# R² Score
r2_score = metrics.r2_score(target_test, target_pred)

print("Root Mean Squared Error: ", rmse)
print("R² Score: ", r2_score)

# Coefficients (Interpretability)
print("Coefficients: ", mlr.coef_)
