# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    insurance_df = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("Error: 'insurance.csv' not found. Please download the dataset and place it in the same directory.")
    exit()
    
insurance_df_encoded = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = insurance_df_encoded.drop('charges', axis=1)
y = insurance_df_encoded['charges']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
print('R2 Score:', r2_score(y_test, y_pred))

# 1. Inspect the first 5 rows of the data
print("First 5 rows of the dataset:")
print(insurance_df.head())

# 2. Get information about the dataset (data types, non-null counts)
print("\nDataset Information:")
insurance_df.info()

# 3. Check for any missing values
print("\nMissing values in each column:")
print(insurance_df.isnull().sum())
# This dataset is clean and has no missing values.

# 4. Get statistical summary of numerical columns
print("\nStatistical Summary:")
print(insurance_df.describe())

# 5. Visualize the distribution of the target variable 'charges'
plt.figure(figsize=(10, 6))
sns.histplot(insurance_df['charges'], kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# 6. Convert categorical features to numerical data
# Machine learning models require all input to be numerical.
# We will use one-hot encoding for this.
insurance_df_encoded = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("\nDataset after encoding categorical variables:")
print(insurance_df_encoded.head())

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = insurance_df_encoded.drop('charges', axis=1)
y = insurance_df_encoded['charges']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

from sklearn.linear_model import LinearRegression

# Create a Linear Regression model instance
lr_model = LinearRegression()

# Train the model on the training data
lr_model.fit(X_train, y_train)

from sklearn.metrics import r2_score

# Make predictions on the training and testing data
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Calculate R-squared score for both sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nR-squared score on Training Data: {r2_train:.4f}")
print(f"R-squared score on Testing Data: {r2_test:.4f}")

# Example: Predict cost for a new person
# This person is a 31-year-old, non-smoking male from the southwest with 2 children and a BMI of 25.74
# Note: The input must match the columns of X_train after encoding.
# 'sex_male'=1, 'smoker_yes'=0, 'region_northwest'=0, 'region_southeast'=0, 'region_southwest'=1

# Create a NumPy array for the new data point
new_data = np.array([[31, 25.74, 2, 1, 0, 0, 0, 1]])

# Predict the insurance cost
predicted_cost = lr_model.predict(new_data)

print(f"\nPredicted Insurance Cost for the new person: ${predicted_cost[0]:.2f}")

