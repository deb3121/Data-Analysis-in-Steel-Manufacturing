#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('DSDataLastThreeMonths.csv')

# Check if 'CASTNO' column exists in the dataset
if 'CASTNO' not in dataset.columns:
    raise ValueError("'CASTNO' column not found in the dataset")

# Drop non-numeric columns from the dataset
non_numeric_columns = []

for column in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[column]):
        non_numeric_columns.append(column)

dataset = dataset.drop(non_numeric_columns, axis=1)

# Drop rows with missing values in both X and y
dataset = dataset.dropna()

# Split the dataset into input features (X) and target variable (y)
X = dataset.drop('DS_S', axis=1)
y = dataset['DS_S']

# Handle missing values using SimpleImputer for input features (X)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the imputed dataset and the cleaned target variable into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")


# In[ ]:




