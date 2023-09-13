#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('DSDataLastThreeMonths.csv')

# Select relevant columns for the model
selected_columns = ['HM_WT', 'AIM_S', 'HM_S', 'HM_C', 'HM_SI', 'HM_TI', 'HM_MN', 'CAC2', 'MG', 'HM_TEMP', 'CAC2_INJ_TIME', 'MG_INJ_TIME', 'DS_S']

dataset = dataset[selected_columns]

# Drop rows with missing values in both X and y
dataset = dataset.dropna()

# Split the dataset into input features (X) and target variable (y)
X = dataset.drop('DS_S', axis=1)
y = dataset['DS_S']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")

# Calculate the Model hit rate (% data point with (Pred DS_S – Act DS_S) between +- 0.003%)
tolerance = 0.003
within_tolerance = abs(y_pred - y_test) <= tolerance
hit_rate = (within_tolerance.sum() / len(y_test)) * 100
print(f"Model hit rate: {hit_rate:.2f}%")


# In[ ]:




