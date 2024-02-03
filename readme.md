## Prediction Model using Machine Learning

This repository contains a simple example of how to create a prediction model using the scikit-learn library in Python.

## Step 1: Install Dependencies
Make sure you have Python installed. Then, install the required libraries:

pip install scikit-learn pandas

## Step 2: Import Libraries
In the Python script, import the necessary libraries:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

## Step 3:Load and Explore Data
Assuming you have a CSV file named 'data.csv' with your features and target variable. Load and explore the data:

data = pd.read_csv('dados.csv')
print(data.head())

## Step 4: Prepare Data
Split the data into training and testing sets:

X = data.drop('target_variable', axis=1)
y = data['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Step 5: Train the Model
Initialize and train a prediction model; in this case, we use a RandomForestRegressor:

model = RandomForestRegressor()
model.fit(X_train, y_train)

## Step 6: Evaluate the Model
Assess the model's performance using Mean Squared Error (MSE):

y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

## Step 7: Save the Trained Model

Finally, save the trained model for future use:

joblib.dump(model, 'prediction_model.joblib')

Now, you have a trained prediction model ready to make predictions on new data.


