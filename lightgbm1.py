import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib

# Load the dataset
data_path = 'C:/Mohit/SIES/Mini Project/ION Product inspection system/dataset'
data = pd.read_csv(os.path.join(data_path, 'jarlids_annots_updated.csv'), encoding='utf-8')

# Label encoding for categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['name'] = label_encoder.fit_transform(data['name'])

# Split features and target variable
column_drop=['no.','filename','target column']
X = data.drop(columns=column_drop)
y=data.get("target column")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100000, learning_rate=0.1, max_depth=5, random_state=42)

# Train the LightGBM model
lgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the LightGBM model
joblib.dump(lgb_model, 'lgb_model.pkl')

# Load the saved LightGBM model
loaded_lgb_model = joblib.load('lgb_model.pkl')

# Make predictions on the test data using the loaded model
loaded_y_pred = loaded_lgb_model.predict(X_test)

# Calculate accuracy using the loaded model
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)
print(f'Loaded Accuracy: {loaded_accuracy:.2f}')