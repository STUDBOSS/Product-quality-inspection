import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = 'C:/Mohit/SIES/Mini Project/ION Product inspection system/dataset'  # Use the path obtained from kagglehub
# data = pd.read_csv(os.path.join(data_path, 'jarlids_annots.csv'))  # Replace with actual file name
data = pd.read_csv(os.path.join(data_path, 'jarlids_annots_updated.csv'), encoding='utf-8')

#print(data['target column'])
# print(data.columns)
# data.columns = data.columns.str.strip()
# print(data.columns)
# # Display the first few rows of the dataset
# print(data.head())

# # Check for missing values
#print(data.isnull().sum())

# # Get basic statistics
#print(data.describe())


# Example: Fill missing values
# data.ffill(inplace=True)
# Update 'target column' to extract the values ("intact" or "damaged")
# import ast

# # Safely parse the stringified dictionary and extract the "type" value
# data['target column'] = data['target column'].apply(lambda x: ast.literal_eval(x).get("type"))

# # Verify the updates by displaying the unique values in 'target column'
# data['target column'].unique(), data.head()


# Example: Encode categorical variables
# data = pd.get_dummies(data, drop_first=True)

from sklearn.preprocessing import LabelEncoder

# Example: Label encoding for categorical data
label_encoder = LabelEncoder()
data['name'] = label_encoder.fit_transform(data['name'])

# Split features and target variable
# X = data.drop(index=[0,1,5]) 
# y = data['target column']
# y=data.drop(index=[0,1,2,3,4,6,7,8,9,10])
column_drop=['no.','filename','target column','file_size','region_id']
X = data.drop(columns=column_drop)

# X=data.get("file_size","region_count")
# # X=data.assign("region_id")
# X=data.assign("name")
# X=data.assign("x")
# X=data.assign("y")
# X=data.assign("width")
# X=data.assign("height")
# X=data.assign
y=data.get("target column")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize the model
model = RandomForestClassifier(n_estimators=600000, random_state=42)

# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


import joblib

# Save the model
joblib.dump(model, 'trained_model.pkl')




