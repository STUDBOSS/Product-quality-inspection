import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data_path = 'C:/Mohit/SIES/Mini Project/ION Product inspection system/dataset'
data = pd.read_csv(os.path.join(data_path, 'jarlids_annots_updated.csv'), encoding='utf-8')

# One-hot encoding for categorical data
one_hot_encoder = OneHotEncoder()
data = pd.get_dummies(data, columns=['name'])
#data['name'] = one_hot_encoder.fit_transform(data['name'].values.reshape(-1, 1)).toarray()

# Split features and target variable
column_drop=['no.','filename','target column']
X = data.drop(columns=column_drop)
y=data.get("target column")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Keras model
keras_model = Sequential()

# Add the input layer
keras_model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))

# Add the output layer
keras_model.add(Dense(1, activation='sigmoid'))

# Compile the model
keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
keras_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the test data
y_pred = keras_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')