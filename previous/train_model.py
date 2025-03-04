import pandas as pd
import numpy as np
import joblib  # For saving the scaler
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Load dataset
data = pd.read_csv('cleaned_patient_data.csv')

# Convert column names to lowercase and strip spaces
data.columns = data.columns.str.strip().str.lower()

# Define target variable (example: predicting 'diabetes' or 'disease')
if 'diagnosis' not in data.columns:
    raise ValueError("The dataset must contain a 'diagnosis' column for prediction.")

# Convert categorical labels to numerical values
data['diagnosis'] = data['diagnosis'].astype('category').cat.codes  # Encode labels

# Define features (X) and target (y)
X = data.drop(columns=['diagnosis'])  # Remove target column
y = data['diagnosis']

# Handle categorical data (e.g., Gender)
if 'gender' in X.columns:
    X = pd.get_dummies(X, columns=['gender'], drop_first=True)

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Define Deep Learning Model (ANN)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(set(y)), activation='softmax')  # Output layer for multi-class classification
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save trained model
model.save('ai_healthcare_model.h5')

print("Deep Learning model saved successfully!")
