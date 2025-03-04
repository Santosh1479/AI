import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load sample health data
data = pd.read_csv('patient_data.csv')

# Convert column names to lowercase and strip spaces to avoid KeyError
data.columns = data.columns.str.strip().str.lower()

# Check available columns
print("Columns in dataset:", data.columns)

# Handle missing values safely
fill_values = {}

if 'glucose' in data.columns:
    fill_values['glucose'] = data['glucose'].median()
if 'blood_pressure' in data.columns:
    fill_values['blood_pressure'] = data['blood_pressure'].mean()
if 'bmi' in data.columns:
    data['bmi'] = data['bmi'].ffill()

data.fillna(fill_values, inplace=True)

# Remove outliers in 'age' column if present
if 'age' in data.columns:
    z_scores = np.abs((data['age'] - data['age'].mean()) / data['age'].std())
    data = data[z_scores < 3]

# Normalize numerical features safely
scaler = StandardScaler()
numeric_features = [col for col in ['age', 'blood_pressure', 'glucose'] if col in data.columns]

if numeric_features:
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

print("Preprocessed Data:")
print(data.head())