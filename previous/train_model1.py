import joblib  # For saving the model
import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.preprocessing import StandardScaler  # For normalizing data
from sklearn.ensemble import RandomForestClassifier  # Machine Learning model

# Step 1: Load Data
df = pd.read_csv("cleaned_patient_data.csv")  # Load dataset

# Convert column names to lowercase and strip spaces
df.columns = df.columns.str.strip().str.lower()

# Step 2: Data Preprocessing
df.fillna(df.mean(), inplace=True)  # Handle missing values

# Convert categorical to numerical (One-Hot Encoding for 'gender')
if 'gender' in df.columns:
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Ensure 'diagnosis' exists in dataset
if 'diagnosis' not in df.columns:
    raise ValueError("Dataset must contain 'diagnosis' column.")

# Step 3: Split Data into Features & Labels
X = df.drop("diagnosis", axis=1)  # Features (patient data)
y = df["diagnosis"]  # Target label (health condition)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Save the Trained Model and Scaler
joblib.dump(model, "healthcare_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete! Model saved as 'healthcare_model.pkl'.")
