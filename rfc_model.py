# uses random forest classifier to train models

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# 1️⃣ Train Diabetes Model
# ---------------------------

# Load dataset
diabetes_data = pd.read_csv("dataset/diabetes.csv")

# Features & target
X_diabetes = diabetes_data.drop(columns=["Outcome"])
y_diabetes = diabetes_data["Outcome"]

# Split data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Standardize features
diabetes_scaler = StandardScaler()
X_train_d_scaled = diabetes_scaler.fit_transform(X_train_d)
X_test_d_scaled = diabetes_scaler.transform(X_test_d)

# Train model
diabetes_model = RandomForestClassifier(n_estimators=400, random_state=42)
diabetes_model.fit(X_train_d_scaled, y_train_d)

# Evaluate model
diabetes_pred = diabetes_model.predict(X_test_d_scaled)
diabetes_acc = accuracy_score(y_test_d, diabetes_pred)
print(f"Diabetes Model Accuracy: {diabetes_acc:.2f}")

# Save model and scaler
joblib.dump(diabetes_model, "trained/diabetes_model.pkl")
joblib.dump(diabetes_scaler, "trained/diabetes_scaler.pkl")


# ---------------------------
# 2️⃣ Train Heart Disease Model
# ---------------------------

# Load dataset
heart_data = pd.read_csv("dataset/heart.csv")

# Features & target
X_heart = heart_data.drop(columns=["target"])  # Adjust if column name is different
y_heart = heart_data["target"]

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Standardize features
heart_scaler = StandardScaler()
X_train_h_scaled = heart_scaler.fit_transform(X_train_h)
X_test_h_scaled = heart_scaler.transform(X_test_h)

# Train model
heart_model = RandomForestClassifier(n_estimators=400, random_state=42)
heart_model.fit(X_train_h_scaled, y_train_h)

# Evaluate model
heart_pred = heart_model.predict(X_test_h_scaled)
heart_acc = accuracy_score(y_test_h, heart_pred)
print(f"Heart Disease Model Accuracy: {heart_acc:.2f}")

# Save model and scaler
joblib.dump(heart_model, "trained/heart_model.pkl")
joblib.dump(heart_scaler, "trained/heart_scaler.pkl")

print("✅ Models and scalers saved successfully!")
