# various algorithms are used to test accuracy of each algorithm
# doesnot generate any .pkl files
    # means this file doesnot save the model.. it only train and evaluate models and check accuracy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
diabetes_df = pd.read_csv("dataset/diabetes.csv")
heart_df = pd.read_csv("dataset/heart.csv")

# Preprocessing Diabetes Dataset
X_diabetes = diabetes_df.drop(columns=['Outcome'])
y_diabetes = diabetes_df['Outcome']
scaler = StandardScaler()
X_diabetes_scaled = scaler.fit_transform(X_diabetes)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42)

# Preprocessing Heart Dataset
X_heart = heart_df.drop(columns=['target'])
y_heart = heart_df['target']
X_heart_scaled = scaler.fit_transform(X_heart)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart_scaled, y_heart, test_size=0.2, random_state=42)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=400),
    'SVM': SVC()
}

# Train and Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_d, y_train_d)
    y_pred_d = model.predict(X_test_d)
    acc_d = accuracy_score(y_test_d, y_pred_d)
    
    model.fit(X_train_h, y_train_h)
    y_pred_h = model.predict(X_test_h)
    acc_h = accuracy_score(y_test_h, y_pred_h)
    
    results[name] = {'Diabetes Accuracy': acc_d, 'Heart Disease Accuracy': acc_h}

# Display Results
for model, scores in results.items():
    print(f"{model} - Diabetes Accuracy: {scores['Diabetes Accuracy']:.4f}, Heart Disease Accuracy: {scores['Heart Disease Accuracy']:.4f}")
