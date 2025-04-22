import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64

import streamlit as st

st.set_page_config(
    page_title="Disease Probability Predictior",
    page_icon="🧬", 
    layout="wide"
)


# Load trained models and scalers
diabetes_model = joblib.load("trained/diabetes_model.pkl")
heart_model = joblib.load("trained/heart_model.pkl")
diabetes_scaler = joblib.load("trained/diabetes_scaler.pkl")
heart_scaler = joblib.load("trained/heart_scaler.pkl")

# Streamlit UI Title
st.title("🩺 Disease Probability Predictor 🧬")

# Sidebar navigation

file_path = "pic1.png"  
with open(file_path, "rb") as img_file:
    encoded_img = base64.b64encode(img_file.read()).decode()

# Inject circular image into the sidebar
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{encoded_img}"
             style="border-radius: 50%; width: 200px; height: 200px; object-fit: cover;">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

st.sidebar.header("Select Disease")
disease_type = st.sidebar.radio("Choose Disease:", ["Diabetes Prediction", "Heart Disease Prediction"])
pregnancies = 0

# Input form for Diabetes
if disease_type == "Diabetes Prediction":
    st.header("🔍 Diabetes Prediction")
    gender = st.selectbox("Select your Gender: ", ["Male", "Female"])
    
    if gender == "Female":
        pregnancies = st.number_input("Pregnancies:", min_value=0, max_value=20, step=1)
    
    glucose = st.number_input("Glucose Level:", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure:", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness:", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin Level:", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI:", min_value=0.0, max_value=100.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age:", min_value=0, max_value=120, step=1)
    
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    user_data_scaled = diabetes_scaler.transform(user_data)

    if st.button("Predict Diabetes"):
        # Simple validation: key fields must not be 0
        if glucose == 0 or age == 0 or blood_pressure == 0 or bmi == 0:
            st.warning("⚠️ Please enter valid input values before prediction.")
        else:
            prediction = diabetes_model.predict(user_data_scaled)[0]
            probability = diabetes_model.predict_proba(user_data_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High Risk! Probability: {probability*100:.2f}%")
            else:
                st.success(f"✅ Low Risk! Probability: {probability*100:.2f}%")

            # Visualization - Line Chart
            st.subheader("📈 Comparison with Healthy Baseline")
            feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Pedigree", "Age"]
            healthy_values = [1, 100, 75, 20, 80, 22.5, 0.3, 30]
            line_color = 'red' if prediction == 1 else 'green'

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(feature_names, healthy_values, marker='o', label='Healthy Person', color='blue', linewidth=2)
            ax.plot(feature_names, user_data[0], marker='o', label='User Input', color=line_color, linewidth=2)

            plt.xticks(rotation=45)
            ax.set_ylabel("Values")
            ax.set_title("User vs Healthy Parameter Levels")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# Input form for Heart Disease
elif disease_type == "Heart Disease Prediction":
    st.header("💖 Heart Disease Prediction")

    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, step=1)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("False", 0), ("True", 1)])
    restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=250, step=1)
    exang = st.selectbox("Exercise-Induced Angina", [("No", 0), ("Yes", 1)])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, step=0.1)
    slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, step=1)

    user_data = np.array([[age, sex[1], cp, trestbps, chol, fbs[1], restecg, thalach, exang[1], oldpeak, slope, ca, thal]])
    user_data_scaled = heart_scaler.transform(user_data)

    if st.button("Predict"):
        # Simple validation: key fields must not be 0
        if age == 0 or trestbps == 0 or chol == 0 or thalach == 0:
            st.warning("⚠️ Please enter valid input values before prediction.")
        else:
            prediction = heart_model.predict(user_data_scaled)[0]
            probability = heart_model.predict_proba(user_data_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High Risk! Probability: {probability*100:.2f}%")
            else:
                st.success(f"✅ Low Risk! Probability: {probability*100:.2f}%")

            # Visualization - Line Chart
            st.subheader("📈 Comparison with Healthy Baseline")
            feature_names = ["Age", "Sex", "CP", "BP", "Chol", "FBS", "ECG", "Max HR", "Angina", "Oldpeak", "Slope", "Vessels", "Thal"]
            healthy_values = [35, 0, 0, 120, 200, 0, 1, 170, 0, 0.1, 1, 0, 1]
            line_color = 'red' if prediction == 1 else 'green'

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(feature_names, healthy_values, marker='o', label='Healthy Person', color='blue', linewidth=2)
            ax.plot(feature_names, user_data[0], marker='o', label='User Input', color=line_color, linewidth=2)

            plt.xticks(rotation=45)
            ax.set_ylabel("Values")
            ax.set_title("User vs Healthy Parameter Levels")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
# Disclaimer note
st.markdown(
    """
    <div style="background-color:#b30000;padding:12px;border-radius:8px;">
        <p style="color:yellow;font-size:16px;text-align:center;">
        ⚠️ <strong>Disclaimer:</strong> This application is intended solely for academic and educational use. It does not constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare professional for any medical concerns or decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📌 Developed by BATCH - C1")
