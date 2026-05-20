import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction Web App")
st.write("Enter the required clinical metrics below to evaluate risk state.")

# Layout input forms neatly in columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=130)
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
with col2:
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    chol = st.number_input("Serum Cholestoral (mg/dl)", value=240)
    thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
with col3:
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])

oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect)", [0, 1, 2, 3])

if st.button("Predict"):
    # Parse inputs exactly in order of features
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)

    if prediction[0] == 0:
        st.success("The model predicts this person is **Healthy**.")
    else:
        st.error("The model predicts signs of **Heart Disease**.")