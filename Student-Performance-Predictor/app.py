import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📚 Student Performance Predictor")

st.write("Predict final exam score using ML")

study = st.number_input("Study Hours per Day", 0.0, 15.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous = st.number_input("Previous Score", 0.0, 100.0)

if st.button("Predict"):
    data = np.array([[study, attendance, previous]])
    prediction = model.predict(data)

    st.success(f"Predicted Final Score: {prediction[0]:.2f}")
