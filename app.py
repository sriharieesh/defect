%%writefile app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open("random_forest.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔍 Defect Detection System (Random Forest Model)")
st.write("Enter the feature values below to predict the defect label.")

# Input fields for the user
surface_roughness = st.number_input("Surface Roughness", min_value=0.0, format="%.4f")
color_variation = st.number_input("Color Variation", min_value=0.0, format="%.4f")
edge_irregularity = st.number_input("Edge Irregularity", min_value=0.0, format="%.4f")
texture_score = st.number_input("Texture Score", min_value=0.0, format="%.4f")
noise_intensity = st.number_input("Noise Intensity", min_value=0.0, format="%.4f")
symmetry_score = st.number_input("Symmetry Score", min_value=0.0, format="%.4f")
defect_probability = st.number_input("Defect Probability", min_value=0.0, max_value=1.0, format="%.4f")

# Prepare input for prediction
input_data = np.array([[surface_roughness, color_variation, edge_irregularity,
                        texture_score, noise_intensity, symmetry_score, defect_probability]])

# Predict button
if st.button("Predict Defect"):
    prediction = model.predict(input_data)[0]

    st.subheader("📌 Prediction Result:")
    if prediction == 0:
        st.success("No Defect Detected ✔️")
    else:
        st.error("⚠️ Defect Detected")
