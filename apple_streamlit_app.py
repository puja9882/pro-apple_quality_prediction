import streamlit as st
import numpy as np
import joblib

# Load the trained KNN model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('knn_scaler.joblib')

# Title and description
st.set_page_config(page_title="Apple Quality Predictor", page_icon="🍎")
st.title("🍎 Apple Quality Predictor")
st.markdown("Enter the values of apple features to check whether it is of **Good** or **Bad** quality.")

# Feature list used in the model
features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

# Collect user input for each feature
user_input = []
for feature in features:
    value = st.number_input(f"{feature}:", min_value=0.0, step=0.1, format="%.2f")
    user_input.append(value)

# Prediction button
if st.button("🔍 Predict Quality"):
    input_array = np.array([user_input])  
    input_scaled = scaler.transform(input_array)  
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.success("✅ The apple is of **Good Quality**!")
    else:
        st.error("❌ The apple is of **Bad Quality**.")