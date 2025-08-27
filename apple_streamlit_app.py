import streamlit as st
import numpy as np
import joblib

# Load the trained KNN model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('knn_scaler.joblib')

# Title and description
st.set_page_config(page_title="Apple Quality Predictor", page_icon="🍎")
st.title("🍎 Apple Quality Predictor")
st.markdown("Enter the values of apple features within the given ranges to check whether it is of **Good** or **Bad** quality.")

# Feature list with ranges and units
feature_ranges = {
    "Size (cm)": (0.001, 15.0),
    "Weight (grams)": (0.01, 1000.0),
    "Sweetness (0–1)": (0.000001, 1.0),
    "Crunchiness (0–1)": (0.000001, 1.0),
    "Juiciness (0–1)": (0.000001, 1.0),
    "Ripeness (0–1)": (0.000001, 1.0),
    "Acidity (0–1)": (0.000001, 1.0)
}

# Collect user input
user_input = []
for feature, (low, high) in feature_ranges.items():
    # Use fewer decimals for big ranges, more for small ranges
    if high > 50:  
        label = f"{feature} [{low:.2f} – {high:.2f}]"
        fmt = "%.2f"
        step = 0.1
    elif high > 1:
        label = f"{feature} [{low:.3f} – {high:.3f}]"
        fmt = "%.3f"
        step = 0.01
    else:
        label = f"{feature} [{low:.6f} – {high:.6f}]"
        fmt = "%.6f"
        step = 0.01
    
    value = st.number_input(
        label,
        min_value=low,
        max_value=high,
        step=step,
        format=fmt
    )
    user_input.append(value)

# Prediction button
if st.button("🔍 Predict Quality"):
    try:
        input_array = np.array([user_input])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        # Display result
        if prediction[0] == 1:
            st.success("✅ The apple is of **Good Quality**!")
        else:
            st.error("❌ The apple is of **Bad Quality**.")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
