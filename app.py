
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model.pkl')

# Page configuration
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# App title
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Input Fields
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["HS-grad", "Assoc", "Some-college", "Bachelors", "Masters", "PhD"])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-Fishing", "Protective-serv",
    "Transport-moving", "Priv-house-serv", "protective-serv", "Armed-Serv"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# Build input DataFrame (NOTE: removed 'experience')
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours_per_week': [hours_per_week]
})

# Display input data
st.write("### Input Data:")
st.write(input_df)

# Preprocessing maps (must match model training)
education_map = {
    "HS-grad": 0,
    "Some-college": 1,
    "Assoc": 2,
    "Bachelors": 3,
    "Masters": 4,
    "PhD": 5
}

occupation_map = {
    "Tech-support": 0,
    "Craft-repair": 1,
    "Other-service": 2,
    "Sales": 3,
    "Exec-managerial": 4,
    "Prof-specialty": 5,
    "Handlers-cleaners": 6,
    "Machine-op-inspct": 7,
    "Adm-clerical": 8,
    "Farming-Fishing": 9,
    "Protective-serv": 10,
    "Transport-moving": 11,
    "Priv-house-serv": 12,
    "protective-serv": 13,
    "Armed-Serv": 14
}

# --- Prediction ---
if st.button("Predict Salary Class"):
    # Encode categorical values
    input_df['education'] = input_df['education'].map(education_map)
    input_df['occupation'] = input_df['occupation'].map(occupation_map)

    # Check for invalid values
    if input_df.isnull().any().any():
        st.error("Error: Invalid category detected. Please check input values.")
    else:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(batch_data.head())

    # Only keep expected columns
    required_columns = ['age', 'education', 'occupation', 'hours_per_week']
    if not all(col in batch_data.columns for col in required_columns):
        st.error("Error: Uploaded CSV must contain exactly these columns: " + ", ".join(required_columns))
    else:
        # Keep only required columns
        batch_data = batch_data[required_columns]

        # Encode categorical columns
        batch_data['education'] = batch_data['education'].map(education_map)
        batch_data['occupation'] = batch_data['occupation'].map(occupation_map)

        if batch_data.isnull().any().any():
            st.error("Error: Invalid categories in uploaded file.")
        else:
            # Perform prediction
            batch_preds = model.predict(batch_data)
            batch_data['Predictions'] = batch_preds

            # Display and download results
            st.write("### Predictions")
            st.write(batch_data.head())

            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='Predictions.csv', mime='text/csv')


