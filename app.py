import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("final_model.pkl")

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊")
st.title("👩‍💼 Employee Attrition Prediction Dashboard")
st.write("Fill out the form below to predict if the employee is likely to leave the company.")

with st.form("employee_form"):
    # Input fields
    job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director'])
    monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
    work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
    distance_from_home = st.slider("Distance from Home", 1, 30, 10)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    remote_work = st.radio("Remote Work", ['Yes', 'No'])  # We'll convert this to 1/0
    years_at_company = st.slider("Years at Company", 0, 40, 3)

    submit = st.form_submit_button("Predict")

if submit:
    # Derived fields
    remote_work_val = 1 if remote_work == 'Yes' else 0
    salary_perf_ratio = monthly_income / (performance_rating if performance_rating != 0 else 1)
    income_joblevel_ratio = monthly_income / (job_level if job_level != 0 else 1)

    # Grouping tenure
    if years_at_company <= 2:
        tenure_group = '0-2'
    elif years_at_company <= 5:
        tenure_group = '3-5'
    elif years_at_company <= 10:
        tenure_group = '6-10'
    else:
        tenure_group = '10+'

    # Prepare the input DataFrame
    input_data = {
        'Years at Company': years_at_company,
        'Job Role': job_role,
        'Monthly Income': monthly_income,
        'Work-Life Balance': work_life_balance,
        'Job Satisfaction': job_satisfaction,
        'Performance Rating': performance_rating,
        'Distance from Home': distance_from_home,
        'Marital Status': marital_status,
        'Job Level': job_level,
        'Remote Work': remote_work_val,
        'Salary_Performance_Ratio': salary_perf_ratio,
        'Tenure_Group': tenure_group,
        'WorkLife_Satisfaction_Score': (work_life_balance + job_satisfaction) / 2,
        'Income_JobLevel_Ratio': income_joblevel_ratio
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns the same way they were during training
    for col in ['Job Role', 'Marital Status', 'Tenure_Group']:
        input_df[col] = input_df[col].astype("category").cat.codes

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error(f"This employee is likely to leave the company. (Probability: {prediction_proba:.2%})")
    else:
        st.success(f"This employee is likely to stay. (Probability: {1 - prediction_proba:.2%})")
