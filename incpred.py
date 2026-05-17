import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Saved Files
# -------------------------------

model = joblib.load('RF_income.pkl')
expected_columns = joblib.load('income_columns.pkl')

# -------------------------------
# App Title
# -------------------------------

st.title("💰 Income Prediction App")
st.markdown("Provide the following details:")
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")

# -------------------------------
# User Inputs
# -------------------------------

age = st.slider("Age", 18, 100, 30)

education_num = st.slider("Education Number", 1, 16, 10)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    max_value=100000,
    value=0
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    max_value=100000,
    value=0
)

hours_per_week = st.number_input(
    "Hours per Week",
    min_value=1,
    max_value=100,
    value=40
)

workclass = st.selectbox(
    "Workclass",
    [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse"
    ]
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Income"):

    # Create dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'education.num': [education_num],
        'sex': [1 if sex == "Male" else 0],
        'capital.gain': [capital_gain],
        'capital.loss': [capital_loss],
        'hours.per.week': [hours_per_week],
        'workclass': [workclass],
        'occupation': [occupation],
        'marital.status': [marital_status]
    })

    # One-hot encoding
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]

    # Output
    if prediction[0] == 1:
        st.success("Predicted Income: >50K")
    else:
        st.error("Predicted Income: <=50K")

    st.write(f"Confidence: {probability:.2%}")

    st.balloons()