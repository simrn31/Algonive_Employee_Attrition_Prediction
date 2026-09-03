import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

artifact = joblib.load("models/employee_attrition_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Employee Attrition Prediction")
st.write(
    "Predict whether an employee is likely to leave the organization "
    "using a Machine Learning model."
)


# --------------------------------------------------
# Employee Information
# --------------------------------------------------

st.header("Employee Information")


col1, col2, col3 = st.columns(3)


with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    BusinessTravel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    DailyRate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

    Department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    DistanceFromHome = st.number_input(
        "Distance From Home",
        min_value=0,
        value=5
    )

    Education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3
    )

    EducationField = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    EnvironmentSatisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    HourlyRate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=65
    )


with col2:

    JobInvolvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    JobLevel = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    JobSatisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=14000
    )

    NumCompaniesWorked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

    OverTime = st.selectbox(
        "Overtime",
        ["Yes", "No"]
    )

    PercentSalaryHike = st.number_input(
        "Percent Salary Hike",
        min_value=0,
        max_value=100,
        value=15
    )


with col3:

    PerformanceRating = st.number_input(
        "Performance Rating",
        min_value=1,
        max_value=4,
        value=3
    )

    RelationshipSatisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    StockOptionLevel = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1
    )

    TotalWorkingYears = st.number_input(
        "Total Working Years",
        min_value=0,
        value=8
    )

    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year",
        min_value=0,
        value=3
    )

    WorkLifeBalance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

    YearsAtCompany = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    YearsInCurrentRole = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=3
    )

    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=1
    )

    YearsWithCurrManager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=3
    )


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame([{

    "Age": Age,
    "BusinessTravel": BusinessTravel,
    "DailyRate": DailyRate,
    "Department": Department,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education,
    "EducationField": EducationField,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "Gender": Gender,
    "HourlyRate": HourlyRate,
    "JobInvolvement": JobInvolvement,
    "JobLevel": JobLevel,
    "JobRole": JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MaritalStatus": MaritalStatus,
    "MonthlyIncome": MonthlyIncome,
    "MonthlyRate": MonthlyRate,
    "NumCompaniesWorked": NumCompaniesWorked,
    "OverTime": OverTime,
    "PercentSalaryHike": PercentSalaryHike,
    "PerformanceRating": PerformanceRating,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "StockOptionLevel": StockOptionLevel,
    "TotalWorkingYears": TotalWorkingYears,
    "TrainingTimesLastYear": TrainingTimesLastYear,
    "WorkLifeBalance": WorkLifeBalance,
    "YearsAtCompany": YearsAtCompany,
    "YearsInCurrentRole": YearsInCurrentRole,
    "YearsSinceLastPromotion": YearsSinceLastPromotion,
    "YearsWithCurrManager": YearsWithCurrManager

}])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Attrition", use_container_width=True):

    probability = model.predict_proba(input_data)[0, 1]

    prediction = int(probability >= threshold)

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Probability of Leaving",
            f"{probability:.1%}"
        )

    with col2:

        if prediction == 1:

            st.error("⚠️ Likely to Leave")

        else:

            st.success("✅ Likely to Stay")

    st.progress(float(probability))

    st.caption(
        f"Decision threshold used: {threshold:.2f}"
    )
