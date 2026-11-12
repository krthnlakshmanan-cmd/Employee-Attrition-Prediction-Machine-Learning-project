
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💻",
    layout="wide"
)

# Load trained model
model = joblib.load("models/attrition_model_new.pkl")

# Application title
st.title("Employee Attrition Prediction")
st.write("Predict employee attrition using Machine Learning.")
st.info(
    "Educational project only. Predictions are estimates, "
    "not actual employee decisions."
)

# Employee details
st.header("Enter Employee Details")

col1, col2, col3 = st.columns(3)

# 1. Personal Information
with col1:
    st.subheader("Personal Information")

    age = st.number_input("Age", 18, 60, 30)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    education_label = st.selectbox(
        "Education",
        [
            "Below College",
            "College",
            "Bachelor",
            "Master",
            "Doctor"
        ]
    )

    education = {
        "Below College": 1,
        "College": 2,
        "Bachelor": 3,
        "Master": 4,
        "Doctor": 5
    }[education_label]

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Other",
            "Human Resources"
        ]
    )

    distance = st.number_input(
        "Distance From Home", 1, 29, 5
    )


# 2. Job Information
with col2:
    st.subheader("Job Information")

    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Research Scientist",
            "Laboratory Technician",
            "Research Director",
            "Manufacturing Director",
            "Healthcare Representative",
            "Sales Executive",
            "Sales Representative",
            "Manager",
            "Human Resources"
        ]
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    daily_rate = st.number_input(
        "Daily Rate", 100, 1500, 500
    )

    monthly_income = st.number_input(
        "Monthly Income", 1000, 200000, 5000
    )

    monthly_rate = st.number_input(
        "Monthly Rate", 2000, 27000, 15000
    )

    hourly_rate = st.number_input(
        "Hourly Rate", 30, 100, 60
    )


# 3. Work & Satisfaction
with col3:
    st.subheader("Work & Satisfaction")

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    environment_label = st.selectbox(
        "Environment Satisfaction",
        ["Low", "Medium", "High", "Very High"]
    )

    job_satisfaction_label = st.selectbox(
        "Job Satisfaction",
        ["Low", "Medium", "High", "Very High"]
    )

    involvement_label = st.selectbox(
        "Job Involvement",
        ["Low", "Medium", "High", "Very High"]
    )

    relationship_label = st.selectbox(
        "Relationship Satisfaction",
        ["Low", "Medium", "High", "Very High"]
    )

    work_life_label = st.selectbox(
        "Work-Life Balance",
        ["Bad", "Good", "Better", "Best"]
    )

    stock_option = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )

    performance = st.selectbox(
        "Performance Rating",
        [3, 4]
    )

    salary_hike = st.number_input(
        "Percent Salary Hike", 11, 25, 15
    )


# Convert satisfaction labels into numbers
environment = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}[environment_label]

job_satisfaction = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}[job_satisfaction_label]

involvement = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}[involvement_label]

relationship = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}[relationship_label]

work_life = {
    "Bad": 1,
    "Good": 2,
    "Better": 3,
    "Best": 4
}[work_life_label]


# 4. Work Experience
st.subheader("Work Experience")

exp1, exp2, exp3 = st.columns(3)

with exp1:
    total_years = st.number_input(
        "Total Working Years", 0, 40, 5
    )

    years_company = st.number_input(
        "Years At Company", 0, 40, 3
    )

    years_role = st.number_input(
        "Years In Current Role", 0, 18, 2
    )

with exp2:
    companies = st.number_input(
        "Number of Companies Worked", 0, 9, 2
    )

    training = st.number_input(
        "Training Times Last Year", 0, 6, 3
    )

    promotion = st.number_input(
        "Years Since Last Promotion", 0, 15, 1
    )

with exp3:
    manager_years = st.number_input(
        "Years With Current Manager", 0, 17, 2
    )


# 5. Prediction
if st.button("Predict Attrition", type="primary"):

    employee = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": companies,
        "OverTime": overtime,
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": promotion,
        "YearsWithCurrManager": manager_years
    }])

    prediction = model.predict(employee)[0]

    if prediction == 1:
        st.warning(
            "Prediction: Employee may leave (Attrition: Yes)"
        )
    else:
        st.success(
            "Prediction: Employee may stay (Attrition: No)"
        )
