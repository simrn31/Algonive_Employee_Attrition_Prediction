# Employee Attrition Prediction

An end-to-end Machine Learning project that predicts whether an employee is likely to leave an organization.

The project includes data analysis, preprocessing, model comparison, hyperparameter tuning, threshold optimization, model interpretation, and deployment using Streamlit.

## 🚀 Live Demo

[Open the Employee Attrition Prediction App](https://employee-attrition-rate.streamlit.app)

## 📌 Project Objective

Employee attrition can negatively affect organizations through recruitment costs, productivity loss, and disruption to teams.

This project uses historical employee data to identify employees who may be at higher risk of leaving, helping organizations take proactive retention measures.

## 📊 Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

- Rows: 1,470
- Columns: 35
- Target variable: `Attrition`
- `No` → Employee stays
- `Yes` → Employee leaves

## 🔍 Exploratory Data Analysis

The analysis examined:

- Dataset structure and data types
- Missing values
- Duplicate records
- Target class distribution
- Numerical feature distributions
- Categorical feature relationships
- Attrition patterns across employee characteristics

One important finding was that employees working overtime showed a substantially higher attrition rate than employees who did not work overtime.

## ⚙️ Machine Learning Pipeline

The preprocessing pipeline includes:

### Numerical Features
- Median imputation
- Standard scaling

### Categorical Features
- Most-frequent imputation
- One-hot encoding

A Scikit-learn `Pipeline` and `ColumnTransformer` were used to keep preprocessing and model training together and reduce the risk of inconsistent transformations.

## 🤖 Models

The following models were evaluated:

### Logistic Regression

A baseline Logistic Regression model was trained first and then optimized using `GridSearchCV`.

Best parameters:

```text
C = 100
class_weight = None
