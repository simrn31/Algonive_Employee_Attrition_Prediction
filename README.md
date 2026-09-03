# Employee Attrition Prediction

A Machine Learning project that predicts whether an employee is likely to leave an organization based on demographic, job-related, satisfaction, income, and experience-related factors.

## 🎯 Project Objective

Employee attrition can increase recruitment and training costs for organizations.

The objective of this project is to build a classification model that can identify employees who may be at higher risk of leaving, allowing HR teams to take preventive action.

## 📊 Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

- Rows: 1,470
- Original columns: 35
- Target variable: `Attrition`
- `No` → Employee stays
- `Yes` → Employee leaves

## 🔎 Exploratory Data Analysis

The analysis included:

- Dataset structure and data types
- Missing-value analysis
- Duplicate-value analysis
- Target class distribution
- Numerical feature analysis
- Categorical feature analysis
- Attrition patterns across employee characteristics

One important finding was that employees working overtime showed substantially higher attrition than employees who did not work overtime.

## 🛠️ Machine Learning Workflow

The project follows this workflow:

1. Data Understanding
2. Exploratory Data Analysis
3. Feature Selection
4. Train/Test Split
5. Data Preprocessing
6. Logistic Regression
7. Model Evaluation
8. Hyperparameter Tuning
9. Threshold Optimization
10. Model Comparison
11. Model Deployment

## ⚙️ Preprocessing

A Scikit-learn preprocessing pipeline was used.

### Numerical Features

- Median imputation
- Standard scaling

### Categorical Features

- Most-frequent imputation
- One-hot encoding

The preprocessing and model were combined into a single pipeline to prevent data leakage and ensure the same transformations are applied during prediction.

## 🤖 Models

Two classification approaches were evaluated:

### Logistic Regression

The tuned Logistic Regression model achieved approximately:

- Accuracy: 85%
- ROC-AUC: 0.81

Threshold optimization was also performed to improve detection of employees belonging to the attrition class.

### Random Forest

Random Forest was also evaluated and tuned.

Although it achieved approximately 85% accuracy, its recall for the attrition class was considerably lower than Logistic Regression.

Therefore, Logistic Regression was selected as the preferred model because identifying potential employees at risk of leaving was the primary business objective.

## 🎚️ Threshold Optimization

The default classification threshold of 0.50 was evaluated against multiple thresholds.

A threshold of 0.20 produced the best F1 score for the attrition class in the test-set experiment.

Final decision rule:

```text
Probability >= 0.20 → Likely to Leave
Probability < 0.20 → Likely to Stay