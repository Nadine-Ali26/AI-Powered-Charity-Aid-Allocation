### AI-Powered-Charity-Aid-Allocation
A Machine Learning solution for beneficiary classification and financial aid prediction, enabling charitable organizations to make accurate, efficient, and data-driven assistance decisions. 


# Overview
This project aims to support charitable organizations by automating the process of evaluating beneficiaries. Based on demographic and financial information, the system predicts the beneficiary category and recommends an appropriate financial aid amount using Machine Learning models.

# Problem Statement
Charitable organizations often receive a large number of aid requests. Evaluating beneficiaries manually can be time-consuming, inconsistent, and prone to human bias,fault, making it difficult to distribute financial assistance fairly and efficiently.

#  Solution
This project leverages Machine Learning to automate beneficiary evaluation by classifying applicants based on their information and predicting an appropriate financial aid amount. The system helps organizations make faster, more consistent, and data-driven decisions.

### Objectives
- Automate beneficiary evaluation.
- Classify applicants based on their eligibility.
- Predict appropriate financial aid amounts.
- Reduce manual effort and improve decision consistency.
- Support data-driven aid allocation.

  
# Features
- Data preprocessing and feature engineering.
- Beneficiary classification using multiple ML algorithms.
- Financial aid amount prediction.
- Hyperparameter tuning with GridSearchCV.
- Model performance comparison.
- Interactive Streamlit web application.

  
### Workflow

Beneficiary Data
        │
        ▼
 Data Preprocessing
        │
        ▼
 Feature Engineering
        │
        ├─────────────┐
        ▼             ▼
Classification   Regression
        │             │
        ▼             ▼
 Beneficiary      Recommended
   Category       Aid Amount

   
##  Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Streamlit
- Matplotlib
- Seaborn
- Joblib

### Machine Learning Models

## Classification
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- SVC 
- XGBoost
- CatBoost
- LightGBM

## Regression
- Linear Regression
- SVR
- XGBoost
- Random Forest Regressor
- AdaBoost
- XGBoost Regressor

| Type           | Models                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------  |
| Classification |  SVC, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM ,KNN ,Gradient Boosting   |
| Regression     | Linear Regression, Random Forest Regressor, Gradient Boosting, XGBoost Regressor,AdaBoost |

### Evaluation Metrics
# Classification
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

# Regression
- MAE
- MSE
- RMSE
- R² Score

### Results
The models were evaluated using standard Machine Learning metrics, and the best-performing models were selected after hyperparameter tuning using GridSearchCV.

### Application & Deployment
The project includes an interactive Streamlit application where users can enter beneficiary information and instantly receive predictions.

### Future Improvements
- Deploy the application online.
- Integrate with a real charity database.
- Add explainable AI (SHAP).
- Improve prediction performance.
- Support multi-language interface.
