
# Loan Prediction ML Project

## 📌 Overview
This project predicts whether a loan application will be approved or not based on applicant details such as income, loan amount, CIBIL score, and asset values.  
The workflow includes **data preprocessing, outlier treatment, feature scaling, categorical encoding, model training, and evaluation**.

---

## ⚙️ Steps in the Project
1. **Data Preprocessing**
   - Outlier treatment using IQR method  
   - Feature scaling with StandardScaler  
   - Encoding categorical variables (LabelEncoder)

2. **Model Training**
   - Logistic Regression  
   - Random Forest  
   - XGBoost  

3. **Model Evaluation**
   - Classification reports  
   - Comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)  
   - Feature importance plots  

3. **Deployement**
   - Deployed an app which can predict whether the loan will approve or reject.
---

##  Results
- **Logistic Regression**: Baseline model  
- **Random Forest**: Improved performance with feature importance insights  
- **XGBoost**: Best performance on tabular data with tuned hyperparameters  

Feature importance analysis shows that **CIBIL score, income, and loan amount** are the most influential features in predicting loan approval.

