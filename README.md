# 📉 Customer Churn Prediction

This project aims to predict customer churn in a telecom company using machine learning techniques. Churn prediction helps businesses identify customers likely to leave, enabling proactive retention strategies.

## 📁 Dataset

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **File**: `Customer-Churn.csv`
- **Target Variable**: `Churn` (Yes/No)

### Key Features:
- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Services: `PhoneService`, `InternetService`, `StreamingTV`, etc.
- Account Info: `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 🧠 Project Workflow

### 1. Data Preprocessing
- Handled missing values in `TotalCharges`
- Converted `TotalCharges` to numeric
- Dropped irrelevant columns like `customerID`
- One-hot encoded categorical variables
- Encoded target variable (`Churn`: Yes → 1, No → 0)

### 2. Feature Scaling
- Used `StandardScaler` to normalize features

### 3. Model Training & Evaluation
Trained and evaluated three models:
- ✅ Logistic Regression
- ✅ Random Forest Classifier
- ✅ XGBoost Classifier (with class imbalance handling)

### 4. Performance Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 🧪 Results Summary

| Model               | Accuracy | Notes                          |
|--------------------|----------|--------------------------------|
| Logistic Regression| ~82%     | Baseline model                 |
| Random Forest      | ~79%     | Better generalization          |
| XGBoost            | ~76%     | Tuned for higher recall        |

---

## 💾 Model Deployment Prep

- Saved trained model and feature list using `joblib`
```python
joblib.dump({
    'model': model_xgb,
    'features': X_train.columns.tolist()
}, 'churn_model_bundle.pkl')
