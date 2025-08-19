# Stroke-Prediction
Machine Learning for Stroke Risk Prediction


---

## 📌 Objectives

- Predict stroke risk using machine learning models
- Compare model performance across two different datasets
- Interpret model predictions using SHAP (SHapley Additive exPlanations)
- Evaluate the impact of features on stroke prediction
- Propose a workflow for potential clinical deployment

---

## 🧪 Datasets Used

1. **Dataset 1 (Primary)**  
   Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
   Contains demographic and lifestyle factors such as age, hypertension, heart disease, BMI, and smoking status.

2. **Dataset 2 (Experimental)**  
   An additional dataset was introduced to assess generalizability and compare feature importance patterns across datasets.
   Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset) 
---

## ⚙️ Models Used

The following machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

All models were evaluated using:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

---

## 🧠 Feature Importance (Explainability)

SHAP was used to explain model decisions and identify the most impactful features.  
Consistently important features across datasets include:
- Age
- Average glucose level
- Hypertension
- Smoking status

---

## ⚖️ Handling Class Imbalance

Both datasets were highly imbalanced.  
To address this, we used the **SMOTE-ENN** technique, which combines:
- **SMOTE**: Generates synthetic samples for the minority class
- **ENN (Edited Nearest Neighbors)**: Removes noisy or misclassified instances from the majority class

---

## 📊 Data Split

Each dataset was split into:
- **Training set**: 70%
- **Testing set**: 30%

---

## 🔍 Key Results

| Model              | Dataset 1 (Best)     | Dataset 2 (Best)     |
|-------------------|----------------------|----------------------|
| **Best Model**     | LightGBM             | Logistic Regression  |
| **Highest Recall** | LightGBM             | Logistic Regression  |
| **Interpretability** | SHAP values consistent across key features |

---

## 🔮 Future Work

- Incorporate additional clinical variables (e.g., blood pressure, cholesterol)
- Validate models on real-world hospital data
- Explore deep learning models like TabNet and FNNs
- Develop an interactive web-based tool for clinicians
- Experiment with cost-sensitive learning to address false negatives in medical diagnosis

---

## 🙋‍♂️ Author

Sameer-Ul-Haq  
MSc Data Science & Analytics  
Toronto Metropolitan University

Feel free to connect on  [Email](mailto:sameerulhaq88@outlook.com)

