# 🩺 Smart Disease Prediction System (ML Project)

An end-to-end **Machine Learning-based Disease Prediction System** that predicts diseases based on patient symptoms and health parameters. The project includes a complete ML pipeline, model comparison, evaluation, and deployment using Streamlit.

🔗 Live App: https://smart-disease-prediction-system-using-machine-learning-cvhyy2r.streamlit.app
---

## 🚀 Overview

This project focuses on building a **robust classification system** for disease prediction using patient data such as symptoms, vital signs, and demographics.

👉 The system:
- Takes patient inputs (symptoms + vitals)
- Predicts the most probable disease
- Provides a **confidence score**
- Displays model insights like feature importance

---

## 🧠 Problem Statement

Disease prediction is a **classification problem**, where:

- **Input:** Symptoms + health parameters  
- **Output:** Disease (Diagnosis)

Initially explored multiple approaches, but refined the solution to a **pure classification pipeline** for accuracy and clarity.

---

## 📊 Dataset

- Patient clinical dataset with:
  - Age, Gender  
  - Symptoms (Symptom_1, Symptom_2, Symptom_3)  
  - Vital signs (Temperature, Oxygen, Heart Rate, BP)  
- Target column: **Diagnosis**

---

## ⚙️ Machine Learning Pipeline

### 🔹 Data Preprocessing
- Removed unnecessary columns (e.g., Patient_ID)
- Handled categorical features using encoding
- Split blood pressure into systolic/diastolic
- Feature scaling using StandardScaler
- Stratified train-test split

---

### 🔹 Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest ⭐ (Final Model)  
- Support Vector Machine (SVM)

---

### 🔹 Model Evaluation

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix (with heatmap)  
- Cross Validation (5-fold)

---

### 🔹 Advanced Techniques

- ✅ **Hyperparameter Tuning** using GridSearchCV  
- ✅ **Feature Importance Analysis** (Random Forest)  
- ✅ **Multi-class ROC Curve (One-vs-Rest)**  
- ✅ Awareness of class imbalance  

---

## 🏆 Model Selection

All models were compared, and the best-performing model was selected automatically.

👉 Final Model:
- **Random Forest Classifier**
- Selected based on performance and generalization

---

## 📈 Results

- High classification accuracy (~94–99%)  
- Strong performance across most classes  
- Slight performance drop for minority classes (handled analytically)

---

## 💾 Model Saving

Saved components:
- `best_model.pkl`  
- `scaler.pkl`  
- `encoders.pkl`  
- `feature_columns.pkl`  

Ensures consistent predictions during deployment.

---

## 🖥️ Streamlit Web App

An interactive UI built using Streamlit:

### Features:
- Input patient details  
- Predict disease  
- Show **confidence score (%)**  
- Display class probabilities  
- Clean and modern UI  

---

## 📁 Project Structure

```
ML/
│
├── dataset/
│   └── disease_diagnosis.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│
├── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│
├── app/
│   └── streamlit_app.py
│
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train model
```bash
python src/train_model.py
```

### 3️⃣ Run Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Key Learnings

- Correct problem formulation (classification vs regression)
- Building modular preprocessing pipelines
- Model comparison and evaluation techniques
- Avoiding data leakage
- Feature importance interpretation
- Deploying ML models using Streamlit

---

## 💬 Interview Explanation

> “We developed a disease prediction system using multiple classification models and selected Random Forest based on performance. The pipeline includes preprocessing, model comparison, evaluation, and deployment using Streamlit. We also incorporated feature importance, cross-validation, and hyperparameter tuning to improve robustness.”

