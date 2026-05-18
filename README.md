# 🩺 Smart Disease Prediction System using Machine Learning

## 📌 Project Overview
This project is a Machine Learning-based Disease Diagnosis System that predicts diseases based on patient symptoms and clinical parameters. The system integrates multiple ML techniques including classification, regression, clustering, and association rule mining. A user-friendly Streamlit web application is developed to provide real-time predictions.

🔗 Live App: https://smart-disease-prediction-system-using-machine-learning-4gw8xid.streamlit.app

## 🚀 Features
- Predict disease using patient inputs  
- Interactive UI built with Streamlit  
- Multiple ML models implemented and compared  
- Clustering using K-Means  
- Association Rule Mining using Apriori  
- Clean and structured project architecture  

## 🧠 Machine Learning Models Used

### 🔹 Classification Models
- Random Forest (Final Model Used in App)  
- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- XGBoost  

### 🔹 Regression Models
- Linear Regression  
- Polynomial Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regression  
- KNN Regression  

### 🔹 Clustering
- K-Means Clustering  

### 🔹 Association Rule Mining
- Apriori Algorithm  

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  

## 📁 Project Structure
dataset/ → disease_diagnosis.csv  
models/ → classification, regression, clustering, association  
comparison/ → model comparison scripts  
saved_models/ → model.pkl, scaler.pkl, encoders.pkl, feature_columns.pkl  
utils/ → eda.py, preprocessing.py, final_model.py  
streamlit_app.py  
requirements.txt  
README.md  

## ⚙️ Installation & Setup

1. Clone the repository  
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  

2. Create virtual environment (recommended)  
python -m venv venv  

Activate environment:  
Windows → venv\Scripts\activate  
Mac/Linux → source venv/bin/activate  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the application  
streamlit run streamlit_app.py  

5. Open in browser  
http://localhost:8501  

## 🖥️ How to Use
1. Enter patient details (Age, Gender, Symptoms, Vitals, Treatment Plan)  
2. Click "Run Diagnostic Analysis"  
3. View predicted disease and confidence score  

## ⚠️ Common Errors & Fixes
- App not running → install requirements  
- Model not loading → check saved_models folder  
- Dataset error → ensure correct path (dataset/disease_diagnosis.csv)  

## ⚠️ Disclaimer
This system is for educational purposes only and should not be used for real medical diagnosis. Always consult a certified medical professional.  

## 👨‍💻 Author
Vineet Rajput  

## 📌 Project Type
Machine Learning Academic Project
