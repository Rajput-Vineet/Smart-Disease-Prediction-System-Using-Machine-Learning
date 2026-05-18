# -------------------------------
# Preprocessing Pipeline
# -------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("disease_diagnosis.csv")

# Remove Patient_ID
df = df.drop("Patient_ID", axis=1)

# Split BP
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
df = df.drop('Blood_Pressure_mmHg', axis=1)

# Encode
le = LabelEncoder()
categorical_columns = [
    'Gender','Symptom_1','Symptom_2','Symptom_3',
    'Diagnosis','Severity','Treatment_Plan'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Features & target
X = df.drop(['Diagnosis', 'Severity'], axis=1)
y = df['Diagnosis']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# XGBoost Model
# -------------------------------

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -------------------------------
# RESULTS
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# ✅ CONFUSION MATRIX GRAPH (FINAL)
# -------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
