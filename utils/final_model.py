import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv("disease_diagnosis.csv")

# Preprocessing
df = df.drop("Patient_ID", axis=1)

# Split BP
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
df = df.drop('Blood_Pressure_mmHg', axis=1)

# ✅ Create encoders dictionary
encoders = {}

categorical_columns = [
    'Gender','Symptom_1','Symptom_2','Symptom_3',
    'Diagnosis','Severity','Treatment_Plan'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & target
X = df.drop(['Diagnosis', 'Severity'], axis=1)
y = df['Diagnosis']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-Test Split (NEW)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# -------------------------------
# ✅ EVALUATION METRICS (NEW)
# -------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

cm = confusion_matrix(y_test, y_pred)

# Simple specificity (for first class)
TN = cm[0][0]
FP = cm[0][1]
specificity = TN / (TN + FP)

roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("\n📊 MODEL EVALUATION")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Specificity:", specificity)
print("ROC-AUC:", roc)

# -------------------------------
# USER INPUT
# -------------------------------

print("\nEnter patient details:")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ")
symptom1 = input("Symptom 1: ")
symptom2 = input("Symptom 2: ")
symptom3 = input("Symptom 3: ")
heart_rate = int(input("Heart Rate: "))
bp = input("Blood Pressure (e.g. 120/80): ")
treatment = input("Treatment Plan: ")
temperature = float(input("Body Temperature (C): "))
oxygen = int(input("Oxygen Saturation (%): "))

# Split BP
systolic, diastolic = bp.split('/')
systolic = int(systolic)
diastolic = int(diastolic)

# Create dataframe
input_data = pd.DataFrame([[
    age, gender, symptom1, symptom2, symptom3,
    temperature, oxygen, heart_rate, treatment, systolic, diastolic
]], columns=[
    'Age','Gender','Symptom_1','Symptom_2','Symptom_3',
    'Body_Temperature_C','Oxygen_Saturation_%',
    'Heart_Rate_bpm','Treatment_Plan','Systolic_BP','Diastolic_BP'
])

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Apply encoders
for col in ['Gender','Symptom_1','Symptom_2','Symptom_3','Treatment_Plan']:
    try:
        input_data[col] = encoders[col].transform(input_data[col])
    except:
        print(f"⚠️ Unknown value in {col}, setting default")
        input_data[col] = 0

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# Decode output
predicted_label = encoders['Diagnosis'].inverse_transform(prediction)

print("\nPredicted Diagnosis:", predicted_label[0])

import pickle

# Save feature column order
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))
