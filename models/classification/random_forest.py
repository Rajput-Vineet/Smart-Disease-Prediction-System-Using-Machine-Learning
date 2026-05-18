# Preprocessing Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
encoders = pickle.load(open("encoders.pkl", "rb"))
from sklearn.metrics import confusion_matrix
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


# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoders['Diagnosis'].classes_,
    yticklabels=encoders['Diagnosis'].classes_
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))
