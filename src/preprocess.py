import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("dataset/disease_diagnosis.csv")

    # Remove unnecessary column
    df = df.drop("Patient_ID", axis=1)

    # Split Blood Pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
    df['Systolic_BP'] = df['Systolic_BP'].astype(int)
    df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)

    # Drop original column
    df = df.drop('Blood_Pressure_mmHg', axis=1)

    # Encode categorical columns
    categorical_columns = [
        'Gender',
        'Symptom_1',
        'Symptom_2',
        'Symptom_3',
        'Diagnosis',
        'Severity',
        'Treatment_Plan'
    ]

    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and target
    X = df.drop(['Diagnosis', 'Severity', 'Treatment_Plan'], axis=1)
    y = df['Diagnosis']

    # Scaling (IMPORTANT: only fit here, split later)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders, X.columns
