import pandas as pd
df = pd.read_csv("disease_diagnosis.csv")
print(df.head())

# Remove Patient_ID column

df = df.drop("Patient_ID", axis=1)

print(df.head())

# Split Blood Pressure into two columns

df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)

# Convert into integer type

df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)

# Remove original Blood Pressure column

df = df.drop('Blood_Pressure_mmHg', axis=1)

# Display first rows

print(df.head())

# Import LabelEncoder

from sklearn.preprocessing import LabelEncoder

# Create encoder object

le = LabelEncoder()

# List of categorical columns

categorical_columns = [
    'Gender',
    'Symptom_1',
    'Symptom_2',
    'Symptom_3',
    'Diagnosis',
    'Severity',
    'Treatment_Plan'
]

# Apply label encoding

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Display dataset

print(df.head())

# Define features and target (classification)

X = df.drop(['Diagnosis', 'Severity'], axis=1)
y_class = df['Diagnosis']

# Define regression target separately

y_reg = df['Severity']

print("Features shape:", X.shape)
print("Classification target shape:", y_class.shape)
print("Regression target shape:", y_reg.shape)

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Apply scaling to features
X_scaled = scaler.fit_transform(X)

print("Scaled features shape:", X_scaled.shape)

from sklearn.model_selection import train_test_split

# Classification split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
