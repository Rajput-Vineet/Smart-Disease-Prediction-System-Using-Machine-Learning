# Preprocessing Pipeline

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

# 🔴 IMPORTANT CHANGE HERE (Regression target)
X = df.drop(['Diagnosis', 'Severity'], axis=1)
y = df['Severity']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Polynomial Regression Model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Convert to polynomial features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create model
model = LinearRegression()

# Train
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

plt.scatter(y_test, y_pred, color='blue')

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted - Polynomial Regression")

plt.show()

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)
