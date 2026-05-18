# Preprocessing Pipeline

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("disease_diagnosis.csv")

# Remove Patient_ID
df = df.drop("Patient_ID", axis=1)

# Split BP
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
df = df.drop('Blood_Pressure_mmHg', axis=1)

# Encode categorical columns
le = LabelEncoder()
categorical_columns = [
    'Gender','Symptom_1','Symptom_2','Symptom_3',
    'Diagnosis','Severity','Treatment_Plan'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Features only
X = df.drop(['Diagnosis', 'Severity'], axis=1)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# K-Means Model
# -------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_

# -------------------------------
# ✅ PCA FOR BETTER GRAPH (IMPORTANT)
# -------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Transform cluster centers also
centers_pca = pca.transform(kmeans.cluster_centers_)

# -------------------------------
# ✅ FINAL GRAPH
# -------------------------------

plt.figure(figsize=(7,5))

# Data points
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap='viridis',
    alpha=0.7
)

# Cluster centers
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centers'
)

plt.title("K-Means Clustering (PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.show()

# -------------------------------
# Add cluster labels to dataset
# -------------------------------

df['Cluster'] = labels

print(df.head())
print(df['Cluster'].value_counts())
print(df.groupby('Cluster').mean())
