import pandas as pd
df = pd.read_csv("disease_diagnosis.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.nunique())
print(df.dtypes)
print(df['Diagnosis'].value_counts())
print(df.corr(numeric_only=True))
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
