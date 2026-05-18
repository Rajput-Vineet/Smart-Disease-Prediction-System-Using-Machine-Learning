# -------------------------------
# PREPROCESSING
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("disease_diagnosis.csv")

# Select relevant columns
df = df[[
    'Gender',
    'Symptom_1',
    'Symptom_2',
    'Symptom_3',
    'Diagnosis',
    'Treatment_Plan'
]]

print(df.head())

# One-hot encoding
df_encoded = pd.get_dummies(df)

print(df_encoded.head())

# -------------------------------
# APRIORI
# -------------------------------

from mlxtend.frequent_patterns import apriori, association_rules

# Frequent itemsets
frequent_items = apriori(df_encoded, min_support=0.1, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_items)

# Association rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:")
print(rules)

# -------------------------------
# ✅ FINAL GRAPH (IMPORTANT)
# -------------------------------

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    rules['support'],
    rules['confidence'],
    c=rules['lift'],
    cmap='viridis',
    s=80,              # bigger dots (better visibility)
    alpha=0.7,
    edgecolors='black' # improves clarity
)

plt.colorbar(scatter, label='Lift')

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Apriori Rules: Support vs Confidence (Lift as Color)")

plt.grid(True)  # makes graph more readable

plt.show()

# -------------------------------
# ✅ BEST RULES (TOP INSIGHT)
# -------------------------------

print("\nTop Rules (by Lift):")
print(rules.sort_values(by='lift', ascending=False).head())
