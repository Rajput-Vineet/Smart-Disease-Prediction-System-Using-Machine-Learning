import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# MODELS
# -------------------------------

models = [
    "Logistic",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN",
    "XGBoost"
]

# -------------------------------
# YOUR REAL VALUES (WEIGHTED AVG)
# -------------------------------

accuracy = [
    0.945,
    0.98,
    0.9925,
    0.9875,
    0.95,
    0.8925,
    0.99
]

precision = [
    0.94,
    0.98,
    0.99,
    0.99,
    0.94,
    0.88,
    0.99
]

recall = [
    0.94,
    0.98,
    0.99,
    0.99,
    0.95,
    0.89,
    0.99
]

f1_score = [
    0.94,
    0.98,
    0.99,
    0.99,
    0.94,
    0.88,
    0.99
]

# -------------------------------
# GRAPH SETUP
# -------------------------------

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12,6))

# Bars
plt.bar(x - 1.5*width, accuracy,  width, label='Accuracy')
plt.bar(x - 0.5*width, precision, width, label='Precision')
plt.bar(x + 0.5*width, recall,    width, label='Recall')
plt.bar(x + 1.5*width, f1_score,  width, label='F1 Score')

# Labels
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Multi-Metric Model Comparison")

plt.xticks(x, models, rotation=20)

# Adjust range for better visibility
plt.ylim(0.85, 1.0)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
