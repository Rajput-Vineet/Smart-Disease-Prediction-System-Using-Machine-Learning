import matplotlib.pyplot as plt

# -------------------------------
# MODELS
# -------------------------------

models = [
    "Logistic Regression",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN"
]

# ✅ YOUR REAL VALUES (UPDATED)
accuracy = [
    0.945,     # Logistic
    0.98,      # Naive Bayes
    0.9925,    # Decision Tree
    0.9875,    # Random Forest
    0.95,      # SVM
    0.8925     # KNN
]

models.append("XGBoost")
accuracy.append(0.99)

# -------------------------------
# GRAPH
# -------------------------------

plt.figure(figsize=(10,6))

bars = plt.bar(models, accuracy, color='skyblue', edgecolor='black')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval,
        round(yval, 4),
        ha='center',
        va='bottom'
    )

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.xticks(rotation=30)

# Adjusted range (important because KNN is lower)
plt.ylim(0.85, 1.0)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
