import matplotlib.pyplot as plt

models = [
    "Linear",
    "Polynomial",
    "Ridge",
    "Lasso",
    "Decision Tree",
    "KNN"
]

r2_scores = [
    1.0,
    1.0,
    0.999999,
    0.984,
    1.0,
    0.949
]

plt.figure(figsize=(10,6))

bars = plt.bar(models, r2_scores, color='lightgreen', edgecolor='black')

# Add values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval,
             round(yval, 3),
             ha='center', va='bottom')

plt.title("Regression Model Comparison (R² Score)")
plt.xlabel("Models")
plt.ylabel("R² Score")

plt.ylim(0.9, 1.0)
plt.xticks(rotation=30)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
