import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------
# Load preprocessed data
# -------------------------------
X, y, scaler, encoders, feature_names = load_and_preprocess_data()

# -------------------------------
# Train-test split (stratified)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 🔥 Hyperparameter Tuning (RF)
# -------------------------------
print("\n🔧 Tuning Random Forest...\n")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print("Best RF Parameters:", grid.best_params_)

# -------------------------------
# Define models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": best_rf,   # ✅ tuned model
    "SVM": SVC(probability=True)
}

results = {}
trained_models = {}

print("\n📊 MODEL COMPARISON:\n")

# -------------------------------
# Train and evaluate models
# -------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

# -------------------------------
# Select best model (prefer RF)
# -------------------------------
if "Random Forest" in results:
    best_model_name = "Random Forest"
else:
    best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]:.4f}")

# -------------------------------
# Cross Validation
# -------------------------------
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\n📊 Cross Validation Score: {cv_scores.mean():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix (Heatmap)
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Feature Importance (RF only)
# -------------------------------
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# -------------------------------
# 🔥 ROC Curve (Multi-class)
# -------------------------------
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(8, 6))

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multi-class)")
plt.legend()
plt.show()

# -------------------------------
# Save model and preprocessing
# -------------------------------
os.makedirs("saved_models", exist_ok=True)

joblib.dump(best_model, "saved_models/best_model.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(encoders, "saved_models/encoders.pkl")
joblib.dump(feature_names, "saved_models/feature_columns.pkl")

print("\n✅ Model and preprocessing objects saved successfully!")

# -------------------------------
# Sample Prediction Probability
# -------------------------------
sample_probs = best_model.predict_proba(X_test[:1])
print("\n📊 Prediction Probabilities:\n", sample_probs)
