
import os
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load cleaned dataset
print("Loading cleaned dataset...")

df = pd.read_csv("data/processed/cleaned_complaints.csv")

X = df["instruction"]
y = df["category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced"
    )
}

results = []
trained_models = {}

# Train and evaluate
for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    })

    trained_models[name] = pipeline

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Macro F1: {f1:.4f}")

# Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    zero_division=0
))

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    best_model.predict(X_test),
    labels=best_model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

fig, ax = plt.subplots(figsize=(12, 9))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

plt.title("Customer Complaint Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

print("Confusion matrix saved successfully!")

# Save model and comparison results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nTraining completed successfully!")
print("Best model saved at: models/best_model.pkl")
print("Comparison saved at: results/model_comparison.csv")
