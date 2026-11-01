"""Train and evaluate the customer complaint classifier.

Improvements over the original script:
1. Cleans duplicate, empty, and conflicting examples.
2. Uses word + character TF-IDF features.
3. Tunes model hyperparameters with stratified cross-validation.
4. Selects the best model using macro F1, which treats small categories fairly.
5. Saves richer reports and misclassified examples for error analysis.
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ==========================================
# 1. SETTINGS
# ==========================================

DATA_PATH = "data/processed/cleaned_complaints.csv"
MODEL_PATH = "models/best_model.pkl"
COMPARISON_PATH = "results/model_comparison.csv"
REPORT_PATH = "results/classification_report.csv"
ERRORS_PATH = "results/misclassified_examples.csv"
CONFUSION_PATH = "results/confusion_matrix.png"
METADATA_PATH = "models/model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==========================================
# 2. LOAD + CLEAN DATA
# ==========================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required_columns = {"instruction", "category"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required column(s): {sorted(missing_columns)}"
    )

df = df[["instruction", "category"]].copy()
df["instruction"] = (
    df["instruction"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df["category"] = df["category"].fillna("").astype(str).str.strip()

# Remove empty rows.
df = df[(df["instruction"] != "") & (df["category"] != "")]

# Remove exact duplicate text/category pairs.
df = df.drop_duplicates(subset=["instruction", "category"])

# A complaint appearing with different labels is ambiguous training data.
# Remove those texts instead of teaching the model contradictory labels.
label_counts_per_text = df.groupby("instruction")["category"].nunique()
conflicting_texts = label_counts_per_text[label_counts_per_text > 1].index
if len(conflicting_texts):
    df = df[~df["instruction"].isin(conflicting_texts)]

print("Clean dataset shape:", df.shape)
print("Categories:", df["category"].nunique())
print("Smallest category:", int(df["category"].value_counts().min()))

if df["category"].value_counts().min() < 2:
    raise ValueError(
        "Every category needs at least 2 examples for a stratified train/test split."
    )

X = df["instruction"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

min_train_examples = int(y_train.value_counts().min())
actual_cv_folds = min(CV_FOLDS, min_train_examples)
if actual_cv_folds < 2:
    raise ValueError("Not enough examples per category for cross-validation.")

cv = StratifiedKFold(
    n_splits=actual_cv_folds,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ==========================================
# 3. FEATURES + MODELS
# ==========================================

def make_features():
    """Combine phrase-level word features with character features.

    Word n-grams capture meaning such as "charged twice".
    Character n-grams help with spelling variations such as "refridgerator"
    and related forms such as "not cooling" / "not-cooling".
    """
    return FeatureUnion(
        [
            (
                "word_features",
                TfidfVectorizer(
                    analyzer="word",
                    strip_accents="unicode",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character_features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={
            "word_features": 1.0,
            "character_features": 0.5,
        },
    )


def make_pipeline(classifier):
    return Pipeline(
        [
            ("features", make_features()),
            ("classifier", classifier),
        ]
    )


models = {
    "Naive Bayes": (
        make_pipeline(MultinomialNB()),
        {"classifier__alpha": [0.1, 0.5, 1.0]},
    ),
    "Logistic Regression": (
        make_pipeline(
            LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        {"classifier__C": [0.5, 1.0, 2.0, 4.0]},
    ),
    "Linear SVM": (
        make_pipeline(
            LinearSVC(
                class_weight="balanced",
                max_iter=5_000,
                random_state=RANDOM_STATE,
            )
        ),
        {"classifier__C": [0.25, 0.5, 1.0, 2.0, 4.0]},
    ),
}


# ==========================================
# 4. TRAIN + EVALUATE
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

best_model = None
best_name = None
best_f1 = -1.0
best_predictions = None
comparison = []

for name, (pipeline, parameter_grid) in models.items():
    print(f"\nTuning: {name}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print("Best parameters:", search.best_params_)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1 * 100, 2), "%")
    print(classification_report(y_test, predictions, zero_division=0))

    comparison.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro F1": macro_f1,
            "Best Parameters": json.dumps(search.best_params_),
        }
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_name = name
        best_predictions = predictions


if best_model is None:
    raise RuntimeError("No model was trained successfully.")


# ==========================================
# 5. SAVE BEST MODEL + METADATA
# ==========================================

joblib.dump(best_model, MODEL_PATH)

metadata = {
    "model_name": best_name,
    "macro_f1": round(float(best_f1), 6),
    "categories": sorted(str(label) for label in best_model.classes_),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cv_folds": actual_cv_folds,
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
    json.dump(metadata, metadata_file, indent=2)

comparison_df = pd.DataFrame(comparison).sort_values(
    "Macro F1", ascending=False
)
comparison_df.to_csv(COMPARISON_PATH, index=False)


# ==========================================
# 6. SAVE REPORTS + ERROR ANALYSIS
# ==========================================

report = classification_report(
    y_test,
    best_predictions,
    labels=best_model.classes_,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose().reset_index()
report_df.rename(columns={"index": "Label"})
report_df.to_csv(REPORT_PATH, index=False)

errors_df = pd.DataFrame(
    {
        "Complaint": X_test.values,
        "Actual Category": y_test.values,
        "Predicted Category": best_predictions,
    }
)
errors_df = errors_df[
    errors_df["Actual Category"] != errors_df["Predicted Category"]
]
errors_df.to_csv(ERRORS_PATH, index=False)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

labels = best_model.classes_
cm = confusion_matrix(y_test, best_predictions, labels=labels)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

fig, ax = plt.subplots(figsize=(13, 10))
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False,
)
ax.set_title(f"Customer Complaint Classification · {best_name}")
plt.tight_layout()
plt.savefig(CONFUSION_PATH, dpi=160, bbox_inches="tight")
plt.close(fig)


print("\nBest model:", best_name)
print("Best Macro F1:", round(best_f1 * 100, 2), "%")
print("Saved:", MODEL_PATH)
print("Saved:", COMPARISON_PATH)
print("Saved:", REPORT_PATH)
print("Saved:", ERRORS_PATH)
print("Saved:", CONFUSION_PATH)
print("\nTraining completed successfully!")
