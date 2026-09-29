"""Train and evaluate StressRisk AI models from the sole CSV in Dataset/."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "saved_model"
ANALYSIS_DIR = BASE_DIR / "analysis"
TARGET = "Stress Level"
EXCLUDED = {"Person ID", "erson ID", "Meal Regularity", "Meditation Practice", TARGET}


def find_dataset() -> Path:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV file exists in Dataset/")
    return files[0]


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def build_models() -> dict:
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=2,
        )
    except (ImportError, RuntimeError, ValueError):
        pass
    return models


def main() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    data_path = find_dataset()
    data = pd.read_csv(data_path)
    data = data.drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    target_counts = data[TARGET].value_counts()
    excluded_singletons = target_counts[target_counts < 2].index.tolist()
    if excluded_singletons:
        data = data[~data[TARGET].isin(excluded_singletons)].copy()
    features = [column for column in data.columns if column not in EXCLUDED]
    X, y = data[features], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = make_preprocessor(X)
    models = build_models()
    results = {}
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = sorted(y.unique())
    xgb_classes_metadata = None
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        fit_y = y_train
        decode = lambda values: values
        if name == "XGBoost":
            xgb_classes = sorted(y_train.unique())
            xgb_classes_metadata = xgb_classes
            encode = {label: index for index, label in enumerate(xgb_classes)}
            decode = lambda values: [xgb_classes[int(value)] for value in values]
            fit_y = y_train.map(encode)
        pipeline.fit(X_train, fit_y)
        predicted = decode(pipeline.predict(X_test))
        report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
        cv_scores = cross_val_score(pipeline, X_train, fit_y, cv=cv, scoring="accuracy", n_jobs=1)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        }
        fitted[name] = pipeline
        ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=labels, xticks_rotation="vertical")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=140)
        plt.close()

    # Prefer Random Forest when it is statistically competitive; otherwise use the best test F1.
    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["accuracy"]))
    if "Random Forest" in results and results["Random Forest"]["f1"] >= results[best_name]["f1"] - 0.02:
        best_name = "Random Forest"
    final_model = fitted[best_name]
    joblib.dump(final_model, MODEL_DIR / "stressrisk_model.pkl")
    joblib.dump(final_model.named_steps["preprocessor"], MODEL_DIR / "preprocessor.pkl")
    joblib.dump(features, MODEL_DIR / "feature_columns.pkl")
    metadata = {
        "dataset": data_path.name,
        "duplicate_rows_removed": int(pd.read_csv(data_path).duplicated().sum()),
        "singleton_target_values_excluded_for_stratification": excluded_singletons,
        "features": features,
        "target": TARGET,
        "target_values": labels,
        "defaults": {column: (float(X[column].median()) if pd.api.types.is_numeric_dtype(X[column]) else str(X[column].mode(dropna=True).iloc[0])) for column in features},
        "selected_model": best_name,
        "xgboost_classes": xgb_classes_metadata,
        "results": results,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    data.describe(include="all").transpose().to_csv(ANALYSIS_DIR / "descriptive_statistics.csv")
    data.isna().sum().rename("missing_values").to_csv(ANALYSIS_DIR / "missing_values.csv")
    data[TARGET].value_counts().sort_index().plot(kind="bar", title="Stress Level distribution")
    plt.tight_layout(); plt.savefig(ANALYSIS_DIR / "target_distribution.png", dpi=140); plt.close()
    numeric = data.select_dtypes(include="number")
    if len(numeric.columns) > 1:
        numeric.corr().to_csv(ANALYSIS_DIR / "correlation_matrix.csv")
    return metadata


if __name__ == "__main__":
    summary = main()
    for model, scores in summary["results"].items():
        print(f"{model}: accuracy={scores['accuracy']:.4f}, precision={scores['precision']:.4f}, recall={scores['recall']:.4f}, f1={scores['f1']:.4f}")
    print(f"Selected model: {summary['selected_model']}")
