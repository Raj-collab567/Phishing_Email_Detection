from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocessing import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "sample_emails.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH).dropna(subset=["email_text", "label"])
    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["label"].isin(["Phishing", "Safe"])]
    if data["label"].nunique() != 2 or len(data) < 20:
        raise ValueError("Dataset must contain at least 20 rows and both labels: Phishing and Safe.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["email_text"].astype(str), data["label"],
        test_size=0.25, random_state=42, stratify=data["label"]
    )

    features = FeatureUnion([
        ("text", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=12000, sublinear_tf=True
        )),
        ("url_indicators", Pipeline([
            ("extract", FunctionTransformer(extract_url_features, validate=False)),
            ("scale", StandardScaler(with_mean=False))
        ]))
    ])

    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_phishing": round(float(precision_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
        "recall_phishing": round(float(recall_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
        "f1_phishing": round(float(f1_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
        "confusion_matrix_labels": ["Phishing", "Safe"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["Phishing", "Safe"]).tolist(),
        "test_samples": int(len(y_test))
    }

    joblib.dump(model, MODEL_DIR / "phishing_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model saved to models/phishing_model.joblib")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("\nIMPORTANT: metrics from the included illustrative demo dataset are not evidence of real-world performance.")

if __name__ == "__main__":
    main()
