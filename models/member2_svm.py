"""
member2_svm.py   (MEMBER 2's solution)
======================================
Method : Linear Support Vector Machine (LinearSVC) + TF-IDF features
Wrapper: OneVsRestClassifier (one SVM per label -> multi-label)

An SVM finds the hyperplane that best separates the two classes with the widest
margin. On sparse high-dimensional text (TF-IDF), a *linear* SVM is a very strong
and fast classifier and is a classic benchmark for text classification.

Run from the project root:
    python -m models.member2_svm
    python -m models.member2_svm --sample 20000
"""

import os
import sys
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import prepare_data
from src.evaluate import evaluate_model, save_result

MODEL_NAME = "Linear SVM (TF-IDF)"
MODEL_PATH = os.path.join("results", "model_svm.joblib")


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced"))),
    ])


def main(sample=None):
    X_train, X_test, y_train, y_test, labels = prepare_data(sample=sample)

    print(f"\nTraining {MODEL_NAME} ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    result = evaluate_model(MODEL_NAME, y_test.values, y_pred, labels)
    save_result(result)

    os.makedirs("results", exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": labels}, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    main(sample=args.sample)
