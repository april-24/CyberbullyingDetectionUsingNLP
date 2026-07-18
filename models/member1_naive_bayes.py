"""
member1_naive_bayes.py   (MEMBER 1's solution)
==============================================
Method : Multinomial Naive Bayes + TF-IDF features
Wrapper: OneVsRestClassifier (trains one NB per label -> multi-label)

Naive Bayes is a fast probabilistic baseline. It assumes word features are
conditionally independent given the class. Despite that "naive" assumption it
works surprisingly well on text.

Run from the project root:
    python -m models.member1_naive_bayes
Add a quick test on a subset:
    python -m models.member1_naive_bayes --sample 20000
"""

import os
import sys
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# allow "python -m models.member1_naive_bayes" to import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import prepare_data
from src.evaluate import evaluate_model, save_result

MODEL_NAME = "Naive Bayes (TF-IDF)"
MODEL_PATH = os.path.join("results", "model_nb.joblib")


def build_pipeline():
    """TF-IDF vectorizer + One-vs-Rest Multinomial Naive Bayes in one pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,      # cap vocabulary size
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,                # ignore words in fewer than 2 docs
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(MultinomialNB(alpha=0.3))),
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
    ap.add_argument("--sample", type=int, default=None,
                    help="use only N rows for a quick run")
    args = ap.parse_args()
    main(sample=args.sample)
