"""
member3_bilstm.py   (MEMBER 3's solution)
=========================================
Method : Bidirectional LSTM neural network with a trainable word-embedding layer
Output : sigmoid layer (one neuron per label) -> multi-label

Unlike Naive Bayes / SVM which treat a comment as a bag of words, the BiLSTM reads
the comment as an ORDERED SEQUENCE and learns context from both directions
(left->right and right->left). This usually captures meaning that bag-of-words
models miss, at the cost of longer training time.

Requires: tensorflow  (pip install tensorflow)

Run from the project root:
    python -m models.member3_bilstm
    python -m models.member3_bilstm --sample 30000 --epochs 3
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import prepare_data
from src.evaluate import evaluate_model, save_result

MODEL_NAME = "BiLSTM (Deep Learning)"
MODEL_PATH = os.path.join("results", "model_bilstm.keras")
TOKENIZER_PATH = os.path.join("results", "bilstm_tokenizer.pkl")

MAX_WORDS = 20000      # vocabulary size
MAX_LEN = 100          # max tokens per comment (pad/truncate to this)
EMBED_DIM = 128


def main(sample=None, epochs=4, batch_size=128):
    # Imported here so the script only needs TensorFlow when actually run.
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D)
    from tensorflow.keras.callbacks import EarlyStopping

    X_train, X_test, y_train, y_test, labels = prepare_data(sample=sample)
    n_labels = len(labels)

    # --- Tokenize text into integer sequences ---
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(X_train)

    Xtr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN,
                        padding="post", truncating="post")
    Xte = pad_sequences(tok.texts_to_sequences(X_test), maxlen=MAX_LEN,
                        padding="post", truncating="post")
    ytr, yte = y_train.values, y_test.values

    # --- Build the network ---
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_labels, activation="sigmoid"),   # multi-label -> sigmoid
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.summary()

    print(f"\nTraining {MODEL_NAME} ...")
    model.fit(
        Xtr, ytr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1,
    )

    # --- Predict (threshold sigmoid probabilities at 0.5) ---
    probs = model.predict(Xte, batch_size=batch_size)
    y_pred = (probs >= 0.5).astype(int)

    result = evaluate_model(MODEL_NAME, yte, y_pred, labels)
    save_result(result)

    os.makedirs("results", exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump({"tokenizer": tok, "labels": labels, "max_len": MAX_LEN}, f)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved tokenizer -> {TOKENIZER_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    main(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size)
