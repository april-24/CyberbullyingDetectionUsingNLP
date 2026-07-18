"""
app.py  -  Streamlit demo UI
============================
A simple web interface for the demo (Week 12-14). Type a comment, pick a model,
and see whether it is flagged as cyberbullying and under which categories.

Run:
    streamlit run app.py

It loads whichever trained models exist in results/. Train at least one model
first (see README).
"""

import os
import pickle
import joblib
import numpy as np
import streamlit as st

from src.preprocessing import clean_text

st.set_page_config(page_title="Cyberbully Comment Detection", page_icon="🛡️")

NB_PATH = "results/model_nb.joblib"
SVM_PATH = "results/model_svm.joblib"
#LSTM_PATH = "results/model_bilstm.keras"
#LSTM_TOK = "results/bilstm_tokenizer.pkl"


@st.cache_resource
def load_sklearn(path):
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
#def load_lstm():
#    if not (os.path.exists(LSTM_PATH) and os.path.exists(LSTM_TOK)):
#        return None
#    import tensorflow as tf
#    model = tf.keras.models.load_model(LSTM_PATH)
#    with open(LSTM_TOK, "rb") as f:
#        meta = pickle.load(f)
#    return {"model": model, **meta}


def predict_sklearn(bundle, text):
    cleaned = clean_text(text)
    pred = bundle["pipeline"].predict([cleaned])[0]
    return dict(zip(bundle["labels"], pred))


#def predict_lstm(bundle, text):
#    from tensorflow.keras.preprocessing.sequence import pad_sequences
#    cleaned = clean_text(text)
#    seq = bundle["tokenizer"].texts_to_sequences([cleaned])
#    seq = pad_sequences(seq, maxlen=bundle["max_len"], padding="post", truncating="post")
#    probs = bundle["model"].predict(seq, verbose=0)[0]
#    return {lab: int(p >= 0.5) for lab, p in zip(bundle["labels"], probs)}


st.title("🛡️ Cyberbully Comment Detection")
st.caption("NLP multi-label classification — AI assignment prototype")

# Which models are available?
available = {}
if load_sklearn(NB_PATH):
    available["Naive Bayes"] = ("nb", load_sklearn(NB_PATH))
if load_sklearn(SVM_PATH):
    available["Linear SVM"] = ("svm", load_sklearn(SVM_PATH))
#if load_lstm():
#    available["BiLSTM"] = ("lstm", load_lstm())

if not available:
    st.warning("No trained models found in results/. Train a model first "
               "(see README), then reload this page.")
    st.stop()

choice = st.selectbox("Choose a model", list(available.keys()))
text = st.text_area("Enter a comment to analyze:",
                    "You are so dumb, nobody likes you.", height=120)

if st.button("Analyze", type="primary"):
    kind, bundle = available[choice]
    #result = predict_lstm(bundle, text) if kind == "lstm" else predict_sklearn(bundle, text)

    flagged = [lab for lab, v in result.items() if v == 1]
    if flagged:
        st.error("⚠️ Flagged as CYBERBULLYING")
        st.write("Categories detected:")
        for lab in flagged:
            st.markdown(f"- **{lab}**")
    else:
        st.success("✅ No cyberbullying detected — comment looks clean.")

    with st.expander("Raw label outputs"):
        st.json(result)
