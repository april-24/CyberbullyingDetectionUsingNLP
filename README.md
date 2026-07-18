# Cyberbully Comment Detection (NLP — Multi-Label Classification)

AI Assignment — Title 4: **Natural Language Processing** — TARUMT Session 202605.

Given an online comment, the system predicts **six binary labels at once**:

| Label | Meaning |
|-------|---------|
| `abusive` | Is the comment offensive or hate speech? (the core cyberbullying flag) |
| `Race` | Does it target a racial/ethnic group? |
| `Religion` | Does it target a religion? |
| `Gender` | Does it target a gender? |
| `Sexual_Orientation` | Does it target a sexual orientation? |
| `Miscellaneous` | Does it target another group (refugee, disability, etc.)? |

Because one comment can carry several of these at once (12,097 comments do), this
is a genuine **multi-label** classification problem, not a single-label one.

**Three members, three different methods, one fair comparison:**

| Member | Method | File |
|--------|--------|------|
| Member 1 | Multinomial **Naive Bayes** + TF-IDF | `models/member1_naive_bayes.py` |
| Member 2 | **Linear SVM** + TF-IDF | `models/member2_svm.py` |
| Member 3 | **BiLSTM** deep neural network + word embeddings | `models/member3_bilstm.py` |

---

## Dataset

**HateXplain** (`final_hateXplain.csv`, 20,109 rows), from the Kaggle set you
selected: <https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification>

Original columns → how we use them:
- `comment` → input text
- `label` (normal / offensive / hatespeech) → builds the `abusive` flag
- `Race`, `Religion`, `Gender`, `Sexual Orientation`, `Miscellaneous` → build the 5 target flags

The CSV is already placed in `data/` so the project runs out of the box. The
label construction lives in `src/data_loader.py` (function `build_labels`).

---

## How this meets the NLP assignment requirements (a–g)

- **(a) Identify an NLP task** — Text Classification: detecting cyberbullying and
  its target communities.
- **(b) Background study** — covered in the documentation (Introduction + Related Work).
- **(c) Web crawler OR dataset from a reliable website** — we take the second
  option and use the public HateXplain dataset from Kaggle. ✅ No crawler needed.
- **(d) Preprocessing** — `src/preprocessing.py` does text cleaning (removing
  URLs, mentions, punctuation, numbers, special characters), **tokenization**, stop-word
  removal, and **lemmatization** (used in place of stemming as it produces real
  dictionary words). **Feature extraction**: TF-IDF for NB/SVM, and a trainable
  **word-embedding** layer for the BiLSTM.
- **(e) Each member implements a DIFFERENT method** — Naive Bayes, Linear SVM,
  and a BiLSTM deep-learning model. ✅
- **(f) Compare & evaluate** — `compare_models.py` reports **Accuracy, Precision,
  Recall, and F1** (per-label + micro/macro). ✅
- **(g) Datasets/ideas** — HateXplain, a well-known academic hate-speech dataset.

---

## 1. Setup

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

> If you only run Naive Bayes + SVM you can skip the heavy `tensorflow` install
> (comment it out in `requirements.txt`). Member 3's BiLSTM needs it.

## 2. Explore the data (for the documentation)

```bash
python run_eda.py
```
Saves 3 charts to `results/` (label counts, comment-length histogram, label
co-occurrence heatmap) for your *Methodology / Dataset* section.

## 3. Train the models

```bash
python -m models.member1_naive_bayes      # Member 1
python -m models.member2_svm              # Member 2
python -m models.member3_bilstm           # Member 3 (needs tensorflow)
```
Add `--sample 20000` for a quick trial run. Each script prints a full metrics
report, saves its model to `results/`, and appends scores to
`results/model_scores.csv`.

## 4. Compare all models

```bash
python compare_models.py
```
Produces `results/comparison_table.csv` and `results/comparison_f1.png`.

## 5. Live demo (Week 12–14 presentation)

```bash
streamlit run app.py
```
Type a comment, pick a model, and see the predicted categories.

---

## Results already obtained on the full dataset

| Model | Micro-F1 | Macro-F1 | Subset Acc. |
|-------|:-------:|:-------:|:-----------:|
| Naive Bayes | 0.60 | 0.41 | 0.29 |
| Linear SVM | **0.69** | **0.65** | **0.35** |

Discussion hook for your report: Naive Bayes has high *precision* but poor
*recall* on rarer labels (Gender, Sexual Orientation, Miscellaneous), while the
class-balanced SVM trades a little precision for much better recall and a far
higher macro-F1 — showing the effect of class imbalance in multi-label text data.
(BiLSTM numbers depend on your training run.)

---

## How it works (for your Q&A)

1. **Preprocessing** (`src/preprocessing.py`) — the same cleaning for all three
   models, so the comparison is fair.
2. **Label construction** (`src/data_loader.py`) — turns HateXplain's categorical
   columns into the 6 binary labels described above.
3. **Feature extraction** — TF-IDF (NB, SVM) / trainable embeddings (BiLSTM).
4. **Multi-label handling** — `OneVsRestClassifier` (NB, SVM) and a per-label
   **sigmoid** output (BiLSTM).
5. **Evaluation** (`src/evaluate.py`) — subset accuracy, Hamming loss, and
   precision/recall/F1 with micro & macro averaging + a per-label report.

## Project structure

```
cyberbully_detection/
├── data/
│   └── final_hateXplain.csv     the dataset (already included)
├── src/
│   ├── data_loader.py           builds the 6 binary labels from HateXplain
│   ├── preprocessing.py         shared text cleaning pipeline
│   ├── common.py                load + clean + train/test split
│   └── evaluate.py              shared multi-label metrics
├── models/
│   ├── member1_naive_bayes.py
│   ├── member2_svm.py
│   └── member3_bilstm.py
├── run_eda.py                   exploratory charts
├── compare_models.py            comparison table + chart
├── app.py                       Streamlit demo UI
├── requirements.txt
└── results/                     models, charts, scores
```

## Troubleshooting

- **NLTK download errors** — the code falls back to a built-in stopword list and
  still runs; lemmatization is skipped in that case.
- **TensorFlow too slow / no GPU** — run `python -m models.member3_bilstm --sample 30000 --epochs 3`,
  or train on Google Colab (Runtime → Change runtime type → GPU) for free.
- **Different results per run** — the split is fixed (`random_state=42`), so
  NB/SVM are reproducible; the BiLSTM varies slightly due to random weight init.
