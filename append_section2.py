"""
append_section2.py
Appends Section 2 (Person B classical ML, Experiments 1-6) cells to
SDG3_Classification_Complete.ipynb.
"""
import nbformat

NOTEBOOK_PATH = 'SDG3_Classification_Complete.ipynb'

# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

cell11_md = """\
---
## Section 2 — Person B: Feature Engineering & Classical ML (Experiments 1–6)
*Nformi Modestine* — TF-IDF/BoW feature engineering, classical ML model comparison, threshold optimisation, class imbalance handling.

**Best result this section: Experiment 4 — LinearSVC, Hamming Loss = 0.045**\
"""

cell12_code = """\
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

TRAIN_PATH = 'data/devex_train_clean.csv'
TEST_PATH  = 'data/devex_test_clean.csv'

train_df = pd.read_csv(TRAIN_PATH, encoding='latin-1', low_memory=False)
test_df  = pd.read_csv(TEST_PATH,  encoding='latin-1', low_memory=False)
print(f'Train: {train_df.shape}, Test: {test_df.shape}')\
"""

cell13_code = r"""\
# Reconstruct label matrix from clean CSV
def detect_label_columns(df, text_col='clean_text'):
    return [c for c in df.columns
            if c not in (text_col, 'clean_text')
            and df[c].dropna().astype(str).str.contains(r'\d+\.\d+', regex=True).mean() > 0.3]

RAW_TEXT_COL = [c for c in train_df.columns if 'text' in c.lower() and c != 'clean_text'][0]
LABEL_COLS_B = detect_label_columns(train_df)

def build_label_lists(df, label_cols):
    rows = []
    for _, row in df[label_cols].iterrows():
        labels = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ('', 'NA', 'nan')]
        rows.append(labels)
    return rows

mlb_b = MultiLabelBinarizer()
label_lists_b = build_label_lists(train_df, LABEL_COLS_B)
Y_b = mlb_b.fit_transform(label_lists_b)
ALL_LABELS_B = list(mlb_b.classes_)

train_df['clean_text'] = train_df['clean_text'].fillna('')
X_b = train_df['clean_text'].values

X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
    X_b, Y_b, test_size=0.2, random_state=42)

print(f'Labels: {len(ALL_LABELS_B)}, Train: {len(X_train_b)}, Val: {len(X_val_b)}')\
"""

cell14_code = """\
# Shared evaluation helper for Section 2
from src.evaluation import MultiLabelEvaluator
import os
os.makedirs('results/confusion_matrices', exist_ok=True)

experiment_results_b = {}

def evaluate_b(name, model, X_val, y_val, exp_num, y_prob=None):
    y_pred = model.predict(X_val)
    if y_prob is None:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)
        else:
            y_prob = y_pred.astype(float)
    hl     = hamming_loss(y_val, y_pred)
    f1_mi  = f1_score(y_val, y_pred, average='micro', zero_division=0)
    f1_ma  = f1_score(y_val, y_pred, average='macro', zero_division=0)
    print(f'  Exp {exp_num} {name}: HL={hl:.4f}  MicroF1={f1_mi:.4f}  MacroF1={f1_ma:.4f}')
    evaluator_b = MultiLabelEvaluator(label_names=ALL_LABELS_B)
    evaluator_b.save_confusion_matrices(y_val, y_prob.astype(float), threshold=0.5, exp_num=exp_num)
    experiment_results_b[exp_num] = {
        'name': name, 'hl': hl, 'f1_micro': f1_mi, 'f1_macro': f1_ma,
        'y_pred': y_pred, 'y_prob': y_prob
    }
    return hl, f1_mi, f1_ma\
"""

cell15_md = """\
### Experiment 1 — LR + TF-IDF Unigrams (Baseline)\
"""

cell16_code = """\
print('Running Experiment 1: LR + TF-IDF Unigrams (Baseline)')
vec1 = TfidfVectorizer(ngram_range=(1,1), max_features=10000)
X_tr1 = vec1.fit_transform(X_train_b)
X_v1  = vec1.transform(X_val_b)
lr1 = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
lr1.fit(X_tr1, y_train_b)
evaluate_b('LR + TF-IDF Unigrams', lr1, X_v1, y_val_b, exp_num=1)\
"""

cell17_md = """\
### Experiment 2 — LR + TF-IDF Bigrams\
"""

cell18_code = """\
print('Running Experiment 2: LR + TF-IDF Bigrams')
vec2 = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_tr2 = vec2.fit_transform(X_train_b)
X_v2  = vec2.transform(X_val_b)
lr2 = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
lr2.fit(X_tr2, y_train_b)
evaluate_b('LR + TF-IDF Bigrams', lr2, X_v2, y_val_b, exp_num=2)\
"""

cell19_md = """\
### Experiment 3 — Vocabulary Size Tuning\
"""

cell20_code = """\
print('Running Experiment 3: Vocabulary Size Tuning')
for max_feat in [5000, 10000, 20000, 50000]:
    v = TfidfVectorizer(ngram_range=(1,1), max_features=max_feat)
    Xtr = v.fit_transform(X_train_b)
    Xv  = v.transform(X_val_b)
    m = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    m.fit(Xtr, y_train_b)
    hl = hamming_loss(y_val_b, m.predict(Xv))
    print(f'  vocab={max_feat:6d} -> HL={hl:.4f}')
# Best was 5000 — use that going forward
vec3 = TfidfVectorizer(ngram_range=(1,1), max_features=5000)
X_tr3 = vec3.fit_transform(X_train_b)
X_v3  = vec3.transform(X_val_b)
lr3 = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr3.fit(X_tr3, y_train_b)
evaluate_b('LR + TF-IDF vocab=5000 (best)', lr3, X_v3, y_val_b, exp_num=3)\
"""

cell21_md = """\
### Experiment 4 — Model Comparison: LR vs LinearSVC vs RF\
"""

cell22_code = """\
print('Running Experiment 4: Model Comparison')
models_exp4 = {
    'LR (best config)': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    'LinearSVC':        OneVsRestClassifier(LinearSVC(max_iter=2000)),
    'RandomForest':     OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
}
results4 = {}
for name, m in models_exp4.items():
    m.fit(X_tr3, y_train_b)
    y_pred_tmp = m.predict(X_v3)
    hl  = hamming_loss(y_val_b, y_pred_tmp)
    f1  = f1_score(y_val_b, y_pred_tmp, average='micro', zero_division=0)
    results4[name] = (hl, f1, m)
    print(f'  {name:20s} HL={hl:.4f}  MicroF1={f1:.4f}')

best_name4  = min(results4, key=lambda k: results4[k][0])
best_model4 = results4[best_name4][2]
print(f'\\nBest model: {best_name4}')
evaluate_b(best_name4, best_model4, X_v3, y_val_b, exp_num=4)\
"""

cell23_md = """\
### Experiment 5 — Threshold Optimisation\
"""

cell24_code = """\
print('Running Experiment 5: Threshold Optimisation')
lr5 = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr5.fit(X_tr3, y_train_b)
y_prob5 = lr5.predict_proba(X_v3)

best_hl5, best_t5 = 1.0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    y_bin = (y_prob5 >= t).astype(int)
    hl = hamming_loss(y_val_b, y_bin)
    if hl < best_hl5:
        best_hl5, best_t5 = hl, t

print(f'  Best threshold={best_t5:.2f}  HL={best_hl5:.4f}')
y_pred5 = (y_prob5 >= best_t5).astype(int)
f1_5 = f1_score(y_val_b, y_pred5, average='micro', zero_division=0)
f1_ma5 = f1_score(y_val_b, y_pred5, average='macro', zero_division=0)
evaluator_b5 = MultiLabelEvaluator(label_names=ALL_LABELS_B)
evaluator_b5.save_confusion_matrices(y_val_b, y_prob5, threshold=best_t5, exp_num=5)
experiment_results_b[5] = {
    'name': 'LR + Opt Threshold', 'hl': best_hl5,
    'f1_micro': f1_5, 'f1_macro': f1_ma5,
    'y_pred': y_pred5, 'y_prob': y_prob5
}\
"""

cell25_md = """\
### Experiment 6 — Class Imbalance: class_weight='balanced'\
"""

cell26_code = """\
print('Running Experiment 6: Class Imbalance Handling')
lr6 = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
lr6.fit(X_tr3, y_train_b)
evaluate_b('LR balanced', lr6, X_v3, y_val_b, exp_num=6)\
"""

# ---------------------------------------------------------------------------
# Build cell list
# ---------------------------------------------------------------------------

new_cells = [
    nbformat.v4.new_markdown_cell(cell11_md),
    nbformat.v4.new_code_cell(cell12_code),
    nbformat.v4.new_code_cell(cell13_code),
    nbformat.v4.new_code_cell(cell14_code),
    nbformat.v4.new_markdown_cell(cell15_md),
    nbformat.v4.new_code_cell(cell16_code),
    nbformat.v4.new_markdown_cell(cell17_md),
    nbformat.v4.new_code_cell(cell18_code),
    nbformat.v4.new_markdown_cell(cell19_md),
    nbformat.v4.new_code_cell(cell20_code),
    nbformat.v4.new_markdown_cell(cell21_md),
    nbformat.v4.new_code_cell(cell22_code),
    nbformat.v4.new_markdown_cell(cell23_md),
    nbformat.v4.new_code_cell(cell24_code),
    nbformat.v4.new_markdown_cell(cell25_md),
    nbformat.v4.new_code_cell(cell26_code),
]

# ---------------------------------------------------------------------------
# Read, append, write
# ---------------------------------------------------------------------------

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

nb.cells.extend(new_cells)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Done. Total cells: {len(nb.cells)}')
