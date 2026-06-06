"""
create_notebook.py
Generates SDG3_Classification_Complete.ipynb with Sections 0 and 1.
"""
import nbformat

nb = nbformat.v4.new_notebook()

# ---------------------------------------------------------------------------
# CELL 1 — Title markdown
# ---------------------------------------------------------------------------
cell1 = nbformat.v4.new_markdown_cell(
    "# SDG 3 Indicator Text Classification\n"
    "**Group Assignment 2 — Complete Experiment Pipeline**\n\n"
    "Run this notebook top-to-bottom in Google Colab. "
    "All dependencies install in Section 0."
)

# ---------------------------------------------------------------------------
# CELL 2 — Colab install + path setup
# ---------------------------------------------------------------------------
cell2 = nbformat.v4.new_code_cell(
    "# Section 0: Install dependencies and configure paths\n"
    "import subprocess, sys, os\n"
    "\n"
    "# Install all requirements\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'], check=True)\n"
    "\n"
    "# NLTK downloads\n"
    "import nltk\n"
    "for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger']:\n"
    "    nltk.download(pkg, quiet=True)\n"
    "\n"
    "# Make sure src/ and models/ are importable\n"
    "if '.' not in sys.path:\n"
    "    sys.path.insert(0, '.')\n"
    "\n"
    "print('Setup complete.')\n"
)

# ---------------------------------------------------------------------------
# CELL 3 — GloVe download
# ---------------------------------------------------------------------------
cell3 = nbformat.v4.new_code_cell(
    "# Download GloVe vectors (needed for Experiment 11)\n"
    "# ~820MB — only downloads if not already present\n"
    "import os\n"
    "if not os.path.exists('glove.6B.300d.txt'):\n"
    "    print('Downloading GloVe 6B vectors...')\n"
    "    os.system('wget -q https://nlp.stanford.edu/data/glove.6B.zip')\n"
    "    os.system('unzip -q glove.6B.zip glove.6B.300d.txt')\n"
    "    os.system('rm glove.6B.zip')\n"
    "    print('GloVe ready.')\n"
    "else:\n"
    "    print('GloVe vectors already present.')\n"
)

# ---------------------------------------------------------------------------
# CELL 4 — Section 1 markdown
# ---------------------------------------------------------------------------
cell4 = nbformat.v4.new_markdown_cell(
    "---\n"
    "## Section 1 — Person A: EDA & Preprocessing\n"
    "*Kayonga Elvis* — dataset understanding, text statistics, preprocessing pipeline.\n"
    "\n"
    "Outputs: `data/devex_train_clean.csv`, `data/devex_test_clean.csv`, visualizations in `outputs/`"
)

# ---------------------------------------------------------------------------
# CELL 5 — Load raw data
# ---------------------------------------------------------------------------
cell5 = nbformat.v4.new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "train_df = pd.read_csv('data/Devex_train.csv', encoding='latin-1', low_memory=False)\n"
    "test_df  = pd.read_csv('data/Devex_test_questions.csv', encoding='latin-1', low_memory=False)\n"
    "\n"
    "print(f'Train shape: {train_df.shape}')\n"
    "print(f'Test shape:  {test_df.shape}')\n"
    "train_df.head(3)\n"
)

# ---------------------------------------------------------------------------
# CELL 6 — Column detection
# ---------------------------------------------------------------------------
cell6 = nbformat.v4.new_code_cell(
    "# Column detection\n"
    "def detect_text_col(df):\n"
    "    obj_cols = [c for c in df.columns if df[c].dtype == object]\n"
    "    return max(obj_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())\n"
    "\n"
    "def detect_id_col(df):\n"
    "    for c in df.columns:\n"
    "        if df[c].nunique() == len(df):\n"
    "            return c\n"
    "    return df.columns[0]\n"
    "\n"
    "def detect_label_cols(df, text_col, id_col):\n"
    "    import re\n"
    r"    return [c for c in df.columns"
    "\n"
    "            if c not in (text_col, id_col)\n"
    r"            and df[c].dropna().astype(str).str.contains(r'\d+\.\d+', regex=True).mean() > 0.3]"
    "\n"
    "\n"
    "TEXT_COL   = detect_text_col(train_df)\n"
    "ID_COL     = detect_id_col(train_df)\n"
    "LABEL_COLS = detect_label_cols(train_df, TEXT_COL, ID_COL)\n"
    "\n"
    "print(f'Text column : {TEXT_COL}')\n"
    "print(f'ID column   : {ID_COL}')\n"
    "print(f'Label cols  : {len(LABEL_COLS)} -> {LABEL_COLS[:3]} ...')\n"
)

# ---------------------------------------------------------------------------
# CELL 7 — Discover unique labels
# ---------------------------------------------------------------------------
cell7 = nbformat.v4.new_code_cell(
    "# Discover unique labels\n"
    "from sklearn.preprocessing import MultiLabelBinarizer\n"
    "\n"
    "def build_label_lists(df, label_cols):\n"
    "    rows = []\n"
    "    for _, row in df[label_cols].iterrows():\n"
    "        labels = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ('', 'NA', 'nan')]\n"
    "        rows.append(labels)\n"
    "    return rows\n"
    "\n"
    "label_lists = build_label_lists(train_df, LABEL_COLS)\n"
    "mlb = MultiLabelBinarizer()\n"
    "Y = mlb.fit_transform(label_lists)\n"
    "ALL_LABELS = list(mlb.classes_)\n"
    "\n"
    "print(f'Unique labels: {len(ALL_LABELS)}')\n"
    "print(f'Label matrix : {Y.shape}')\n"
)

# ---------------------------------------------------------------------------
# CELL 8 — Label distribution plot
# ---------------------------------------------------------------------------
cell8 = nbformat.v4.new_code_cell(
    "# Label distribution\n"
    "import os\n"
    "os.makedirs('outputs', exist_ok=True)\n"
    "\n"
    "label_counts = Y.sum(axis=0)\n"
    "sorted_idx = np.argsort(label_counts)[::-1]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "ax.bar(range(len(ALL_LABELS)), label_counts[sorted_idx], color='steelblue')\n"
    "ax.set_xticks(range(len(ALL_LABELS)))\n"
    "ax.set_xticklabels([ALL_LABELS[i][:20] for i in sorted_idx], rotation=90, fontsize=7)\n"
    "ax.set_title('Label Frequency Distribution (SDG 3 Indicators)')\n"
    "ax.set_ylabel('Count')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/label_distribution.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'Imbalance ratio: {label_counts.max() / label_counts.min():.2f}')\n"
)

# ---------------------------------------------------------------------------
# CELL 9 — Text statistics
# ---------------------------------------------------------------------------
cell9 = nbformat.v4.new_code_cell(
    "# Text statistics\n"
    "doc_lengths = train_df[TEXT_COL].dropna().astype(str).str.split().str.len()\n"
    "print(f'Avg tokens : {doc_lengths.mean():.1f}')\n"
    "print(f'Median     : {doc_lengths.median():.0f}')\n"
    "print(f'Min / Max  : {doc_lengths.min()} / {doc_lengths.max()}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].hist(doc_lengths, bins=50, color='steelblue', edgecolor='white')\n"
    "axes[0].set_title('Document Length (tokens)')\n"
    "axes[0].set_xlabel('Tokens')\n"
    "char_lengths = train_df[TEXT_COL].dropna().astype(str).str.len()\n"
    "axes[1].hist(char_lengths, bins=50, color='coral', edgecolor='white')\n"
    "axes[1].set_title('Document Length (characters)')\n"
    "axes[1].set_xlabel('Characters')\n"
    "plt.tight_layout()\n"
    "plt.savefig('outputs/document_length_histogram.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
)

# ---------------------------------------------------------------------------
# CELL 10 — Preprocessing pipeline
# ---------------------------------------------------------------------------
cell10 = nbformat.v4.new_code_cell(
    "# Preprocessing pipeline (Person A)\n"
    "import re\n"
    "import nltk\n"
    "from nltk.tokenize import RegexpTokenizer\n"
    "from nltk.corpus import stopwords\n"
    "from nltk.stem import WordNetLemmatizer\n"
    "\n"
    "DOMAIN_ACRONYMS = {'sdg', 'who', 'hiv', 'tb', 'usaid', 'ngo', 'oda', 'oecd', 'wash', 'ncds'}\n"
    "STOP_WORDS = set(stopwords.words('english'))\n"
    "lemmatizer = WordNetLemmatizer()\n"
    r"tokenizer  = RegexpTokenizer(r'\b[a-zA-Z]+\b')"
    "\n"
    "\n"
    "def clean_html(text):\n"
    r"    text = re.sub(r'<[^>]+>', ' ', text)"
    "\n"
    r"    text = re.sub(r'&[a-z]+;', ' ', text)"
    "\n"
    "    return text\n"
    "\n"
    "def preprocess_text(text):\n"
    "    if pd.isna(text): return ''\n"
    "    text = clean_html(str(text))\n"
    "    text = text.lower()\n"
    "    tokens = tokenizer.tokenize(text)\n"
    "    cleaned = []\n"
    "    for tok in tokens:\n"
    "        if tok.upper() in {a.upper() for a in DOMAIN_ACRONYMS}:\n"
    "            cleaned.append(tok.upper())\n"
    "            continue\n"
    "        if tok in STOP_WORDS: continue\n"
    "        lemma = lemmatizer.lemmatize(tok)\n"
    "        if len(lemma) > 1:\n"
    "            cleaned.append(lemma)\n"
    "    return ' '.join(cleaned)\n"
    "\n"
    "# Skip if clean CSVs already exist\n"
    "if not os.path.exists('data/devex_train_clean.csv'):\n"
    "    print('Preprocessing train...')\n"
    "    train_df['clean_text'] = train_df[TEXT_COL].apply(preprocess_text)\n"
    "    train_df.to_csv('data/devex_train_clean.csv', index=False)\n"
    "    print('Preprocessing test...')\n"
    "    test_df['clean_text'] = test_df[TEXT_COL].apply(preprocess_text)\n"
    "    test_df.to_csv('data/devex_test_clean.csv', index=False)\n"
    "    print('Clean CSVs saved.')\n"
    "else:\n"
    "    print('Clean CSVs already exist — skipping preprocessing.')\n"
)

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

output_path = "SDG3_Classification_Complete.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {output_path} ({len(nb.cells)} cells)")
