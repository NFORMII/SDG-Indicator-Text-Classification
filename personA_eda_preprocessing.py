import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    from wordcloud import WordCloud
except ImportError as exc:
    raise ImportError(
        "wordcloud is required. Install with: pip install wordcloud"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "Devex_train.csv"
TEST_PATH = BASE_DIR / "Devex_test_questions.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

ACRONYMS = {
    "SDG",
    "WHO",
    "HIV",
    "TB",
    "AIDS",
    "UN",
    "UNICEF",
    "USAID",
    "LMIC",
    "NCD",
    "WASH",
    "MDG",
    "IHR",
    "COVID",
    "SARS",
    "MERS",
    "NGO",
    "ODA",
    "RSV",
    "HCV",
    "HBV",
    "HIV/AIDS",
    "STI",
    "EPI",
    "GDP",
    "US",
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&nbsp;|&amp;|&quot;|&lt;|&gt;")


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise


def detect_text_column(df):
    object_cols = [c for c in df.columns if df[c].dtype in ["object", "string", "str"]]
    if not object_cols:
        raise ValueError("No object columns found to infer text column.")

    name_matches = [c for c in object_cols if "text" in c.lower()]
    candidates = name_matches if name_matches else object_cols

    lengths = {
        c: df[c].fillna("").astype(str).str.len().mean() for c in candidates
    }
    text_col = max(lengths, key=lengths.get)
    return text_col


def detect_label_columns(df, text_col):
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if label_cols:
        return label_cols

    label_like = []
    for col in df.columns:
        if col == text_col:
            continue
        if df[col].dtype not in ["object", "string", "str"]:
            continue
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        sdg_ratio = sample.str.contains(r"\d+(\.\d+)+", regex=True).mean()
        if sdg_ratio >= 0.5:
            label_like.append(col)
    return label_like


def detect_id_column(df, text_col, label_cols):
    excluded = set([text_col]) | set(label_cols)
    id_candidates = [c for c in df.columns if "id" in c.lower() and c not in excluded]

    def uniqueness_ratio(series):
        if series.empty:
            return 0.0
        return series.nunique(dropna=True) / max(len(series), 1)

    if id_candidates:
        ratios = {c: uniqueness_ratio(df[c]) for c in id_candidates}
        return max(ratios, key=ratios.get)

    for col in df.columns:
        if col in excluded:
            continue
        if uniqueness_ratio(df[col]) >= 0.95:
            return col
    return None


def detect_label_format(df, label_cols):
    if not label_cols:
        return "none"
    numeric_cols = [c for c in label_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == len(label_cols):
        unique_vals = set(pd.unique(df[label_cols].values.ravel()))
        unique_vals = {v for v in unique_vals if not pd.isna(v)}
        if unique_vals.issubset({0, 1}):
            return "binary"
    return "multi-column"


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = HTML_ENTITY_RE.sub(" ", text)
    return text


def basic_tokenize(text):
    text = normalize_text(text)
    return TOKEN_RE.findall(text.lower())


def sentence_count(text):
    text = normalize_text(text)
    chunks = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return len(chunks)


def build_label_lists(df, label_cols, label_format):
    if label_format == "binary":
        labels = [
            [col for col in label_cols if row.get(col) == 1]
            for _, row in df[label_cols].iterrows()
        ]
    else:
        labels = [
            [
                str(v).strip()
                for v in row
                if pd.notna(v)
                and str(v).strip()
                and str(v).strip().upper() != "NA"
            ]
            for row in df[label_cols].values.tolist()
        ]
    return labels


def get_text_stats(df, text_col):
    word_counts = df[text_col].fillna("").astype(str).apply(
        lambda x: len(basic_tokenize(x))
    )
    char_counts = df[text_col].fillna("").astype(str).apply(
        lambda x: len(normalize_text(x))
    )
    sentence_counts = df[text_col].fillna("").astype(str).apply(sentence_count)

    return pd.DataFrame(
        {
            "word_count": word_counts,
            "char_count": char_counts,
            "sentence_count": sentence_counts,
        }
    )


def build_vocabulary(tokens_list):
    vocab = set()
    for tokens in tokens_list:
        vocab.update(tokens)
    return vocab


def preprocess_text(text, stop_words, lemmatizer):
    text = normalize_text(text)
    tokens = TOKEN_RE.findall(text)
    cleaned = []
    for tok in tokens:
        upper = tok.upper()
        if upper in ACRONYMS:
            cleaned.append(upper)
            continue
        tok = tok.lower()
        if tok.isnumeric():
            continue
        if tok in stop_words:
            continue
        tok = lemmatizer.lemmatize(tok)
        cleaned.append(tok)
    return " ".join(cleaned)


def ensure_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def plot_label_distribution(label_counts, output_path):
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind="bar")
    plt.title("Label Frequency Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_histogram(series, title, xlabel, output_path):
    plt.figure(figsize=(10, 5))
    sns.histplot(series, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_wordcloud(texts, output_path):
    combined_text = " ".join(texts)
    wc = WordCloud(width=1200, height=600, background_color="white").generate(
        combined_text
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cooccurrence_heatmap(matrix, labels, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        cbar=True,
    )
    plt.title("Label Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(TRAIN_PATH)
    test_df = load_csv(TEST_PATH)

    text_col = detect_text_column(train_df)
    label_cols = detect_label_columns(train_df, text_col)
    label_format = detect_label_format(train_df, label_cols)
    id_col = detect_id_column(train_df, text_col, label_cols)

    if not text_col:
        raise ValueError("Text column could not be determined.")
    if not label_cols:
        raise ValueError("Label columns could not be determined.")
    if text_col not in test_df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in test data."
        )

    labels_list = build_label_lists(train_df, label_cols, label_format)
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels_list)
    label_names = list(mlb.classes_)

    label_counts = pd.Series(label_matrix.sum(axis=0), index=label_names).sort_values(
        ascending=False
    )
    label_freq_df = pd.DataFrame(
        {
            "label": label_counts.index,
            "count": label_counts.values,
            "percentage": (label_counts.values / len(train_df) * 100).round(2),
        }
    )

    text_stats = get_text_stats(train_df, text_col)
    tokens_list = train_df[text_col].fillna("").astype(str).apply(basic_tokenize)
    vocab = build_vocabulary(tokens_list)

    dataset_summary = {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "total_features_train": len(train_df.columns),
        "total_features_test": len(test_df.columns),
        "text_column": text_col,
        "id_column": id_col if id_col else "not_detected",
        "label_columns": len(label_cols),
        "label_format": label_format,
        "unique_labels": len(label_names),
        "avg_doc_length_tokens": round(text_stats["word_count"].mean(), 2),
        "median_doc_length_tokens": int(text_stats["word_count"].median()),
        "min_doc_length_tokens": int(text_stats["word_count"].min()),
        "max_doc_length_tokens": int(text_stats["word_count"].max()),
        "avg_char_count": round(text_stats["char_count"].mean(), 2),
        "vocabulary_size": len(vocab),
        "missing_values_train": int(train_df.isna().sum().sum()),
        "missing_values_test": int(test_df.isna().sum().sum()),
    }

    dataset_summary_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in dataset_summary.items()]
    )

    dataset_summary_path = OUTPUT_DIR / "dataset_summary.csv"
    label_freq_path = OUTPUT_DIR / "label_frequencies.csv"
    label_distribution_path = OUTPUT_DIR / "label_distribution.png"
    doc_length_hist_path = OUTPUT_DIR / "document_length_histogram.png"
    char_length_hist_path = OUTPUT_DIR / "character_length_histogram.png"
    wordcloud_path = OUTPUT_DIR / "wordcloud.png"
    cooccurrence_path = OUTPUT_DIR / "label_cooccurrence_heatmap.png"

    dataset_summary_df.to_csv(dataset_summary_path, index=False)
    label_freq_df.to_csv(label_freq_path, index=False)

    sns.set_theme(style="whitegrid")
    plot_label_distribution(label_counts, label_distribution_path)
    plot_histogram(
        text_stats["word_count"],
        "Document Length Distribution (Words)",
        "Word Count",
        doc_length_hist_path,
    )
    plot_histogram(
        text_stats["char_count"],
        "Character Length Distribution",
        "Character Count",
        char_length_hist_path,
    )

    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    train_df["clean_text"] = train_df[text_col].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )
    test_df["clean_text"] = test_df[text_col].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )

    plot_wordcloud(train_df["clean_text"].tolist(), wordcloud_path)

    cooccurrence_matrix = np.dot(label_matrix.T, label_matrix)
    cooccurrence_df = pd.DataFrame(
        cooccurrence_matrix, index=label_names, columns=label_names
    )
    plot_cooccurrence_heatmap(cooccurrence_df, label_names, cooccurrence_path)

    train_output_path = BASE_DIR / "devex_train_clean.csv"
    test_output_path = BASE_DIR / "devex_test_clean.csv"
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    most_common_labels = label_counts.head(5)
    rare_labels = label_counts.tail(5)
    imbalance_ratio = (
        round(label_counts.max() / max(label_counts.min(), 1), 2)
        if len(label_counts) > 1
        else 1.0
    )

    token_counter = Counter(
        token for tokens in train_df["clean_text"].str.split() for token in tokens
    )
    top_terms = ", ".join([w for w, _ in token_counter.most_common(10)])

    cooccurrence_pairs = []
    for i, j in combinations(range(len(label_names)), 2):
        count = cooccurrence_matrix[i, j]
        if count > 0:
            cooccurrence_pairs.append((label_names[i], label_names[j], int(count)))
    cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = "; ".join(
        [f"{a} & {b} ({c})" for a, b, c in cooccurrence_pairs[:5]]
    )

    summary_lines = [
        "# Person A Summary",
        "",
        "## Dataset Description Findings",
        f"Train samples: {dataset_summary['train_samples']}",
        f"Test samples: {dataset_summary['test_samples']}",
        f"Detected text column: {dataset_summary['text_column']}",
        f"Detected ID column: {dataset_summary['id_column']}",
        f"Label columns: {dataset_summary['label_columns']} (format: {dataset_summary['label_format']})",
        f"Unique labels: {dataset_summary['unique_labels']}",
        f"Average document length: {dataset_summary['avg_doc_length_tokens']} tokens (median: {dataset_summary['median_doc_length_tokens']}, min: {dataset_summary['min_doc_length_tokens']}, max: {dataset_summary['max_doc_length_tokens']})",
        f"Average character count: {dataset_summary['avg_char_count']}",
        f"Vocabulary size (lowercased tokens): {dataset_summary['vocabulary_size']}",
        f"Missing values (train/test): {dataset_summary['missing_values_train']} / {dataset_summary['missing_values_test']}",
        "",
        "## Label Imbalance Findings",
        f"Most common labels: {', '.join(most_common_labels.index.tolist())}",
        f"Rarest labels: {', '.join(rare_labels.index.tolist())}",
        f"Imbalance ratio (max/min): {imbalance_ratio}",
        "",
        "## Text Characteristic Findings",
        f"Average sentence count: {round(text_stats['sentence_count'].mean(), 2)}",
        f"Top domain terms (cleaned): {top_terms}",
        "",
        "## Co-occurrence Findings",
        f"Strongest co-occurring label pairs: {top_pairs if top_pairs else 'No co-occurring labels detected.'}",
        "",
        "## Preprocessing Decisions",
        "Lowercasing, HTML tag removal, punctuation and special character filtering, whitespace normalization via token re-join.",
        "Tokenization with regex word boundaries.",
        "Stopword removal using NLTK English list.",
        "Lemmatization using WordNetLemmatizer.",
        "Preserved domain acronyms (e.g., SDG, WHO, HIV, TB, USAID) in uppercase.",
        "Numerical-only tokens removed.",
    ]

    summary_path = BASE_DIR / "personA_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    generated_files = [
        dataset_summary_path,
        label_freq_path,
        label_distribution_path,
        doc_length_hist_path,
        char_length_hist_path,
        wordcloud_path,
        cooccurrence_path,
        train_output_path,
        test_output_path,
        summary_path,
    ]

    print("Generated files:")
    for path in generated_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
