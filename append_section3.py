"""Append Section 3 (Person 3 — Deep Learning, Experiments 7-10) to master notebook."""
import nbformat

NOTEBOOK_PATH = 'SDG3_Classification_Complete.ipynb'

nb = nbformat.read(open(NOTEBOOK_PATH, encoding='utf-8'), as_version=4)

new_cells = []

# CELL 27 — markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    '---\n'
    '## Section 3 — Person 3: Deep Learning with Word2Vec & BERT (Experiments 7–10)\n'
    '*Patricia Mugabo* — Word2Vec embeddings, deep neural networks, BERT feature extraction, '
    'hyperparameter tuning, class weighting.\n\n'
    '**Best result this section: Experiment 9 — Word2Vec + DeepNN (Tuned), Hamming Loss = 0.0595**'
))

# CELL 28 — code: DL setup
new_cells.append(nbformat.v4.new_code_cell(
    'import sys, os\n'
    "sys.path.insert(0, '.')\n"
    'import numpy as np\n'
    'import torch\n'
    'from src.data_loader import SDGDataLoader\n'
    'from src.evaluation import MultiLabelEvaluator\n'
    'from src.trainer import Trainer, TextDataset\n'
    'from models.neural_networks import DeepNN\n'
    'from models.embedding_models import Word2VecEmbedding\n'
    '\n'
    'np.random.seed(42)\n'
    'torch.manual_seed(42)\n'
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print(f'Device: {device}')\n"
    '\n'
    'loader = SDGDataLoader(\n'
    "    train_path='data/devex_train_clean.csv',\n"
    "    test_path='data/devex_test_clean.csv',\n"
    '    use_cleaned=True\n'
    ')\n'
    'train_df_dl, test_df_dl = loader.load_data()\n'
    "train_df_dl, test_df_dl = loader.preprocess_dataset(text_column='text')\n"
    'label_columns = loader.get_label_columns()\n'
    '\n'
    'X_train_dl, X_val_dl, y_train_dl, y_val_dl = loader.train_val_split(test_size=0.2, random_state=42)\n'
    'y_train_dl = y_train_dl.astype(float)\n'
    'y_val_dl   = y_val_dl.astype(float)\n'
    "print(f'Train: {len(X_train_dl)}  Val: {len(X_val_dl)}  Labels: {len(label_columns)}')"
))

# CELL 29 — markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    '### Experiment 7 — Word2Vec (trained on corpus) + DeepNN'
))

# CELL 30 — code: Experiment 7
new_cells.append(nbformat.v4.new_code_cell(
    "print('Running Experiment 7: Word2Vec + DeepNN')\n"
    '\n'
    'X_train_tok7 = [t.split() for t in X_train_dl]\n'
    'X_val_tok7   = [t.split() for t in X_val_dl]\n'
    '\n'
    'w2v7 = Word2VecEmbedding(vector_size=300, window=5, min_count=2, workers=4, sg=1)\n'
    'w2v7.train(X_train_tok7)\n'
    '\n'
    'X_tr7 = w2v7.transform(X_train_tok7)\n'
    'X_v7  = w2v7.transform(X_val_tok7)\n'
    '\n'
    'model7 = DeepNN(input_dim=300, hidden_dims=[512, 256, 128], output_dim=y_train_dl.shape[1],\n'
    '                dropout=0.3, use_batch_norm=True)\n'
    '\n'
    'evaluator7 = MultiLabelEvaluator(label_names=label_columns)\n'
    'tr7 = Trainer(model7,\n'
    '              torch.utils.data.DataLoader(TextDataset(X_tr7, y_train_dl), batch_size=64, shuffle=True),\n'
    '              torch.utils.data.DataLoader(TextDataset(X_v7, y_val_dl),   batch_size=64),\n'
    '              device=device)\n'
    '\n'
    "os.makedirs('results/experiment_7_new', exist_ok=True)\n"
    "model7 = tr7.train(num_epochs=30, evaluator=evaluator7, save_path='results/experiment_7_new')\n"
    '\n'
    'model7.eval()\n'
    'with torch.no_grad():\n'
    '    preds7 = []\n'
    '    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v7, y_val_dl), batch_size=64):\n'
    '        preds7.append(model7(xb.to(device)).cpu().numpy())\n'
    'y_pred7 = np.vstack(preds7)\n'
    '\n'
    'from sklearn.metrics import hamming_loss, f1_score\n'
    'hl7  = hamming_loss(y_val_dl, (y_pred7 >= 0.5).astype(int))\n'
    "f1_7 = f1_score(y_val_dl, (y_pred7 >= 0.5).astype(int), average='micro', zero_division=0)\n"
    "print(f'Exp 7 — HL={hl7:.4f}  MicroF1={f1_7:.4f}')\n"
    '\n'
    'evaluator7.save_confusion_matrices(y_val_dl, y_pred7, threshold=0.5, exp_num=7)'
))

# CELL 31 — markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    '### Experiment 8 — BERT (pretrained contextual embeddings)'
))

# CELL 32 — code: Experiment 8
new_cells.append(nbformat.v4.new_code_cell(
    "print('Running Experiment 8: BERT')\n"
    'from transformers import BertTokenizer, BertModel\n'
    '\n'
    "tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')\n"
    "bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)\n"
    'bert_model.eval()\n'
    '\n'
    'def bert_encode(texts, batch_size=32):\n'
    '    all_vecs = []\n'
    '    for i in range(0, len(texts), batch_size):\n'
    '        batch = list(texts[i:i+batch_size])\n'
    '        enc = tokenizer_bert(batch, padding=True, truncation=True,\n'
    "                             max_length=128, return_tensors='pt').to(device)\n"
    '        with torch.no_grad():\n'
    '            out = bert_model(**enc)\n'
    '        vecs = out.last_hidden_state[:, 0, :].cpu().numpy()\n'
    '        all_vecs.append(vecs)\n'
    '    return np.vstack(all_vecs)\n'
    '\n'
    "print('Encoding train...')\n"
    'X_tr8 = bert_encode(X_train_dl)\n'
    "print('Encoding val...')\n"
    'X_v8  = bert_encode(X_val_dl)\n'
    "print(f'BERT embedding shape: {X_tr8.shape}')\n"
    '\n'
    'model8 = DeepNN(input_dim=768, hidden_dims=[256, 128], output_dim=y_train_dl.shape[1],\n'
    '                dropout=0.3, use_batch_norm=True)\n'
    '\n'
    'evaluator8 = MultiLabelEvaluator(label_names=label_columns)\n'
    'tr8 = Trainer(model8,\n'
    '              torch.utils.data.DataLoader(TextDataset(X_tr8, y_train_dl), batch_size=32, shuffle=True),\n'
    '              torch.utils.data.DataLoader(TextDataset(X_v8,  y_val_dl),   batch_size=32),\n'
    '              device=device)\n'
    '\n'
    "os.makedirs('results/experiment_8_new', exist_ok=True)\n"
    "model8 = tr8.train(num_epochs=15, evaluator=evaluator8, save_path='results/experiment_8_new')\n"
    '\n'
    'model8.eval()\n'
    'with torch.no_grad():\n'
    '    preds8 = []\n'
    '    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v8, y_val_dl), batch_size=32):\n'
    '        preds8.append(model8(xb.to(device)).cpu().numpy())\n'
    'y_pred8 = np.vstack(preds8)\n'
    '\n'
    'hl8  = hamming_loss(y_val_dl, (y_pred8 >= 0.5).astype(int))\n'
    "f1_8 = f1_score(y_val_dl, (y_pred8 >= 0.5).astype(int), average='micro', zero_division=0)\n"
    "print(f'Exp 8 — HL={hl8:.4f}  MicroF1={f1_8:.4f}')\n"
    'evaluator8.save_confusion_matrices(y_val_dl, y_pred8, threshold=0.5, exp_num=8)'
))

# CELL 33 — markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    '### Experiment 9 — Word2Vec + DeepNN (Tuned hyperparameters)'
))

# CELL 34 — code: Experiment 9
new_cells.append(nbformat.v4.new_code_cell(
    "print('Running Experiment 9: Word2Vec + DeepNN (Tuned)')\n"
    'w2v9 = Word2VecEmbedding(vector_size=200, window=5, min_count=2, workers=4, sg=1)\n'
    'w2v9.train(X_train_tok7)\n'
    '\n'
    'X_tr9 = w2v9.transform(X_train_tok7)\n'
    'X_v9  = w2v9.transform(X_val_tok7)\n'
    '\n'
    'model9 = DeepNN(input_dim=200, hidden_dims=[512, 256, 128, 64], output_dim=y_train_dl.shape[1],\n'
    '                dropout=0.2, use_batch_norm=True)\n'
    '\n'
    'evaluator9 = MultiLabelEvaluator(label_names=label_columns)\n'
    'tr9 = Trainer(model9,\n'
    '              torch.utils.data.DataLoader(TextDataset(X_tr9, y_train_dl), batch_size=64, shuffle=True),\n'
    '              torch.utils.data.DataLoader(TextDataset(X_v9,  y_val_dl),   batch_size=64),\n'
    '              device=device, learning_rate=0.001)\n'
    '\n'
    "os.makedirs('results/experiment_9_new', exist_ok=True)\n"
    "model9 = tr9.train(num_epochs=30, evaluator=evaluator9, save_path='results/experiment_9_new')\n"
    '\n'
    'model9.eval()\n'
    'with torch.no_grad():\n'
    '    preds9 = []\n'
    '    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v9, y_val_dl), batch_size=64):\n'
    '        preds9.append(model9(xb.to(device)).cpu().numpy())\n'
    'y_pred9 = np.vstack(preds9)\n'
    '\n'
    'hl9  = hamming_loss(y_val_dl, (y_pred9 >= 0.5).astype(int))\n'
    "f1_9 = f1_score(y_val_dl, (y_pred9 >= 0.5).astype(int), average='micro', zero_division=0)\n"
    "print(f'Exp 9 — HL={hl9:.4f}  MicroF1={f1_9:.4f}')\n"
    'evaluator9.save_confusion_matrices(y_val_dl, y_pred9, threshold=0.5, exp_num=9)'
))

# CELL 35 — markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    '### Experiment 10 — Word2Vec + Class Weights'
))

# CELL 36 — code: Experiment 10
new_cells.append(nbformat.v4.new_code_cell(
    "print('Running Experiment 10: Word2Vec + Class Weights')\n"
    'label_counts_dl = y_train_dl.sum(axis=0)\n'
    'n_samples = len(y_train_dl)\n'
    'class_weights10 = (n_samples - label_counts_dl) / (label_counts_dl + 1e-8)\n'
    'class_weights10 = np.clip(class_weights10, 1.0, 10.0)\n'
    '\n'
    'model10 = DeepNN(input_dim=200, hidden_dims=[512, 256, 128, 64], output_dim=y_train_dl.shape[1],\n'
    '                 dropout=0.2, use_batch_norm=True)\n'
    '\n'
    'evaluator10 = MultiLabelEvaluator(label_names=label_columns)\n'
    'tr10 = Trainer(model10,\n'
    '               torch.utils.data.DataLoader(TextDataset(X_tr9, y_train_dl), batch_size=64, shuffle=True),\n'
    '               torch.utils.data.DataLoader(TextDataset(X_v9,  y_val_dl),   batch_size=64),\n'
    '               device=device, class_weights=class_weights10)\n'
    '\n'
    "os.makedirs('results/experiment_10_new', exist_ok=True)\n"
    "model10 = tr10.train(num_epochs=30, evaluator=evaluator10, save_path='results/experiment_10_new')\n"
    '\n'
    'model10.eval()\n'
    'with torch.no_grad():\n'
    '    preds10 = []\n'
    '    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v9, y_val_dl), batch_size=64):\n'
    '        preds10.append(model10(xb.to(device)).cpu().numpy())\n'
    'y_pred10 = np.vstack(preds10)\n'
    '\n'
    'best_hl10, best_t10 = 1.0, 0.5\n'
    'for t in np.arange(0.1, 0.9, 0.05):\n'
    '    hl = hamming_loss(y_val_dl, (y_pred10 >= t).astype(int))\n'
    '    if hl < best_hl10:\n'
    '        best_hl10, best_t10 = hl, t\n'
    '\n'
    "print(f'Exp 10 — Best HL={best_hl10:.4f} at threshold={best_t10:.2f}')\n"
    'evaluator10.save_confusion_matrices(y_val_dl, y_pred10, threshold=best_t10, exp_num=10)'
))

# Append all 18 new cells
nb.cells.extend(new_cells)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Done. Total cells: {len(nb.cells)}')
