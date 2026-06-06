import nbformat

NOTEBOOK_PATH = 'SDG3_Classification_Complete.ipynb'

nb = nbformat.read(open(NOTEBOOK_PATH, encoding='utf-8'), as_version=4)

# Cell 37 — markdown: Section 4 header
cell37 = nbformat.v4.new_markdown_cell(
    '---\n'
    '## Section 4 — Lorita: New Embedding Experiments (11–12)\n'
    'Comparing pretrained static embeddings (GloVe) and subword-aware embeddings (FastText) against trained Word2Vec, '
    'using the same DeepNN architecture as Experiment 9 for fair comparison.'
)

# Cell 38 — markdown: Experiment 11 header
cell38 = nbformat.v4.new_markdown_cell(
    '### Experiment 11 — GloVe (pretrained, 6B.300d) + DeepNN\n'
    '**Hypothesis:** Pretrained embeddings from 6B tokens should give richer semantic representations than Word2Vec trained on ~3,000 documents.'
)

# Cell 39 — code: load GloVe and transform
cell39 = nbformat.v4.new_code_cell(
    "print('Running Experiment 11: GloVe + DeepNN')\n"
    "from models.embedding_models import GloVeEmbedding\n"
    "import os\n"
    "\n"
    "GLOVE_PATH = 'glove.6B.300d.txt'\n"
    "assert os.path.exists(GLOVE_PATH), f'GloVe file not found at {GLOVE_PATH}. Run the setup cell in Section 0.'\n"
    "\n"
    "glove11 = GloVeEmbedding(embedding_dim=300)\n"
    "glove11.load(GLOVE_PATH)\n"
    "\n"
    "X_train_tok11 = [t.split() for t in X_train_dl]\n"
    "X_val_tok11   = [t.split() for t in X_val_dl]\n"
    "\n"
    "X_tr11 = glove11.transform(X_train_tok11)\n"
    "X_v11  = glove11.transform(X_val_tok11)\n"
    "print(f'GloVe embedding shape: {X_tr11.shape}')\n"
)

# Cell 40 — code: train and evaluate
cell40 = nbformat.v4.new_code_cell(
    "# Same architecture as Exp 9 (tuned) for fair comparison\n"
    "model11 = DeepNN(input_dim=300, hidden_dims=[512, 256, 128, 64], output_dim=y_train_dl.shape[1],\n"
    "                 dropout=0.2, use_batch_norm=True)\n"
    "\n"
    "evaluator11 = MultiLabelEvaluator(label_names=label_columns)\n"
    "tr11 = Trainer(model11,\n"
    "               torch.utils.data.DataLoader(TextDataset(X_tr11, y_train_dl), batch_size=64, shuffle=True),\n"
    "               torch.utils.data.DataLoader(TextDataset(X_v11,  y_val_dl),   batch_size=64),\n"
    "               device=device, learning_rate=0.001)\n"
    "\n"
    "os.makedirs('results/experiment_11', exist_ok=True)\n"
    "model11 = tr11.train(num_epochs=30, evaluator=evaluator11, save_path='results/experiment_11')\n"
    "\n"
    "model11.eval()\n"
    "with torch.no_grad():\n"
    "    preds11 = []\n"
    "    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v11, y_val_dl), batch_size=64):\n"
    "        preds11.append(model11(xb.to(device)).cpu().numpy())\n"
    "y_pred11 = np.vstack(preds11)\n"
    "\n"
    "from sklearn.metrics import hamming_loss, f1_score\n"
    "hl11     = hamming_loss(y_val_dl, (y_pred11 >= 0.5).astype(int))\n"
    "f1_11    = f1_score(y_val_dl, (y_pred11 >= 0.5).astype(int), average='micro', zero_division=0)\n"
    "f1_mac11 = f1_score(y_val_dl, (y_pred11 >= 0.5).astype(int), average='macro', zero_division=0)\n"
    "print(f'Exp 11 GloVe — HL={hl11:.4f}  MicroF1={f1_11:.4f}  MacroF1={f1_mac11:.4f}')\n"
    "\n"
    "import json\n"
    "with open('results/experiment_11/results.json', 'w') as f:\n"
    "    json.dump({'experiment_id': 11, 'experiment_name': 'GloVe + DeepNN',\n"
    "               'results': {'hamming_loss': hl11, 'f1_micro': f1_11, 'f1_macro': f1_mac11}}, f, indent=2)\n"
    "\n"
    "evaluator11.save_confusion_matrices(y_val_dl, y_pred11, threshold=0.5, exp_num=11)\n"
    "print('Experiment 11 complete.')\n"
)

nb.cells.extend([cell37, cell38, cell39, cell40])

nbformat.write(nb, open(NOTEBOOK_PATH, 'w', encoding='utf-8'))
print(f'Done. Total cells: {len(nb.cells)}')
