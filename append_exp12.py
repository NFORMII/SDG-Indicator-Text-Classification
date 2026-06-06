import nbformat

nb = nbformat.read(open('SDG3_Classification_Complete.ipynb', encoding='utf-8'), as_version=4)

cell41 = nbformat.v4.new_markdown_cell(
    "### Experiment 12 — FastText (trained on corpus, subword) + DeepNN\n"
    "**Hypothesis:** Subword information helps represent rare/technical health terms (e.g. 'hepatitis', 'neonatal') that Word2Vec might miss due to low frequency."
)

cell42 = nbformat.v4.new_code_cell(
    "print('Running Experiment 12: FastText + DeepNN')\n"
    "from models.embedding_models import FastTextEmbedding\n"
    "\n"
    "ft12 = FastTextEmbedding(vector_size=300, window=5, min_count=1, workers=4)\n"
    "ft12.train(X_train_tok11)\n"
    "\n"
    "X_tr12 = ft12.transform(X_train_tok11)\n"
    "X_v12  = ft12.transform(X_val_tok11)\n"
    "print(f'FastText embedding shape: {X_tr12.shape}')"
)

cell43 = nbformat.v4.new_code_cell(
    "# Same architecture as Exp 9/11 for fair comparison\n"
    "model12 = DeepNN(input_dim=300, hidden_dims=[512, 256, 128, 64], output_dim=y_train_dl.shape[1],\n"
    "                 dropout=0.2, use_batch_norm=True)\n"
    "\n"
    "evaluator12 = MultiLabelEvaluator(label_names=label_columns)\n"
    "tr12 = Trainer(model12,\n"
    "               torch.utils.data.DataLoader(TextDataset(X_tr12, y_train_dl), batch_size=64, shuffle=True),\n"
    "               torch.utils.data.DataLoader(TextDataset(X_v12,  y_val_dl),   batch_size=64),\n"
    "               device=device, learning_rate=0.001)\n"
    "\n"
    "os.makedirs('results/experiment_12', exist_ok=True)\n"
    "model12 = tr12.train(num_epochs=30, evaluator=evaluator12, save_path='results/experiment_12')\n"
    "\n"
    "model12.eval()\n"
    "with torch.no_grad():\n"
    "    preds12 = []\n"
    "    for xb, _ in torch.utils.data.DataLoader(TextDataset(X_v12, y_val_dl), batch_size=64):\n"
    "        preds12.append(model12(xb.to(device)).cpu().numpy())\n"
    "y_pred12 = np.vstack(preds12)\n"
    "\n"
    "from sklearn.metrics import hamming_loss, f1_score\n"
    "hl12     = hamming_loss(y_val_dl, (y_pred12 >= 0.5).astype(int))\n"
    "f1_12    = f1_score(y_val_dl, (y_pred12 >= 0.5).astype(int), average='micro', zero_division=0)\n"
    "f1_mac12 = f1_score(y_val_dl, (y_pred12 >= 0.5).astype(int), average='macro', zero_division=0)\n"
    "print(f'Exp 12 FastText — HL={hl12:.4f}  MicroF1={f1_12:.4f}  MacroF1={f1_mac12:.4f}')\n"
    "\n"
    "import json\n"
    "with open('results/experiment_12/results.json', 'w') as f:\n"
    "    json.dump({'experiment_id': 12, 'experiment_name': 'FastText + DeepNN',\n"
    "               'results': {'hamming_loss': hl12, 'f1_micro': f1_12, 'f1_macro': f1_mac12}}, f, indent=2)\n"
    "\n"
    "evaluator12.save_confusion_matrices(y_val_dl, y_pred12, threshold=0.5, exp_num=12)\n"
    "print('Experiment 12 complete.')"
)

cell44 = nbformat.v4.new_markdown_cell(
    "---\n"
    "*End of Section 4. All 12 experiments defined. Proceed to Section 5 for the unified comparison table.*"
)

nb.cells.extend([cell41, cell42, cell43, cell44])

nbformat.write(nb, open('SDG3_Classification_Complete.ipynb', 'w', encoding='utf-8'))
print(f'Done. Total cells: {len(nb.cells)}')
