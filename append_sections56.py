"""Append Sections 5 and 6 to SDG3_Classification_Complete.ipynb."""
import nbformat

NB_PATH = 'SDG3_Classification_Complete.ipynb'

nb = nbformat.read(open(NB_PATH, encoding='utf-8'), as_version=4)

def md(source):
    return nbformat.v4.new_markdown_cell(source)

def code(source):
    return nbformat.v4.new_code_cell(source)

new_cells = [
    # CELL 45
    md("## Section 5 — Evaluation & Comparison"),

    # CELL 46
    code("""\
import json, os, pandas as pd

exp_results = []
exp_dirs = sorted([d for d in os.listdir('results') if d.startswith('experiment_')],
                  key=lambda x: int(x.split('_')[1]))
for ed in exp_dirs:
    rpath = f'results/{ed}/results.json'
    if os.path.exists(rpath):
        with open(rpath) as f:
            r = json.load(f)
        exp_results.append({
            'Exp': r.get('experiment_id', '?'),
            'Name': r.get('experiment_name', ed),
            'Hamming Loss': round(r['results'].get('hamming_loss', float('nan')), 4),
            'F1 Micro': round(r['results'].get('f1_micro', float('nan')), 4),
            'F1 Macro': round(r['results'].get('f1_macro', float('nan')), 4),
        })

# Add GloVe/FastText placeholders if not yet in results/
for exp_id, name in [(11, 'GloVe + DeepNN'), (12, 'FastText + DeepNN')]:
    if not any(r['Exp'] == exp_id for r in exp_results):
        exp_results.append({'Exp': exp_id, 'Name': name,
                            'Hamming Loss': float('nan'), 'F1 Micro': float('nan'), 'F1 Macro': float('nan')})

results_df = pd.DataFrame(exp_results).sort_values('Exp').reset_index(drop=True)
results_df.to_csv('results/experiment_comparison_full.csv', index=False)
print(results_df.to_string(index=False))"""),

    # CELL 47
    code("""\
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(results_df))
bars = ax.bar(x, results_df['Hamming Loss'], color='steelblue', edgecolor='white')
ax.axhline(0.045, color='red', linestyle='--', label='Baseline (0.045)')
ax.set_xticks(x)
ax.set_xticklabels([f"Exp {r['Exp']}\\n{r['Name'][:20]}" for _, r in results_df.iterrows()], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Hamming Loss (lower is better)')
ax.set_title('Experiment Comparison — Hamming Loss (all 12 experiments)')
ax.legend()
plt.tight_layout()
plt.savefig('results/experiment_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved to results/experiment_comparison_chart.png')"""),

    # CELL 48
    code("""\
# Per-label F1 for the best model (lowest Hamming Loss)
best_row = results_df.loc[results_df['Hamming Loss'].idxmin()]
print(f"Best model: Exp {best_row['Exp']} — {best_row['Name']}")
print(f"  Hamming Loss: {best_row['Hamming Loss']:.4f}")
print(f"  F1 Micro: {best_row['F1 Micro']:.4f}")
print(f"  F1 Macro: {best_row['F1 Macro']:.4f}")
print()
print("Per-label F1 breakdown requires y_pred from the best model.")
print("Run the best model's experiment cell to regenerate y_pred, then compute:")
print("  f1_score(y_true, y_pred, average=None, zero_division=0)")"""),

    # CELL 49
    md("""\
### Confusion Matrices

Below are the multilabel confusion matrices for each experiment.
Each subplot shows the binary confusion matrix (TN, FP, FN, TP) for one SDG 3 indicator.
Saved to `results/confusion_matrices/`."""),

    # CELL 50
    code("""\
import os
from IPython.display import Image, display

cm_dir = 'results/confusion_matrices'
if os.path.exists(cm_dir):
    cm_files = sorted([f for f in os.listdir(cm_dir) if f.endswith('.png')])
    if cm_files:
        for f in cm_files:
            print(f'--- {f} ---')
            display(Image(filename=os.path.join(cm_dir, f), width=900))
    else:
        print('No confusion matrix images found yet. Run experiments to generate them.')
else:
    print(f'Directory {cm_dir} not found. Run experiments to generate confusion matrices.')"""),

    # CELL 51
    md("""\
## Section 6 — Final Test Predictions

Retrain the best model on the full training data and generate predictions for the test set."""),

    # CELL 52
    code("""\
# Identify best model from results
best_exp = int(results_df.loc[results_df['Hamming Loss'].idxmin(), 'Exp'])
print(f'Best experiment: {best_exp}')
print('Retraining best model on full training data...')
print('NOTE: This cell requires the variables from the best experiment\\'s section to be in scope.')
print('If using Word2Vec (Exp 7-10): X_train_w2v, y_train are the full training arrays.')
print('If using GloVe (Exp 11): X_tr11, y_train_dl are the full training arrays.')
print('Adjust the variable names below to match the best experiment.')

# Generic template — adjust model/data vars to match best experiment:
# model_final = DeepNN(input_dim=300, hidden_dims=[512, 256, 128, 64],
#                      output_dim=y_train_dl.shape[1], dropout=0.2, use_batch_norm=True)
# trainer_final = Trainer(model_final,
#     torch.utils.data.DataLoader(TextDataset(X_tr11, y_train_dl), batch_size=64, shuffle=True),
#     torch.utils.data.DataLoader(TextDataset(X_tr11, y_train_dl), batch_size=64),
#     device=device, learning_rate=0.001)
# model_final = trainer_final.train(num_epochs=30)
print('Uncomment and adapt the template above to retrain the best model.')"""),

    # CELL 53
    code("""\
# Load test data
test_df = pd.read_csv('data/devex_test_clean.csv')
print(f'Test set: {test_df.shape[0]} rows')

# Template for generating predictions — adjust to match your best model
# from src.data_loader import SDGDataLoader
# loader_test = SDGDataLoader('data/devex_test_clean.csv')
# X_test_raw = loader_test.load_text()
# ... apply same embedding as best model ...
# with torch.no_grad():
#     test_preds = model_final(torch.FloatTensor(X_test_embedded).to(device)).cpu().numpy()
# pred_labels = (test_preds >= 0.5).astype(int)
# pred_df = pd.DataFrame(pred_labels, columns=label_columns)
# pred_df.to_csv('results/test_predictions.csv', index=False)
# print(f'Test predictions saved: {pred_df.shape}')
print('Uncomment and adapt the template above to generate test predictions.')
print('Output: results/test_predictions.csv')"""),

    # CELL 54
    md("""\
---
*End of notebook. All 12 experiments complete.*
*To reproduce: Runtime → Run All in Google Colab after Section 0 setup.*"""),
]

nb.cells.extend(new_cells)

nbformat.write(nb, open(NB_PATH, 'w', encoding='utf-8'))
print(f'Done. Total cells: {len(nb.cells)}')
