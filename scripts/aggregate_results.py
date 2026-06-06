"""
Aggregate all experiment results and create comparison tables
"""

import json
import pandas as pd
import os

print("="*60)
print("Aggregating Experiment Results")
print("="*60)

# Load all experiment results
experiments = []

# Experiment 5: Word2Vec
with open('results/experiment_5/results.json', 'r') as f:
    exp5 = json.load(f)
    experiments.append({
        'Experiment': '5',
        'Name': 'Word2Vec + DeepNN',
        'Embedding': 'Word2Vec',
        'Hamming Loss': exp5['results']['hamming_loss'],
        'HL (Optimal)': exp5['results']['hamming_loss_optimal'],
        'F1 Micro': exp5['results']['f1_micro'],
        'F1 Micro (Optimal)': exp5['results']['f1_micro_optimal'],
        'F1 Macro': exp5['results']['f1_macro'],
        'Optimal Threshold': exp5['results']['optimal_threshold']
    })

# Experiment 7: BERT
with open('results/experiment_7/results.json', 'r') as f:
    exp7 = json.load(f)
    experiments.append({
        'Experiment': '7',
        'Name': 'BERT',
        'Embedding': 'BERT',
        'Hamming Loss': exp7['results']['hamming_loss'],
        'HL (Optimal)': exp7['results']['hamming_loss_optimal'],
        'F1 Micro': exp7['results']['f1_micro'],
        'F1 Micro (Optimal)': exp7['results']['f1_micro_optimal'],
        'F1 Macro': exp7['results']['f1_macro'],
        'Optimal Threshold': exp7['results']['optimal_threshold']
    })

# Experiment 8: Hyperparameter Tuning (best config)
with open('results/experiment_8/best_config.json', 'r') as f:
    exp8 = json.load(f)
    experiments.append({
        'Experiment': '8',
        'Name': 'Word2Vec + DeepNN (Tuned)',
        'Embedding': 'Word2Vec (tuned)',
        'Hamming Loss': exp8['hamming_loss'],
        'HL (Optimal)': exp8['hamming_loss'],  # No optimal threshold in this result
        'F1 Micro': exp8['f1_micro'],
        'F1 Micro (Optimal)': exp8['f1_micro'],
        'F1 Macro': exp8['f1_macro'],
        'Optimal Threshold': 'N/A'
    })

# Experiment 9: Class Imbalance
with open('results/experiment_9/results.json', 'r') as f:
    exp9 = json.load(f)
    experiments.append({
        'Experiment': '9',
        'Name': 'Word2Vec + DeepNN + Class Weights',
        'Embedding': 'Word2Vec + Class Weights',
        'Hamming Loss': exp9['results']['hamming_loss'],
        'HL (Optimal)': exp9['results']['hamming_loss_optimal'],
        'F1 Micro': exp9['results']['f1_micro'],
        'F1 Micro (Optimal)': exp9['results']['f1_micro_optimal'],
        'F1 Macro': exp9['results']['f1_macro'],
        'Optimal Threshold': exp9['results']['optimal_threshold']
    })

# Create DataFrame
df = pd.DataFrame(experiments)

# Add Person 2's baseline
baseline = pd.DataFrame([{
    'Experiment': 'Baseline',
    'Name': 'LinearSVC (Person 2)',
    'Embedding': 'TF-IDF',
    'Hamming Loss': 0.045,
    'HL (Optimal)': 0.045,
    'F1 Micro': 0.6156,
    'F1 Micro (Optimal)': 0.6156,
    'F1 Macro': 0.5149,
    'Optimal Threshold': 'N/A'
}])

# Combine
df_all = pd.concat([baseline, df], ignore_index=True)

# Sort by Hamming Loss
df_all = df_all.sort_values('Hamming Loss')

print("\nExperiment Comparison Table:")
print("="*60)
print(df_all.to_string(index=False))

# Save to CSV
df_all.to_csv('results/experiment_comparison.csv', index=False)
print("\nComparison table saved to results/experiment_comparison.csv")

# Create summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline (Person 2 - LinearSVC): 0.045")
print(f"Best Person 3 Model: {df_all.iloc[1]['Name']} with HL {df_all.iloc[1]['Hamming Loss']:.4f}")
print(f"Improvement over baseline: {(df_all.iloc[1]['Hamming Loss'] - 0.045):.4f}")
if df_all.iloc[1]['Hamming Loss'] < 0.045:
    print("✓ Beat baseline!")
else:
    print("✗ Did not beat baseline")

# Save summary as JSON
summary = {
    'baseline_hamming_loss': 0.045,
    'best_model': {
        'name': df_all.iloc[1]['Name'],
        'experiment_id': df_all.iloc[1]['Experiment'],
        'hamming_loss': float(df_all.iloc[1]['Hamming Loss']),
        'f1_micro': float(df_all.iloc[1]['F1 Micro']),
        'f1_macro': float(df_all.iloc[1]['F1 Macro'])
    },
    'all_experiments': experiments,
    'beats_baseline': bool(df_all.iloc[1]['Hamming Loss'] < 0.045)
}

with open('results/experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to results/experiment_summary.json")
print("="*60)
