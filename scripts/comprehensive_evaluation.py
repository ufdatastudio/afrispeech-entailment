#!/usr/bin/env python3
"""
Comprehensive evaluation script for all audio language models.
Creates a unified table with all models grouped by difficulty.
"""

import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import os

# Define difficulty mapping based on hypothesis content
DIFFICULTY_MAPPING = {}

def load_difficulty_mapping():
    """Load the difficulty mapping from an AudioFlamingo3 file that has difficulty labels."""
    af3_file = "outputs/AudioFlamingo3/interview_nli/results/AudioFlamingo3_interview_nli.jsonl"
    if not os.path.exists(af3_file):
        print(f"Warning: {af3_file} not found, cannot map difficulties")
        return
    
    with open(af3_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            item_id = data['item_id']
            file_name = data.get('file_name', '')
            hypothesis = data.get('hypothesis', '')
            difficulty = data.get('difficulty', 'unknown')
            
            # Store mapping by item_id, file_name+hypothesis, and hypothesis alone
            DIFFICULTY_MAPPING[item_id] = difficulty
            if file_name and hypothesis:
                key = f"{file_name}::{hypothesis}"
                DIFFICULTY_MAPPING[key] = difficulty

def add_difficulty_to_record(record):
    """Add difficulty field to records that don't have it."""
    if 'difficulty' in record:
        return record
    
    item_id = record.get('item_id', '')
    file_name = record.get('file_name', '')
    hypothesis = record.get('hypothesis', '')
    
    # Try multiple mapping strategies
    if item_id in DIFFICULTY_MAPPING:
        record['difficulty'] = DIFFICULTY_MAPPING[item_id]
    elif file_name and hypothesis:
        key = f"{file_name}::{hypothesis}"
        if key in DIFFICULTY_MAPPING:
            record['difficulty'] = DIFFICULTY_MAPPING[key]
        else:
            # Try just the hypothesis
            for mapped_key, diff in DIFFICULTY_MAPPING.items():
                if '::' in mapped_key and mapped_key.split('::')[1] == hypothesis:
                    record['difficulty'] = diff
                    break
            else:
                record['difficulty'] = 'unknown'
    else:
        record['difficulty'] = 'unknown'
    
    return record

def load_model_results(jsonl_path, model_name):
    """Load and process model results from JSONL file."""
    results = []
    
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found, skipping {model_name}")
        return results
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                data = add_difficulty_to_record(data)
                data['model'] = model_name
                results.append(data)
            except json.JSONDecodeError:
                continue
    
    return results

def compute_metrics(df):
    """Compute evaluation metrics for a dataframe."""
    if df.empty:
        return {}
    
    # Map labels to numbers for sklearn
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2, 'unparseable': 3}
    
    # Convert predictions and gold labels (handle case insensitivity)
    y_true = df['gold'].astype(str).str.lower().map(label_map)
    y_pred = df['pred'].astype(str).str.lower().map(label_map)
    
    # Handle missing mappings
    valid_indices = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    
    if len(y_true) == 0:
        return {}
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Class-specific accuracies
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    class_acc = {}
    class_names = ['entailment', 'contradiction', 'neutral', 'unparseable']
    
    for i, class_name in enumerate(class_names):
        if i < cm.shape[0] and cm[i].sum() > 0:
            class_acc[f'CACC_{class_name}'] = cm[i, i] / cm[i].sum()
        else:
            class_acc[f'CACC_{class_name}'] = 0.0
    
    return {
        'N': len(y_true),
        'ACC': accuracy,
        'P_macro': precision,
        'R_macro': recall,
        'F1_macro': f1,
        **class_acc
    }

def main():
    # Load difficulty mapping from AudioFlamingo3
    load_difficulty_mapping()
    
    # Define models to evaluate (including all models)
    models_to_evaluate = [
        ('outputs/AudioFlamingo3/interview_nli/results/AudioFlamingo3_interview_nli.jsonl', 'AudioFlamingo3'),
        ('outputs/AudioFlamingo3_v2/interview_nli/results/AudioFlamingo3_v2_interview_nli.jsonl', 'AudioFlamingo3_v2'),
        ('outputs/AudioFlamingo3_v3/interview_nli/results/AudioFlamingo3_v3_interview_nli.jsonl', 'AudioFlamingo3_v3'),
        ('outputs/AudioFlamingo3_v4/interview_nli/results/AudioFlamingo3_v4_interview_nli.jsonl', 'AudioFlamingo3_v4'),
        ('outputs/AudioFlamingo2/interview_nli/results/AudioFlamingo2_interview_nli.jsonl', 'AudioFlamingo2'),
        ('outputs/GAMA/interview_nli/results/GAMA_interview_nli.jsonl', 'GAMA'),
        ('outputs/Kimi/interview_nli/results/Kimi_interview_nli.jsonl', 'Kimi'),
        ('outputs/LTU/interview_nli/results/LTU_interview_nli.jsonl', 'LTU'),
        ('outputs/Qwen2.5Omni/interview_nli/results/Qwen2.5Omni_interview_nli.jsonl', 'Qwen2.5Omni'),
        ('outputs/Qwen2AudioInstruct/interview_nli/results/Qwen2AudioInstruct_interview_nli.jsonl', 'Qwen2AudioInstruct'),
        ('outputs/SALMONN/interview_nli/results/SALMONN_interview_nli.jsonl', 'SALMONN'),
    ]
    
    # Collect all results
    all_results = []
    for jsonl_path, model_name in models_to_evaluate:
        model_results = load_model_results(jsonl_path, model_name)
        all_results.extend(model_results)
        print(f"Loaded {len(model_results)} results from {model_name}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Debug: Check what models and difficulties we have before filtering
    print("Before filtering:")
    print(f"Models found: {df['model'].value_counts()}")
    print(f"Difficulties found: {df['difficulty'].value_counts()}")
    
    # Filter out records with unknown difficulty
    df = df[df['difficulty'].isin(['easy', 'medium', 'hard'])]
    
    print(f"Total valid records: {len(df)}")
    print(f"Models after filtering: {df['model'].unique()}")
    print(f"Difficulties after filtering: {df['difficulty'].unique()}")
    
    # Compute metrics for each model and difficulty
    results = []
    
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        
        for difficulty in ['easy', 'medium', 'hard']:
            subset = model_df[model_df['difficulty'] == difficulty]
            if subset.empty:
                continue
                
            metrics = compute_metrics(subset)
            if metrics:
                result = {
                    'task': 'interview_nli',
                    'dataset': 'interview',
                    'alm': model,
                    'difficulty': difficulty,
                    **metrics
                }
                results.append(result)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Save CSV
    output_csv = 'outputs/comprehensive_interview_nli_evaluation.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"Saved comprehensive results to {output_csv}")
    
    # Create LaTeX table
    latex_content = create_latex_table(results_df)
    output_tex = 'outputs/comprehensive_interview_nli_evaluation.tex'
    with open(output_tex, 'w') as f:
        f.write(latex_content)
    print(f"Saved LaTeX table to {output_tex}")
    
    # Print summary
    print("\\nSummary:")
    print(results_df.groupby('alm')['ACC'].mean().sort_values(ascending=False))

def create_latex_table(df):
    """Create a LaTeX table from the results DataFrame."""
    
    latex = """\\begin{table}[ht]
\\centering
\\caption{Comprehensive Audio Language Model Evaluation on Interview NLI Task}
\\label{tab:comprehensive_interview_nli}
\\begin{tabular}{llcccccc}
\\toprule
Model & Difficulty & N & Accuracy & F1-macro & Precision & Recall & CACC-Ent \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        model = row['alm'].replace('_', '\\_')
        difficulty = row['difficulty']
        n = int(row['N'])
        acc = f"{row['ACC']:.3f}"
        f1 = f"{row['F1_macro']:.3f}"
        prec = f"{row['P_macro']:.3f}"
        recall = f"{row['R_macro']:.3f}"
        cacc_ent = f"{row.get('CACC_entailment', 0):.3f}"
        
        latex += f"{model} & {difficulty} & {n} & {acc} & {f1} & {prec} & {recall} & {cacc_ent} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

if __name__ == "__main__":
    main()