import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_results(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                   data.append(json.loads(line))
                except json.JSONDecodeError:
                   pass
    
    df = pd.DataFrame(data)
    
    # Required columns: task, dataset, alm, N, ACC, P_macro, R_macro, F1_macro, EACC, NACC, CACC
    results = []
    
    # Filter out FAILED predictions for metric calculation (or treat as specific class? usually filtered or counted as wrong)
    # The reference table likely counts N as total samples.
    # If we want to strictly match the reference which seems to be standard NLP metrics:
    # We should probably treat FAILED as a wrong prediction (which it is).
    # However, sklearn requires consistent labels. 
    # Let's clean the labels to be consistent. 
    
    models = df['model'].unique()
    
    # Label mapping for per-class acc
    # EACC -> Entailment
    # NACC -> Neutral
    # CACC -> Contradiction
    
    for model in sorted(models):
        model_df = df[df['model'] == model].copy()
        
        # We need to handle FAILED. If we treat them as "WRONG", we can fill them with a placeholder that is NOT gold.
        # But for sklearn metrics, better to just let them be "FAILED" and ensure "FAILED" is not in gold.
        model_df['pred'] = model_df['pred'].fillna("FAILED")
        
        y_true = model_df['gold'].str.upper()
        y_pred = model_df['pred'].str.upper()
        
        # Compute overall stats
        N = len(model_df)
        acc = accuracy_score(y_true, y_pred)
        
        # Macro metrics (unweighted mean of per-class metrics)
        # We use labels=['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'] to ensure we target the classes of interest
        # Any 'FAILED' pred will count as a mismatch.
        labels = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=labels, zero_division=0
        )
        
        # Per-class Accuracy (Recall per class)
        # EACC = Recall of Entailment
        # NACC = Recall of Neutral
        # CACC = Recall of Contradiction
        # precision_recall_fscore_support returns (p, r, f, s) for each label sorted or specified
        _, recalls, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        
        e_acc = recalls[0] # Entailment
        n_acc = recalls[1] # Neutral
        c_acc = recalls[2] # Contradiction
        
        results.append({
            "task": "nli",
            "dataset": "Medical", # As implied by the directory name and content
            "alm": model,
            "N": N,
            "ACC": acc,
            "P_macro": p_macro,
            "R_macro": r_macro,
            "F1_macro": f1_macro,
            "EACC": e_acc,
            "NACC": n_acc,
            "CACC": c_acc
        })
    
    results_df = pd.DataFrame(results)
    
    # Print as CSV
    print(results_df.to_csv(index=False))

if __name__ == "__main__":
    INPUT_FILE = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/CaptionBeforeReasoning/standardized_predictions.jsonl"
    evaluate_results(INPUT_FILE)
