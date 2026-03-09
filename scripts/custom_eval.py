import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import argparse

def compute_metrics(df):
    # Group by task, dataset, alm
    group_cols = ["task", "dataset", "alm"]
    records = []
    
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["gold", "pred"])
        if g.empty:
            continue
            
        y_true = g["gold"].astype(str).to_numpy()
        y_pred = g["pred"].astype(str).to_numpy()
        
        # Labels: inconsistent, consistent? 
        # evaluation.py infers labels if not strictly provided.
        # User requested consistency task.
        # labels = ["CONSISTENT", "INCONSISTENT"] ? 
        # In the data we saw "CONSISTENT", "INCONSISTENT".
        # Let's infer labels to be safe but ensure consistent order?
        # evaluation.py infers sorted labels if not specified.
        labels = sorted(set(y_true).union(set(y_pred)))
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        
        row = dict(zip(group_cols, keys))
        row["N"] = len(g)
        row["ACC"] = acc
        row["P_macro"] = p
        row["R_macro"] = r
        row["F1_macro"] = f1
        
        # Per class accuracy
        for lab in labels:
            mask = (y_true == lab)
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == lab).mean()
            else:
                class_acc = np.nan
            
            # format column name like ACC_CONSISTENT
            col_name = f"ACC_{lab.upper().replace(' ', '_')}"
            row[col_name] = class_acc

        records.append(row)

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    print(f"Reading {args.predictions_csv}...")
    df = pd.read_csv(args.predictions_csv)
    print(f"Loaded {len(df)} rows.")
    
    # Run metrics
    results = compute_metrics(df)
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "metrics_consistency_custom.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    print(results)

if __name__ == "__main__":
    main()
