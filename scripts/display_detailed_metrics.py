#!/usr/bin/env python3
"""
Display comprehensive evaluation results with all metrics.
"""

import pandas as pd

def main():
    # Load the comprehensive results
    df = pd.read_csv('outputs/comprehensive_interview_nli_evaluation.csv')
    
    # Define metric columns
    metrics = ['ACC', 'P_macro', 'R_macro', 'F1_macro', 'CACC_entailment', 'CACC_contradiction', 'CACC_neutral']
    
    # Create summary by model (average across difficulties)
    model_summary = df.groupby('alm')[metrics].mean().round(3)
    model_summary = model_summary.sort_values('ACC', ascending=False)
    
    print("=== OVERALL MODEL RANKING (Averaged Across Difficulties) ===")
    print("")
    print(f"{'Model':<20} {'ACC':<7} {'F1':<7} {'Prec':<7} {'Recall':<7} {'E-ACC':<7} {'C-ACC':<7} {'N-ACC':<7}")
    print("=" * 85)
    
    for model, row in model_summary.iterrows():
        print(f"{model:<20} {row['ACC']:<7.3f} {row['F1_macro']:<7.3f} {row['P_macro']:<7.3f} "
              f"{row['R_macro']:<7.3f} {row['CACC_entailment']:<7.3f} {row['CACC_contradiction']:<7.3f} "
              f"{row['CACC_neutral']:<7.3f}")
    
    print("\n\n=== DETAILED RESULTS BY DIFFICULTY ===")
    print("")
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"=== {difficulty.upper()} DIFFICULTY ===")
        subset = df[df['difficulty'] == difficulty].sort_values('ACC', ascending=False)
        
        print(f"{'Model':<20} {'ACC':<7} {'F1':<7} {'Prec':<7} {'Recall':<7} {'E-ACC':<7} {'C-ACC':<7} {'N-ACC':<7}")
        print("-" * 85)
        
        for _, row in subset.iterrows():
            print(f"{row['alm']:<20} {row['ACC']:<7.3f} {row['F1_macro']:<7.3f} {row['P_macro']:<7.3f} "
                  f"{row['R_macro']:<7.3f} {row['CACC_entailment']:<7.3f} {row['CACC_contradiction']:<7.3f} "
                  f"{row['CACC_neutral']:<7.3f}")
        print("")

if __name__ == "__main__":
    main()