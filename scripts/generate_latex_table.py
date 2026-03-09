#!/usr/bin/env python3
"""
Generate comprehensive LaTeX table for all interview NLI evaluation results.
"""

import pandas as pd

def create_comprehensive_latex_table(df):
    """Create a comprehensive LaTeX table with all models and metrics."""
    
    latex = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{rotating}
\\usepackage{array}
\\usepackage{longtable}
\\usepackage{geometry}
\\geometry{landscape, margin=0.5in}

\\begin{document}

\\begin{sidewaystable}
\\centering
\\caption{Comprehensive Audio Language Model Evaluation on Interview NLI Task by Difficulty}
\\label{tab:comprehensive_interview_nli_all}
\\footnotesize
\\begin{longtable}{llccccccc}
\\toprule
\\textbf{Model} & \\textbf{Difficulty} & \\textbf{N} & \\textbf{ACC} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Recall} & \\textbf{E-ACC} & \\textbf{C-ACC} & \\textbf{N-ACC} \\\\
\\midrule
\\endfirsthead

\\multicolumn{9}{c}{\\tablename\\ \\thetable{} -- \\textit{Continued from previous page}} \\\\
\\toprule
\\textbf{Model} & \\textbf{Difficulty} & \\textbf{N} & \\textbf{ACC} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Recall} & \\textbf{E-ACC} & \\textbf{C-ACC} & \\textbf{N-ACC} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{9}{r}{\\textit{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

"""
    
    # Sort by overall performance (average accuracy across difficulties)
    model_avg_acc = df.groupby('alm')['ACC'].mean().sort_values(ascending=False)
    
    for model_name in model_avg_acc.index:
        model_data = df[df['alm'] == model_name].sort_values('difficulty')
        
        # Add model separator
        if model_name != model_avg_acc.index[0]:  # Not the first model
            latex += "\\midrule\n"
        
        for i, (_, row) in enumerate(model_data.iterrows()):
            model_display = model_name.replace('_', '\\_') if i == 0 else ""
            difficulty = row['difficulty']
            n = int(row['N'])
            acc = f"{row['ACC']:.3f}"
            f1 = f"{row['F1_macro']:.3f}"
            prec = f"{row['P_macro']:.3f}"
            recall = f"{row['R_macro']:.3f}"
            e_acc = f"{row['CACC_entailment']:.3f}"
            c_acc = f"{row['CACC_contradiction']:.3f}"
            n_acc = f"{row['CACC_neutral']:.3f}"
            
            latex += f"{model_display} & {difficulty} & {n} & {acc} & {f1} & {prec} & {recall} & {e_acc} & {c_acc} & {n_acc} \\\\\n"
    
    latex += """
\\end{longtable}
\\end{sidewaystable}

\\vspace{1cm}

\\begin{table}[ht]
\\centering
\\caption{Model Rankings by Overall Performance (Averaged Across Difficulties)}
\\label{tab:model_rankings}
\\begin{tabular}{lcccccccc}
\\toprule
\\textbf{Rank} & \\textbf{Model} & \\textbf{ACC} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Recall} & \\textbf{E-ACC} & \\textbf{C-ACC} & \\textbf{N-ACC} \\\\
\\midrule
"""
    
    # Add summary rankings
    summary_df = df.groupby('alm')[['ACC', 'F1_macro', 'P_macro', 'R_macro', 'CACC_entailment', 'CACC_contradiction', 'CACC_neutral']].mean()
    summary_df = summary_df.sort_values('ACC', ascending=False)
    
    for rank, (model, row) in enumerate(summary_df.iterrows(), 1):
        model_clean = model.replace('_', '\\_')
        latex += f"{rank} & {model_clean} & {row['ACC']:.3f} & {row['F1_macro']:.3f} & {row['P_macro']:.3f} & {row['R_macro']:.3f} & {row['CACC_entailment']:.3f} & {row['CACC_contradiction']:.3f} & {row['CACC_neutral']:.3f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}

\\vspace{1cm}

\\section*{Notes}
\\begin{itemize}
\\item \\textbf{ACC}: Overall accuracy
\\item \\textbf{F1}: Macro F1-score
\\item \\textbf{Prec}: Macro precision  
\\item \\textbf{Recall}: Macro recall
\\item \\textbf{E-ACC}: Entailment class accuracy
\\item \\textbf{C-ACC}: Contradiction class accuracy
\\item \\textbf{N-ACC}: Neutral class accuracy
\\item \\textbf{N}: Number of test instances per difficulty level (69 each)
\\item Models are ranked by overall accuracy averaged across all difficulty levels
\\item AudioFlamingo3\\_v2, v3, v4 represent different prompt engineering variants
\\end{itemize}

\\end{document}"""
    
    return latex

def main():
    # Load the comprehensive results
    df = pd.read_csv('outputs/comprehensive_interview_nli_evaluation.csv')
    
    # Generate LaTeX table
    latex_content = create_comprehensive_latex_table(df)
    
    # Save LaTeX file
    output_file = 'outputs/comprehensive_interview_nli_evaluation_complete.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✅ Comprehensive LaTeX table saved to: {output_file}")
    print(f"📊 Total models: {df['alm'].nunique()}")
    print(f"📈 Total records: {len(df)}")
    
    # Show top 5 performers
    summary = df.groupby('alm')['ACC'].mean().sort_values(ascending=False)
    print("\n🏆 Top 5 Models by Overall Accuracy:")
    for i, (model, acc) in enumerate(summary.head().items(), 1):
        print(f"   {i}. {model}: {acc:.3f}")

if __name__ == "__main__":
    main()