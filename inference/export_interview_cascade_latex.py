#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "asr_model": "ASR",
        "text_llm": "TextLLM",
        "alm": "System",
        "difficulty": "Difficulty",
        "N": "N",
        "ACC": "ACC",
        "P_macro": "P",
        "R_macro": "R",
        "F1_macro": "F1",
        "EACC": "EACC",
        "NACC": "NACC",
        "CACC": "CACC",
        "unparseable_rate": "Unparseable",
    }
    out = out.rename(columns=rename_map)
    return out


def to_latex(csv_path: Path, tex_path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(csv_path)
    df = format_table(df)

    latex = df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label,
        escape=True,
    )

    tex_path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export interview cascade metrics CSVs to LaTeX tables.")
    parser.add_argument(
        "--root",
        default="/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/cascade_entailment",
        help="Directory containing interview_cascade_metrics_overall.csv and interview_cascade_metrics_by_difficulty.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    overall_csv = root / "interview_cascade_metrics_overall.csv"
    diff_csv = root / "interview_cascade_metrics_by_difficulty.csv"

    overall_tex = root / "interview_cascade_metrics_overall.tex"
    diff_tex = root / "interview_cascade_metrics_by_difficulty.tex"

    to_latex(
        overall_csv,
        overall_tex,
        caption="Interview NLI cascade results (ASR + Text LLM), overall.",
        label="tab:interview_cascade_overall",
    )
    to_latex(
        diff_csv,
        diff_tex,
        caption="Interview NLI cascade results (ASR + Text LLM), grouped by hypothesis difficulty.",
        label="tab:interview_cascade_by_difficulty",
    )

    print(f"Wrote: {overall_tex}")
    print(f"Wrote: {diff_tex}")


if __name__ == "__main__":
    main()
