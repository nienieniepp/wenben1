"""评估基线和 Transformer 模型的 ROUGE。"""
from __future__ import annotations

import argparse
import pandas as pd
import evaluate

from baseline_tfidf import summarize_tfidf
from inference import Summarizer
from utils import DATA_DIR, RESULTS_DIR, ensure_dirs


def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
    return {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "rouge_scores.csv"))
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(args.test_file)
    refs = df["summary"].astype(str).tolist()

    baseline_preds = df["text"].astype(str).map(summarize_tfidf).tolist()
    baseline_scores = compute_rouge(baseline_preds, refs)

    summarizer = Summarizer()
    transformer_preds = [summarizer.summarize(t) for t in df["text"].astype(str).tolist()]
    transformer_scores = compute_rouge(transformer_preds, refs)

    out_df = pd.DataFrame([
        {"model": "baseline_tfidf", **baseline_scores},
        {"model": "transformer_pegasus", **transformer_scores},
    ])
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out_df)
    print(f"ROUGE结果已保存: {args.output}")


if __name__ == "__main__":
    main()
