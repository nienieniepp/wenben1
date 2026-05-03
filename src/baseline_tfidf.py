"""TF-IDF 抽取式摘要基线。"""
from __future__ import annotations

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import DATA_DIR, RESULTS_DIR, ensure_dirs, split_sentences_zh


def summarize_tfidf(text: str, max_sentences: int = 2) -> str:
    """使用句子 TF-IDF 分数抽取关键句。"""
    sents = split_sentences_zh(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return "".join(sents)

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(sents)
    scores = mat.sum(axis=1).A1
    top_idx = sorted(scores.argsort()[-max_sentences:])
    return "".join(sents[i] for i in top_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "baseline_predictions.csv"))
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(args.input)
    df["baseline_pred"] = df["text"].map(summarize_tfidf)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"基线摘要已保存: {args.output}")


if __name__ == "__main__":
    main()
