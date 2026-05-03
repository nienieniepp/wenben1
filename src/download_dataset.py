"""下载 LCSTS 数据集并保存子集到 CSV。"""
from __future__ import annotations

import argparse
from datasets import load_dataset
import pandas as pd

from utils import DATA_DIR, ensure_dirs


def build_df(split_data, size: int) -> pd.DataFrame:
    """根据大小抽样并标准化字段名。"""
    subset = split_data.select(range(min(size, len(split_data))))
    df = pd.DataFrame(subset)

    # LCSTS 常见字段为 content(原文) 与 summary(摘要)
    if "content" in df.columns:
        df = df.rename(columns={"content": "text"})
    if "text" not in df.columns:
        raise ValueError(f"未找到原文字段，当前列: {df.columns.tolist()}")
    if "summary" not in df.columns:
        raise ValueError(f"未找到摘要字段，当前列: {df.columns.tolist()}")

    return df[["text", "summary"]].dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--valid_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    args = parser.parse_args()

    ensure_dirs()

    dataset = load_dataset("hugcyp/LCSTS")

    train_df = build_df(dataset["train"], args.train_size)
    valid_df = build_df(dataset.get("validation", dataset["train"]), args.valid_size)
    test_df = build_df(dataset.get("test", dataset["train"]), args.test_size)

    train_df.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(DATA_DIR / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8-sig")

    print("数据集保存完成：")
    print(DATA_DIR / "train.csv")
    print(DATA_DIR / "valid.csv")
    print(DATA_DIR / "test.csv")


if __name__ == "__main__":
    main()
