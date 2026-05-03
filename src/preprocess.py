"""数据预处理脚本（去空白与长度裁剪示例）。"""
from __future__ import annotations

import argparse
import pandas as pd

from utils import DATA_DIR


def clean_text(s: str) -> str:
    """简单清洗：去除首尾空格。"""
    return str(s).strip()


def process_file(filename: str) -> None:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df["summary"] = df["summary"].map(clean_text)
    df = df[(df["text"] != "") & (df["summary"] != "")]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已处理: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=["train.csv", "valid.csv", "test.csv"])
    args = parser.parse_args()

    for name in args.files:
        process_file(name)


if __name__ == "__main__":
    main()
