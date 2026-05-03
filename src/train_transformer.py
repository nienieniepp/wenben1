"""微调 Pegasus 中文摘要模型。"""
from __future__ import annotations

import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import DATA_DIR, MODELS_DIR, ensure_dirs

MODEL_NAME = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese"


def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["text", "summary"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--valid_file", default=str(DATA_DIR / "valid.csv"))
    parser.add_argument("--output_dir", default=str(MODELS_DIR / "pegasus_lcsts"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    train_ds = load_csv_dataset(args.train_file)
    valid_ds = load_csv_dataset(args.valid_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["text"], max_length=args.max_source_length, truncation=True
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_valid = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-5,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
