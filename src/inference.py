"""加载模型并生成摘要。"""
from __future__ import annotations

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import MODELS_DIR

DEFAULT_MODEL_DIR = MODELS_DIR / "pegasus_lcsts"
DEFAULT_BASE_MODEL = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese"


class Summarizer:
    """摘要推理类。"""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        model_path = str(model_dir if model_dir.exists() else DEFAULT_BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def summarize(self, text: str, max_length: int = 64) -> str:
        """生成中文摘要。"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            do_sample=False,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
