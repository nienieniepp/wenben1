"""Flask 中文摘要演示。"""
from __future__ import annotations

from pathlib import Path
import sys
from flask import Flask, render_template, request

# 兼容从 app 目录启动
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from inference import Summarizer  # noqa: E402

app = Flask(__name__)
summarizer = Summarizer()


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = ""
    text_len = 0
    summary_len = 0

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            summary = summarizer.summarize(text)
            text_len = len(text)
            summary_len = len(summary)

    return render_template(
        "index.html",
        text=text,
        summary=summary,
        text_len=text_len,
        summary_len=summary_len,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
