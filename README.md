# 中文抽象文本摘要系统（Transformer）

## 项目背景
在数字媒体场景中，新闻、社媒与评论内容增长迅速，人工阅读成本高。自动摘要可以帮助用户快速获取关键信息。

## 项目目标
构建一个中文摘要系统：输入中文文章/段落，输出简洁摘要；并比较传统基线与深度学习模型效果。

## 数据集介绍
- 数据集：`hugcyp/LCSTS`（Hugging Face）
- 特点：大规模中文短文本摘要数据
- 下载方式：代码自动下载，无需手动下载

## 模型方法
1. **基线模型（Extractive）**：TF-IDF 句子打分抽取关键句。  
2. **主模型（Abstractive）**：`IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese`。  
3. 在 LCSTS 子集上进行轻量微调并保存最优模型。  
4. 使用 ROUGE-1/2/L 评估并比较。

## 实验结果
运行 `python src/evaluate.py` 后，会生成：
- `results/rouge_scores.csv`：包含基线与 Transformer 的 ROUGE 对比结果。

## 如何运行
```bash
pip install -r requirements.txt
python src/download_dataset.py --train_size 2000
python src/preprocess.py
python src/baseline_tfidf.py
python src/train_transformer.py
python src/evaluate.py
python app/app.py
```

## 小组分工
- 成员A：数据下载与预处理（`download_dataset.py`, `preprocess.py`）
- 成员B：基线方法与评估（`baseline_tfidf.py`, `evaluate.py`）
- 成员C：Transformer 微调与推理（`train_transformer.py`, `inference.py`）
- 成员D：Web Demo 与文档（`app.py`, `README`, `slides`, `report`）
