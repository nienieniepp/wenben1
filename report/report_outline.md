# COMM7340 Project Report Outline (English)

## 1. Introduction
- Background of Chinese text summarization in digital media.
- Why abstractive summarization is useful.

## 2. Objective
- Build a Chinese summarization system using Transformer.
- Compare TF-IDF extractive baseline vs fine-tuned Transformer.

## 3. Methods
### 3.1 Dataset
- Hugging Face `hugcyp/LCSTS`
- Automatic download by code
- Subset strategy for lightweight training

### 3.2 Baseline
- TF-IDF sentence scoring and extraction

### 3.3 Transformer Model
- Pretrained model: `IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese`
- Fine-tuning setup (epochs, batch size, max length)

### 3.4 Evaluation
- ROUGE-1 / ROUGE-2 / ROUGE-L
- Comparison protocol on test split

## 4. Experiments
- Hardware and software environment
- Data subset sizes
- Training configuration

## 5. Results
- ROUGE table from `results/rouge_scores.csv`
- Qualitative examples of generated summaries

## 6. Discussion
- Why Transformer outperforms baseline
- Error cases and limitations
- Effect of subset size and short training

## 7. Conclusion
- Summary of findings and practical value
- Future work: larger data, longer training, human evaluation

## 8. Contributions
- Member-wise contribution summary and coding responsibilities
