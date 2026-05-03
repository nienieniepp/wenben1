# Slide 1: Topic & Team
- Chinese Abstractive Text Summarization System based on Transformer
- COMM7340 Group Project
- Team members and roles

# Slide 2: Objective
- Input Chinese article/paragraph
- Output concise Chinese summary
- Compare baseline and Transformer performance

# Slide 3: Dataset
- Hugging Face: hugcyp/LCSTS
- Large-scale Chinese text-summary pairs
- Automatic download in code

# Slide 4: Overall Method
- Data pipeline
- Baseline (extractive)
- Transformer (abstractive)
- Evaluation and demo

# Slide 5: Baseline
- TF-IDF sentence scoring
- Extract top-k sentences as summary
- Lightweight and interpretable

# Slide 6: Transformer Model
- IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese
- Fine-tuning on LCSTS subset
- Generate abstractive summaries

# Slide 7: Training Setup
- Subset size (e.g., train=2000)
- Key hyperparameters (batch size, epochs, max lengths)
- Model saving strategy

# Slide 8: Results
- ROUGE-1 / ROUGE-2 / ROUGE-L
- Baseline vs Transformer comparison table

# Slide 9: Demo
- Flask web app
- Input Chinese text
- Output summary + text length + summary length

# Slide 10: Discussion
- Strengths and limitations
- Typical good/bad examples

# Slide 11: Conclusion
- Achievements against objective
- Real-world value in digital media

# Slide 12: Contribution
- Member-wise coding and documentation contributions
- Future improvement directions
