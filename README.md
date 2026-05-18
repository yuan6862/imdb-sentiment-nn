---
license: mit
language:
- en
tags:
- sentiment-analysis
- text-classification
- pytorch
- distilbert
- imdb
datasets:
- imdb
metrics:
- accuracy
pipeline_tag: text-classification
---

# IMDB Sentiment Analysis — DistilBERT Fine-tuned

Sentiment analysis model fine-tuned from `distilbert-base-uncased` on the full IMDB 50K dataset, auto-deployed via **GitHub Actions CI/CD → Hugging Face Hub**.

## Model Architecture

| Component | Details |
|-----------|---------|
| Base model | `distilbert-base-uncased` (66M parameters) |
| Fine-tuning | Full model fine-tuning on IMDB 50K |
| Max sequence length | 256 tokens |
| Output | Binary: NEGATIVE / POSITIVE |
| Task | Binary sentiment classification |

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="enzoliao/imdb-sentiment-nn")
result = clf("This movie was absolutely fantastic! I loved every minute.")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]
```

## Training

- **Dataset**: IMDB Full 50K (25,000 train / 25,000 test, standard split)
- **Base model**: `distilbert-base-uncased`
- **Optimizer**: AdamW (default HF Trainer settings)
- **Epochs**: 3
- **Batch size**: 16

## CI/CD Pipeline

Code pushed to GitHub → GitHub Actions fine-tunes DistilBERT → uploads model to this repo automatically. No manual uploads.

## Repository Structure

```
imdb-sentiment-nn/
├── data/imdb_top_500.csv
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .github/workflows/train-and-upload.yml
```
