---
license: mit
language:
- en
tags:
- sentiment-analysis
- text-classification
- pytorch
- tfidf
- imdb
datasets:
- imdb
metrics:
- accuracy
pipeline_tag: text-classification
---

# IMDB Sentiment Analysis — TF-IDF + Neural Network

Sentiment analysis model trained on IMDB Top 500 movie reviews, auto-deployed via **GitHub Actions CI/CD → Hugging Face Hub**.

## Model Architecture

| Component | Details |
|-----------|---------|
| Feature extraction | TF-IDF (15000 features, unigrams + bigrams, sublinear_tf) |
| Input layer | 15000-dim TF-IDF vector |
| Hidden layer 1 | Linear(15000→512) + BatchNorm + LeakyReLU + Dropout(0.4) |
| Hidden layer 2 | Linear(512→256) + BatchNorm + LeakyReLU + Dropout(0.3) |
| Hidden layer 3 | Linear(256→64) + BatchNorm + LeakyReLU + Dropout(0.2) |
| Output layer | Linear(64→1) + Sigmoid |
| Task | Binary sentiment classification |

## Usage

```python
import torch
import torch.nn as nn
import pickle, json, re
from huggingface_hub import hf_hub_download

class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# Download artifacts
repo = "enzoliao/imdb-sentiment-nn"
vectorizer_path = hf_hub_download(repo_id=repo, filename="vectorizer.pkl")
model_path      = hf_hub_download(repo_id=repo, filename="model.pt")
config_path     = hf_hub_download(repo_id=repo, filename="config.json")

with open(config_path) as f:
    config = json.load(f)
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

model = SentimentNN(config["input_dim"])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Predict
import numpy as np
text = "This movie was absolutely fantastic!"
text = re.sub(r"<[^>]+>", " ", text).strip().lower()
vec  = vectorizer.transform([text]).toarray().astype("float32")
with torch.no_grad():
    prob = model(torch.tensor(vec)).item()
label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
print(f"{label} ({prob:.4f})")
```

## Training

- **Dataset**: IMDB Top 500 (400 train / 100 test, stratified split)
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Epochs**: 100 (best checkpoint saved)
- **Batch size**: 32

## CI/CD Pipeline

Code pushed to GitHub → GitHub Actions trains the model → uploads artifacts to this repo automatically. No manual uploads.

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
