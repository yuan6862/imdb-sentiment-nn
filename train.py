import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import re
import os

MAX_FEATURES = 5000


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs("model", exist_ok=True)

    df = pd.read_csv("data/imdb_top_500.csv")
    df["text"] = df["text"].apply(clean_text)
    X, y = df["text"].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    X_test_vec = vectorizer.transform(X_test).toarray().astype(np.float32)

    X_tr = torch.tensor(X_train_vec)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test_vec)
    y_te = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    model = SentimentNN(MAX_FEATURES)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/30  loss={epoch_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = (model(X_te) >= 0.5).float().numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_te.numpy(), y_pred, target_names=["negative", "positive"]))

    torch.save(model.state_dict(), "model/model.pt")
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    config = {
        "model_type": "TF-IDF + Feedforward Neural Network",
        "input_dim": MAX_FEATURES,
        "hidden_layers": [256, 64],
        "dropout": 0.3,
        "ngram_range": [1, 2],
        "task": "sentiment-analysis",
        "dataset": "IMDB Top 500",
        "epochs": 30,
    }
    with open("model/config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved: model/model.pt, model/vectorizer.pkl, model/config.json, model/metrics.json")


if __name__ == "__main__":
    main()
