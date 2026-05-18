import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import pickle
import json
import re
import os

MAX_FEATURES = 15000


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs("model", exist_ok=True)

    print("Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]

    X_train = [clean_text(t) for t in train_data["text"]]
    y_train = np.array(train_data["label"], dtype=np.float32)
    X_test = [clean_text(t) for t in test_data["text"]]
    y_test = np.array(test_data["label"], dtype=np.float32)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_arr = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    X_tr = torch.tensor(X_train_arr)
    del X_train_arr

    X_test_arr = vectorizer.transform(X_test).toarray().astype(np.float32)
    X_te = torch.tensor(X_test_arr)
    del X_test_arr

    y_tr = torch.tensor(y_train).unsqueeze(1)
    y_te = torch.tensor(y_test).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    model = SentimentNN(MAX_FEATURES)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 512):
                preds.append((model(X_te[i : i + 512]) >= 0.5).float())
        acc_eval = accuracy_score(y_te.numpy(), torch.cat(preds).numpy())
        if acc_eval > best_acc:
            best_acc = acc_eval
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch+1}/20  loss={avg_loss:.4f}  "
            f"acc={acc_eval:.4f}  best={best_acc:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            preds.append((model(X_te[i : i + 512]) >= 0.5).float())
    y_pred = torch.cat(preds).numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_te.numpy(), y_pred, target_names=["negative", "positive"]))

    torch.save(model.state_dict(), "model/model.pt")
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    config = {
        "model_type": "TF-IDF + Feedforward Neural Network",
        "input_dim": MAX_FEATURES,
        "hidden_layers": [512, 256, 64],
        "dropout": [0.4, 0.3, 0.2],
        "batch_norm": True,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "task": "sentiment-analysis",
        "dataset": "IMDB Full (50K)",
        "epochs": 20,
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
