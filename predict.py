import torch
import torch.nn as nn
import pickle
import re
import sys
import json


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


def predict(text, model_dir="model"):
    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)
    with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    model = SentimentNN(config["input_dim"])
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
    model.eval()

    import numpy as np
    vec = vectorizer.transform([clean_text(text)]).toarray().astype("float32")
    tensor = torch.tensor(vec)

    with torch.no_grad():
        prob = model(tensor).item()

    return {"label": "POSITIVE" if prob >= 0.5 else "NEGATIVE", "score": round(prob, 4)}


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "This movie was absolutely fantastic! I loved every minute."
    result = predict(text)
    print(f"Text   : {text[:80]}...")
    print(f"Label  : {result['label']}")
    print(f"Score  : {result['score']}")
