import sys
from transformers import pipeline


def predict(text, model_dir="model"):
    clf = pipeline("text-classification", model=model_dir, tokenizer=model_dir)
    result = clf(text, truncation=True, max_length=256)[0]
    label = result["label"].upper()
    return {"label": label, "score": round(result["score"], 4)}


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "This movie was absolutely fantastic! I loved every minute."
    result = predict(text)
    print(f"Text   : {text[:80]}...")
    print(f"Label  : {result['label']}")
    print(f"Score  : {result['score']}")
