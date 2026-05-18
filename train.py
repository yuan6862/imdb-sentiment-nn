import numpy as np
import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 16


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def main():
    os.makedirs("model", exist_ok=True)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, batch_size=1000, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=200,
        no_cuda=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"\nTest Accuracy: {acc:.4f}")

    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "train_samples": len(tokenized["train"]),
        "test_samples": len(tokenized["test"]),
        "base_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved to model/")


if __name__ == "__main__":
    main()
