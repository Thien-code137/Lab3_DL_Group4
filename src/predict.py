"""Inference from an already exported local fine-tuned checkpoint."""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(**tokenizer(args.text, return_tensors="pt", truncation=True))["logits"], dim=-1)[0]
    index = int(probabilities.argmax())
    print({"label": "POSITIVE" if index else "NEGATIVE", "confidence": float(probabilities[index])})


if __name__ == "__main__":
    main()
