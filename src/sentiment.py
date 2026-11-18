"""Exercise 1: pretrained Hugging Face sentiment inference, no training."""
import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_CHECKPOINT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def infer(text, checkpoint=DEFAULT_CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(probabilities.argmax())
    return {"text": text, "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            "input_ids": inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "label": model.config.id2label[index], "score": float(probabilities[index])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I love this movie. It is amazing!")
    args = parser.parse_args()
    print(infer(args.text))
