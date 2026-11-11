"""Khởi tạo tokenizer và mô hình phân loại cảm xúc hai lớp."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    return tokenizer, model
