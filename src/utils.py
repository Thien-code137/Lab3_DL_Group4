"""Kiểm tra cấu hình trước khi train và lưu JSON."""
import json
import math
from pathlib import Path


def validate_config(cfg):
    required = ("seed", "dataset", "model", "train_size", "validation_size",
                "test_size", "max_length", "epochs", "batch_size", "learning_rate",
                "weight_decay", "selection_metric")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration: {missing}")
    for key in ("train_size", "validation_size", "test_size", "max_length", "epochs", "batch_size"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["train_size"] % 2 or cfg["validation_size"] % 2:
        raise ValueError("Balanced train/validation sizes must be even")
    if not (0 < cfg["learning_rate"] < 1):
        raise ValueError("Invalid learning rate")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if cfg["selection_metric"] != "eval_loss":
        raise ValueError("Use eval_loss for checkpoint selection")
    return cfg


def read_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
