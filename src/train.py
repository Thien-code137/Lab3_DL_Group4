"""Huấn luyện DistilBERT trên IMDb và xuất kết quả cho từng lần chạy."""
import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from src.data import prepare_splits
from src.evaluate import compute_metrics, evaluate_and_save
from src.model import load_model_and_tokenizer
from src.utils import read_config, save_json
from src.visualize import save_training_history


def train(config_path="configs/default.json", output_dir="outputs/new_run"):
    config = read_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])

    # Ba tập đảm nhiệm vai trò riêng: train, chọn checkpoint, test cuối.
    train_set, val_set, test_set = prepare_splits(config)
    save_json(output / "data_summary.json", {
        name: {"size": len(ds), "labels": dict(Counter(ds["label"]))}
        for name, ds in (("train", train_set), ("validation", val_set), ("test", test_set))
    })
    tokenizer, model = load_model_and_tokenizer(config["model"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config["max_length"])

    train_tokens = train_set.map(tokenize, batched=True)
    val_tokens = val_set.map(tokenize, batched=True)
    test_tokens = test_set.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        seed=config["seed"], data_seed=config["seed"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model=config["selection_metric"],
        greater_is_better=False, save_total_limit=2, report_to="none", optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tokens, eval_dataset=val_tokens,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = datetime.now(timezone.utc).isoformat()
    train_result = trainer.train()
    trainer.save_state()
    save_training_history(trainer.state.log_history, output)

    # load_best_model_at_end khôi phục trọng số từ checkpoint tốt nhất theo validation.
    test_output = trainer.predict(test_tokens)
    summary = evaluate_and_save(test_set, test_output.predictions, output,
                                trainer.state.best_model_checkpoint, trainer.state.best_metric)

    trainer.save_model(str(output / "best_model"))
    tokenizer.save_pretrained(str(output / "best_model"))
    save_json(output / "run_info.json", {
        "started_at_utc": start_time,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(args.device),
        "training_loss": float(train_result.training_loss),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_loss": trainer.state.best_metric,
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved to:", output.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with IMDb")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="outputs/new_run")
    options = parser.parse_args()
    train(options.config, options.output)


if __name__ == "__main__":
    main()
