"""Lưu lịch sử huấn luyện và đồ thị loss."""
import csv
from pathlib import Path


def rows_from_log_history(log_history):
    return [{"epoch": item.get("epoch", ""), "step": item.get("step", ""),
             "train_loss": item.get("loss", ""), "val_loss": item.get("eval_loss", ""),
             "val_accuracy": item.get("eval_accuracy", ""),
             "val_f1": item.get("eval_f1_positive", "")}
            for item in log_history]


def save_training_history(log_history, output_dir):
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    rows = rows_from_log_history(log_history)
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "step", "train_loss",
                                                  "val_loss", "val_accuracy", "val_f1"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, legend in (("train_loss", "Training loss"), ("val_loss", "Validation loss")):
        points = [(float(row["epoch"]), float(row[name])) for row in rows
                  if row["epoch"] != "" and row[name] != ""]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker="o", label=legend)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss during fine-tuning")
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curves.png", dpi=180)
    plt.close(fig)
