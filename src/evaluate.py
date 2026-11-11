"""Tính các chỉ số và xuất kết quả đánh giá test."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.utils import save_json


def scores(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have equal shapes")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "precision_positive": float(precision),
            "recall_positive": float(recall), "f1_positive": float(f1)}


def compute_metrics(eval_prediction):
    pred = np.argmax(eval_prediction.predictions, axis=-1)
    # Hugging Face Trainer uses the returned names prefixed with eval_ on validation.
    return scores(eval_prediction.label_ids, pred)


def evaluate_and_save(dataset, logits, output_dir, best_checkpoint, best_val_loss):
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    actual = np.asarray(dataset["label"])
    predicted = np.asarray(logits).argmax(axis=-1)
    result = {"test_samples": len(actual), **scores(actual, predicted),
              "best_val_loss": None if best_val_loss is None else float(best_val_loss),
              "best_checkpoint": best_checkpoint}
    save_json(output / "test_metrics.json", result)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", "NEGATIVE", "POSITIVE"])
        for label, row in zip(["NEGATIVE", "POSITIVE"], matrix):
            writer.writerow([label, *row])
    (output / "classification_report.txt").write_text(
        classification_report(actual, predicted, labels=[0, 1],
                              target_names=["NEGATIVE", "POSITIVE"], zero_division=0),
        encoding="utf-8")

    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "true_label", "predicted_label", "correct"])
        for text, y, prediction in zip(dataset["text"], actual, predicted):
            writer.writerow([text, int(y), int(prediction), int(y == prediction)])

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], ["NEGATIVE", "POSITIVE"])
    ax.set_yticks([0, 1], ["NEGATIVE", "POSITIVE"])
    ax.set(xlabel="Predicted label", ylabel="True label", title="IMDb confusion matrix")
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return result
