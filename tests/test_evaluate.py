"""Kiểm tra metrics bằng dữ liệu nhỏ, không tải IMDb."""
import unittest

from src.evaluate import scores
from src.visualize import rows_from_log_history


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics(self):
        result = scores([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision_positive"], 2 / 3)
        self.assertAlmostEqual(result["recall_positive"], 1.0)

    def test_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            scores([0, 1], [1])

    def test_log_history(self):
        history = rows_from_log_history([{"epoch": 1, "loss": 0.5},
                                         {"epoch": 1, "eval_loss": 0.4, "eval_accuracy": 0.8}])
        self.assertEqual(history[0]["train_loss"], 0.5)
        self.assertEqual(history[1]["val_loss"], 0.4)


if __name__ == "__main__":
    unittest.main()
