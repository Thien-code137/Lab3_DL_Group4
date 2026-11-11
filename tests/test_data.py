"""Dependency-free checks for configuration and pipeline source."""
import json
import unittest
from pathlib import Path
from src.utils import validate_config


class ConfigTests(unittest.TestCase):
    def test_config_splits(self):
        cfg = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["train_size"], 1000)
        self.assertEqual(cfg["validation_size"], 200)
        self.assertEqual(cfg["selection_metric"], "eval_loss")
        self.assertEqual(cfg["train_size"] % 2, 0)

    def test_invalid_train_size(self):
        cfg = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
        cfg["train_size"] = 999
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_invalid_selection_metric(self):
        cfg = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
        cfg["selection_metric"] = "eval_accuracy"
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
