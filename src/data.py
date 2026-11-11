"""Chia dữ liệu IMDb thành train, validation và test."""
from collections import Counter
from datasets import load_dataset


def balanced_subset(dataset, n, seed):
    """Lấy số lượng mẫu bằng nhau ở hai lớp theo seed cố định."""
    if n % 2:
        raise ValueError("Balanced binary subset size must be even")
    selected = []
    for label in (0, 1):
        pool = dataset.filter(lambda row: row["label"] == label)
        if len(pool) < n // 2:
            raise ValueError(f"Not enough examples for label {label}")
        selected.append(pool.shuffle(seed=seed).select(range(n // 2)))
    from datasets import concatenate_datasets
    result = concatenate_datasets(selected).shuffle(seed=seed)
    assert Counter(result["label"]) == {0: n // 2, 1: n // 2}
    return result


def prepare_splits(config):
    raw = load_dataset(config["dataset"])
    # Chia train/validation trước khi lấy mẫu để tránh dùng chung một dòng.
    split = raw["train"].train_test_split(test_size=0.1, seed=config["seed"], stratify_by_column="label")
    train = balanced_subset(split["train"], config["train_size"], config["seed"])
    validation = balanced_subset(split["test"], config["validation_size"], config["seed"])
    test = raw["test"].shuffle(seed=config["seed"]).select(range(config["test_size"]))
    # Kiểm tra văn bản của train và validation không trùng nhau.
    assert set(train["text"]).isdisjoint(validation["text"])
    assert len(train) == config["train_size"]
    assert len(validation) == config["validation_size"]
    assert len(test) == config["test_size"]
    return train, validation, test
