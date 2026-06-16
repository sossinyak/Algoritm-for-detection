"""
Расчет пиксельных метрик для бинарных масок изменений.

Во всех масках значение 0 означает "нет изменения", а любое значение больше
127 считается классом "изменение".
"""

from __future__ import annotations

import numpy as np


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Приводит маску к бинарному виду 0/1 независимо от исходного диапазона."""
    values = np.asarray(mask)
    if values.dtype == np.bool_:
        return values.astype(np.uint8)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.uint8)

    numeric = values.astype(np.float32, copy=False)
    if float(np.max(numeric)) <= 1.0:
        return (numeric > 0).astype(np.uint8)
    return (numeric > 127).astype(np.uint8)


def calculate_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """Считает TP, TN, FP, FN, Precision, Recall, F1 и Accuracy."""
    pred = binarize_mask(pred_mask)
    true = binarize_mask(true_mask)

    # Матрица ошибок считается попиксельно: каждый пиксель является отдельным объектом оценки.
    tp = int(np.sum((pred == 1) & (true == 1)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
