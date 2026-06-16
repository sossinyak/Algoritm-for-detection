"""Анализ типов ошибок для бинарных карт изменений.

Классификация эвристическая: она не заменяет экспертную разметку, но помогает
понять, где конвейер ошибается чаще всего.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import binarize_mask, calculate_metrics
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from pipelines.baseline_methods import build_classical_methods
from utils.data_loader import LEVIRCDLoader
from utils.pipeline_config import build_adaptive_params


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _local_texture(gray: np.ndarray) -> np.ndarray:
    gray32 = gray.astype(np.float32)
    mean = cv2.blur(gray32, (9, 9))
    sq_mean = cv2.blur(gray32 * gray32, (9, 9))
    return np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))


def classify_error_pixels(img_a: np.ndarray, img_b: np.ndarray, pred: np.ndarray, label: np.ndarray) -> pd.DataFrame:
    """Классифицирует FP/FN пиксели по простым признакам сцены."""
    pred_bin = binarize_mask(pred)
    label_bin = binarize_mask(label)
    fp = (pred_bin == 1) & (label_bin == 0)
    fn = (pred_bin == 0) & (label_bin == 1)

    gray_a = _gray(img_a).astype(np.float32)
    gray_b = _gray(img_b).astype(np.float32)
    brightness_delta = gray_b - gray_a
    abs_delta = np.abs(brightness_delta)
    texture = (_local_texture(gray_a) + _local_texture(gray_b)) / 2.0
    edges = cv2.Canny(_gray(img_b).astype(np.uint8), 50, 150) > 0

    masks = {
        "shadow_or_darkening_fp": fp & (brightness_delta < -20),
        "illumination_fp": fp & (abs_delta > 35) & (texture < 18),
        "texture_fp": fp & (texture >= 18),
        "edge_fp": fp & edges,
        "missed_low_contrast_fn": fn & (abs_delta <= 20),
        "missed_small_or_thin_fn": fn & edges,
        "other_fp": fp,
        "other_fn": fn,
    }
    used_fp = np.zeros(fp.shape, dtype=bool)
    used_fn = np.zeros(fn.shape, dtype=bool)
    rows = []
    for name, mask in masks.items():
        if name.endswith("_fp"):
            effective = mask & ~used_fp
            used_fp |= effective
            error_kind = "FP"
        elif name.endswith("_fn"):
            effective = mask & ~used_fn
            used_fn |= effective
            error_kind = "FN"
        else:
            effective = mask
            error_kind = "other"
        rows.append({"error_type": name, "error_kind": error_kind, "pixels": int(np.sum(effective))})
    return pd.DataFrame(rows)


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _build_method(method_name: str, config: dict) -> object:
    """Создает метод для анализа ошибок."""
    if method_name == "Adaptive PCA-CVA":
        return AdaptiveChangeDetection(**build_adaptive_params(config))
    methods = build_classical_methods("research")
    if method_name not in methods:
        available = list(methods) + ["Adaptive PCA-CVA"]
        raise KeyError(f"Метод не найден: {method_name}. Доступно: {', '.join(available)}")
    return methods[method_name]


def analyze_method_errors(
    dataset_path: Path,
    method_name: str,
    split: str = "test",
    max_samples: int | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запускает метод и собирает метрики ошибок по датасету."""
    method = _build_method(method_name, config or {})
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split, max_pairs=max_samples) if pair.get("label") is not None]
    rows = []
    metric_rows = []
    for pair in pairs:
        pred = method.process(pair["img_a"], pair["img_b"])
        metrics = calculate_metrics(pred, pair["label"])
        metric_rows.append({"dataset": dataset_path.name, "split": split, "method": method_name, "patch_name": pair["name"], **metrics})
        errors = classify_error_pixels(pair["img_a"], pair["img_b"], pred, pair["label"])
        errors.insert(0, "patch_name", pair["name"])
        errors.insert(0, "method", method_name)
        errors.insert(0, "dataset", dataset_path.name)
        rows.append(errors)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Эвристический анализ FP/FN ошибок.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--method", default="Adaptive PCA-CVA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/error_analysis"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    errors, metrics = analyze_method_errors(
        args.data_path,
        args.method,
        args.split,
        args.max_samples,
        config=_load_config(args.config),
    )
    error_csv = args.results_dir / f"{args.data_path.name}_{args.method}_errors.csv"
    metrics_csv = args.results_dir / f"{args.data_path.name}_{args.method}_metrics.csv"
    errors.to_csv(error_csv, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary = errors.groupby(["dataset", "method", "error_type"], as_index=False)["pixels"].sum() if not errors.empty else pd.DataFrame()
    summary_csv = args.results_dir / f"{args.data_path.name}_{args.method}_error_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"errors_csv": str(error_csv.resolve()), "metrics_csv": str(metrics_csv.resolve()), "summary_csv": str(summary_csv.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
