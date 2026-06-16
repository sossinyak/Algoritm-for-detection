"""Запуск сравнения базовых методов и итогового адаптивного алгоритма."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.metrics import calculate_metrics
from analysis.plot_style import BAR_PALETTE, apply_chart_style, method_label, save_chart
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from pipelines.baseline_methods import build_classical_methods
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Переводит OpenCV BGR в RGB для matplotlib."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Накладывает маску на снимок, чтобы сравнение методов было видно сразу на сцене."""
    image = _to_rgb(image_bgr).astype(np.float32)
    overlay = image.copy()
    active = mask > 127
    overlay[active] = 0.55 * image[active] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _empty_result_bucket() -> dict:
    """Создает накопитель метрик для одного метода."""
    return {
        "f1": [],
        "time_ms": [],
        "pred_positive_fraction": [],
        "gt_positive_fraction": [],
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def _summarize(values: dict) -> dict:
    """Собирает итоговые метрики по суммарной матрице ошибок."""
    tp = values["tp"]
    tn = values["tn"]
    fp = values["fp"]
    fn = values["fn"]

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "f1_std": float(np.std(values["f1"])) if values["f1"] else 0.0,
        "pred_positive_fraction": (
            float(np.mean(values["pred_positive_fraction"])) if values["pred_positive_fraction"] else 0.0
        ),
        "gt_positive_fraction": (
            float(np.mean(values["gt_positive_fraction"])) if values["gt_positive_fraction"] else 0.0
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "time_ms": float(np.mean(values["time_ms"])) if values["time_ms"] else 0.0,
    }


def _representative_indices(length: int, count: int) -> list[int]:
    """Возвращает индексы, равномерно покрывающие отсортированный набор патчей."""
    if length <= 0:
        return []

    count = max(1, min(count, length))
    if count == 1:
        return [length // 2]

    raw_indices = np.linspace(0, length - 1, num=count)
    indices: list[int] = []
    for value in raw_indices:
        index = int(round(float(value)))
        if index not in indices:
            indices.append(index)

    for index in range(length):
        if len(indices) >= count:
            break
        if index not in indices:
            indices.append(index)

    return sorted(indices)


def _select_visualization_pairs(pair_rows: pd.DataFrame, count: int) -> list[str]:
    """Выбирает несколько репрезентативных патчей по F1 адаптивного метода."""
    changed_df = pair_rows[pair_rows["gt_positive_fraction"] > 0].copy()
    source_df = changed_df if not changed_df.empty else pair_rows.copy()
    if source_df.empty:
        return []

    ordered = source_df.sort_values("adaptive_f1").reset_index(drop=True)
    return [str(ordered.iloc[index]["patch_name"]) for index in _representative_indices(len(ordered), count)]


def _save_comparison_visual(
    pair: dict,
    predictions: dict[str, np.ndarray],
    metrics_by_method: dict[str, dict],
    output_path: Path,
) -> None:
    """Сохраняет одну страницу сравнения всех методов на выбранном патче."""
    panels = [
        ("T1", _to_rgb(pair["img_a"]), "rgb"),
        ("T2", _to_rgb(pair["img_b"]), "rgb"),
        ("Эталонная маска", _overlay_mask(pair["img_b"], pair["label"], (0, 255, 0)), "rgb"),
    ]

    for method_name, prediction in predictions.items():
        metrics = metrics_by_method[method_name]
        panels.append(
            (
                f"{method_label(method_name).replace(chr(10), ' ')}\nF1={metrics['f1']:.3f} точность={metrics['precision']:.3f} полнота={metrics['recall']:.3f}",
                _overlay_mask(pair["img_b"], prediction, (255, 0, 0)),
                "rgb",
            )
        )

    cols = 4
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, image, mode) in zip(axes, panels):
        if mode == "rgb":
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=mode, vmin=0, vmax=255 if mode == "gray" else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(pair["name"], fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_visuals(
    labeled_pairs: list[dict],
    methods: dict[str, object],
    pair_rows: pd.DataFrame,
    results_dir: Path,
    count: int,
) -> dict[str, object]:
    """Строит сравнение методов для нескольких репрезентативных патчей."""
    selected_names = _select_visualization_pairs(pair_rows, count=count)
    if not selected_names:
        return {"selected_patches": [], "output_dir": None}

    pair_map = {pair["name"]: pair for pair in labeled_pairs}
    visuals_dir = results_dir / "comparison_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    saved_images = []
    for patch_name in selected_names:
        pair = pair_map.get(patch_name)
        if pair is None:
            continue

        predictions: dict[str, np.ndarray] = {}
        metrics_by_method: dict[str, dict] = {}
        for method_name, method in methods.items():
            prediction = method.process(pair["img_a"], pair["img_b"])
            predictions[method_name] = prediction
            metrics_by_method[method_name] = calculate_metrics(prediction, pair["label"])

        output_path = visuals_dir / f"{Path(patch_name).stem}_comparison.png"
        _save_comparison_visual(pair, predictions, metrics_by_method, output_path)
        saved_images.append(str(output_path.resolve()))

        adaptive_metrics = metrics_by_method["Adaptive PCA-CVA"]
        selection_rows.append(
            {
                "patch_name": patch_name,
                "adaptive_precision": adaptive_metrics["precision"],
                "adaptive_recall": adaptive_metrics["recall"],
                "adaptive_f1": adaptive_metrics["f1"],
                "image_path": str(output_path.resolve()),
            }
        )

    selection_csv = visuals_dir / "selected_samples.csv"
    pd.DataFrame(selection_rows).to_csv(selection_csv, index=False, encoding="utf-8-sig")
    return {
        "selected_patches": selected_names,
        "output_dir": str(visuals_dir.resolve()),
        "selection_csv": str(selection_csv.resolve()),
        "images": saved_images,
    }


def _save_f1_plot(summary: dict[str, dict], output_path: Path) -> None:
    """Сохраняет столбчатый график F1 для всех сравниваемых методов."""
    method_names = sorted(summary.keys(), key=lambda name: float(summary[name]["f1"]))
    f1_scores = [float(summary[name]["f1"]) for name in method_names]

    fig_width = max(8.8, min(14, 1.08 * len(method_names)))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    colors = [BAR_PALETTE[index % len(BAR_PALETTE)] for index, _ in enumerate(method_names)]
    x = list(range(len(method_names)))
    bars = ax.bar(x, f1_scores, color=colors, edgecolor="white", linewidth=0.7)

    apply_chart_style(fig, ax)
    ax.set_title("Сравнение выбранных методов", fontsize=15, fontweight="bold", loc="left")
    ax.set_ylabel("F1-мера, доли ед.")
    max_f1 = max(f1_scores) if f1_scores else 0.0
    ax.set_ylim(0, max(1.0, max_f1 * 1.18))
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(method) for method in method_names], rotation=0, ha="center")

    for bar, value in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    save_chart(fig, output_path)


def run_comparison_experiment(config: dict) -> dict:
    """Обрабатывает тестовую выборку, сохраняет CSV и визуализации сравнения методов."""
    experiment_cfg = config.get("experiments", {})
    split = experiment_cfg.get("split", "test")
    max_samples = experiment_cfg.get("max_samples")
    save_visualizations = bool(experiment_cfg.get("save_visualizations", True))
    visualization_samples = int(experiment_cfg.get("visualization_samples", 3))
    results_dir = Path(experiment_cfg.get("results_dir", "results"))

    pairs = load_configured_pairs(config, split=split, max_pairs=max_samples)
    labeled_pairs = [pair for pair in pairs if pair.get("label") is not None]
    if not pairs:
        data_path = config.get("data", {}).get("data_path", "./data/LEVIR-CD-filtred")
        raise RuntimeError(f"Не найдены пары изображений в {Path(data_path) / split}")
    if not labeled_pairs:
        data_path = config.get("data", {}).get("data_path", "./data/LEVIR-CD-filtred")
        raise RuntimeError(f"Не найдены размеченные пары в {Path(data_path) / split / 'label'}")

    # Сравниваем классические базовые подходы с основным адаптивным методом.
    method_set = str(experiment_cfg.get("method_set", "individual"))
    methods = dict(build_classical_methods(method_set=method_set))
    methods["Adaptive PCA-CVA"] = AdaptiveChangeDetection(**build_adaptive_params(config))
    results = {name: _empty_result_bucket() for name in methods}
    pair_rows: list[dict] = []

    for pair in labeled_pairs:
        img_a = pair["img_a"]
        img_b = pair["img_b"]
        label = pair["label"]
        pair_row = {
            "patch_name": pair["name"],
            "gt_positive_fraction": float(np.mean(label > 127)),
        }

        for name, method in methods.items():
            start_time = time.perf_counter()
            pred_mask = method.process(img_a, img_b)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            metrics = calculate_metrics(pred_mask, label)
            results[name]["f1"].append(metrics["f1"])
            for count_name in ("tp", "tn", "fp", "fn"):
                results[name][count_name] += int(metrics[count_name])

            # Эти доли нужны для отчета: они сразу показывают проблему низкой Precision.
            results[name]["pred_positive_fraction"].append(float(np.mean(pred_mask > 127)))
            results[name]["gt_positive_fraction"].append(float(np.mean(label > 127)))
            results[name]["time_ms"].append(elapsed_ms)

            if name == "Adaptive PCA-CVA":
                pair_row["adaptive_precision"] = metrics["precision"]
                pair_row["adaptive_recall"] = metrics["recall"]
                pair_row["adaptive_f1"] = metrics["f1"]

        pair_rows.append(pair_row)

    summary = {name: _summarize(values) for name, values in results.items()}

    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary).T
    df.index.name = "method"
    comparison_csv = results_dir / "comparison_results.csv"
    df.to_csv(comparison_csv, encoding="utf-8-sig")

    quality_rows = [pair.get("quality_report", {}) for pair in labeled_pairs if pair.get("quality_report")]
    quality_report_csv = None
    if quality_rows:
        quality_report_csv = results_dir / "input_quality_report.csv"
        pd.DataFrame(quality_rows).to_csv(quality_report_csv, index=False, encoding="utf-8-sig")

    comparison_plot = results_dir / "comparison_f1.png"
    _save_f1_plot(summary, comparison_plot)

    artifacts = {
        "comparison_results_csv": str(comparison_csv.resolve()),
        "comparison_f1_plot": str(comparison_plot.resolve()),
    }
    if quality_report_csv is not None:
        artifacts["input_quality_report_csv"] = str(quality_report_csv.resolve())
    if save_visualizations:
        visual_methods = methods
        if len(methods) > 16:
            visual_methods = {
                name: methods[name]
                for name in list(methods.keys())[:12]
                + (["Adaptive PCA-CVA"] if "Adaptive PCA-CVA" in methods else [])
            }
        visuals = _save_comparison_visuals(
            labeled_pairs=labeled_pairs,
            methods=visual_methods,
            pair_rows=pd.DataFrame(pair_rows),
            results_dir=results_dir,
            count=visualization_samples,
        )
        artifacts["visuals"] = visuals

    summary["artifacts"] = artifacts
    return summary
