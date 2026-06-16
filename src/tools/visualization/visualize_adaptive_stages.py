"""Визуализация промежуточных этапов адаптивного CVA с МГК."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import calculate_metrics
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Переводит OpenCV BGR в RGB для matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _error_map(pred_mask: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """Строит карту ошибок: TP - белый, FP - красный, FN - зеленый."""
    pred = pred_mask > 127
    true = true_mask > 127
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[pred & true] = (255, 255, 255)
    rgb[pred & ~true] = (255, 0, 0)
    rgb[~pred & true] = (0, 255, 0)
    return rgb


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Накладывает бинарную маску на изображение для лучшей интерпретации результата."""
    image = _to_rgb(image_bgr).astype(np.float32)
    overlay = image.copy()
    active = mask > 127
    overlay[active] = 0.55 * image[active] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _score_pairs(pairs: list[dict], method: AdaptiveChangeDetection) -> list[dict]:
    """Считает F1 адаптивного метода для всех доступных патчей."""
    rows = []
    for pair in pairs:
        pred_mask = method.process(pair["img_a"], pair["img_b"])
        metrics = calculate_metrics(pred_mask, pair["label"])
        rows.append(
            {
                "pair": pair,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "gt_positive_fraction": float(np.mean(pair["label"] > 127)),
            }
        )
    return rows


def _representative_indices(length: int, count: int) -> list[int]:
    """Возвращает индексы, равномерно покрывающие отсортированный список патчей."""
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


def _select_pairs(scored_pairs: list[dict], selection: str, count: int) -> list[dict]:
    """Выбирает несколько патчей для визуализации."""
    changed_pairs = [row for row in scored_pairs if row["gt_positive_fraction"] > 0]
    source = changed_pairs if changed_pairs else scored_pairs
    ordered = sorted(source, key=lambda row: row["f1"])
    if not ordered:
        return []

    count = max(1, min(count, len(ordered)))
    if selection == "best":
        return list(reversed(ordered[-count:]))
    if selection == "worst":
        return ordered[:count]
    if selection == "median":
        center = len(ordered) // 2
        start = max(0, center - count // 2)
        end = min(len(ordered), start + count)
        start = max(0, end - count)
        return ordered[start:end]

    return [ordered[index] for index in _representative_indices(len(ordered), count)]


def _stage_rows(intermediate: dict, label: np.ndarray) -> list[dict]:
    """Считает метрики для масок после каждого этапа."""
    stage_names = [
        ("threshold_mask", "Пороговая маска"),
        ("median_mask", "Медианная фильтрация"),
        ("opening_mask", "Размыкание"),
        ("closing_mask", "Замыкание"),
        ("area_mask", "Фильтрация компонент"),
        ("filled_mask", "Заполнение полостей"),
        ("final_mask", "Итоговая маска"),
    ]

    rows = []
    for key, title in stage_names:
        mask = intermediate.get(key)
        if mask is None:
            continue
        # Этот CSV нужен не только для иллюстрации, но и для быстрой проверки,
        # на каком этапе растут FP/FN после порогования.
        metrics = calculate_metrics(mask, label)
        rows.append(
            {
                "stage": title,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pred_positive_fraction": float(np.mean(mask > 127)),
            }
        )
    return rows


def _save_figure(
    pair: dict,
    intermediate: dict,
    output_path: Path,
    final_metrics: dict,
) -> None:
    """Сохраняет картинку с промежуточными результатами одного патча."""
    img_a = pair["img_a"]
    img_b = pair["img_b"]
    label = pair["label"]

    panels = [
        ("T1", _to_rgb(img_a), "rgb"),
        ("T2", _to_rgb(img_b), "rgb"),
        ("Эталонная маска", _overlay_mask(img_b, label, (0, 255, 0)), "rgb"),
        ("Карта изменений", intermediate["change_map"], "viridis"),
        ("Пороговая маска", intermediate["threshold_mask"], "gray"),
        ("Медианная фильтрация", intermediate["median_mask"], "gray"),
        ("Размыкание", intermediate["opening_mask"], "gray"),
        ("Замыкание", intermediate["closing_mask"], "gray"),
        ("Фильтрация компонент", intermediate["area_mask"], "gray"),
        ("Заполнение полостей", intermediate["filled_mask"], "gray"),
        ("Итоговое наложение", _overlay_mask(img_b, intermediate["final_mask"], (255, 0, 0)), "rgb"),
        ("Ошибки", _error_map(intermediate["final_mask"], label), "rgb"),
    ]

    plt.figure(figsize=(16, 12))
    for index, (title, image, mode) in enumerate(panels, start=1):
        ax = plt.subplot(3, 4, index)
        if mode == "rgb":
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=mode, vmin=0, vmax=255 if mode == "gray" else None)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(
        (
            f"{pair['name']} | "
            f"F1={final_metrics['f1']:.3f} "
            f"точность={final_metrics['precision']:.3f} "
            f"полнота={final_metrics['recall']:.3f}"
        ),
        fontsize=14,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Точка входа для построения поэтапной визуализации."""
    parser = argparse.ArgumentParser(description="Визуализировать этапы адаптивного CVA с МГК")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--selection",
        type=str,
        choices=["representative", "best", "median", "worst"],
        default="representative",
    )
    parser.add_argument("--sample-name", nargs="+", default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage_visualization"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.data_path is not None:
        config.setdefault("data", {})["data_path"] = args.data_path
    pairs = load_configured_pairs(config, split=args.split, max_pairs=args.max_pairs)
    pairs = [pair for pair in pairs if pair.get("label") is not None]

    method = AdaptiveChangeDetection(**build_adaptive_params(config))

    if args.sample_name is not None:
        selected = []
        pair_map = {pair["name"]: pair for pair in pairs}
        for sample_name in args.sample_name:
            pair = pair_map.get(sample_name)
            if pair is None:
                raise ValueError(f"Sample {sample_name} not found in split {args.split}")
            selected.append(
                {
                    "pair": pair,
                    "f1": None,
                    "precision": None,
                    "recall": None,
                }
            )
    else:
        scored_pairs = _score_pairs(pairs, method)
        selected = _select_pairs(scored_pairs, args.selection, args.count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []

    for row in selected:
        pair = row["pair"]
        final_mask = method.process(pair["img_a"], pair["img_b"])
        intermediate = method.get_intermediate_results()
        intermediate["final_mask"] = final_mask

        final_metrics = calculate_metrics(final_mask, pair["label"])
        metrics_rows = _stage_rows(intermediate, pair["label"])
        metrics_path = args.output_dir / f"{pair['name']}_stage_metrics.csv"
        image_path = args.output_dir / f"{pair['name']}_stages.png"

        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        _save_figure(pair, intermediate, image_path, final_metrics)

        selection_rows.append(
            {
                "patch_name": pair["name"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1": final_metrics["f1"],
                "metrics_csv": str(metrics_path.resolve()),
                "image_path": str(image_path.resolve()),
            }
        )

    selection_csv = args.output_dir / "selected_samples.csv"
    pd.DataFrame(selection_rows).to_csv(selection_csv, index=False, encoding="utf-8-sig")

    print(selection_csv)
    for row in selection_rows:
        print(row["image_path"])


if __name__ == "__main__":
    main()
