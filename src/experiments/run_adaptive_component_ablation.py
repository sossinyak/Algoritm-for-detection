"""Оценка отдельных этапов Adaptive PCA-CVA без добавления внешних комбинаций.

Скрипт показывает, что дает каждый структурный блок итогового алгоритма:
карта PCA-CVA, порог Оцу и последовательные операции очистки маски. Это дает проверяемую таблицу вклада этапов.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.metrics import calculate_metrics
from analysis.plot_style import apply_chart_style, dataset_label, save_chart
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from postprocessing.area_filter import filter_components
from utils.data_loader import LEVIRCDLoader
from utils.pipeline_config import build_adaptive_params


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_adaptive_params(config: dict, best_summary_csv: Path | None, dataset: str) -> dict:
    adaptive_params = build_adaptive_params(config)
    if best_summary_csv is None or not best_summary_csv.exists():
        return adaptive_params
    summary = pd.read_csv(best_summary_csv)
    rows = summary[(summary["dataset"] == dataset) & (summary["method"] == "Adaptive PCA-CVA")]
    if rows.empty:
        return adaptive_params
    params = json.loads(str(rows.sort_values(["f1", "precision"], ascending=False).iloc[0]["best_params"]))
    adaptive_params.update(params.get("adaptive_params", {}))
    return adaptive_params


def _stage_masks(method: AdaptiveChangeDetection, img_a, img_b) -> dict[str, object]:
    """Возвращает маски после отдельных этапов Adaptive PCA-CVA."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    absdiff = cv2.absdiff(gray_b, gray_a)
    _, absdiff_threshold = cv2.threshold(absdiff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    change_map = method.compute_change_map(img_a, img_b)
    threshold_mask = method.threshold_change_map(change_map)
    masks: dict[str, object] = {
        "01_AbsDiff_Otsu_baseline": absdiff_threshold,
        "02_PCA_CVA_Otsu": threshold_mask,
    }

    result = threshold_mask.copy()
    if method.median_kernel > 1:
        result = cv2.medianBlur(result, method.median_kernel)
    masks["03_Median"] = result.copy()

    if method.opening_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (method.opening_kernel, method.opening_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    masks["04_Opening"] = result.copy()

    if method.closing_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (method.closing_kernel, method.closing_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    masks["05_Closing"] = result.copy()

    result = filter_components(
        result,
        min_area=method.min_area,
        max_aspect_ratio=method.max_aspect_ratio,
        min_solidity=method.min_solidity,
        min_extent=method.min_extent,
    )
    masks["06_Area_shape_filter"] = result.copy()

    if method.fill_holes:
        result = method.postprocess_mask(threshold_mask)
    masks["07_Fill_holes_final"] = result.copy()
    return masks


def evaluate_components(config: dict, data_path: Path, split: str, max_samples: int | None, best_summary_csv: Path | None) -> pd.DataFrame:
    pairs = [pair for pair in LEVIRCDLoader(str(data_path)).load_split(split=split, max_pairs=max_samples) if pair.get("label") is not None]
    method = AdaptiveChangeDetection(**_load_adaptive_params(config, best_summary_csv, data_path.name))
    rows = []
    for pair in pairs:
        start = time.perf_counter()
        masks = _stage_masks(method, pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for stage, mask in masks.items():
            if stage.endswith("score"):
                continue
            metrics = calculate_metrics(mask, pair["label"])
            rows.append(
                {
                    "dataset": data_path.name,
                    "split": split,
                    "patch_name": pair["name"],
                    "stage": stage,
                    "time_ms_total_pipeline": elapsed_ms,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, group in frame.groupby("stage"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]) if "dataset" in group.columns and not group.empty else "",
                "stage": stage,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "samples": int(len(group)),
                "time_ms_total_pipeline": float(group["time_ms_total_pipeline"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)
    if not summary.empty:
        summary["delta_precision"] = summary["precision"].diff().fillna(summary["precision"])
        summary["delta_recall"] = summary["recall"].diff().fillna(summary["recall"])
        summary["delta_f1"] = summary["f1"].diff().fillna(summary["f1"])
    return summary




def _plot_component_contribution(summary: pd.DataFrame, output_path: Path) -> None:
    """Строит график вклада этапов: Precision, Recall и F1 после каждого компонента."""
    if summary.empty:
        return
    data = summary.copy().sort_values("stage")
    stage_labels = {
        "01_PCA_CVA_map": "Карта CVA\nс МГК",
        "02_otsu_threshold": "Порог\nОцу",
        "03_median": "Медианная\nфильтрация",
        "04_opening": "Размыкание",
        "05_closing": "Замыкание",
        "06_area_shape_filter": "Фильтрация\nкомпонент",
        "07_fill_holes_final": "Заполнение\nполостей",
    }
    labels = data["stage"].astype(str).map(lambda value: stage_labels.get(value, value))

    fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(11.5, 8.0), gridspec_kw={"height_ratios": [2.2, 1.0]})
    x = list(range(len(data)))
    width = 0.25
    ax.bar([value - width for value in x], data["precision"], width=width, label="Точность", color="#3b6ea8", edgecolor="white", linewidth=0.6)
    ax.bar(x, data["recall"], width=width, label="Полнота", color="#4f9a8b", edgecolor="white", linewidth=0.6)
    bars_f1 = ax.bar([value + width for value in x], data["f1"], width=width, label="F1", color="#d39b46", edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, min(1.05, max(1.0, float(data[["precision", "recall", "f1"]].max().max()) * 1.12)))
    ax.set_ylabel("Метрика, доли ед.")
    ax.set_xlabel("Этап адаптивного CVA с МГК")
    dataset = str(data["dataset"].iloc[0]) if "dataset" in data.columns and not data.empty else "dataset"
    ax.set_title(f"Вклад компонентов адаптивного CVA с МГК: {dataset_label(dataset)}", fontsize=15, fontweight="bold", loc="left")
    ax.legend(loc="best")

    for bar, value in zip(bars_f1, data["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)

    delta_values = data["delta_f1"] if "delta_f1" in data else data["f1"].diff().fillna(data["f1"])
    delta_colors = ["#4f9a8b" if value >= 0 else "#c76b5a" for value in delta_values]
    delta_bars = delta_ax.bar(x, delta_values, color=delta_colors, edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, delta_ax)
    delta_ax.axhline(0, color="#555555", linewidth=0.8)
    delta_ax.set_xticks(x)
    delta_ax.set_xticklabels(labels, rotation=20, ha="right")
    delta_ax.set_ylabel("Вклад в F1")
    for bar, value in zip(delta_bars, delta_values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.006 if value >= 0 else -0.006
        delta_ax.text(bar.get_x() + bar.get_width() / 2, float(value) + offset, f"{float(value):+.3f}", ha="center", va=va, fontsize=8)

    fig.tight_layout()
    save_chart(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценить вклад этапов Adaptive PCA-CVA.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--best-summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_component_ablation"))
    args = parser.parse_args()

    config = _load_config(args.config)
    data_path = args.data_path or Path(config.get("data", {}).get("data_path", "data/LEVIR-CD-filtred"))
    max_samples = None if args.max_samples is not None and args.max_samples <= 0 else args.max_samples
    rows = evaluate_components(config, data_path=data_path, split=args.split, max_samples=max_samples, best_summary_csv=args.best_summary_csv)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = args.output_dir / "component_stage_pair_metrics.csv"
    summary_csv = args.output_dir / "component_stage_summary.csv"
    plot_png = args.output_dir / "component_contribution_f1_precision_recall.png"
    rows.to_csv(rows_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    _plot_component_contribution(summary, plot_png)
    print(json.dumps({"rows_csv": str(rows_csv.resolve()), "summary_csv": str(summary_csv.resolve()), "plot_png": str(plot_png.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
