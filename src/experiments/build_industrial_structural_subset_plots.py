"""Графики для промышленно-структурных подмножеств датасетов.

Подмножество берется из strict feature-based отбора run_selected_pairs_thresholds.py.
Для LEVIR-CD-filtred и JL1-CD используется только строгий preset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from analysis.plot_style import BAR_PALETTE, apply_chart_style, dataset_label, method_label, save_chart
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from pipelines.method_metadata import COMPARISON_METHODS
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


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _load_subset(selection_csv: Path, max_pairs: int | None) -> pd.DataFrame:
    selected = pd.read_csv(selection_csv)
    if "selection_score" in selected.columns:
        selected = selected.sort_values(["selection_score", "patch_name"], ascending=[False, True])
    else:
        selected = selected.sort_values("patch_name")
    if max_pairs is not None:
        selected = selected.head(max_pairs)
    return selected.reset_index(drop=True)


def _aggregate_method_metrics(pair_metrics_csv: Path, dataset: str, selected_names: set[str]) -> pd.DataFrame:
    rows = pd.read_csv(pair_metrics_csv)
    rows = rows[(rows["dataset"] == dataset) & (rows["patch_name"].astype(str).isin(selected_names))]
    rows = rows[rows["method"].isin(COMPARISON_METHODS)]
    rows = rows.copy()
    summary_rows = []
    for method, group in rows.groupby("method"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "samples": int(group["patch_name"].nunique()),
                "time_ms": float(group["time_ms"].mean()) if "time_ms" in group else 0.0,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("f1", ascending=False).reset_index(drop=True)


def _plot_methods(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    data = summary.sort_values("f1", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(8.8, 1.08 * len(data)), 5.6))
    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(data))]
    x = range(len(data))
    bars = ax.bar(x, data["f1"], color=colors, edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels([method_label(method) for method in data["method"]], rotation=0, ha="center")
    ax.set_ylabel("F1-мера, доли ед.")
    ax.set_title(f"{dataset_label(dataset)}: выбранное промышленное подмножество, {int(data['samples'].max())} пар", fontsize=15, fontweight="bold", loc="left")
    ax.set_ylim(0, min(1.05, max(1.0, float(data["f1"].max()) * 1.15)))
    for bar, value in zip(bars, data["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_chart(fig, output_path)


def _stage_masks(method: AdaptiveChangeDetection, img_a, img_b) -> dict[str, object]:
    import cv2

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


def _component_metrics(config: dict, dataset_path: Path, selected_names: set[str], split: str, best_summary_csv: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split=split) if pair.get("label") is not None]
    pairs = [pair for pair in pairs if str(pair["name"]) in selected_names]
    method = AdaptiveChangeDetection(**_load_adaptive_params(config, best_summary_csv, dataset_path.name))
    rows = []
    for pair in pairs:
        start = time.perf_counter()
        masks = _stage_masks(method, pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for stage, mask in masks.items():
            metrics = calculate_metrics(mask, pair["label"])
            rows.append({"dataset": dataset_path.name, "split": split, "patch_name": pair["name"], "stage": stage, "time_ms_total_pipeline": elapsed_ms, **metrics})
    pair_rows = pd.DataFrame(rows)
    summary_rows = []
    for stage, group in pair_rows.groupby("stage"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        summary_rows.append({"dataset": dataset_path.name, "stage": stage, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "samples": int(group["patch_name"].nunique())})
    summary = pd.DataFrame(summary_rows).sort_values("stage").reset_index(drop=True)
    if not summary.empty:
        summary["delta_precision"] = summary["precision"].diff().fillna(summary["precision"])
        summary["delta_recall"] = summary["recall"].diff().fillna(summary["recall"])
        summary["delta_f1"] = summary["f1"].diff().fillna(summary["f1"])
    return pair_rows, summary


def _plot_components(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    data = summary.sort_values("stage").reset_index(drop=True)
    stage_labels = {
        "01_AbsDiff_Otsu_baseline": "Абс. разность\n+ порог Оцу",
        "02_PCA_CVA_Otsu": "CVA с МГК\n+ порог Оцу",
        "03_Median": "Медианная\nфильтрация",
        "04_Opening": "Размыкание",
        "05_Closing": "Замыкание",
        "06_Area_shape_filter": "Фильтрация\nкомпонент",
        "07_Fill_holes_final": "Заполнение\nполостей",
    }
    labels = data["stage"].astype(str).map(lambda value: stage_labels.get(value, value))
    fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(11.5, 8.0), gridspec_kw={"height_ratios": [2.2, 1.0]})
    x = list(range(len(data)))
    width = 0.25
    bars_precision = ax.bar([value - width for value in x], data["precision"], width=width, label="Точность", color="#3b6ea8", edgecolor="white", linewidth=0.6)
    bars_recall = ax.bar(x, data["recall"], width=width, label="Полнота", color="#4f9a8b", edgecolor="white", linewidth=0.6)
    bars_f1 = ax.bar([value + width for value in x], data["f1"], width=width, label="F1", color="#d39b46", edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, min(1.05, max(1.0, float(data[["precision", "recall", "f1"]].max().max()) * 1.12)))
    ax.set_ylabel("Метрика, доли ед.")
    ax.set_xlabel("")
    ax.set_title(f"Вклад этапов адаптивного CVA с МГК: {dataset_label(dataset)}", fontsize=15, fontweight="bold", loc="left")
    ax.legend(loc="best")
    for bar, value in zip(bars_f1, data["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)

    delta_values = data["delta_f1"] if "delta_f1" in data else data["f1"].diff().fillna(data["f1"])
    delta_colors = ["#4f9a8b" if value >= 0 else "#c76b5a" for value in delta_values]
    delta_bars = delta_ax.bar(x, delta_values, color=delta_colors, edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, delta_ax)
    delta_ax.axhline(0, color="#555555", linewidth=0.8)
    delta_ax.set_xticks(x)
    delta_ax.set_xticklabels(labels, rotation=18, ha="right")
    delta_ax.set_ylabel("Вклад в F1")
    for bar, value in zip(delta_bars, delta_values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.006 if value >= 0 else -0.006
        delta_ax.text(bar.get_x() + bar.get_width() / 2, float(value) + offset, f"{float(value):+.3f}", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    save_chart(fig, output_path)


def build_for_dataset(dataset: str, selection_csv: Path, max_pairs: int | None, args: argparse.Namespace) -> dict[str, str]:
    out_dir = args.output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = _load_subset(selection_csv, max_pairs=max_pairs)
    subset.to_csv(out_dir / "industrial_structural_subset.csv", index=False, encoding="utf-8-sig")
    selected_names = set(subset["patch_name"].astype(str))

    method_summary = _aggregate_method_metrics(args.pair_metrics_csv, dataset, selected_names)
    method_summary.to_csv(out_dir / "method_f1_summary.csv", index=False, encoding="utf-8-sig")
    method_plot = out_dir / "method_f1_industrial_structural_subset.png"
    _plot_methods(method_summary, dataset, method_plot)

    config = _load_config(args.config)
    best_summary_csv = args.pair_metrics_csv.parent / "full_test_best_params_summary.csv"
    pair_rows, component_summary = _component_metrics(config, args.data_root / dataset, selected_names, args.split, best_summary_csv)
    pair_rows.to_csv(out_dir / "component_stage_pair_metrics.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(out_dir / "component_stage_summary.csv", index=False, encoding="utf-8-sig")
    component_plot = out_dir / "component_contribution_industrial_structural_subset.png"
    _plot_components(component_summary, dataset, component_plot)

    return {
        "dataset": dataset,
        "pairs": str(len(subset)),
        "subset_csv": str((out_dir / "industrial_structural_subset.csv").resolve()),
        "method_summary_csv": str((out_dir / "method_f1_summary.csv").resolve()),
        "method_plot": str(method_plot.resolve()),
        "component_summary_csv": str((out_dir / "component_stage_summary.csv").resolve()),
        "component_plot": str(component_plot.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить графики для строгих промышленно-структурных подмножеств.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--pair-metrics-csv", type=Path, default=Path("results/full_protocol/full_test_best_params_pair_metrics.csv"))
    parser.add_argument("--selection-root", type=Path, default=Path("results/industrial_structural_subsets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/industrial_structural_subset_plots"))
    args = parser.parse_args()

    plan = {
        "LEVIR-CD-filtred": None,
        "JL1-CD": None,
    }
    outputs = []
    for dataset, max_pairs in plan.items():
        selection_csv = args.selection_root / dataset / "selected_pairs.csv"
        if not selection_csv.exists():
            raise FileNotFoundError(selection_csv)
        outputs.append(build_for_dataset(dataset, selection_csv, max_pairs, args))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
