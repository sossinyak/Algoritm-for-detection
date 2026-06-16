"""Построение сравнительных графиков F1-меры по методам."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plot_style import BAR_PALETTE, apply_chart_style, dataset_label, method_label, save_chart
from pipelines.method_metadata import COMPARISON_METHODS


def _prepare_comparison_methods(summary: pd.DataFrame) -> pd.DataFrame:
    """Оставляет единый финальный набор методов сравнения."""
    filtered = summary[summary["method"].isin(COMPARISON_METHODS)].copy()
    filtered["method"] = pd.Categorical(filtered["method"], categories=COMPARISON_METHODS, ordered=True)
    return filtered.sort_values(["dataset", "method"]).reset_index(drop=True)


def _plot_grouped_bars(summary: pd.DataFrame, methods: list[str], datasets: list[str], title: str, output_path: Path) -> None:
    """Строит grouped bar chart для выбранных методов и датасетов."""
    plot_df = summary[summary["dataset"].isin(datasets)].pivot(index="method", columns="dataset", values="f1")
    plot_df = plot_df.reindex(methods).dropna(how="all")
    order = plot_df.mean(axis=1).sort_values(ascending=True).index
    plot_df = plot_df.loc[order]
    if plot_df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 0.72), 6.8))
    x = np.arange(len(plot_df.index))
    width = 0.72 / max(len(datasets), 1)

    for index, dataset in enumerate(datasets):
        if dataset not in plot_df.columns:
            continue
        values = plot_df[dataset].values
        offset = (index - (len(datasets) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=dataset_label(dataset),
            color=BAR_PALETTE[index % len(BAR_PALETTE)],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    apply_chart_style(fig, ax)
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
    ax.set_ylabel("F1-мера, доли ед.")
    ax.set_xlabel("")
    max_value = float(np.nanmax(plot_df.values)) if np.isfinite(plot_df.values).any() else 1.0
    ax.set_ylim(0, min(1.08, max(1.0, max_value * 1.16)))
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(method) for method in plot_df.index], rotation=0, ha="center")
    ax.legend(title="Датасет")
    fig.tight_layout()
    save_chart(fig, output_path, dpi=220)


def _plot_dataset_ranking(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    """Строит отдельный рейтинг всех методов для одного датасета."""
    dataset_df = summary[(summary["dataset"] == dataset) & (summary["method"].isin(COMPARISON_METHODS))].sort_values("f1", ascending=True).copy()
    if dataset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.8, 1.08 * len(dataset_df)), 5.6))
    colors = [BAR_PALETTE[idx % len(BAR_PALETTE)] for idx in range(len(dataset_df))]
    x = np.arange(len(dataset_df))
    bars = ax.bar(x, dataset_df["f1"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, min(1.05, max(1.0, float(dataset_df["f1"].max()) * 1.15)))
    apply_chart_style(fig, ax)
    ax.set_ylabel("F1-мера, доли ед.")
    ax.set_title(f"{dataset_label(dataset)}: сравнение выбранных методов", fontsize=15, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(method) for method in dataset_df["method"]], rotation=0, ha="center")
    for bar, value in zip(bars, dataset_df["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.01, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_chart(fig, output_path, dpi=220)


def build_f1_plots(summary_csv: Path, output_dir: Path) -> dict[str, str]:
    """Создает графики F1 и возвращает пути к артефактам."""
    summary = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = _prepare_comparison_methods(summary)
    comparison_summary_csv = output_dir / "f1_comparison_methods_summary.csv"
    comparison.to_csv(comparison_summary_csv, index=False, encoding="utf-8-sig")

    artifacts: dict[str, str] = {"comparison_summary_csv": str(comparison_summary_csv.resolve())}
    synthetic_path = output_dir / "f1_comparison_methods_synthetic.png"
    real_path = output_dir / "f1_comparison_methods_real_datasets.png"
    _plot_grouped_bars(
        comparison,
        COMPARISON_METHODS,
        ["synthetic-lab"],
        "Синтетический набор: сравнение выбранных методов",
        synthetic_path,
    )
    _plot_grouped_bars(
        comparison,
        COMPARISON_METHODS,
        ["JL1-CD", "LEVIR-CD-filtred"],
        "Реальные датасеты: сравнение выбранных методов",
        real_path,
    )
    artifacts["comparison_synthetic_png"] = str(synthetic_path.resolve())
    artifacts["comparison_real_png"] = str(real_path.resolve())

    for dataset in sorted(summary["dataset"].unique()):
        dataset_path = output_dir / f"f1_ranking_{dataset}.png"
        _plot_dataset_ranking(summary, dataset, dataset_path)
        artifacts[f"ranking_{dataset}"] = str(dataset_path.resolve())

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить графики F1-меры по методам.")
    parser.add_argument("--summary-csv", type=Path, default=Path("results/parameter_study/all_parameter_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_plots"))
    args = parser.parse_args()

    print(json.dumps(build_f1_plots(args.summary_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
