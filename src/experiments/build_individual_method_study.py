"""Построение таблиц и графиков чувствительности для итогового набора методов."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.method_metadata import INDIVIDUAL_METHODS, METHOD_THEORY
from analysis.plot_style import BAR_PALETTE, apply_chart_style, dataset_label, method_label, save_chart


PARAMETERS = [
    "threshold",
    "threshold_scale",
    "sigma",
    "median_kernel",
    "morph_kernel",
    "min_area",
    "adaptive_block_size",
    "adaptive_c",
]

PARAMETER_LABELS = {
    "threshold": "тип порога",
    "threshold_scale": "масштаб порога, доли ед.",
    "sigma": "sigma сглаживания, пикс.",
    "median_kernel": "размер медианного ядра, пикс.",
    "morph_kernel": "размер морфологического ядра, пикс.",
    "min_area": "минимальная площадь компоненты, пикс.",
    "adaptive_block_size": "размер блока локального порога, пикс.",
    "adaptive_c": "смещение локального порога, уровни яркости",
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _line_plot(frame: pd.DataFrame, x_col: str, y_col: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    data = frame.groupby(x_col, dropna=False)[y_col].mean().reset_index()
    data = data.assign(_sort_key=data[x_col].astype(str)).sort_values("_sort_key").drop(columns="_sort_key")
    labels = data[x_col].astype(str)
    bars = ax.bar(labels, data[y_col], color=BAR_PALETTE[0], edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_title(title)
    ax.set_xlabel(PARAMETER_LABELS.get(x_col, f"{x_col}, значение"))
    ax.set_ylabel("F1 на валидации, доли ед.")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, data[y_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.006, f"{float(value):.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_chart(fig, out)


def _bar_plot(frame: pd.DataFrame, dataset: str, out: Path) -> None:
    data = frame[frame["dataset"] == dataset].sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(max(8.8, 1.08 * len(data)), 5.6))
    colors = [BAR_PALETTE[index % len(BAR_PALETTE)] for index in range(len(data))]
    x = list(range(len(data)))
    bars = ax.bar(x, data["f1"], color=colors, edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(method) for method in data["method"]], rotation=0, ha="center")
    ax.set_ylabel("F1 на test, доли ед.")
    ax.set_title(f"Отдельные алгоритмы на {dataset_label(dataset)}")
    for bar, value in zip(bars, data["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.008, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_chart(fig, out)


def build(
    full_protocol_dir: Path = PROJECT_ROOT / "results" / "full_protocol",
    output_dir: Path = PROJECT_ROOT / "results" / "individual_methods",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(full_protocol_dir / "full_test_best_params_summary.csv")
    trials = pd.read_csv(full_protocol_dir / "parameter_study" / "all_parameter_trials.csv")
    pairs = pd.read_csv(full_protocol_dir / "full_test_best_params_pair_metrics.csv")

    summary = summary[summary["method"].isin(INDIVIDUAL_METHODS)].copy()
    trials = trials[trials["method"].isin(INDIVIDUAL_METHODS)].copy()
    pairs = pairs[pairs["method"].isin(INDIVIDUAL_METHODS)].copy()

    summary.to_csv(output_dir / "individual_summary.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(output_dir / "individual_parameter_trials.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output_dir / "individual_pair_metrics.csv", index=False, encoding="utf-8-sig")

    theory_rows = []
    for method in INDIVIDUAL_METHODS:
        info = METHOD_THEORY[method]
        theory_rows.append({"method": method, **info})
    pd.DataFrame(theory_rows).to_csv(output_dir / "individual_method_theory.csv", index=False, encoding="utf-8-sig")

    best_params = (
        summary[["dataset", "method", "best_params", "precision", "recall", "f1", "time_ms"]]
        .sort_values(["dataset", "f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best_params.to_csv(output_dir / "individual_best_params.csv", index=False, encoding="utf-8-sig")

    for dataset in sorted(summary["dataset"].unique()):
        _bar_plot(summary, dataset, plots_dir / f"{_safe_name(dataset)}_individual_f1.png")

    sensitivity_rows = []
    for (dataset, method), group in trials.groupby(["dataset", "method"]):
        for param in PARAMETERS:
            if param not in group.columns:
                continue
            values = group[param].dropna().astype(str)
            if values.nunique() < 2:
                continue
            out = plots_dir / _safe_name(dataset) / _safe_name(method) / f"f1_by_{param}.png"
            label = PARAMETER_LABELS.get(param, param)
            method_title = method_label(method).replace("\n", " ")
            _line_plot(group, param, "tune_f1", f"{method_title}: влияние параметра {label} ({dataset_label(dataset)})", out)
            best = group.sort_values("tune_f1", ascending=False).iloc[0]
            sensitivity_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "parameter": param,
                    "best_value": best.get(param),
                    "best_validation_f1": best.get("tune_f1"),
                    "plot": str(out.resolve()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "individual_parameter_sensitivity.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "summary": str((output_dir / "individual_summary.csv").resolve()),
        "trials": str((output_dir / "individual_parameter_trials.csv").resolve()),
        "pairs": str((output_dir / "individual_pair_metrics.csv").resolve()),
        "theory": str((output_dir / "individual_method_theory.csv").resolve()),
        "best_params": str((output_dir / "individual_best_params.csv").resolve()),
        "sensitivity": str((output_dir / "individual_parameter_sensitivity.csv").resolve()),
        "plots": str(plots_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить таблицы и графики чувствительности итогового набора методов.")
    parser.add_argument("--full-protocol-dir", type=Path, default=PROJECT_ROOT / "results" / "full_protocol")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "individual_methods")
    args = parser.parse_args()
    print(json.dumps(build(args.full_protocol_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
