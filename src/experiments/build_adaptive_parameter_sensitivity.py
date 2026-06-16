"""Построение графиков чувствительности параметров Adaptive PCA-CVA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.plot_style import BAR_PALETTE, apply_chart_style, dataset_label, save_chart


ADAPTIVE_METHOD = "Adaptive PCA-CVA"

PARAMETER_LABELS = {
    "threshold": "тип порога",
    "threshold_method": "локальный/глобальный порог",
    "radiometric_normalization": "радиометрическая нормализация",
    "otsu_scale": "масштаб порога Оцу",
    "adaptive_block_size": "размер блока локального порога",
    "adaptive_c": "смещение локального порога",
    "kimura_k": "коэффициент Кимуры",
    "sauvola_k": "коэффициент Саволы",
    "opening_kernel": "ядро морфологического открытия",
    "closing_kernel": "ядро морфологического закрытия",
    "min_area": "минимальная площадь компоненты",
    "max_aspect_ratio": "максимальная вытянутость",
    "min_solidity": "минимальная плотность компоненты",
    "min_extent": "минимальная заполненность рамки",
}

PLOT_ORDER = [
    "threshold",
    "threshold_method",
    "radiometric_normalization",
    "otsu_scale",
    "adaptive_block_size",
    "adaptive_c",
    "kimura_k",
    "sauvola_k",
    "opening_kernel",
    "closing_kernel",
    "min_area",
    "max_aspect_ratio",
    "min_solidity",
    "min_extent",
]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _parse_json_dict(value: object) -> dict:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _flatten_adaptive_trials(trials: pd.DataFrame) -> pd.DataFrame:
    adaptive = trials[trials["method"] == ADAPTIVE_METHOD].copy()
    if adaptive.empty:
        return adaptive
    records = []
    for _, row in adaptive.iterrows():
        record = row.to_dict()
        adaptive_params = _parse_json_dict(record.get("adaptive_params"))
        for key, value in adaptive_params.items():
            record[key] = value
        if not record.get("threshold_method"):
            record["threshold_method"] = record.get("threshold")
        records.append(record)
    return pd.DataFrame(records)


def _plot_parameter(frame: pd.DataFrame, dataset: str, parameter: str, output_path: Path) -> dict | None:
    data = frame[frame["dataset"] == dataset].copy()
    if parameter not in data.columns:
        return None
    values = data[parameter].dropna().astype(str)
    if values.nunique() < 2:
        return None
    data[parameter] = data[parameter].astype(str)
    grouped = (
        data.groupby(parameter, dropna=False)
        .agg(mean_f1=("tune_f1", "mean"), best_f1=("tune_f1", "max"), trials=("tune_f1", "count"))
        .reset_index()
    )
    grouped = grouped.assign(_sort_key=grouped[parameter].astype(str)).sort_values("_sort_key").drop(columns="_sort_key")

    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(grouped)), 4.6))
    x = list(range(len(grouped)))
    width = 0.36
    ax.bar([value - width / 2 for value in x], grouped["mean_f1"], width=width, label="средняя F1", color=BAR_PALETTE[0], edgecolor="white", linewidth=0.6)
    best_bars = ax.bar([value + width / 2 for value in x], grouped["best_f1"], width=width, label="лучшая F1", color=BAR_PALETTE[2], edgecolor="white", linewidth=0.6)
    apply_chart_style(fig, ax)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped[parameter], rotation=25, ha="right")
    label = PARAMETER_LABELS.get(parameter, parameter)
    ax.set_title(f"Адаптивный CVA с МГК: влияние параметра {label} ({dataset_label(dataset)})", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel(label)
    ax.set_ylabel("F1 на валидации, доли ед.")
    ax.legend(loc="best")
    max_value = float(grouped[["mean_f1", "best_f1"]].max().max())
    ax.set_ylim(0, min(1.05, max(1.0, max_value * 1.15)))
    for bar, value in zip(best_bars, grouped["best_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.01, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_chart(fig, output_path, dpi=190)

    best = grouped.sort_values(["best_f1", "mean_f1"], ascending=False).iloc[0]
    return {
        "dataset": dataset,
        "parameter": parameter,
        "best_value": best[parameter],
        "best_validation_f1": float(best["best_f1"]),
        "mean_validation_f1_at_best_value": float(best["mean_f1"]),
        "trials_at_best_value": int(best["trials"]),
        "plot": str(output_path.resolve()),
    }


def build(trials_csv: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(trials_csv)
    adaptive = _flatten_adaptive_trials(trials)
    flattened_csv = output_dir / "adaptive_parameter_trials_flat.csv"
    adaptive.to_csv(flattened_csv, index=False, encoding="utf-8-sig")

    sensitivity_rows = []
    for dataset in sorted(adaptive["dataset"].dropna().unique()):
        for parameter in PLOT_ORDER:
            result = _plot_parameter(
                adaptive,
                str(dataset),
                parameter,
                output_dir / _safe_name(str(dataset)) / f"adaptive_f1_by_{_safe_name(parameter)}.png",
            )
            if result is not None:
                sensitivity_rows.append(result)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_csv = output_dir / "adaptive_parameter_sensitivity.csv"
    sensitivity.to_csv(sensitivity_csv, index=False, encoding="utf-8-sig")
    manifest = {
        "flattened_trials_csv": str(flattened_csv.resolve()),
        "sensitivity_csv": str(sensitivity_csv.resolve()),
        "plots_dir": str(output_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить графики чувствительности параметров Adaptive PCA-CVA.")
    parser.add_argument("--trials-csv", type=Path, default=Path("results/full_protocol/parameter_study/all_parameter_trials.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_protocol/adaptive_parameter_sensitivity"))
    args = parser.parse_args()
    print(json.dumps(build(args.trials_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
