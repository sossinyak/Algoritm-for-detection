"""Общий стиль графиков итогового отчета."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PLOT_BACKGROUND = "#ffffff"
GRID_COLOR = "#d8d8d8"
SPINE_COLOR = "#444444"
TEXT_COLOR = "#151515"
BAR_PALETTE = [
    "#3f78ad",
    "#5fa696",
    "#d9a545",
    "#aa6ab0",
    "#c96d5f",
    "#879f56",
    "#6c87b8",
    "#7f7f7f",
    "#2b7a78",
]


def method_label(method: str) -> str:
    labels = {
        "AbsDiff": "Абсолютная\nразность",
        "LogRatio": "Логарифмическое\nотношение",
        "RGB-CVA": "CVA\nпо RGB",
        "PCA-CVA": "CVA\nс МГК",
        "Adaptive PCA-CVA": "Адаптивный\nCVA с МГК",
    }
    return labels.get(method, method.replace("-", "\n"))


def dataset_label(dataset: str) -> str:
    labels = {
        "LEVIR-CD-filtred": "LEVIR-CD",
        "LEVIR-CD": "LEVIR-CD",
        "synthetic-lab": "синтетический набор",
    }
    return labels.get(dataset, dataset)


def apply_chart_style(fig, ax) -> None:
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    ax.set_facecolor(PLOT_BACKGROUND)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors="#555555")
    ax.title.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)


def save_chart(fig, output_path: Path, dpi: int = 180) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
