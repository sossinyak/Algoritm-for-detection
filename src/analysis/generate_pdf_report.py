"""Формирование PDF-отчета по CSV-таблицам и PNG-артефактам эксперимента."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plot_style import dataset_label, method_label
from pipelines.method_metadata import COMPARISON_METHODS


PAGE_SIZE = (11.69, 8.27)  # A4 landscape, inches.
KEY_ADAPTIVE_PARAMETERS = {
    "threshold_method",
    "radiometric_normalization",
    "otsu_scale",
    "min_area",
    "max_aspect_ratio",
}
REPORT_PARAMETER_DATASET = "LEVIR-CD-filtred"

COLUMN_LABELS = {
    "dataset": "Датасет",
    "target_split": "Разбиение",
    "samples": "Пар",
    "method": "Метод",
    "precision": "Точность",
    "recall": "Полнота",
    "f1": "F1",
    "time_ms": "Время, мс",
    "subset": "Подмножество",
    "baseline": "Базовый метод",
    "candidate": "Кандидат",
    "delta_mean": "Средняя разница F1",
    "ci_low": "CI 2.5%",
    "ci_high": "CI 97.5%",
    "p_improvement": "P(улучшение)",
    "pairs": "Пар",
    "aligned_pairs": "Выровнено",
    "warnings_before": "Предупр. до",
    "warnings_after": "Предупр. после",
    "mean_shift_before": "Средний сдвиг до",
    "mean_shift_after": "Средний сдвиг после",
    "parameter": "Параметр",
    "best_value": "Лучшее значение",
    "best_validation_f1": "Лучший F1 на валидации",
    "mean_validation_f1_at_best_value": "Средний F1 на валидации",
    "trials_at_best_value": "Число trials",
}


def _read_csv(path: Path | None) -> pd.DataFrame:
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _collect_split_counts(data_root: Path = Path("data")) -> pd.DataFrame:
    split_csv = data_root / "split_summary.csv"
    if split_csv.exists():
        splits = pd.read_csv(split_csv)
        if {"dataset", "target_split"}.issubset(splits.columns):
            return splits.groupby(["dataset", "target_split"]).size().reset_index(name="samples")
        return splits

    rows = []
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for split in ("train", "val", "test"):
            label_dir = dataset_dir / split / "label"
            if label_dir.is_dir():
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "target_split": split,
                        "samples": sum(1 for path in label_dir.iterdir() if path.is_file()),
                    }
                )
    return pd.DataFrame(rows)


def _format_frame(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 24) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"message": ["Нет данных"]})
    view = df.copy()
    if columns:
        view = view[[column for column in columns if column in view.columns]]
    view = view.head(max_rows)
    for column in ["dataset", "subset"]:
        if column in view.columns:
            view[column] = view[column].astype(str).map(dataset_label)
    for column in ["method", "baseline", "candidate"]:
        if column in view.columns:
            view[column] = view[column].astype(str).map(lambda value: method_label(value).replace("\n", " "))
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    view = view.rename(columns={column: COLUMN_LABELS.get(column, column) for column in view.columns})
    return view.astype(str)


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, title, fontsize=22, fontweight="bold", color="#102a43")
    y = 0.78
    for line in lines:
        fig.text(0.06, y, line, fontsize=11.5, color="#1f2933", wrap=True)
        y -= 0.055
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 24) -> None:
    view = _format_frame(df, columns=columns, max_rows=max_rows)
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#102a43", pad=16)
    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(weight="bold", color="#102a43")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str | None = None) -> None:
    if not image_path.exists():
        return
    with Image.open(image_path) as image:
        width, height = image.size
        fig_w, fig_h = PAGE_SIZE
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        ax.axis("off")
        if title:
            ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#102a43", pad=12)
        ax.imshow(image)
        ax.set_aspect("equal")
        if width / max(height, 1) > fig_w / fig_h:
            ax.set_xlim(0, width)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _quality_summary(quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty:
        return quality
    rows = []
    for dataset, group in quality.groupby("dataset"):
        before = (
            pd.to_numeric(group["estimated_shift_magnitude"], errors="coerce")
            if "estimated_shift_magnitude" in group.columns
            else pd.Series(dtype=float)
        )
        before_response = (
            pd.to_numeric(group["estimated_shift_response"], errors="coerce")
            if "estimated_shift_response" in group.columns
            else pd.Series(1.0, index=group.index)
        )
        before_reliable = (
            group["estimated_shift_reliable"].fillna(False).astype(bool)
            if "estimated_shift_reliable" in group.columns
            else before_response >= 0.05
        )
        after = (
            pd.to_numeric(group["residual_shift_magnitude"], errors="coerce")
            if "residual_shift_magnitude" in group.columns
            else pd.Series(dtype=float)
        )
        after_response = (
            pd.to_numeric(group["residual_shift_response"], errors="coerce")
            if "residual_shift_response" in group.columns
            else pd.Series(1.0, index=group.index)
        )
        after_reliable = (
            group["residual_shift_reliable"].fillna(False).astype(bool)
            if "residual_shift_reliable" in group.columns
            else after_response >= 0.05
        )
        before_valid = before[before_reliable]
        after_valid = after[after_reliable]
        aligned = group.get("alignment_applied", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        rows.append(
            {
                "dataset": dataset,
                "pairs": int(len(group)),
                "aligned_pairs": int(aligned.sum()),
                "unreliable_estimates": int((~before_reliable).sum()),
                "warnings_before": int((before_valid > 1.5).sum()),
                "warnings_after": int((after_valid > 1.5).sum()) if not after_valid.empty else "",
                "mean_shift_before": float(before_valid.mean()) if not before_valid.empty else 0.0,
                "mean_shift_after": float(after_valid.mean()) if not after_valid.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _subset_summary(results_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((results_dir / "industrial_structural_subset_plots").glob("*/method_f1_summary.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        frame = frame[frame["method"].isin(COMPARISON_METHODS)].copy()
        frame = frame.sort_values(["f1", "precision"], ascending=False).head(3).copy()
        frame.insert(0, "subset", csv_path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _conclusion_lines(summary: pd.DataFrame, subset: pd.DataFrame, stats: pd.DataFrame, quality: pd.DataFrame) -> list[str]:
    lines = [
        "Итоговый алгоритм выбирается по полному test-разбиению после автоподбора параметров на валидационной выборке.",
    ]
    if not summary.empty:
        selected = summary.sort_values(["dataset", "f1", "precision"], ascending=[True, False, False]).groupby("dataset").head(1)
        for _, row in selected.sort_values("dataset").iterrows():
            lines.append(
                f"{dataset_label(str(row['dataset']))}: выбран {method_label(str(row['method'])).replace(chr(10), ' ')} с F1={float(row['f1']):.3f}, "
                f"точность={float(row['precision']):.3f}, полнота={float(row['recall']):.3f}."
            )
    if not subset.empty:
        best_subset = subset.sort_values(["subset", "f1", "precision"], ascending=[True, False, False]).groupby("subset").head(1)
        for _, row in best_subset.iterrows():
            lines.append(f"Промышленно-структурное подмножество {row['subset']}: лучший метод {method_label(str(row['method'])).replace(chr(10), ' ')}, F1={float(row['f1']):.3f}.")
    if not stats.empty:
        stable = stats[stats.get("p_improvement", 0) >= 0.95]
        lines.append(f"Bootstrap: статистически устойчивое улучшение относительно абсолютной разности получено для {len(stable)} из {stats['dataset'].nunique()} датасетов.")
    if not quality.empty and "warnings_after" in quality:
        before = int(pd.to_numeric(quality["warnings_before"], errors="coerce").fillna(0).sum())
        after = int(pd.to_numeric(quality["warnings_after"], errors="coerce").fillna(0).sum())
        lines.append(f"Геометрия: предупреждений о сдвиге до выравнивания {before}, после автоматической коррекции {after}.")
    lines.append("Синтетический набор используется как лабораторная проверка достижимости высокого качества; реальные датасеты ограничены RGB-каналами и неоднородным геометрическим совмещением.")
    return lines


def _plot_paths(results_dir: Path) -> list[Path]:
    paths: list[Path] = [
        results_dir / "final_plots" / "f1_ranking_LEVIR-CD-filtred.png",
        results_dir / "final_plots" / "f1_ranking_JL1-CD.png",
        results_dir / "final_plots" / "f1_ranking_synthetic-lab.png",
    ]
    paths.extend(sorted((results_dir / "industrial_structural_subset_plots").glob("*/method_f1_industrial_structural_subset.png")))
    paths.extend(sorted((results_dir / "adaptive_component_ablation").glob("*/component_contribution_f1_precision_recall.png")))

    for dataset_dir in sorted((results_dir / "adaptive_parameter_sensitivity").iterdir() if (results_dir / "adaptive_parameter_sensitivity").is_dir() else []):
        if not dataset_dir.is_dir() or dataset_dir.name != REPORT_PARAMETER_DATASET:
            continue
        for parameter in sorted(KEY_ADAPTIVE_PARAMETERS):
            paths.append(dataset_dir / f"adaptive_f1_by_{parameter}.png")

    for dataset_dir in sorted((results_dir / "stage_visualization").iterdir() if (results_dir / "stage_visualization").is_dir() else []):
        if dataset_dir.is_dir():
            selected_csv = dataset_dir / "selected_samples.csv"
            chosen = None
            if selected_csv.exists():
                selected = pd.read_csv(selected_csv)
                if "f1" in selected.columns and "image_path" in selected.columns:
                    selected["f1"] = pd.to_numeric(selected["f1"], errors="coerce")
                    selected = selected.sort_values(["f1", "precision", "recall"], ascending=False)
                    for _, row in selected.iterrows():
                        stored_path = Path(str(row["image_path"]))
                        candidates = [dataset_dir / stored_path.name, stored_path]
                        if not stored_path.is_absolute():
                            candidates.insert(0, dataset_dir / stored_path)
                        for image_path in dict.fromkeys(candidates):
                            if image_path.exists():
                                chosen = image_path
                                break
                        if chosen is not None:
                            break
            if chosen is None:
                chosen = next(iter(sorted(dataset_dir.glob("*_stages.png"))), None)
            if chosen is not None:
                paths.append(chosen)
    return [path for path in dict.fromkeys(paths) if path.exists()]


def build_pdf_report(results_dir: Path, output_pdf: Path, data_root: Path = Path("data")) -> Path:
    summary_csv = _first_existing(
        [
            results_dir / "full_test_best_params_summary.csv",
            results_dir / "parameter_study" / "all_parameter_summary.csv",
        ]
    )
    quality_csv = _first_existing([results_dir / "input_quality_report.csv", results_dir / "parameter_study" / "all_input_quality_report.csv"])
    adaptive_sensitivity_csv = results_dir / "adaptive_parameter_sensitivity" / "adaptive_parameter_sensitivity.csv"
    stats_paths = sorted(results_dir.glob("statistical_validation*.csv"))

    summary = _read_csv(summary_csv)
    if not summary.empty and "method" in summary.columns:
        summary = summary[summary["method"].isin(COMPARISON_METHODS)].copy()
    quality = _read_csv(quality_csv)
    adaptive_sensitivity = _read_csv(adaptive_sensitivity_csv)
    stats = pd.concat([pd.read_csv(path) for path in stats_paths], ignore_index=True) if stats_paths else pd.DataFrame()
    splits = _collect_split_counts(data_root)
    geometry = _quality_summary(quality)
    subset = _subset_summary(results_dir)
    adaptive_sensitivity = (
        adaptive_sensitivity[
            (adaptive_sensitivity["parameter"].isin(KEY_ADAPTIVE_PARAMETERS))
            & (adaptive_sensitivity["dataset"] == REPORT_PARAMETER_DATASET)
        ].copy()
        if "parameter" in adaptive_sensitivity.columns
        else adaptive_sensitivity
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        _add_text_page(pdf, "Ключевые выводы", _conclusion_lines(summary, subset, stats, geometry))
        _add_text_page(
            pdf,
                "Литературное обоснование выбранного конвейера",
            [
                "Адаптивный CVA с МГК рассматривается как итоговый алгоритм, потому что он развивает классическую схему CVA с МГК, остается интерпретируемым и в целевом промышленно-структурном сценарии дает лучший результат среди фиксированного набора сравнения.",
                "CVA является базовой идеей дистанционного обнаружения изменений: Malila, 1980, Change Vector Analysis.",
                "Метод главных компонент как способ выделения информативного пространства изменений и PCA + K-средних как классический базовый метод описаны в Celik, 2009, IEEE GRSL.",
                "Метод Оцу используется как автоматический глобальный порог: Otsu, 1979, IEEE Transactions on Systems, Man, and Cybernetics.",
                "Локальные пороги проверяются как альтернатива глобальному порогу для неоднородного освещения: Sauvola, Pietikainen, 2000, Pattern Recognition.",
                "Радиометрическая нормализация перед сравнением разновременных снимков обоснована литературой по относительной радиометрической нормализации; согласование гистограмм рассматривается как распространенный вариант такого подхода.",
                "Источники: https://docs.lib.purdue.edu/lars_symp/385/ ; https://perso.telecom-paristech.fr/gousseau/IMA201/change.pdf ; https://skynet.ecn.purdue.edu/~ace/vip/A_Threshold_Selection_Method_from_Gray-Level_Histograms_otsu.pdf ; https://www.sciencedirect.com/science/article/pii/S0031320399000552 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11014200/",
            ],
        )
        _add_table_page(pdf, "Размеры train/val/test", splits)
        _add_table_page(
            pdf,
            "Топ-методы на промышленно-структурных подмножествах",
            subset,
            ["subset", "method", "precision", "recall", "f1", "samples"],
            max_rows=8,
        )
        _add_table_page(pdf, "Bootstrap-проверка улучшения относительно абсолютной разности", stats)
        _add_table_page(
            pdf,
            "Сводка геометрического совмещения",
            geometry,
            [
                "dataset",
                "pairs",
                "aligned_pairs",
                "unreliable_estimates",
                "warnings_before",
                "warnings_after",
                "mean_shift_before",
                "mean_shift_after",
            ],
            max_rows=12,
        )
        _add_table_page(
            pdf,
            "Ключевые параметры адаптивного CVA с МГК",
            adaptive_sensitivity,
            ["dataset", "parameter", "best_value", "best_validation_f1", "mean_validation_f1_at_best_value", "trials_at_best_value"],
            max_rows=24,
        )
        for path in _plot_paths(results_dir):
            title = os.path.relpath(path, results_dir)
            _add_image_page(pdf, path, title=title)
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать PDF-отчет по исследовательскому протоколу.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/full_protocol"))
    parser.add_argument("--output-pdf", type=Path, default=Path("results/full_protocol/research_report.pdf"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = build_pdf_report(args.results_dir, args.output_pdf, args.data_root)
    print(json.dumps({"report_pdf": str(path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
