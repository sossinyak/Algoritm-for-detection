"""Генерация HTML-отчета по исследовательскому протоколу."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd


def _first_existing(paths: list[Path]) -> Path | None:
    """Возвращает первый существующий путь из списка кандидатов."""
    for path in paths:
        if path.exists():
            return path
    return None


def _read_csv(path: Path | None) -> pd.DataFrame:
    """Безопасно читает CSV, если файл существует."""
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()


def _display_path(path: Path | None) -> str | None:
    """Форматирует путь для HTML-отчета без Windows-экранирования обратных слешей."""
    return path.as_posix() if path else None


def _collect_split_counts(data_root: Path = Path("data")) -> pd.DataFrame:
    """Считает размер train/val/test по структуре папок, если split_summary.csv отсутствует."""
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
                samples = sum(1 for path in label_dir.iterdir() if path.is_file())
                rows.append({"dataset": dataset_dir.name, "target_split": split, "samples": samples})
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 40) -> str:
    if df.empty:
        return "<p>Нет данных.</p>"
    view = df.copy()
    if columns:
        view = view[[column for column in columns if column in view.columns]]
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    return view.head(max_rows).to_html(index=False, escape=True, classes="data-table")


def _image_gallery(paths: list[Path], title: str) -> str:
    if not paths:
        return ""
    blocks = [f"<h2>{html.escape(title)}</h2><div class='gallery'>"]
    for path in paths:
        rel = html.escape(path.as_posix())
        blocks.append(f"<figure><img src='{rel}' alt='{html.escape(path.name)}'><figcaption>{html.escape(path.name)}</figcaption></figure>")
    blocks.append("</div>")
    return "\n".join(blocks)


def _relative_display_paths(paths: list[Path], output_html: Path) -> list[Path]:
    """Строит корректные относительные пути к изображениям от папки HTML-отчета."""
    base_dir = output_html.parent.resolve()
    relative_paths = []
    for path in paths:
        absolute_path = path.resolve()
        relative_paths.append(Path(os.path.relpath(absolute_path, base_dir)))
    return relative_paths


def _read_stage_selection(stage_root: Path | None) -> pd.DataFrame:
    """Собирает таблицу выбранных патчей для визуализации промежуточных этапов."""
    if not stage_root or not stage_root.exists():
        return pd.DataFrame()
    frames = []
    for csv_path in sorted(stage_root.glob("*/selected_samples.csv")):
        frame = pd.read_csv(csv_path)
        frame.insert(0, "dataset", csv_path.parent.name)
        for column in ("metrics_csv", "image_path"):
            if column in frame.columns:
                frame[column] = frame[column].map(lambda value: Path(str(value)).name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_report(results_dir: Path, output_html: Path) -> Path:
    """Собирает HTML-отчет из CSV/YAML/PNG артефактов."""
    root_results_dir = results_dir.parent if results_dir.name == "full_test" else results_dir
    parameter_study_dir = _first_existing(
        [
            results_dir / "parameter_study",
            root_results_dir / "parameter_study",
        ]
    )
    summary_csv = _first_existing(
        [
            results_dir / "full_test_best_params_summary.csv",
            results_dir / "all_parameter_summary.csv",
            results_dir / "parameter_study" / "all_parameter_summary.csv",
            root_results_dir / "full_test" / "full_test_best_params_summary.csv",
            root_results_dir / "parameter_study" / "all_parameter_summary.csv",
        ]
    )
    trials_csv = _first_existing(
        [
            results_dir / "parameter_study" / "all_parameter_trials.csv",
            root_results_dir / "parameter_study" / "all_parameter_trials.csv",
        ]
    )
    quality_csv = _first_existing(
        [
            results_dir / "input_quality_report.csv",
            results_dir / "all_input_quality_report.csv",
            results_dir / "parameter_study" / "input_quality_report.csv",
            results_dir / "parameter_study" / "all_input_quality_report.csv",
            root_results_dir / "input_quality_report.csv",
            root_results_dir / "parameter_study" / "all_input_quality_report.csv",
        ]
    )
    stats_paths = sorted(results_dir.glob("statistical_validation*.csv")) + sorted(root_results_dir.glob("statistical_validation*.csv"))

    summary = _read_csv(summary_csv)
    trials = _read_csv(trials_csv)
    quality = _read_csv(quality_csv)
    stats = pd.concat([pd.read_csv(path) for path in stats_paths], ignore_index=True) if stats_paths else pd.DataFrame()
    split_counts = _collect_split_counts()

    plot_paths = sorted(parameter_study_dir.glob("**/plots/*.png")) if parameter_study_dir else []
    final_plots = sorted((results_dir / "final_plots").glob("*.png")) + sorted((root_results_dir / "full_test" / "final_plots").glob("*.png"))
    adaptive_parameter_plots = sorted((results_dir / "adaptive_parameter_sensitivity").glob("**/*.png"))
    component_plots = sorted((results_dir / "adaptive_component_ablation").glob("**/*.png"))
    subset_plots = sorted((results_dir / "industrial_structural_subset_plots").glob("**/*.png"))
    individual_plots = sorted((results_dir / "individual_methods").glob("**/*.png"))
    stage_root = _first_existing(
        [
            results_dir / "stage_visualization",
            root_results_dir / "full_test" / "stage_visualization",
            root_results_dir / "stage_visualization",
        ]
    )
    stage_plots = sorted(stage_root.glob("*/*_stages.png")) if stage_root else []
    stage_selection = _read_stage_selection(stage_root)
    unique_images = list(dict.fromkeys(final_plots + subset_plots + component_plots + adaptive_parameter_plots + individual_plots + plot_paths[:12]))
    image_paths = _relative_display_paths(unique_images, output_html)
    stage_image_paths = _relative_display_paths(stage_plots, output_html)

    best = summary.sort_values(["dataset", "f1", "precision"], ascending=[True, False, False]).groupby("dataset").head(3) if not summary.empty else pd.DataFrame()

    payload = {
        "summary_csv": _display_path(summary_csv),
        "trials_csv": _display_path(trials_csv),
        "quality_csv": _display_path(quality_csv),
        "stats_csv": [_display_path(path) for path in stats_paths],
        "stage_visualization": _display_path(stage_root),
        "split_source": "data/split_summary.csv или подсчет файлов в data/<dataset>/<split>/label",
    }
    html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Исследовательский протокол обнаружения изменений</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; line-height: 1.45; }}
    h1, h2 {{ color: #102a43; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    .data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; }}
    .data-table th {{ background: #f0f4f8; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    figcaption {{ font-size: 13px; color: #52606d; margin-top: 4px; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Исследовательский протокол обнаружения изменений</h1>
  <p>Отчет собран автоматически из артефактов экспериментов. Основная метрика сравнения: F1-мера.</p>
  <p>Параметры подбираются на <code>val</code>, финальная оценка выполняется на независимом <code>test</code>.</p>

  <h2>Артефакты</h2>
  <pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>

  <h2>Распределение train/val/test</h2>
  {_table(split_counts)}

  <h2>Контроль геометрического соответствия входных пар</h2>
  {_table(quality, ["dataset", "split", "patch_name", "a_shape", "b_shape_original", "label_shape_original", "b_resized", "label_resized", "estimated_shift_magnitude", "estimated_shift_response", "alignment_applied", "residual_shift_magnitude", "warning"], max_rows=80)}

  <h2>Лучшие методы по датасетам</h2>
  {_table(best, ["dataset", "method", "precision", "recall", "f1", "time_ms", "tune_split", "eval_split"])}

  <h2>Статистическая проверка</h2>
  {_table(stats)}

  <h2>Фрагмент исследования параметров</h2>
  {_table(trials, ["dataset", "method", "threshold", "sigma", "threshold_scale", "adaptive_block_size", "tune_precision", "tune_recall", "tune_f1"], max_rows=60)}

  <h2>Промежуточные этапы адаптивного CVA с МГК</h2>
  <p>Для выбранных test-патчей показаны исходные снимки, эталонная маска, карта изменений, пороговая маска, морфологическая обработка, финальная маска и карта ошибок.</p>
  {_table(stage_selection, ["dataset", "patch_name", "precision", "recall", "f1", "metrics_csv", "image_path"], max_rows=30)}
  {_image_gallery(stage_image_paths, "Визуализация промежуточных этапов")}

  {_image_gallery(image_paths, "Графики")}
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать HTML-отчет по исследовательскому протоколу.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-html", type=Path, default=Path("results/research_report.html"))
    args = parser.parse_args()
    path = build_report(args.results_dir, args.output_html)
    print(json.dumps({"report_html": str(path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
