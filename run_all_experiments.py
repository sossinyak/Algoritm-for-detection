"""Единый запуск исследовательского протокола."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.experiment_logger import ExperimentLogger
from utils.synthetic import create_synthetic_dataset


def _run(command: list[str], logger: ExperimentLogger | None = None, step: str | None = None) -> None:
    """Печатает команду и запускает ее как отдельный этап протокола."""
    print(json.dumps({"run": command}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    if logger:
        logger.log_event("stage_started", {"step": step, "command": command})
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if logger:
            logger.log_event(
                "stage_failed",
                {
                    "step": step,
                    "command": command,
                    "returncode": error.returncode,
                    "duration_sec": round(time.perf_counter() - started, 3),
                },
            )
        raise
    if logger:
        logger.log_metrics({"duration_sec": round(time.perf_counter() - started, 3)}, step=step)
        logger.log_event(
            "stage_finished",
            {"step": step, "command": command, "duration_sec": round(time.perf_counter() - started, 3)},
        )


def _best_methods(summary_csv: Path) -> list[dict]:
    """Возвращает лучший метод для каждого датасета по F1."""
    if not summary_csv.exists():
        return []
    df = pd.read_csv(summary_csv)
    if df.empty:
        return []
    best = df.sort_values(["dataset", "f1", "precision"], ascending=[True, False, False]).groupby("dataset").head(1)
    return best[["dataset", "method"]].to_dict("records")


def _add_optional_int(command: list[str], name: str, value: int | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить полный исследовательский протокол.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/full_protocol"))
    parser.add_argument("--datasets", nargs="*", default=["LEVIR-CD-filtred", "JL1-CD", "synthetic-lab"])
    parser.add_argument("--max-tune-samples", type=int, default=24)
    parser.add_argument("--max-eval-samples", type=int, default=80)
    parser.add_argument("--monte-carlo-trials", type=int, default=48)
    parser.add_argument("--full-eval-samples", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--error-samples", type=int, default=30)
    parser.add_argument("--ablation-samples", type=int, default=160)
    parser.add_argument("--stage-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resplit", action="store_true", help="Пересобрать train/val/test перед экспериментами.")
    parser.add_argument("--skip-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-train-pairs", type=int, default=24)
    parser.add_argument("--synthetic-val-pairs", type=int, default=12)
    parser.add_argument("--synthetic-test-pairs", type=int, default=24)
    parser.add_argument("--synthetic-image-size", type=int, default=160)
    parser.add_argument("--report-format", choices=["none", "pdf", "html", "both"], default="pdf")
    args = parser.parse_args()

    logger = ExperimentLogger(
        args.results_dir / "runs",
        "research_protocol",
        tags={"stage": "run_all_experiments", "results_dir": args.results_dir},
    )
    logger.log_params({"args": vars(args)})

    python = sys.executable
    datasets = list(dict.fromkeys(args.datasets or []))
    if not datasets:
        raise RuntimeError("Не задан ни один датасет для запуска.")
    try:
        if args.refresh_synthetic and "synthetic-lab" in datasets:
            synthetic_root = create_synthetic_dataset(
                root=args.data_root / "synthetic-lab",
                train_pairs=args.synthetic_train_pairs,
                val_pairs=args.synthetic_val_pairs,
                test_pairs=args.synthetic_test_pairs,
                image_size=args.synthetic_image_size,
                seed=args.seed,
            )
            logger.log_event(
                "synthetic_refreshed",
                {
                    "root": str(synthetic_root),
                    "train_pairs": args.synthetic_train_pairs,
                    "val_pairs": args.synthetic_val_pairs,
                    "test_pairs": args.synthetic_test_pairs,
                    "image_size": args.synthetic_image_size,
                },
            )

        if args.resplit and not args.skip_split:
            split_command = [
                python,
                str(SRC_ROOT / "tools" / "data" / "split_datasets.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.split_root),
                "--seed",
                str(args.seed),
            ]
            split_command.extend(["--datasets", *datasets])
            if args.split_root.resolve() == args.data_root.resolve():
                split_command.append("--in-place")
            _run(split_command, logger=logger, step="split_datasets")

        study_dir = args.results_dir / "parameter_study"
        parameter_command = [
            python,
            str(SRC_ROOT / "analysis" / "parameter_analyzer.py"),
            "--config",
            str(args.config),
            "--data-root",
            str(args.split_root),
            "--results-dir",
            str(study_dir),
            "--max-tune-samples",
            str(args.max_tune_samples),
            "--max-eval-samples",
            str(args.max_eval_samples),
            "--monte-carlo-trials",
            str(args.monte_carlo_trials),
            "--seed",
            str(args.seed),
            "--datasets",
            *datasets,
        ]
        _run(parameter_command, logger=logger, step="parameter_study")

        study_summary_csv = study_dir / "all_parameter_summary.csv"
        full_eval_command = [
            python,
            str(SRC_ROOT / "experiments" / "evaluate_best_params_full_test.py"),
            "--config",
            str(args.config),
            "--data-root",
            str(args.split_root),
            "--best-summary-csv",
            str(study_summary_csv),
            "--output-dir",
            str(args.results_dir),
            "--split",
            "test",
            "--seed",
            str(args.seed),
            "--datasets",
            *datasets,
        ]
        _add_optional_int(full_eval_command, "--max-samples", args.full_eval_samples)
        _run(full_eval_command, logger=logger, step="full_test_best_params")

        pair_metrics_csv = args.results_dir / "full_test_best_params_pair_metrics.csv"
        summary_csv = args.results_dir / "full_test_best_params_summary.csv"
        best_methods = _best_methods(summary_csv)
        logger.log_metrics({"best_methods": len(best_methods)}, step="select_best_methods")
        for item in best_methods:
            output_csv = args.results_dir / f"statistical_validation_{item['dataset']}.csv"
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "statistical_validation.py"),
                    "--pair-metrics-csv",
                    str(pair_metrics_csv),
                    "--baseline",
                    "AbsDiff",
                    "--candidate",
                    item["method"],
                    "--dataset",
                    item["dataset"],
                    "--output-csv",
                    str(output_csv),
                    "--n-bootstrap",
                    str(args.bootstrap_iterations),
                    "--seed",
                    str(args.seed),
                ],
                logger=logger,
                step=f"statistical_validation:{item['dataset']}",
            )
            logger.log_artifact(output_csv)

        error_dir = args.results_dir / "error_analysis"
        for item in best_methods:
            dataset_path = args.split_root / item["dataset"]
            if not dataset_path.exists():
                continue
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "error_analysis.py"),
                    "--data-path",
                    str(dataset_path),
                    "--method",
                    item["method"],
                    "--config",
                    str(args.config),
                    "--split",
                    "test",
                    "--max-samples",
                    str(args.error_samples),
                    "--results-dir",
                    str(error_dir),
                ],
                logger=logger,
                step=f"error_analysis:{item['dataset']}",
            )

        final_plots_dir = args.results_dir / "final_plots"
        _run(
            [
                python,
                str(SRC_ROOT / "analysis" / "plot_f1_results.py"),
                "--summary-csv",
                str(summary_csv),
                "--output-dir",
                str(final_plots_dir),
            ],
            logger=logger,
            step="plot_f1_results",
        )

        individual_dir = args.results_dir / "individual_methods"
        _run(
            [
                python,
                str(SRC_ROOT / "experiments" / "build_individual_method_study.py"),
                "--full-protocol-dir",
                str(args.results_dir),
                "--output-dir",
                str(individual_dir),
            ],
            logger=logger,
            step="individual_method_study",
        )

        adaptive_sensitivity_dir = args.results_dir / "adaptive_parameter_sensitivity"
        _run(
            [
                python,
                str(SRC_ROOT / "experiments" / "build_adaptive_parameter_sensitivity.py"),
                "--trials-csv",
                str(study_dir / "all_parameter_trials.csv"),
                "--output-dir",
                str(adaptive_sensitivity_dir),
            ],
            logger=logger,
            step="adaptive_parameter_sensitivity",
        )

        component_root = args.results_dir / "adaptive_component_ablation"
        for dataset in datasets:
            dataset_path = args.split_root / dataset
            if not dataset_path.exists():
                continue
            command = [
                python,
                str(SRC_ROOT / "experiments" / "run_adaptive_component_ablation.py"),
                "--config",
                str(args.config),
                "--data-path",
                str(dataset_path),
                "--split",
                "test",
                "--output-dir",
                str(component_root / dataset),
                "--best-summary-csv",
                str(summary_csv),
            ]
            _add_optional_int(command, "--max-samples", args.ablation_samples)
            _run(command, logger=logger, step=f"adaptive_component_ablation:{dataset}")

        subset_root = args.results_dir / "industrial_structural_subsets"
        subset_datasets = [dataset for dataset in ("LEVIR-CD-filtred", "JL1-CD") if dataset in datasets]
        for dataset in subset_datasets:
            command = [
                python,
                str(SRC_ROOT / "experiments" / "run_selected_pairs_thresholds.py"),
                "--config",
                str(args.config),
                "--data-root",
                str(args.split_root),
                "--dataset",
                dataset,
                "--split",
                "test",
                "--output-dir",
                str(subset_root / dataset),
                "--visual-limit",
                "12",
            ]
            _run(command, logger=logger, step=f"industrial_strict_subset_selection:{dataset}")

        subset_plots_dir = args.results_dir / "industrial_structural_subset_plots"
        if subset_datasets:
            _run(
                [
                    python,
                    str(SRC_ROOT / "experiments" / "build_industrial_structural_subset_plots.py"),
                    "--config",
                    str(args.config),
                    "--data-root",
                    str(args.split_root),
                    "--split",
                    "test",
                    "--pair-metrics-csv",
                    str(pair_metrics_csv),
                    "--selection-root",
                    str(subset_root),
                    "--output-dir",
                    str(subset_plots_dir),
                ],
                logger=logger,
                step="industrial_structural_strict_subset_plots",
            )

        stage_root = args.results_dir / "stage_visualization"
        for item in best_methods:
            dataset_path = args.split_root / item["dataset"]
            if not dataset_path.exists():
                continue
            _run(
                [
                    python,
                    str(SRC_ROOT / "tools" / "visualization" / "visualize_adaptive_stages.py"),
                    "--config",
                    str(args.config),
                    "--data_path",
                    str(dataset_path),
                    "--split",
                    "test",
                    "--selection",
                    "representative",
                    "--count",
                    str(args.stage_samples),
                    "--output-dir",
                    str(stage_root / item["dataset"]),
                ],
                logger=logger,
                step=f"stage_visualization:{item['dataset']}",
            )

        report_artifacts = []
        if args.report_format in {"html", "both"}:
            html_report_path = args.results_dir / "research_report.html"
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "generate_report.py"),
                    "--results-dir",
                    str(args.results_dir),
                    "--output-html",
                    str(html_report_path),
                ],
                logger=logger,
                step="generate_html_report",
            )
            report_artifacts.append(html_report_path)
        if args.report_format in {"pdf", "both"}:
            pdf_report_path = args.results_dir / "research_report.pdf"
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "generate_pdf_report.py"),
                    "--results-dir",
                    str(args.results_dir),
                    "--output-pdf",
                    str(pdf_report_path),
                    "--data-root",
                    str(args.data_root),
                ],
                logger=logger,
                step="generate_pdf_report",
            )
            report_artifacts.append(pdf_report_path)
        logger.log_artifacts(
            [
                summary_csv,
                pair_metrics_csv,
                study_dir / "all_parameter_trials.csv",
                study_dir / "best_params.yaml",
                final_plots_dir,
                individual_dir,
                adaptive_sensitivity_dir,
                component_root,
                subset_root,
                subset_plots_dir,
                stage_root,
                *report_artifacts,
            ]
        )
        logger.finish()
    except Exception as error:
        logger.finish(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
