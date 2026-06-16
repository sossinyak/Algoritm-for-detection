"""Систематическое исследование параметров для исследовательского протокола."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.experiment_logger import ExperimentLogger
from analysis.metrics import calculate_metrics
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from pipelines.baseline_methods import build_tunable_classical_method
from pipelines.method_metadata import COMPARISON_METHODS
from utils.data_loader import LEVIRCDLoader
from utils.pipeline_config import build_adaptive_params


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _has_split(dataset_path: Path, split: str) -> bool:
    return all((dataset_path / split / folder).is_dir() for folder in ("A", "B", "label"))


def _load_pairs(dataset_path: Path, split: str, max_pairs: int | None, seed: int) -> list[dict]:
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split) if pair.get("label") is not None]
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    return [pairs[index] for index in sorted(indices[:max_pairs])]


def _dataset_splits(dataset_path: Path) -> tuple[str, str]:
    """Выбирает val для подбора и test для финальной оценки."""
    if _has_split(dataset_path, "val") and _has_split(dataset_path, "test"):
        return "val", "test"
    if _has_split(dataset_path, "train") and _has_split(dataset_path, "test"):
        return "train", "test"
    if _has_split(dataset_path, "test"):
        return "test", "test"
    if _has_split(dataset_path, "val"):
        return "val", "val"
    return "train", "train"


def _score_method(method: object, pairs: list[dict]) -> tuple[dict, pd.DataFrame]:
    tp = tn = fp = fn = 0
    times = []
    rows = []
    for pair in pairs:
        start = time.perf_counter()
        pred = method.process(pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = calculate_metrics(pred, pair["label"])
        tp += int(metrics["tp"])
        tn += int(metrics["tn"])
        fp += int(metrics["fp"])
        fn += int(metrics["fn"])
        times.append(elapsed_ms)
        rows.append({"patch_name": pair["name"], **metrics, "time_ms": elapsed_ms})
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return (
        {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "time_ms": float(np.mean(times)) if times else 0.0,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "samples": len(pairs),
        },
        pd.DataFrame(rows),
    )


def _method_from_params(params: dict, config: dict) -> object:
    method_type = params["method_type"]
    if method_type == "classical":
        return build_tunable_classical_method(
            score_name=params["score_name"],
            threshold=params["threshold"],
            threshold_scale=params.get("threshold_scale", 1.0),
            postprocess=params.get("postprocess", "area"),
            median_kernel=params.get("median_kernel", 3),
            morph_kernel=params.get("morph_kernel", 3),
            min_area=params.get("min_area"),
            adaptive_block_size=params.get("adaptive_block_size", 35),
            adaptive_c=params.get("adaptive_c", -2.0),
            sigma=params.get("sigma"),
        )
    if method_type == "adaptive_pca":
        adaptive_params = build_adaptive_params(config)
        adaptive_params.update(params.get("adaptive_params", {}))
        return AdaptiveChangeDetection(**adaptive_params)
    raise ValueError(f"Неизвестный тип метода: {method_type}")


def build_parameter_grid(image_area: int, config: dict) -> list[dict]:
    """Формирует компактную, но систематическую сетку параметров."""
    min_areas = sorted({16, 64, max(16, int(image_area * 0.0015))})
    rows: list[dict] = []
    thresholds = [("otsu", 0.8), ("otsu", 1.0), ("otsu", 1.2), ("adaptive", 1.0), ("kmeans", 1.0)]
    for score_name in [method for method in COMPARISON_METHODS if method != "Adaptive PCA-CVA"]:
        for threshold, scale in thresholds:
            for sigma in [None, "auto", 1.0]:
                base = {
                    "method": score_name,
                    "method_type": "classical",
                    "score_name": score_name,
                    "threshold": threshold,
                    "threshold_scale": scale,
                    "sigma": sigma,
                    "postprocess": "area",
                    "median_kernel": 3,
                    "morph_kernel": 3,
                    "min_area": min_areas[-1],
                    "adaptive_block_size": 35,
                    "adaptive_c": -2.0,
                }
                rows.append(base)

    for otsu_scale, opening, closing, min_area in itertools.product([0.75, 0.9, 1.05], [1, 3, 5], [3, 5], min_areas):
        rows.append(
            {
                "method": "Adaptive PCA-CVA",
                "method_type": "adaptive_pca",
                "threshold": "otsu",
                "adaptive_params": {
                    "threshold_value": None,
                    "otsu_scale": otsu_scale,
                    "opening_kernel": opening,
                    "closing_kernel": closing,
                    "min_area": min_area,
                },
            }
        )
    for threshold_method, radiometric_normalization in itertools.product(
        ["otsu", "adaptive", "kimura", "sauvola"],
        ["none", "histogram_match", "quantile"],
    ):
        rows.append(
            {
                "method": "Adaptive PCA-CVA",
                "method_type": "adaptive_pca",
                "threshold": threshold_method,
                "adaptive_params": {
                    "threshold_value": None,
                    "threshold_method": threshold_method,
                    "radiometric_normalization": radiometric_normalization,
                    "otsu_scale": 0.9781,
                    "adaptive_block_size": 35,
                    "adaptive_c": -2.0,
                    "kimura_k": 0.35,
                    "sauvola_k": 0.20,
                    "opening_kernel": 5,
                    "closing_kernel": 5,
                    "min_area": min_areas[-1],
                    "max_aspect_ratio": 8.0,
                    "min_solidity": 0.2,
                    "min_extent": 0.08,
                },
            }
        )
    return rows


def build_monte_carlo_candidates(image_area: int, trials: int, seed: int) -> list[dict]:
    """Добавляет случайный поиск поверх сетки параметров.

    Метод Монте-Карло полезен здесь не как замена сетке, а как способ проверить
    промежуточные комбинации локальных порогов, min_area, sigma и параметров
    морфологии.
    """
    if trials <= 0:
        return []
    rng = random.Random(seed)
    score_names = [method for method in COMPARISON_METHODS if method != "Adaptive PCA-CVA"]
    min_area_low = max(8, int(image_area * 0.0004))
    min_area_high = max(32, int(image_area * 0.004))
    candidates: list[dict] = []
    for trial in range(trials):
        method_family = rng.choices(
            ["classical", "adaptive_pca"],
            weights=[0.70, 0.30],
            k=1,
        )[0]
        threshold = rng.choice(["otsu", "otsu", "adaptive", "kmeans"])
        common = {
            "threshold": threshold,
            "threshold_scale": round(rng.uniform(0.65, 1.25), 3),
            "sigma": rng.choice([None, "auto", 0.7, 1.0, 1.5, 2.0]),
            "postprocess": "area",
            "median_kernel": rng.choice([1, 3, 5]),
            "morph_kernel": rng.choice([1, 3, 5]),
            "min_area": int(rng.randint(min_area_low, min_area_high)),
            "adaptive_block_size": rng.choice([15, 25, 35, 51, 75]),
            "adaptive_c": round(rng.uniform(-6.0, 3.0), 2),
        }
        if method_family == "classical":
            score_name = rng.choice(score_names)
            candidates.append(
                {
                    "method": score_name,
                    "method_type": "classical",
                    "score_name": score_name,
                    **common,
                }
            )
        else:
            threshold_method = rng.choice(["otsu", "adaptive", "kimura", "sauvola"])
            candidates.append(
                {
                    "method": "Adaptive PCA-CVA",
                    "method_type": "adaptive_pca",
                    "threshold": threshold_method,
                    "adaptive_params": {
                        "threshold_value": None,
                        "threshold_method": threshold_method,
                        "radiometric_normalization": rng.choice(["none", "histogram_match", "quantile"]),
                        "otsu_scale": round(rng.uniform(0.55, 1.25), 3),
                        "adaptive_block_size": rng.choice([15, 25, 35, 51, 75]),
                        "adaptive_c": round(rng.uniform(-6.0, 2.0), 2),
                        "kimura_k": round(rng.uniform(0.15, 0.65), 3),
                        "sauvola_k": round(rng.uniform(0.10, 0.40), 3),
                        "opening_kernel": rng.choice([1, 3, 5]),
                        "closing_kernel": rng.choice([1, 3, 5, 7]),
                        "min_area": int(rng.randint(min_area_low, min_area_high)),
                        "max_aspect_ratio": rng.choice([None, 6.0, 8.0, 12.0]),
                        "min_solidity": rng.choice([None, 0.15, 0.2, 0.3]),
                        "min_extent": rng.choice([None, 0.05, 0.08, 0.12]),
                    },
                }
            )
        candidates[-1]["search_source"] = "monte_carlo"
        candidates[-1]["mc_trial"] = trial
    return candidates


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def analyze_dataset(
    dataset_path: Path,
    config: dict,
    results_dir: Path,
    max_tune_samples: int,
    max_eval_samples: int,
    seed: int,
    monte_carlo_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Подбирает параметры на train/val и оценивает лучшие варианты на test."""
    tune_split, eval_split = _dataset_splits(dataset_path)
    tune_pairs = _load_pairs(dataset_path, tune_split, max_tune_samples, seed)
    eval_pairs = _load_pairs(dataset_path, eval_split, max_eval_samples, seed + 1)
    if not tune_pairs or not eval_pairs:
        raise RuntimeError(f"Нет размеченных пар для {dataset_path}")
    image_area = int(tune_pairs[0]["label"].shape[0] * tune_pairs[0]["label"].shape[1])
    grid = build_parameter_grid(image_area, config)
    for row in grid:
        row.setdefault("search_source", "grid")
        row.setdefault("mc_trial", None)
    dataset_seed = seed + sum(ord(char) for char in dataset_path.name)
    grid.extend(build_monte_carlo_candidates(image_area, trials=monte_carlo_trials, seed=dataset_seed))

    trial_rows = []
    best_by_method: dict[str, dict] = {}
    for trial_index, params in enumerate(grid):
        method = _method_from_params(params, config)
        metrics, _ = _score_method(method, tune_pairs)
        row = {
            "dataset": dataset_path.name,
            "trial_index": trial_index,
            "tune_split": tune_split,
            "eval_split": eval_split,
            **{k: (_safe_json(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()},
            **{f"tune_{k}": v for k, v in metrics.items()},
        }
        trial_rows.append(row)
        current = best_by_method.get(params["method"])
        key = (metrics["f1"], metrics["precision"], -metrics["time_ms"])
        if current is None or key > current["key"]:
            best_by_method[params["method"]] = {"key": key, "params": params, "tune_metrics": metrics}

    summary_rows = []
    pair_metric_frames = []
    for method_name, payload in best_by_method.items():
        method = _method_from_params(payload["params"], config)
        eval_metrics, pair_df = _score_method(method, eval_pairs)
        pair_df.insert(0, "method", method_name)
        pair_df.insert(0, "dataset", dataset_path.name)
        pair_metric_frames.append(pair_df)
        summary_rows.append(
            {
                "dataset": dataset_path.name,
                "method": method_name,
                "tune_split": tune_split,
                "eval_split": eval_split,
                "best_params": _safe_json(payload["params"]),
                **{f"tune_{k}": v for k, v in payload["tune_metrics"].items()},
                **eval_metrics,
            }
        )

    trials = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(summary_rows)
    pair_metrics = pd.concat(pair_metric_frames, ignore_index=True) if pair_metric_frames else pd.DataFrame()
    dataset_dir = results_dir / dataset_path.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(dataset_dir / "parameter_trials.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(dataset_dir / "parameter_summary.csv", index=False, encoding="utf-8-sig")
    pair_metrics.to_csv(dataset_dir / "pair_metrics.csv", index=False, encoding="utf-8-sig")
    quality_rows = [
        {**pair.get("quality_report", {}), "split": split}
        for split, pairs in ((tune_split, tune_pairs), (eval_split, eval_pairs))
        for pair in pairs
        if pair.get("quality_report")
    ]
    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(dataset_dir / "input_quality_report.csv", index=False, encoding="utf-8-sig")
    _save_parameter_plots(trials, dataset_dir)
    _save_best_yaml(summary, dataset_dir / "best_params.yaml")
    return trials, summary, pair_metrics


def _save_parameter_plots(trials: pd.DataFrame, output_dir: Path) -> None:
    """Строит графики зависимости tune F1 от ключевых параметров."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "sigma": "sigma сглаживания, пикс.",
        "threshold": "тип порога",
        "threshold_scale": "масштаб порога, доли ед.",
        "adaptive_block_size": "размер блока локального порога, пикс.",
    }
    for column in ["sigma", "threshold", "threshold_scale", "adaptive_block_size"]:
        if column not in trials.columns or trials[column].dropna().empty:
            continue
        grouped = trials.groupby(column, dropna=False)["tune_f1"].mean().reset_index()
        grouped = grouped.assign(_sort_key=grouped[column].astype(str)).sort_values("_sort_key").drop(columns="_sort_key")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(grouped[column].astype(str), grouped["tune_f1"], color="#3b6ea8", edgecolor="white", linewidth=0.6)
        label = labels.get(column, f"{column}, значение")
        ax.set_title(f"Средняя F1 на валидации по параметру: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("F1 на валидации, доли ед.")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(plot_dir / f"f1_by_{column}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def _save_best_yaml(summary: pd.DataFrame, output_path: Path) -> None:
    """Сохраняет лучшие параметры каждого метода в YAML."""
    payload: dict[str, dict[str, object]] = {}
    for _, row in summary.iterrows():
        dataset = str(row["dataset"])
        method = str(row["method"])
        payload.setdefault(dataset, {})
        payload[dataset][method] = {
            "f1": float(row["f1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "params": json.loads(row["best_params"]),
        }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter study для классических и комбинированных методов.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/parameter_study"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-tune-samples", type=int, default=24)
    parser.add_argument("--max-eval-samples", type=int, default=80)
    parser.add_argument("--monte-carlo-trials", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    logger = ExperimentLogger(
        args.results_dir / "runs",
        "parameter_study",
        tags={"stage": "parameter_study", "data_root": args.data_root},
    )
    logger.log_params({"config": config, "args": vars(args)})

    try:
        dataset_paths = [path for path in sorted(args.data_root.iterdir()) if path.is_dir()]
        if args.datasets:
            names = set(args.datasets)
            dataset_paths = [path for path in dataset_paths if path.name in names]
        summaries = []
        trials = []
        pair_metrics = []
        for dataset_path in dataset_paths:
            if not any(_has_split(dataset_path, split) for split in ("train", "val", "test")):
                continue
            dataset_trials, dataset_summary, dataset_pairs = analyze_dataset(
                dataset_path,
                config,
                args.results_dir,
                args.max_tune_samples,
                args.max_eval_samples,
                args.seed,
                args.monte_carlo_trials,
            )
            trials.append(dataset_trials)
            summaries.append(dataset_summary)
            pair_metrics.append(dataset_pairs)
            best_row = dataset_summary.sort_values(["f1", "precision"], ascending=False).iloc[0]
            logger.log_metrics(
                {
                    "dataset": dataset_path.name,
                    "best_method": best_row["method"],
                    "best_f1": float(best_row["f1"]),
                    "best_precision": float(best_row["precision"]),
                    "best_recall": float(best_row["recall"]),
                    "methods": int(dataset_summary["method"].nunique()),
                    "trials": int(len(dataset_trials)),
                },
                step=dataset_path.name,
            )

        if not summaries:
            raise RuntimeError("Не найдено датасетов для исследования параметров.")
        all_trials = pd.concat(trials, ignore_index=True)
        all_summary = pd.concat(summaries, ignore_index=True)
        all_pairs = pd.concat(pair_metrics, ignore_index=True)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        trials_csv = args.results_dir / "all_parameter_trials.csv"
        summary_csv = args.results_dir / "all_parameter_summary.csv"
        pairs_csv = args.results_dir / "all_pair_metrics.csv"
        best_yaml = args.results_dir / "best_params.yaml"
        all_trials.to_csv(trials_csv, index=False, encoding="utf-8-sig")
        all_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        all_pairs.to_csv(pairs_csv, index=False, encoding="utf-8-sig")
        quality_frames = []
        for dataset_path in dataset_paths:
            quality_csv = args.results_dir / dataset_path.name / "input_quality_report.csv"
            if quality_csv.exists():
                frame = pd.read_csv(quality_csv)
                frame.insert(0, "dataset", dataset_path.name)
                quality_frames.append(frame)
        if quality_frames:
            pd.concat(quality_frames, ignore_index=True).to_csv(
                args.results_dir / "all_input_quality_report.csv",
                index=False,
                encoding="utf-8-sig",
            )
        _save_best_yaml(all_summary, best_yaml)
        logger.log_artifacts([trials_csv, summary_csv, pairs_csv, best_yaml])
        logger.finish()
        print(json.dumps({"trials_csv": str(trials_csv.resolve()), "summary_csv": str(summary_csv.resolve()), "pair_metrics_csv": str(pairs_csv.resolve()), "best_yaml": str(best_yaml.resolve()), "run_dir": str(logger.run_dir.resolve())}, ensure_ascii=False, indent=2))
    except Exception as error:
        logger.finish(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
