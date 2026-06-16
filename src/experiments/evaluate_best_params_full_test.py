"""Оценка подобранных на валидации параметров на полном test-разбиении."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.parameter_analyzer import _method_from_params, _score_method
from utils.data_loader import LEVIRCDLoader


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_pairs(dataset_path: Path, split: str, max_pairs: int | None, seed: int) -> list[dict]:
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split=split) if pair.get("label") is not None]
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    return [pairs[index] for index in sorted(indices[:max_pairs])]


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_best_params(summary_csv: Path, datasets: list[str] | None) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    if datasets:
        summary = summary[summary["dataset"].isin(set(datasets))].copy()
    if summary.empty:
        raise RuntimeError(f"Не найдены строки с лучшими параметрами в {summary_csv}")
    return summary.sort_values(["dataset", "method"]).reset_index(drop=True)


def evaluate_full_test(
    config: dict,
    data_root: Path,
    best_summary: pd.DataFrame,
    output_dir: Path,
    split: str,
    max_samples: int | None,
    seed: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []

    for dataset, dataset_rows in best_summary.groupby("dataset"):
        dataset_path = data_root / str(dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        pairs = _load_pairs(dataset_path, split=split, max_pairs=max_samples, seed=seed + len(summary_rows))
        if not pairs:
            raise RuntimeError(f"Не найдены размеченные пары для {dataset_path / split}")

        quality_rows = [pair.get("quality_report", {}) for pair in pairs if pair.get("quality_report")]
        if quality_rows:
            quality_frame = pd.DataFrame(quality_rows)
            quality_frame.insert(0, "dataset", dataset)
            quality_frame.insert(1, "split", split)
            quality_frames.append(quality_frame)

        dataset_dir = output_dir / str(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_summary_rows = []
        dataset_pair_frames = []
        for _, row in dataset_rows.iterrows():
            method_name = str(row["method"])
            params = json.loads(row["best_params"])
            method = _method_from_params(params, config)
            metrics, pair_df = _score_method(method, pairs)
            pair_df.insert(0, "method", method_name)
            pair_df.insert(0, "dataset", dataset)
            dataset_pair_frames.append(pair_df)
            dataset_summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method_name,
                    "selection_source": "validation_best_params",
                    "test_split": split,
                    "best_params": _safe_json(params),
                    "tune_f1": row.get("tune_f1"),
                    "tune_precision": row.get("tune_precision"),
                    "tune_recall": row.get("tune_recall"),
                    **metrics,
                }
            )

        dataset_summary = pd.DataFrame(dataset_summary_rows).sort_values(["f1", "precision"], ascending=False)
        dataset_pairs = pd.concat(dataset_pair_frames, ignore_index=True)
        dataset_summary.to_csv(dataset_dir / "full_test_best_params_summary.csv", index=False, encoding="utf-8-sig")
        dataset_pairs.to_csv(dataset_dir / "full_test_best_params_pair_metrics.csv", index=False, encoding="utf-8-sig")
        summary_rows.extend(dataset_summary_rows)
        pair_frames.append(dataset_pairs)

    full_summary = pd.DataFrame(summary_rows).sort_values(["dataset", "f1", "precision"], ascending=[True, False, False])
    full_pairs = pd.concat(pair_frames, ignore_index=True)
    summary_csv = output_dir / "full_test_best_params_summary.csv"
    pair_metrics_csv = output_dir / "full_test_best_params_pair_metrics.csv"
    full_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    full_pairs.to_csv(pair_metrics_csv, index=False, encoding="utf-8-sig")

    artifacts = {
        "summary_csv": str(summary_csv.resolve()),
        "pair_metrics_csv": str(pair_metrics_csv.resolve()),
    }
    if quality_frames:
        quality_csv = output_dir / "input_quality_report.csv"
        pd.concat(quality_frames, ignore_index=True).to_csv(quality_csv, index=False, encoding="utf-8-sig")
        artifacts["input_quality_report_csv"] = str(quality_csv.resolve())

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценить подобранные на валидации параметры на полном test-разбиении.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--best-summary-csv", type=Path, default=Path("results/full_protocol/parameter_study/all_parameter_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_protocol"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    best_summary = _load_best_params(args.best_summary_csv, args.datasets)
    artifacts = evaluate_full_test(
        config=config,
        data_root=args.data_root,
        best_summary=best_summary,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
