"""Статистическая проверка различий между методами по F1 на уровне патчей."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_f1_difference(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Оценивает доверительный интервал разницы F1 через парный bootstrap."""
    baseline_values = np.asarray(baseline_values, dtype=np.float32)
    candidate_values = np.asarray(candidate_values, dtype=np.float32)
    n = min(len(baseline_values), len(candidate_values))
    if n == 0:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_improvement": 0.0, "samples": 0}
    baseline_values = baseline_values[:n]
    candidate_values = candidate_values[:n]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float32)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(candidate_values[indices] - baseline_values[indices]))
    return {
        "delta_mean": float(np.mean(candidate_values - baseline_values)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_improvement": float(np.mean(deltas > 0)),
        "samples": int(n),
    }


def validate_from_pair_metrics(
    pair_metrics_csv: Path,
    baseline_method: str,
    candidate_method: str,
    dataset_name: str | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Читает CSV с колонками dataset/method/patch_name/f1 и сравнивает методы."""
    df = pd.read_csv(pair_metrics_csv)
    required = {"dataset", "method", "patch_name", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {sorted(missing)}")
    if dataset_name is not None:
        df = df[df["dataset"] == dataset_name].copy()
    rows = []
    for dataset, dataset_df in df.groupby("dataset"):
        base = dataset_df[dataset_df["method"] == baseline_method][["patch_name", "f1"]]
        cand = dataset_df[dataset_df["method"] == candidate_method][["patch_name", "f1"]]
        merged = base.merge(cand, on="patch_name", suffixes=("_baseline", "_candidate"))
        stats = bootstrap_f1_difference(
            merged["f1_baseline"].to_numpy(),
            merged["f1_candidate"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({"dataset": dataset, "baseline": baseline_method, "candidate": candidate_method, **stats})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap-проверка улучшения F1.")
    parser.add_argument("--pair-metrics-csv", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("results/statistical_validation.csv"))
    args = parser.parse_args()

    result = validate_from_pair_metrics(
        args.pair_metrics_csv,
        args.baseline,
        args.candidate,
        dataset_name=args.dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"output_csv": str(args.output_csv.resolve()), "rows": result.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
