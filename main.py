"""Точка входа для запуска итогового сравнения методов обнаружения изменений."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел для повторяемого запуска."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str | Path) -> dict:
    """Загружает параметры эксперимента из YAML-файла."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    """Точка входа: запускает финальное сравнение методов и сохраняет CSV."""
    parser = argparse.ArgumentParser(description="Запуск алгоритма обнаружения изменений")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument(
        "--method_set",
        type=str,
        default=None,
        choices=["core", "individual", "research"],
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Запустить сравнение на всех датасетах в ./data со структурой split/A, split/B и split/label.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Создать небольшой synthetic-lab и запустить на нем алгоритм.",
    )
    parser.add_argument("--synthetic_path", type=str, default="./data/synthetic-lab")
    parser.add_argument("--synthetic_train_pairs", type=int, default=24)
    parser.add_argument("--synthetic_val_pairs", type=int, default=12)
    parser.add_argument("--synthetic_test_pairs", type=int, default=24)
    parser.add_argument("--synthetic_image_size", type=int, default=160)
    args = parser.parse_args()

    config = load_config(args.config)

    # Эти override нужны для быстрых разовых запусков без редактирования YAML.
    if args.data_path:
        config["data"]["data_path"] = args.data_path
    if args.max_samples is not None:
        config.setdefault("experiments", {})["max_samples"] = args.max_samples
    if args.results_dir is not None:
        config.setdefault("experiments", {})["results_dir"] = args.results_dir
    if args.method_set is not None:
        config.setdefault("experiments", {})["method_set"] = args.method_set

    set_seed(config.get("seed", 42))

    if args.synthetic:
        from utils.synthetic import create_synthetic_dataset

        synthetic_root = create_synthetic_dataset(
            root=Path(args.synthetic_path),
            train_pairs=args.synthetic_train_pairs,
            val_pairs=args.synthetic_val_pairs,
            test_pairs=args.synthetic_test_pairs,
            image_size=args.synthetic_image_size,
            seed=config.get("seed", 42),
        )
        config.setdefault("data", {})["dataset"] = "synthetic"
        config["data"]["data_path"] = str(synthetic_root)
        config.setdefault("experiments", {})["split"] = "test"

    from experiments.run_comparison import run_comparison_experiment

    if args.all_datasets:
        split = config.get("experiments", {}).get("split", "test")
        data_root = Path("data")
        dataset_paths = [
            path
            for path in sorted(data_root.iterdir())
            if (path / split / "A").is_dir()
            and (path / split / "B").is_dir()
            and (path / split / "label").is_dir()
        ]
        all_rows = []
        all_summary = {}
        base_results_dir = Path(config.get("experiments", {}).get("results_dir", "results"))
        for dataset_path in dataset_paths:
            dataset_config = copy.deepcopy(config)
            dataset_config.setdefault("data", {})["dataset"] = dataset_path.name
            dataset_config["data"]["data_path"] = str(dataset_path)
            dataset_config.setdefault("experiments", {})["results_dir"] = str(base_results_dir / dataset_path.name)
            summary = run_comparison_experiment(dataset_config)
            all_summary[dataset_path.name] = summary
            for method, values in summary.items():
                if method == "artifacts":
                    continue
                row = {"dataset": dataset_path.name, "method": method}
                row.update(values)
                all_rows.append(row)

        if not dataset_paths:
            raise RuntimeError(f"Не найдены датасеты в {data_root.resolve()}")

        import pandas as pd

        base_results_dir.mkdir(parents=True, exist_ok=True)
        aggregate_csv = base_results_dir / "all_datasets_comparison.csv"
        pd.DataFrame(all_rows).to_csv(aggregate_csv, index=False, encoding="utf-8-sig")
        print(json.dumps({"aggregate_csv": str(aggregate_csv.resolve())}, indent=2, ensure_ascii=False))
        print(
            json.dumps(
                {k: {m: v for m, v in s.items() if m != "artifacts"} for k, s in all_summary.items()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    summary = run_comparison_experiment(config)
    printable_summary = {
        method: values
        for method, values in summary.items()
        if method != "artifacts"
    }
    print(json.dumps(printable_summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary.get("artifacts", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
