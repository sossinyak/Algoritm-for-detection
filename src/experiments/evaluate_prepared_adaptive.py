"""Оценка Adaptive PCA-CVA на подготовленном наборе патчей."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import calculate_metrics
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from utils.data_loader import LEVIRCDLoader
from utils.pipeline_config import build_adaptive_params


PATCH_SCENE_PATTERN = re.compile(r"^(?P<scene>.+)_y\d+_x\d+\.\w+$")


def _scene_name_from_patch(patch_name: str) -> str:
    """Восстанавливает имя исходной сцены по имени патча."""
    match = PATCH_SCENE_PATTERN.match(patch_name)
    if match is None:
        return Path(patch_name).stem
    return match.group("scene")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _summarize(rows: pd.DataFrame) -> dict:
    """Считает итоговые метрики по общей матрице ошибок."""
    tp = int(rows["tp"].sum())
    tn = int(rows["tn"].sum())
    fp = int(rows["fp"].sum())
    fn = int(rows["fn"].sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {
        "samples": int(len(rows)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "time_ms": float(rows["time_ms"].mean()) if not rows.empty else 0.0,
    }


def evaluate(data_path: Path, split: str, config: dict, max_samples: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Запускает Adaptive PCA-CVA и возвращает per-patch таблицу и summary."""
    loader = LEVIRCDLoader(str(data_path))
    pairs = [pair for pair in loader.load_split(split=split, max_pairs=max_samples) if pair.get("label") is not None]
    if not pairs:
        raise RuntimeError(f"Нет размеченных пар: {data_path / split}")
    method = AdaptiveChangeDetection(**build_adaptive_params(config))
    rows = []
    for pair in tqdm(pairs, desc="Adaptive PCA-CVA", unit="patch"):
        start = time.perf_counter()
        pred = method.process(pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = calculate_metrics(pred, pair["label"])
        rows.append(
            {
                "split": split,
                "patch_name": pair["name"],
                "scene_name": _scene_name_from_patch(pair["name"]),
                **metrics,
                "time_ms": elapsed_ms,
            }
        )
    df = pd.DataFrame(rows)
    return df, _summarize(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценить Adaptive PCA-CVA на подготовленном датасете.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results/adaptive_prepared"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    data_path = args.data_path or Path(config.get("data", {}).get("data_path", "data/LEVIR-CD-filtred"))
    per_patch, summary = evaluate(data_path, args.split, config, args.max_samples)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_patch_csv = args.results_dir / "per_patch_metrics.csv"
    summary_json = args.results_dir / "summary.json"
    per_patch.to_csv(per_patch_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"per_patch_csv": str(per_patch_csv.resolve()), "summary_json": str(summary_json.resolve()), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
