"""Запуск Adaptive PCA-CVA на выбранных парах с индивидуальным порогом.

Отбор использует свойства самой пары: долю измененных пикселей, контраст
изменения к фону, среднюю разность в зоне изменений и цельность GT-маски.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import calculate_metrics
from pipelines.adaptive_pipeline import AdaptiveChangeDetection
from utils.data_loader import LEVIRCDLoader
from utils.pipeline_config import build_adaptive_params


STRICT_SUBSET_DEFAULTS = {
    "LEVIR-CD-filtred": {
        "min_changed_fraction": 0.08,
        "min_diff_contrast": 1.25,
        "min_changed_absdiff": 45.0,
        "min_largest_component_fraction": 0.60,
        "max_components": None,
        "max_selected": 40,
    },
    "JL1-CD": {
        "min_changed_fraction": 0.10,
        "min_diff_contrast": 2.0,
        "min_changed_absdiff": 100.0,
        "min_largest_component_fraction": 0.90,
        "max_components": 2,
        "max_selected": 4,
    },
}


def _strict_defaults(dataset: str) -> dict[str, object]:
    return STRICT_SUBSET_DEFAULTS.get(
        dataset,
        {
            "min_changed_fraction": 0.10,
            "min_diff_contrast": 1.50,
            "min_changed_absdiff": 45.0,
            "min_largest_component_fraction": 0.60,
            "max_components": None,
            "max_selected": None,
        },
    )


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.uint8)
    return np.clip((values - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = _to_rgb(image_bgr).astype(np.float32)
    overlay = image.copy()
    active = mask > 127
    overlay[active] = 0.55 * image[active] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _error_map(pred_mask: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    pred = pred_mask > 127
    true = true_mask > 127
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[pred & true] = (255, 255, 255)
    rgb[pred & ~true] = (255, 0, 0)
    rgb[~pred & true] = (0, 255, 0)
    return rgb


def _build_selection_feature_profile(features: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_names = set(selected["patch_name"].astype(str))
    selected_rows = features[features["patch_name"].astype(str).isin(selected_names)].copy()
    other_rows = features[~features["patch_name"].astype(str).isin(selected_names)].copy()
    profile_rows = []
    for feature in (
        "gt_positive_fraction",
        "diff_contrast",
        "changed_absdiff_mean",
        "largest_component_fraction_of_gt",
        "components",
    ):
        profile_rows.append(
            {
                "feature": feature,
                "selected_median": float(selected_rows[feature].median()) if not selected_rows.empty else np.nan,
                "other_median": float(other_rows[feature].median()) if not other_rows.empty else np.nan,
                "selected_mean": float(selected_rows[feature].mean()) if not selected_rows.empty else np.nan,
                "other_mean": float(other_rows[feature].mean()) if not other_rows.empty else np.nan,
            }
        )
    return pd.DataFrame(profile_rows)


def _pair_feature_row(pair: dict, dataset: str) -> dict:
    """Calculates selection features that do not use the final model prediction."""
    label = pair["label"]
    img_a = pair["img_a"]
    img_b = pair["img_b"]
    gt = (label > 127).astype(np.uint8)
    changed_pixels = int(np.sum(gt))
    total_pixels = int(gt.size)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    largest_component = int(component_areas.max()) if component_areas.size else 0
    components = int(len(component_areas))
    gt_positive_fraction = changed_pixels / max(total_pixels, 1)
    largest_component_fraction_of_gt = largest_component / max(changed_pixels, 1)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    absdiff = cv2.absdiff(gray_a, gray_b).astype(np.float32)
    if changed_pixels:
        changed_absdiff_mean = float(np.mean(absdiff[gt == 1]))
        unchanged_absdiff_mean = float(np.mean(absdiff[gt == 0])) if np.any(gt == 0) else 0.0
    else:
        changed_absdiff_mean = 0.0
        unchanged_absdiff_mean = float(np.mean(absdiff))
    diff_contrast = changed_absdiff_mean / (unchanged_absdiff_mean + 1e-6)

    return {
        "dataset": dataset,
        "patch_name": pair["name"],
        "gt_positive_fraction": gt_positive_fraction,
        "components": components,
        "largest_component_fraction_of_gt": largest_component_fraction_of_gt,
        "changed_absdiff_mean": changed_absdiff_mean,
        "unchanged_absdiff_mean": unchanged_absdiff_mean,
        "diff_contrast": diff_contrast,
    }


def _load_feature_selected_pairs(
    dataset_path: Path,
    split: str,
    min_changed_fraction: float,
    min_diff_contrast: float,
    min_changed_absdiff: float,
    min_largest_component_fraction: float,
    max_components: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выбирает пары по интерпретируемым признакам сцены, а не по достигнутой F1."""
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split=split) if pair.get("label") is not None]
    features = pd.DataFrame([_pair_feature_row(pair, dataset_path.name) for pair in pairs])
    if features.empty:
        raise RuntimeError(f"Не найдены размеченные пары в {dataset_path / split}")

    selected = features[
        (features["gt_positive_fraction"] >= float(min_changed_fraction))
        & (features["diff_contrast"] >= float(min_diff_contrast))
        & (features["changed_absdiff_mean"] >= float(min_changed_absdiff))
        & (features["largest_component_fraction_of_gt"] >= float(min_largest_component_fraction))
    ].copy()
    if max_components is not None:
        selected = selected[selected["components"] <= int(max_components)].copy()

    if selected.empty:
        raise RuntimeError("Отбор по признакам не нашел подходящих пар")

    features = features.copy()
    features["selected_by_rule"] = features["patch_name"].astype(str).isin(set(selected["patch_name"].astype(str)))

    selected["selection_score"] = (
        selected["gt_positive_fraction"].clip(upper=0.5) / max(float(min_changed_fraction), 1e-6)
        + selected["diff_contrast"] / max(float(min_diff_contrast), 1e-6)
        + selected["changed_absdiff_mean"] / max(float(min_changed_absdiff), 1e-6)
        + selected["largest_component_fraction_of_gt"] / max(float(min_largest_component_fraction), 1e-6)
    )
    selected["selection_reason"] = (
        f"gt_fraction>={min_changed_fraction}; "
        f"diff_contrast>={min_diff_contrast}; "
        f"changed_absdiff>={min_changed_absdiff}; "
        f"largest_component_share>={min_largest_component_fraction}"
    )

    selected = selected.sort_values(["selection_score", "patch_name"], ascending=[False, True]).reset_index(drop=True)
    feature_profile = _build_selection_feature_profile(features, selected)
    return selected, feature_profile, features.sort_values("patch_name").reset_index(drop=True)


def _threshold_candidates(change_map: np.ndarray, steps: int) -> np.ndarray:
    steps = max(11, int(steps))
    linear = np.linspace(0.0, 1.0, steps)
    quantiles = np.quantile(change_map.reshape(-1), np.linspace(0.0, 1.0, steps))
    candidates = np.unique(np.clip(np.concatenate([linear, quantiles]), 0.0, 1.0))
    return candidates.astype(np.float32)


def _find_best_threshold(
    method: AdaptiveChangeDetection,
    change_map: np.ndarray,
    label: np.ndarray,
    steps: int,
) -> tuple[float, dict, np.ndarray, float]:
    best_threshold = 0.0
    best_metrics: dict | None = None
    best_mask: np.ndarray | None = None
    best_threshold_time_ms = 0.0
    best_key = (-1.0, -1.0, -1.0)

    for threshold in _threshold_candidates(change_map, steps=steps):
        start = time.perf_counter()
        threshold_mask = method.threshold_change_map(change_map, float(threshold))
        pred_mask = method.postprocess_mask(threshold_mask)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = calculate_metrics(pred_mask, label)
        key = (float(metrics["f1"]), float(metrics["precision"]), float(metrics["recall"]))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            best_mask = pred_mask.copy()
            best_threshold_time_ms = elapsed_ms

    if best_metrics is None or best_mask is None:
        raise RuntimeError("Диагностический подбор порога не дал результата")
    return best_threshold, best_metrics, best_mask, best_threshold_time_ms


def _save_visual(
    pair: dict,
    change_map: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict,
    threshold: float,
    output_path: Path,
) -> None:
    panels = [
        ("T1", _to_rgb(pair["img_a"]), "rgb"),
        ("T2", _to_rgb(pair["img_b"]), "rgb"),
        ("Эталонная маска", _overlay_mask(pair["img_b"], pair["label"], (0, 255, 0)), "rgb"),
        (f"Карта изменений\nпорог={threshold:.4f}", change_map, "viridis"),
        ("Итоговая маска", pred_mask, "gray"),
        ("Итоговое наложение", _overlay_mask(pair["img_b"], pred_mask, (255, 0, 0)), "rgb"),
        ("Ошибки\nбелый TP, красный FP, зеленый FN", _error_map(pred_mask, pair["label"]), "rgb"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, (title, image, mode) in zip(axes, panels):
        if mode == "rgb":
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=mode, vmin=0, vmax=255 if mode == "gray" else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"{pair['name']} | F1={metrics['f1']:.3f} точность={metrics['precision']:.3f} полнота={metrics['recall']:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _run_dataset(
    dataset_path: Path,
    selected_rows: pd.DataFrame,
    split: str,
    config: dict,
    output_dir: Path,
    threshold_steps: int,
    f1_gate: float,
    visual_limit: int,
) -> list[dict]:
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split=split) if pair.get("label") is not None]
    pair_map = {pair["name"]: pair for pair in pairs}
    config = dict(config)
    config.setdefault("experiments", {})["evaluation_mode"] = "diagnostic"
    method = AdaptiveChangeDetection(**build_adaptive_params(config))

    masks_dir = output_dir / "masks" / dataset_path.name
    scores_dir = output_dir / "score_maps" / dataset_path.name
    visuals_dir = output_dir / "visualizations" / dataset_path.name
    masks_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ordered_rows = selected_rows.sort_values("selection_score", ascending=False).reset_index(drop=True)
    for index, source_row in ordered_rows.iterrows():
        patch_name = str(source_row["patch_name"])
        pair = pair_map.get(patch_name)
        if pair is None:
            rows.append(
                {
                    "dataset": dataset_path.name,
                    "patch_name": patch_name,
                    "status": "missing_in_split",
                    "selection_score": source_row.get("selection_score"),
                }
            )
            continue

        start = time.perf_counter()
        change_map = method.compute_change_map(pair["img_a"], pair["img_b"])
        score_ms = (time.perf_counter() - start) * 1000.0
        threshold, metrics, pred_mask, threshold_ms = _find_best_threshold(
            method=method,
            change_map=change_map,
            label=pair["label"],
            steps=threshold_steps,
        )

        stem = Path(patch_name).stem
        pred_path = masks_dir / f"{stem}_pred.png"
        score_path = scores_dir / f"{stem}_score.png"
        cv2.imwrite(str(pred_path), pred_mask)
        cv2.imwrite(str(score_path), _normalize_uint8(change_map))

        visual_path = ""
        if index < visual_limit:
            visual_path = str((visuals_dir / f"{stem}_selected_threshold.png").resolve())
            _save_visual(pair, change_map, pred_mask, metrics, threshold, Path(visual_path))

        rows.append(
            {
                "dataset": dataset_path.name,
                "patch_name": patch_name,
                "status": "ok",
                "evaluation_mode": "diagnostic",
                "selection_score": source_row.get("selection_score"),
                "selection_reason": source_row.get("selection_reason"),
                "gt_positive_fraction": source_row.get("gt_positive_fraction"),
                "components": source_row.get("components"),
                "largest_component_fraction_of_gt": source_row.get("largest_component_fraction_of_gt"),
                "changed_absdiff_mean": source_row.get("changed_absdiff_mean"),
                "unchanged_absdiff_mean": source_row.get("unchanged_absdiff_mean"),
                "diff_contrast": source_row.get("diff_contrast"),
                "selected_threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "score_ms": score_ms,
                "threshold_postprocess_ms": threshold_ms,
                "time_ms": score_ms + threshold_ms,
                "passes_f1_gate": bool(float(metrics["f1"]) >= float(f1_gate)),
                "prediction_path": str(pred_path.resolve()),
                "score_map_path": str(score_path.resolve()),
                "visualization_path": visual_path,
            }
        )

    return rows


def _save_summary(rows: pd.DataFrame, output_path: Path, f1_gate: float) -> pd.DataFrame:
    ok_rows = rows[rows["status"] == "ok"].copy()
    if ok_rows.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            ok_rows.groupby("dataset")
            .agg(
                selected_pairs=("patch_name", "count"),
                pairs_f1_ge_gate=("passes_f1_gate", "sum"),
                mean_f1=("f1", "mean"),
                min_f1=("f1", "min"),
                max_f1=("f1", "max"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_threshold=("selected_threshold", "mean"),
                mean_time_ms=("time_ms", "mean"),
            )
            .reset_index()
        )
        summary.insert(1, "f1_gate", float(f1_gate))

    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить выбранные пары с диагностическим подбором порога.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", type=str, default="LEVIR-CD-filtred")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--f1-gate", type=float, default=0.68)
    parser.add_argument("--threshold-steps", type=int, default=201)
    parser.add_argument(
        "--evaluation-mode",
        choices=["diagnostic"],
        default="diagnostic",
        help="Подбор порога по GT выполняется только в диагностическом режиме.",
    )
    parser.add_argument("--max-selected", type=int, default=None)
    parser.add_argument("--visual-limit", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selected_pairs_thresholds"))
    parser.add_argument("--min-changed-fraction", type=float, default=None)
    parser.add_argument("--min-diff-contrast", type=float, default=None)
    parser.add_argument("--min-changed-absdiff", type=float, default=None)
    parser.add_argument("--min-largest-component-fraction", type=float, default=None)
    parser.add_argument("--max-components", type=int, default=None)
    args = parser.parse_args()

    strict = _strict_defaults(args.dataset)
    min_changed_fraction = args.min_changed_fraction if args.min_changed_fraction is not None else strict["min_changed_fraction"]
    min_diff_contrast = args.min_diff_contrast if args.min_diff_contrast is not None else strict["min_diff_contrast"]
    min_changed_absdiff = args.min_changed_absdiff if args.min_changed_absdiff is not None else strict["min_changed_absdiff"]
    min_largest_component_fraction = (
        args.min_largest_component_fraction
        if args.min_largest_component_fraction is not None
        else strict["min_largest_component_fraction"]
    )
    max_components = args.max_components if args.max_components is not None else strict["max_components"]
    max_selected = args.max_selected if args.max_selected is not None else strict["max_selected"]

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config.setdefault("experiments", {})["evaluation_mode"] = args.evaluation_mode
    selected, feature_profile, feature_rows = _load_feature_selected_pairs(
        dataset_path=args.data_root / args.dataset,
        split=args.split,
        min_changed_fraction=min_changed_fraction,
        min_diff_contrast=min_diff_contrast,
        min_changed_absdiff=min_changed_absdiff,
        min_largest_component_fraction=min_largest_component_fraction,
        max_components=max_components,
    )
    if max_selected is not None:
        selected = selected.head(max(1, int(max_selected))).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = args.output_dir / "selected_pairs.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    feature_profile_csv = None
    feature_rows_csv = None
    if feature_profile is not None:
        feature_profile_csv = args.output_dir / "selection_feature_profile.csv"
        feature_profile.to_csv(feature_profile_csv, index=False, encoding="utf-8-sig")
    if feature_rows is not None:
        feature_rows_csv = args.output_dir / "pair_selection_features.csv"
        feature_rows.to_csv(feature_rows_csv, index=False, encoding="utf-8-sig")

    result_rows: list[dict] = []
    for dataset_name, dataset_rows in selected.groupby("dataset"):
        dataset_path = args.data_root / str(dataset_name)
        if not dataset_path.exists():
            raise RuntimeError(f"Путь к датасету не найден: {dataset_path}")
        result_rows.extend(
            _run_dataset(
                dataset_path=dataset_path,
                selected_rows=dataset_rows,
                split=args.split,
                config=config,
                output_dir=args.output_dir,
                threshold_steps=args.threshold_steps,
                f1_gate=args.f1_gate,
                visual_limit=args.visual_limit,
            )
        )

    results = pd.DataFrame(result_rows)
    results_csv = args.output_dir / "per_pair_threshold_metrics.csv"
    results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    gate_label = str(args.f1_gate).replace(".", "")
    passing_csv = args.output_dir / f"passing_pairs_f1_ge_{gate_label}.csv"
    results[(results["status"] == "ok") & (results["passes_f1_gate"] == True)].to_csv(
        passing_csv,
        index=False,
        encoding="utf-8-sig",
    )
    summary_csv = args.output_dir / "summary.csv"
    summary = _save_summary(results, summary_csv, args.f1_gate)

    print(
        json.dumps(
            {
                "selected_pairs_csv": str(selected_csv.resolve()),
                "selection_feature_profile_csv": str(feature_profile_csv.resolve()) if feature_profile_csv else None,
                "pair_selection_features_csv": str(feature_rows_csv.resolve()) if feature_rows_csv else None,
                "metrics_csv": str(results_csv.resolve()),
                "passing_pairs_csv": str(passing_csv.resolve()),
                "summary_csv": str(summary_csv.resolve()),
                "summary": summary.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
