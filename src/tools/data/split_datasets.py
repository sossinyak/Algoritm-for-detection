"""Создание воспроизводимого train/val/test split для датасетов A/B/label.

По умолчанию split применяется прямо внутри `data/<dataset>`: скрипт создает
временную папку, собирает там новое разбиение и затем заменяет только папки
`train`, `val`, `test`. Служебные файлы датасета остаются на месте.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PATCH_SCENE_RE = re.compile(r"^(?P<scene>.+)_y\d+_x\d+$")


def _has_triplet(split_dir: Path) -> bool:
    """Проверяет наличие A/B/label внутри split."""
    return all((split_dir / folder).is_dir() for folder in ("A", "B", "label"))


def _collect_samples(dataset_path: Path) -> list[dict]:
    """Собирает все синхронизированные A/B/label пары."""
    samples: list[dict] = []
    for split_dir in sorted(dataset_path.iterdir()):
        if not split_dir.is_dir() or not _has_triplet(split_dir):
            continue
        a_dir = split_dir / "A"
        b_dir = split_dir / "B"
        label_dir = split_dir / "label"
        for a_path in sorted(a_dir.iterdir()):
            if a_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            b_path = b_dir / a_path.name
            label_path = label_dir / a_path.name
            if b_path.exists() and label_path.exists():
                samples.append(
                    {
                        "source_split": split_dir.name,
                        "name": a_path.name,
                        "a_path": a_path,
                        "b_path": b_path,
                        "label_path": label_path,
                    }
                )
    return samples


def _group_key(sample: dict) -> str:
    """Возвращает ключ сцены для группового split."""
    stem = Path(sample["name"]).stem
    match = PATCH_SCENE_RE.match(stem)
    if match:
        return f"{sample['source_split']}::{match.group('scene')}"
    return f"{sample['source_split']}::{stem}"


def _assign_splits(samples: list[dict], ratios: tuple[float, float, float], seed: int) -> list[dict]:
    """Назначает train/val/test по группам сцен, чтобы снизить утечку."""
    if not samples:
        return []
    total = sum(ratios)
    train_ratio, val_ratio, _ = [value / total for value in ratios]
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(_group_key(sample), []).append(sample)

    rng = random.Random(seed)
    ordered_keys = sorted(groups)
    rng.shuffle(ordered_keys)
    n_groups = len(ordered_keys)
    train_end = int(round(n_groups * train_ratio))
    val_end = train_end + int(round(n_groups * val_ratio))
    train_end = min(max(train_end, 1 if n_groups >= 3 else n_groups), n_groups)
    val_end = min(max(val_end, train_end + (1 if n_groups >= 3 else 0)), n_groups)

    assigned: list[dict] = []
    for index, group_key in enumerate(ordered_keys):
        if index < train_end:
            target_split = "train"
        elif index < val_end:
            target_split = "val"
        else:
            target_split = "test"
        for sample in groups[group_key]:
            sample["target_split"] = target_split
            sample["group_key"] = group_key
            assigned.append(sample)
    return assigned


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Создает hardlink/symlink/copy для одного файла."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def split_dataset(
    dataset_path: Path,
    output_root: Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    mode: str = "hardlink",
) -> pd.DataFrame:
    """Создает новый train/val/test split и возвращает manifest."""
    samples = _collect_samples(dataset_path)
    assigned = _assign_splits(samples, ratios=ratios, seed=seed)
    name_counts: dict[str, int] = {}
    for sample in assigned:
        name_counts[sample["name"]] = name_counts.get(sample["name"], 0) + 1
    target_dataset = output_root / dataset_path.name
    rows = []
    for sample in assigned:
        split = sample["target_split"]
        name = sample["name"]
        if name_counts[name] > 1:
            name = f"{sample['source_split']}_{name}"
        _link_or_copy(sample["a_path"], target_dataset / split / "A" / name, mode)
        _link_or_copy(sample["b_path"], target_dataset / split / "B" / name, mode)
        _link_or_copy(sample["label_path"], target_dataset / split / "label" / name, mode)
        rows.append(
            {
                "dataset": dataset_path.name,
                "target_split": split,
                "source_split": sample["source_split"],
                "group_key": sample.get("group_key", ""),
                "source_name": sample["name"],
                "name": name,
                "a_path": str(sample["a_path"]),
                "b_path": str(sample["b_path"]),
                "label_path": str(sample["label_path"]),
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        target_dataset.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(target_dataset / "split_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def _replace_split_dirs(source_dataset: Path, target_dataset: Path) -> None:
    """Заменяет в целевом датасете только папки train/val/test.

    Служебные файлы рядом со split-папками, например `patch_metadata.csv`,
    остаются на месте. Это позволяет применять воспроизводимый split прямо
    внутри `data/<dataset>` без потери метаданных.
    """
    target_dataset.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        source_split = source_dataset / split
        target_split = target_dataset / split
        if not source_split.is_dir():
            continue
        if target_split.exists():
            shutil.rmtree(target_split)
        shutil.move(str(source_split), str(target_split))

    source_manifest = source_dataset / "split_manifest.csv"
    if source_manifest.exists():
        shutil.move(str(source_manifest), str(target_dataset / "split_manifest.csv"))


def apply_split_root_to_data(split_root: Path, data_root: Path) -> list[str]:
    """Переносит уже созданные split-датасеты обратно в `data`."""
    applied: list[str] = []
    for source_dataset in sorted(path for path in split_root.iterdir() if path.is_dir()):
        _replace_split_dirs(source_dataset, data_root / source_dataset.name)
        applied.append(source_dataset.name)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать воспроизводимый train/val/test split.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Создать split во временной папке и заменить train/val/test прямо внутри data-root.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    in_place = bool(args.in_place or args.output_root.resolve() == args.data_root.resolve())
    if in_place:
        output_root = args.data_root.parent / f".{args.data_root.name}_split_tmp"
        if output_root.exists():
            shutil.rmtree(output_root)

    dataset_paths = [path for path in sorted(args.data_root.iterdir()) if path.is_dir()]
    if args.datasets:
        names = set(args.datasets)
        dataset_paths = [path for path in dataset_paths if path.name in names]
    if not dataset_paths:
        raise RuntimeError(f"Датасеты не найдены: {args.data_root.resolve()}")

    manifests = []
    for dataset_path in dataset_paths:
        manifest = split_dataset(
            dataset_path,
            output_root=output_root,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
            mode=args.mode,
        )
        if not manifest.empty:
            manifests.append(manifest)
    if not manifests:
        raise RuntimeError("Не найдено ни одной синхронизированной A/B/label пары.")

    summary = pd.concat(manifests, ignore_index=True)
    summary_path = output_root / "split_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    applied = []
    if in_place:
        applied = apply_split_root_to_data(output_root, args.data_root)
        shutil.rmtree(output_root)
        summary_path = args.data_root / "split_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    counts = summary.groupby(["dataset", "target_split"]).size().reset_index(name="samples")
    print(
        json.dumps(
            {
                "summary_csv": str(summary_path.resolve()),
                "in_place": bool(in_place),
                "applied_datasets": applied,
                "counts": counts.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
