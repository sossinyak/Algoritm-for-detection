from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from utils.image_io import write_image


def create_synthetic_dataset(
    root: Path,
    train_pairs: int = 24,
    val_pairs: int = 12,
    test_pairs: int = 24,
    image_size: int = 160,
    seed: int = 7,
) -> Path:
    root = root.resolve()
    rng = np.random.default_rng(seed)

    split_counts = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    for split, count in split_counts.items():
        for folder in ("A", "B", "label"):
            folder_path = root / split / folder
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            before, after, mask = _generate_pair(rng, image_size)
            name = f"sample_{index:03d}"
            write_image(root / split / "A" / f"{name}.png", (before * 255).astype(np.uint8))
            write_image(root / split / "B" / f"{name}.png", (after * 255).astype(np.uint8))
            write_image(root / split / "label" / f"{name}.png", (mask * 255).astype(np.uint8))

    return root


def _generate_pair(rng: np.random.Generator, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = width = image_size
    background = np.zeros((height, width, 3), dtype=np.float32)

    xv, yv = np.meshgrid(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        np.linspace(0.0, 1.0, height, dtype=np.float32),
    )
    background[..., 0] = 0.18 + 0.22 * xv
    background[..., 1] = 0.28 + 0.18 * yv
    background[..., 2] = 0.22 + 0.10 * (1.0 - xv)

    before = background.copy()
    after = background.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(10):
        x1, y1, x2, y2 = _random_rectangle(rng, width, height, 10, 24)
        color = _building_color(rng)
        cv2.rectangle(before, (x1, y1), (x2, y2), color.tolist(), thickness=-1)
        cv2.rectangle(after, (x1, y1), (x2, y2), color.tolist(), thickness=-1)

    for _ in range(4):
        x1, y1, x2, y2 = _random_rectangle(rng, width, height, 12, 28)
        color = _building_color(rng)
        cv2.rectangle(after, (x1, y1), (x2, y2), color.tolist(), thickness=-1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)

    for _ in range(2):
        x1, y1, x2, y2 = _random_rectangle(rng, width, height, 12, 26)
        color = _building_color(rng)
        cv2.rectangle(before, (x1, y1), (x2, y2), color.tolist(), thickness=-1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)

    noise_before = rng.normal(0.0, 0.015, size=before.shape).astype(np.float32)
    noise_after = rng.normal(0.0, 0.015, size=after.shape).astype(np.float32)
    before = np.clip(before + noise_before, 0.0, 1.0)
    after = np.clip(after + noise_after, 0.0, 1.0)

    before = cv2.GaussianBlur(before, (3, 3), 0.5)
    after = cv2.GaussianBlur(after, (3, 3), 0.8)
    return before.astype(np.float32), after.astype(np.float32), mask.astype(np.uint8)


def _random_rectangle(
    rng: np.random.Generator,
    width: int,
    height: int,
    min_side: int,
    max_side: int,
) -> tuple[int, int, int, int]:
    rect_w = int(rng.integers(min_side, max_side))
    rect_h = int(rng.integers(min_side, max_side))
    x1 = int(rng.integers(0, max(width - rect_w - 1, 1)))
    y1 = int(rng.integers(0, max(height - rect_h - 1, 1)))
    x2 = x1 + rect_w
    y2 = y1 + rect_h
    return x1, y1, x2, y2


def _building_color(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(0.65, 0.88),
            rng.uniform(0.62, 0.86),
            rng.uniform(0.58, 0.82),
        ],
        dtype=np.float32,
    )
