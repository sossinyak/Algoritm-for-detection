from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray, extension: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = extension or path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"Failed to encode image for: {path}")
    encoded.tofile(str(path))
