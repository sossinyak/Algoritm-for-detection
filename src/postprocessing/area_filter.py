"""Фильтрация бинарной маски по геометрии связных компонент."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Приводит маску к формату uint8 со значениями 0 и 255."""
    if mask.dtype == np.uint8:
        return mask.copy()
    values = mask.astype(np.float32, copy=False)
    if float(np.max(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def filter_by_area(mask: np.ndarray, min_area: Optional[int] = None) -> np.ndarray:
    """Удаляет предсказанные компоненты меньше min_area пикселей."""
    return filter_components(mask, min_area=min_area)


def filter_components(
    mask: np.ndarray,
    min_area: Optional[int] = None,
    max_aspect_ratio: Optional[float] = None,
    min_solidity: Optional[float] = None,
    min_extent: Optional[float] = None,
    max_extent: Optional[float] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
) -> np.ndarray:
    """Удаляет компоненты по площади и простым shape-признакам.

    Эти признаки помогают отсечь типичные линейные артефакты промышленных сцен:
    дороги, длинные тени и тонкие шумовые полосы. Значение None отключает
    соответствующее правило.
    """
    src = _as_uint8_mask(mask)
    if (
        (min_area is None or min_area <= 0)
        and max_aspect_ratio is None
        and min_solidity is None
        and min_extent is None
        and max_extent is None
        and min_width is None
        and min_height is None
    ):
        return src

    filtered_mask = np.zeros_like(src)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((src > 127).astype(np.uint8), connectivity=8)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if min_area is not None and min_area > 0 and area < int(min_area):
            continue
        if min_width is not None and width < int(min_width):
            continue
        if min_height is not None and height < int(min_height):
            continue

        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if max_aspect_ratio is not None and aspect_ratio > float(max_aspect_ratio):
            continue

        bbox_area = max(width * height, 1)
        extent = area / bbox_area
        if min_extent is not None and extent < float(min_extent):
            continue
        if max_extent is not None and extent > float(max_extent):
            continue

        if min_solidity is not None:
            component = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area = sum(float(cv2.contourArea(contour)) for contour in contours)
            hull_area = 0.0
            for contour in contours:
                if len(contour) >= 3:
                    hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = contour_area / max(hull_area, 1.0)
            if solidity < float(min_solidity):
                continue

        filtered_mask[labels == label_id] = 255
    return filtered_mask
