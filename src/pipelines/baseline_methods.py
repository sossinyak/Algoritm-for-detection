"""Классические методы итогового сравнительного протокола."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from pipelines.method_metadata import COMPARISON_METHODS
from postprocessing.area_filter import filter_by_area


ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _odd_kernel(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def estimate_noise_sigma(img1: np.ndarray, img2: np.ndarray | None = None) -> float:
    """Estimate a conservative Gaussian smoothing sigma from the high-frequency residual."""

    def _one(image: np.ndarray) -> float:
        gray = _to_gray(image).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray - blur
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigma = 1.4826 * mad / 255.0
        if sigma < 0.01:
            return 0.0
        if sigma < 0.025:
            return 0.5
        if sigma < 0.05:
            return 0.8
        if sigma < 0.09:
            return 1.2
        return 1.6

    if img2 is None:
        return _one(img1)
    return float(np.clip((_one(img1) + _one(img2)) / 2.0, 0.5, 3.0))


def _apply_gaussian_auto(image: np.ndarray, sigma: float | str | None, other: np.ndarray | None = None) -> np.ndarray:
    if sigma is None or sigma == 0:
        return image
    sigma_value = estimate_noise_sigma(image, other) if sigma == "auto" else float(sigma)
    if sigma_value <= 0:
        return image
    kernel = _odd_kernel(int(round(sigma_value * 6 + 1)), minimum=3)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma_value, sigmaY=sigma_value)


def _otsu_mask(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    score = _normalize_uint8(values)
    threshold, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (score > np.clip(threshold * float(scale), 0.0, 255.0)).astype(np.uint8) * 255


def _triangle_mask(values: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(_normalize_uint8(values), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return mask


def _kmeans_mask(values: np.ndarray) -> np.ndarray:
    score = _normalize_uint8(values)
    samples = score.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels, centers = cv2.kmeans(samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    changed_cluster = int(np.argmax(centers.reshape(-1)))
    return (labels.reshape(score.shape) == changed_cluster).astype(np.uint8) * 255


def _adaptive_mask(values: np.ndarray, block_size: int = 35, c_value: float = -2.0) -> np.ndarray:
    score = _normalize_uint8(values)
    block_size = _odd_kernel(block_size, minimum=3)
    return cv2.adaptiveThreshold(
        score,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        float(c_value),
    )


def _gray_absdiff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.abs(_to_gray(img2).astype(np.float32) - _to_gray(img1).astype(np.float32))


def _gray_log_ratio(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    gray1 = _to_gray(img1).astype(np.float32) + 1.0
    gray2 = _to_gray(img2).astype(np.float32) + 1.0
    return np.abs(np.log(gray2 / gray1))


def _rgb_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.ndim != 3 or img2.ndim != 3:
        return _gray_absdiff(img1, img2)
    diff = img2.astype(np.float32) - img1.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=2))


def _pca_projected_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.ndim != 3 or img2.ndim != 3:
        return _gray_absdiff(img1, img2)
    height, width = img1.shape[:2]
    pixels1 = img1.reshape(-1, 3).astype(np.float32) / 255.0
    pixels2 = img2.reshape(-1, 3).astype(np.float32) / 255.0
    mean, eigenvectors = cv2.PCACompute(np.vstack([pixels1, pixels2]), mean=None, maxComponents=2)
    projected1 = cv2.PCAProject(pixels1, mean, eigenvectors)
    projected2 = cv2.PCAProject(pixels2, mean, eigenvectors)
    diff = projected2 - projected1
    return np.sqrt(np.sum(diff * diff, axis=1)).reshape(height, width)


def classical_score_functions() -> "OrderedDict[str, ScoreFn]":
    """Return only the four classical methods kept for the final comparison."""

    return OrderedDict(
        [
            ("AbsDiff", _gray_absdiff),
            ("LogRatio", _gray_log_ratio),
            ("RGB-CVA", _rgb_cva),
            ("PCA-CVA", _pca_projected_cva),
        ]
    )


@dataclass
class TunableClassicalPipeline:
    score_fn: ScoreFn
    threshold: str = "otsu"
    threshold_scale: float = 1.0
    postprocess: str = "area"
    median_kernel: int = 3
    morph_kernel: int = 3
    min_area: int | None = None
    adaptive_block_size: int = 35
    adaptive_c: float = -2.0
    sigma: float | str | None = None

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        img1p = _apply_gaussian_auto(img1, self.sigma, img2)
        img2p = _apply_gaussian_auto(img2, self.sigma, img1)
        score = self.score_fn(img1p, img2p)
        return self._postprocess(self._threshold(score))

    def _threshold(self, score: np.ndarray) -> np.ndarray:
        if self.threshold == "kmeans":
            return _kmeans_mask(score)
        if self.threshold == "triangle":
            return _triangle_mask(score)
        if self.threshold == "adaptive":
            return _adaptive_mask(score, self.adaptive_block_size, self.adaptive_c)
        return _otsu_mask(score, scale=self.threshold_scale)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if self.postprocess == "raw":
            return mask.astype(np.uint8, copy=True)
        result = mask.astype(np.uint8, copy=True)
        median_kernel = _odd_kernel(self.median_kernel)
        if median_kernel > 1:
            result = cv2.medianBlur(result, median_kernel)
        morph_kernel = max(1, int(self.morph_kernel))
        if morph_kernel > 1:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        if self.postprocess == "area":
            min_area = self.min_area
            if min_area is None:
                min_area = max(16, int(mask.shape[0] * mask.shape[1] * 0.0008))
            result = filter_by_area(result, min_area=int(min_area))
        return result


def build_tunable_classical_method(
    score_name: str,
    threshold: str = "otsu",
    threshold_scale: float = 1.0,
    postprocess: str = "area",
    median_kernel: int = 3,
    morph_kernel: int = 3,
    min_area: int | None = None,
    adaptive_block_size: int = 35,
    adaptive_c: float = -2.0,
    sigma: float | str | None = None,
) -> TunableClassicalPipeline:
    scores = classical_score_functions()
    if score_name not in scores:
        raise KeyError(f"Неизвестная карта изменений итогового сравнения: {score_name}")
    return TunableClassicalPipeline(
        score_fn=scores[score_name],
        threshold=threshold,
        threshold_scale=threshold_scale,
        postprocess=postprocess,
        median_kernel=median_kernel,
        morph_kernel=morph_kernel,
        min_area=min_area,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        sigma=sigma,
    )


def build_individual_classical_methods() -> "OrderedDict[str, object]":
    """Возвращает фиксированный набор сравнения без Adaptive PCA-CVA."""

    methods: OrderedDict[str, object] = OrderedDict()
    for score_name in [method for method in COMPARISON_METHODS if method != "Adaptive PCA-CVA"]:
        methods[score_name] = build_tunable_classical_method(score_name)
    return methods


def build_classical_methods(method_set: str = "individual") -> "OrderedDict[str, object]":
    method_set = str(method_set).strip().lower()
    if method_set in {"individual", "research"}:
        return build_individual_classical_methods()
    if method_set == "core":
        selected = [method for method in COMPARISON_METHODS if method != "Adaptive PCA-CVA"]
        return OrderedDict((name, build_tunable_classical_method(name, postprocess="raw")) for name in selected)
    raise ValueError(f"Неизвестный method_set: {method_set}")
