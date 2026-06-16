"""
Итоговый адаптивный алгоритм обнаружения изменений: PCA + CVA.

Порядок обработки:
1. RGB-снимки переводятся из диапазона 0..255 в диапазон [0, 1].
2. Для каждого пикселя формируется вектор признаков: RGB или RGB-окрестность.
3. PCA обучается на объединенных признаках двух снимков одной пары.
4. Оба снимка проецируются в PCA-пространство.
5. CVA считает длину вектора различий между проекциями.
6. Карта изменений бинаризуется порогом, подобранным на валидационной выборке.
7. Итоговая маска очищается медианным и морфологическим фильтрами.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from postprocessing.area_filter import filter_components
from segmentation.adaptive_threshold import (
    global_otsu_threshold,
    global_otsu_threshold_scaled,
    kimura_threshold,
    local_adaptive_threshold,
    sauvola_threshold,
)


def _odd_kernel(value: int, minimum: int = 1) -> int:
    """Приводит размер ядра к нечетному числу, как требуется в OpenCV."""
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _normalize01(values: np.ndarray) -> np.ndarray:
    """Масштабирует карту значений в [0, 1]."""
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Переводит uint8-изображение 0..255 в float32 0..1."""
    return image.astype(np.float32) / 255.0


def _histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Подгоняет CDF одного канала source к reference."""
    source_u8 = np.clip(source, 0, 255).astype(np.uint8)
    reference_u8 = np.clip(reference, 0, 255).astype(np.uint8)
    src_hist = np.bincount(source_u8.ravel(), minlength=256).astype(np.float64)
    ref_hist = np.bincount(reference_u8.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist) / max(float(source_u8.size), 1.0)
    ref_cdf = np.cumsum(ref_hist) / max(float(reference_u8.size), 1.0)
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return lut[source_u8]


def _histogram_match_pair(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Согласует радиометрию T2 с T1 через histogram matching."""
    if img1.ndim == 2 or img2.ndim == 2:
        return img1.copy(), _histogram_match_channel(img2, img1)
    matched = np.empty_like(img2)
    for channel in range(img2.shape[2]):
        matched[:, :, channel] = _histogram_match_channel(img2[:, :, channel], img1[:, :, channel])
    return img1.copy(), matched


def _quantile_normalize_channel(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Приводит два канала к общей средней квантильной шкале."""
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    order_a = np.argsort(flat_a, kind="mergesort")
    order_b = np.argsort(flat_b, kind="mergesort")
    common = ((np.sort(flat_a).astype(np.float32) + np.sort(flat_b).astype(np.float32)) / 2.0).astype(np.uint8)

    out_a = np.empty_like(flat_a, dtype=np.uint8)
    out_b = np.empty_like(flat_b, dtype=np.uint8)
    out_a[order_a] = common
    out_b[order_b] = common
    return out_a.reshape(a.shape), out_b.reshape(b.shape)


def _quantile_normalize_pair(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Согласует два снимка по каналам через quantile normalization."""
    img1_u8 = np.clip(img1, 0, 255).astype(np.uint8)
    img2_u8 = np.clip(img2, 0, 255).astype(np.uint8)
    if img1_u8.ndim == 2 or img2_u8.ndim == 2:
        return _quantile_normalize_channel(img1_u8, img2_u8)

    norm1 = np.empty_like(img1_u8)
    norm2 = np.empty_like(img2_u8)
    for channel in range(img1_u8.shape[2]):
        norm1[:, :, channel], norm2[:, :, channel] = _quantile_normalize_channel(
            img1_u8[:, :, channel],
            img2_u8[:, :, channel],
        )
    return norm1, norm2


def _apply_radiometric_normalization(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет парную радиометрическую нормализацию перед PCA-CVA."""
    method = str(method or "none").lower()
    if method in {"none", "off", "false"}:
        return img1, img2
    if method in {"histogram_match", "histogram_matching", "hist_match"}:
        return _histogram_match_pair(img1, img2)
    if method in {"quantile", "quantile_normalization", "qn"}:
        return _quantile_normalize_pair(img1, img2)
    raise ValueError(f"Неизвестная радиометрическая нормализация: {method}")


def _extract_patch_features(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Формирует признаки для каждого пикселя.

    При patch_size=1 признак равен RGB-вектору пикселя. При большем окне к признаку
    добавляются RGB-значения соседних пикселей.
    """
    patch_size = _odd_kernel(patch_size, minimum=1)
    if len(image.shape) == 2:
        image = image[:, :, None]

    height, width, channels = image.shape
    if patch_size == 1:
        return image.reshape(height * width, channels).astype(np.float32, copy=False)

    radius = patch_size // 2
    padded = cv2.copyMakeBorder(
        image,
        radius,
        radius,
        radius,
        radius,
        borderType=cv2.BORDER_REFLECT_101,
    )

    features = []
    for row_shift in range(patch_size):
        for col_shift in range(patch_size):
            patch = padded[row_shift : row_shift + height, col_shift : col_shift + width, :]
            features.append(patch.reshape(height * width, channels))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _fit_pca(
    features1: np.ndarray,
    features2: np.ndarray,
    n_components: int = 3,
    variance_ratio: Optional[float] = None,
    whitening: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обучает PCA на признаках двух снимков одной пары."""
    samples = np.vstack([features1, features2]).astype(np.float32, copy=False)
    mean = np.mean(samples, axis=0, dtype=np.float64).astype(np.float32)
    centered = samples - mean

    # Для небольшого числа признаков матрица ковариации быстрее полного SVD.
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if variance_ratio is not None:
        total = float(np.sum(np.maximum(eigvals, 0.0)))
        if total > 0:
            cumulative = np.cumsum(np.maximum(eigvals, 0.0)) / total
            n_components = int(np.searchsorted(cumulative, float(variance_ratio)) + 1)

    n_components = max(1, min(int(n_components), eigvecs.shape[1]))
    components = eigvecs[:, :n_components].astype(np.float32)

    if whitening:
        scale = np.sqrt(np.maximum(eigvals[:n_components], 1e-8)).astype(np.float32)
        components = components / scale[None, :]

    return mean, components, eigvals[:n_components].astype(np.float32)


def _project(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Проецирует признаки в пространство главных компонент."""
    return (features - mean) @ components


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние дыры в бинарной маске через flood fill от границы."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


class AdaptiveChangeDetection:
    """Адаптивный PCA+CVA-алгоритм для пары разновременных RGB-снимков."""

    def __init__(
        self,
        patch_size: int = 1,
        pca_components: int = 3,
        pca_variance_ratio: Optional[float] = None,
        whitening: bool = True,
        evaluation_mode: str = "production",
        radiometric_normalization: str = "none",
        threshold_method: str = "otsu",
        threshold_value: Optional[float] = None,
        otsu_scale: float = 0.85,
        adaptive_block_size: int = 35,
        adaptive_c: float = -2.0,
        sauvola_k: float = 0.20,
        kimura_k: float = 0.35,
        kimura_sigma_max: float = 64.0,
        median_kernel: int = 3,
        opening_kernel: int = 3,
        closing_kernel: int = 3,
        min_area: Optional[int] = 100,
        max_aspect_ratio: Optional[float] = None,
        min_solidity: Optional[float] = None,
        min_extent: Optional[float] = None,
        fill_holes: bool = False,
    ):
        self.patch_size = _odd_kernel(patch_size, minimum=1)
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.whitening = bool(whitening)
        self.evaluation_mode = str(evaluation_mode or "production").lower()
        if self.evaluation_mode not in {"production", "diagnostic"}:
            raise ValueError("evaluation_mode должен быть production или diagnostic")
        self.radiometric_normalization = str(radiometric_normalization or "none").lower()
        self.threshold_method = str(threshold_method or "otsu").lower()
        self.threshold_value = threshold_value
        self.otsu_scale = float(otsu_scale)
        self.adaptive_block_size = _odd_kernel(adaptive_block_size, minimum=3)
        self.adaptive_c = float(adaptive_c)
        self.sauvola_k = float(sauvola_k)
        self.kimura_k = float(kimura_k)
        self.kimura_sigma_max = float(kimura_sigma_max)
        self.median_kernel = _odd_kernel(median_kernel, minimum=1)
        self.opening_kernel = _odd_kernel(opening_kernel, minimum=1)
        self.closing_kernel = _odd_kernel(closing_kernel, minimum=1)
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        self.min_extent = min_extent
        self.fill_holes = bool(fill_holes)
        self.intermediate_results: Dict[str, object] = {}

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Возвращает бинарную маску изменений со значениями 0 и 255."""
        change_map = self.compute_change_map(img1, img2)
        mask = self.threshold_change_map(change_map)
        self.intermediate_results["threshold_mask"] = mask.copy()
        final_mask = self.postprocess_mask(mask)
        self.intermediate_results["final_mask"] = final_mask
        return final_mask

    def compute_change_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Строит непрерывную карту изменений PCA-CVA для одной пары снимков."""
        self.intermediate_results = {}
        height, width = img1.shape[:2]

        img1p, img2p = _apply_radiometric_normalization(img1, img2, self.radiometric_normalization)
        self.intermediate_results["radiometric_normalization"] = self.radiometric_normalization

        norm1 = _to_float01(img1p)
        norm2 = _to_float01(img2p)
        features1 = _extract_patch_features(norm1, self.patch_size)
        features2 = _extract_patch_features(norm2, self.patch_size)

        mean, components, eigvals = _fit_pca(
            features1,
            features2,
            n_components=self.pca_components,
            variance_ratio=self.pca_variance_ratio,
            whitening=self.whitening,
        )
        projected1 = _project(features1, mean, components)
        projected2 = _project(features2, mean, components)

        # CVA: больше расстояние между векторами "до" и "после" - вероятнее изменение.
        change_vectors = projected1 - projected2
        magnitude = np.sqrt(np.sum(change_vectors * change_vectors, axis=1))
        change_map = _normalize01(magnitude.reshape(height, width))

        self.intermediate_results["change_map"] = change_map
        self.intermediate_results["pca_eigenvalues"] = eigvals
        return change_map

    def threshold_change_map(self, change_map: np.ndarray, threshold_value: Optional[float] = None) -> np.ndarray:
        """Бинаризует карту изменений указанным порогом или штатным правилом алгоритма."""
        if threshold_value is not None:
            if self.evaluation_mode != "diagnostic":
                raise ValueError("Per-image threshold override is allowed only in diagnostic mode")
            return (change_map >= float(threshold_value)).astype(np.uint8) * 255
        return self._threshold(change_map)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Применяет штатную очистку к бинарной маске."""
        return self._postprocess(mask)

    def _threshold(self, change_map: np.ndarray) -> np.ndarray:
        """Преобразует карту изменений в бинарную маску."""
        if self.threshold_value is not None:
            return (change_map >= float(self.threshold_value)).astype(np.uint8) * 255

        score_uint8 = np.clip(change_map * 255.0, 0, 255).astype(np.uint8)
        if self.threshold_method in {"adaptive", "local_adaptive"}:
            return local_adaptive_threshold(score_uint8, self.adaptive_block_size, self.adaptive_c)
        if self.threshold_method == "sauvola":
            return sauvola_threshold(score_uint8, self.adaptive_block_size, self.sauvola_k)
        if self.threshold_method == "kimura":
            return kimura_threshold(score_uint8, self.adaptive_block_size, self.kimura_k, self.kimura_sigma_max)
        if self.threshold_method not in {"otsu", "scaled_otsu"}:
            raise ValueError(f"Неизвестный threshold_method: {self.threshold_method}")
        if abs(self.otsu_scale - 1.0) < 1e-6:
            return global_otsu_threshold(score_uint8)
        return global_otsu_threshold_scaled(score_uint8, scale=self.otsu_scale)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Очищает бинарную маску от мелкого шума."""
        result = mask.astype(np.uint8, copy=True)

        if self.median_kernel > 1:
            result = cv2.medianBlur(result, self.median_kernel)
        self.intermediate_results["median_mask"] = result.copy()

        if self.opening_kernel > 1:
            kernel = np.ones((self.opening_kernel, self.opening_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        self.intermediate_results["opening_mask"] = result.copy()

        if self.closing_kernel > 1:
            kernel = np.ones((self.closing_kernel, self.closing_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        self.intermediate_results["closing_mask"] = result.copy()

        result = filter_components(
            result,
            min_area=self.min_area,
            max_aspect_ratio=self.max_aspect_ratio,
            min_solidity=self.min_solidity,
            min_extent=self.min_extent,
        )
        self.intermediate_results["area_mask"] = result.copy()

        if self.fill_holes:
            # Заполняем только внутренние пустоты, чтобы маска крупных объектов была стабильнее.
            result = _fill_binary_holes(result)
        self.intermediate_results["filled_mask"] = result.copy()

        return result

    def get_intermediate_results(self) -> Dict[str, object]:
        """Возвращает промежуточные карты для визуализации и ручной проверки."""
        return self.intermediate_results
