"""Пороговая бинаризация карты изменений."""

import cv2
import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Переводит карту float [0,1] или uint8 [0,255] в uint8."""
    if image.dtype == np.uint8:
        return image.copy()
    img = image.astype(np.float32, copy=False)
    if float(np.max(img)) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def global_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Бинаризация методом Оцу без дополнительного масштабирования порога."""
    img_uint8 = to_uint8(image)
    _, binary = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def global_otsu_threshold_scaled(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Бинаризация методом Оцу с коэффициентом к найденному порогу.

    scale < 1 делает маску шире и повышает Recall, scale > 1 делает маску
    строже и обычно повышает Precision.
    """
    img_uint8 = to_uint8(image)
    threshold, _ = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    scaled_threshold = float(np.clip(threshold * float(scale), 0.0, 255.0))
    return (img_uint8 > scaled_threshold).astype(np.uint8) * 255


def local_adaptive_threshold(image: np.ndarray, block_size: int = 35, c_value: float = -2.0) -> np.ndarray:
    """Локально-адаптивная бинаризация карты изменений."""
    img_uint8 = to_uint8(image)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img_uint8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        float(c_value),
    )


def sauvola_threshold(image: np.ndarray, block_size: int = 35, k: float = 0.20, r_value: float = 128.0) -> np.ndarray:
    """Бинаризация Саволы для неоднородной карты изменений."""
    img = to_uint8(image).astype(np.float32)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1

    mean = cv2.boxFilter(img, ddepth=-1, ksize=(block_size, block_size), normalize=True)
    sq_mean = cv2.boxFilter(img * img, ddepth=-1, ksize=(block_size, block_size), normalize=True)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    threshold = mean * (1.0 + float(k) * (std / max(float(r_value), 1e-6) - 1.0))
    return (img > threshold).astype(np.uint8) * 255


def kimura_threshold(image: np.ndarray, block_size: int = 35, k: float = 0.35, sigma_max: float = 64.0) -> np.ndarray:
    """Локальный порог на основе среднего и текстуры: T = m + k * min(sigma, sigma_max)."""
    img = to_uint8(image).astype(np.float32)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1

    mean = cv2.boxFilter(img, ddepth=-1, ksize=(block_size, block_size), normalize=True)
    sq_mean = cv2.boxFilter(img * img, ddepth=-1, ksize=(block_size, block_size), normalize=True)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    threshold = mean + float(k) * np.minimum(std, float(sigma_max))
    return (img > threshold).astype(np.uint8) * 255
