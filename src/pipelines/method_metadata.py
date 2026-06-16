"""Метаданные фиксированного набора методов для кода, отчетов и ВКР."""

from __future__ import annotations


COMPARISON_METHODS = [
    "AbsDiff",
    "LogRatio",
    "RGB-CVA",
    "PCA-CVA",
    "Adaptive PCA-CVA",
]

INDIVIDUAL_METHODS = COMPARISON_METHODS.copy()


METHOD_THEORY = {
    "AbsDiff": {
        "formula": "D(x)=|I2(x)-I1(x)|",
        "parameters": "sigma, threshold, threshold_scale, median_kernel, morph_kernel, min_area",
        "ranges": "sigma: 0-3 пикс.; масштаб порога: 0,6-1,4; ядра: 1-7 пикс.",
        "role": "Базовое сравнение абсолютной яркостной разности.",
    },
    "LogRatio": {
        "formula": "D(x)=|log((I2(x)+1)/(I1(x)+1))|",
        "parameters": "sigma, threshold, threshold_scale, median_kernel, morph_kernel, min_area",
        "ranges": "sigma: 0-3 пикс.; масштаб порога: 0,6-1,4",
        "role": "Базовый метод отношения, более устойчивый к мультипликативному сдвигу яркости.",
    },
    "RGB-CVA": {
        "formula": "D(x)=||RGB2(x)-RGB1(x)||2",
        "parameters": "sigma, threshold, threshold_scale, median_kernel, morph_kernel, min_area",
        "ranges": "sigma: 0-3 пикс.; масштаб порога: 0,6-1,4",
        "role": "Классический анализ вектора изменений в исходном RGB-пространстве.",
    },
    "PCA-CVA": {
        "formula": "D(x)=||PCA(I2)(x)-PCA(I1)(x)||2",
        "parameters": "n_components, whiten, sigma, threshold, threshold_scale, median_kernel, morph_kernel, min_area",
        "ranges": "компоненты: 1-3; масштаб порога: 0,6-1,4",
        "role": "Анализ вектора изменений после классического метода главных компонент.",
    },
    "Adaptive PCA-CVA": {
        "formula": "карта CVA с МГК после радиометрической нормализации, пороговая бинаризация и объектная фильтрация",
        "parameters": "radiometric_normalization, threshold_method, otsu_scale, adaptive_block_size, adaptive_c, median_kernel, opening_kernel, closing_kernel, min_area, max_aspect_ratio, min_solidity, min_extent, fill_holes",
        "ranges": "компоненты: 2-3; порог: Оцу/локальный/Кимуры/Саволы; min_area: 16-500 пикс.",
        "role": "Итоговый интерпретируемый конвейер, выбранный по валидационной выборке.",
    },
}
