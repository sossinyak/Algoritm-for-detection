"""Преобразование YAML-конфига в параметры объектов программы."""

from __future__ import annotations

from typing import Dict

from utils.data_loader import LEVIRCDLoader


def build_adaptive_params(config: dict) -> Dict[str, object]:
    """Собирает параметры итогового адаптивного алгоритма."""
    experiment_cfg = config.get("experiments", {})
    adaptive_cfg = config.get("adaptive_algorithm", {})
    threshold = adaptive_cfg.get("threshold", {})
    post = adaptive_cfg.get("postprocessing", {})
    shape = post.get("shape_filter", {})

    return {
        "patch_size": adaptive_cfg.get("patch_size", 1),
        "pca_components": adaptive_cfg.get("components", 3),
        "pca_variance_ratio": adaptive_cfg.get("variance_ratio"),
        "whitening": adaptive_cfg.get("whitening", True),
        "evaluation_mode": experiment_cfg.get("evaluation_mode", "production"),
        "radiometric_normalization": adaptive_cfg.get("radiometric_normalization", "none"),
        "threshold_method": threshold.get("method", "otsu"),
        "threshold_value": threshold.get("value"),
        "otsu_scale": threshold.get("otsu_scale", 0.85),
        "adaptive_block_size": threshold.get("adaptive_block_size", 35),
        "adaptive_c": threshold.get("adaptive_c", -2.0),
        "sauvola_k": threshold.get("sauvola_k", 0.20),
        "kimura_k": threshold.get("kimura_k", 0.35),
        "kimura_sigma_max": threshold.get("kimura_sigma_max", 64.0),
        "median_kernel": post.get("median_kernel", 3),
        "opening_kernel": post.get("opening_kernel", 3),
        "closing_kernel": post.get("closing_kernel", 3),
        "min_area": post.get("min_area", 100),
        "max_aspect_ratio": shape.get("max_aspect_ratio"),
        "min_solidity": shape.get("min_solidity"),
        "min_extent": shape.get("min_extent"),
        "fill_holes": post.get("fill_holes", False),
    }


def load_configured_pairs(config: dict, split: str = "test", max_pairs: int | None = None) -> list[dict]:
    """Загружает пары снимков по пути из config.yaml."""
    data_cfg = config.get("data", {})
    alignment_cfg = data_cfg.get("alignment", {})
    loader = LEVIRCDLoader(
        data_cfg.get("data_path", "./data/LEVIR-CD-filtred"),
        alignment_warning_px=data_cfg.get("alignment_warning_px", 1.5),
        auto_align_b=alignment_cfg.get("auto_align_b", data_cfg.get("auto_align_b", True)),
        max_alignment_shift_px=alignment_cfg.get("max_shift_px", data_cfg.get("max_alignment_shift_px", 12.0)),
        min_alignment_response=alignment_cfg.get("min_response", data_cfg.get("min_alignment_response", 0.05)),
        align_label=alignment_cfg.get("align_label", data_cfg.get("align_label", False)),
    )
    return loader.load_split(split=split, max_pairs=max_pairs)
