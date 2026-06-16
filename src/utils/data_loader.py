"""Dataset loader for A/B/label change-detection pairs.

JL1-CD is evaluated as 256x256 patches, matching the LEVIR-CD patch scale.
The original files are kept intact; patching happens while loading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.image_io import read_image


@dataclass
class PairQualityReport:
    patch_name: str
    a_shape: str
    b_shape_original: str
    label_shape_original: str
    b_resized: bool
    label_resized: bool
    estimated_shift_x: float
    estimated_shift_y: float
    estimated_shift_magnitude: float
    estimated_shift_response: float
    estimated_shift_reliable: bool
    alignment_applied: bool
    alignment_shift_x: float
    alignment_shift_y: float
    residual_shift_x: float
    residual_shift_y: float
    residual_shift_magnitude: float
    residual_shift_response: float
    residual_shift_reliable: bool
    warning: str
    source_name: str = ""
    patch_y: int = 0
    patch_x: int = 0
    patch_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _shape_text(image: np.ndarray | None) -> str:
    if image is None:
        return ""
    return "x".join(str(value) for value in image.shape[:2])


def _estimate_translation(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float, float, float]:
    """Estimate the translation from A to B using phase correlation."""

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    gray_a = gray_a.astype(np.float32)
    gray_b = gray_b.astype(np.float32)
    if gray_a.shape != gray_b.shape or min(gray_a.shape[:2]) < 8:
        return 0.0, 0.0, 0.0, 0.0
    window = cv2.createHanningWindow((gray_a.shape[1], gray_a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(gray_a, gray_b, window)
    shift_x, shift_y = float(shift[0]), float(shift[1])
    return shift_x, shift_y, float(np.hypot(shift_x, shift_y)), float(response)


def _align_translation(image: np.ndarray, shift_x: float, shift_y: float, interpolation: int) -> np.ndarray:
    """Warp B into A coordinates; phaseCorrelate returns the A -> B shift."""

    matrix = np.array([[1.0, 0.0, -float(shift_x)], [0.0, 1.0, -float(shift_y)]], dtype=np.float32)
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


class LEVIRCDLoader:
    """Read image pairs and optional GT masks with geometry diagnostics."""

    def __init__(
        self,
        data_path: str,
        alignment_warning_px: float = 1.5,
        auto_align_b: bool = True,
        max_alignment_shift_px: float = 12.0,
        min_alignment_response: float = 0.05,
        align_label: bool = False,
        patch_size: int | None = None,
        patch_stride: int | None = None,
    ):
        self.data_path = Path(data_path)
        self.alignment_warning_px = float(alignment_warning_px)
        self.auto_align_b = bool(auto_align_b)
        self.max_alignment_shift_px = float(max_alignment_shift_px)
        self.min_alignment_response = float(min_alignment_response)
        self.align_label = bool(align_label)
        self.patch_size = int(patch_size) if patch_size else self._default_patch_size()
        self.patch_stride = int(patch_stride) if patch_stride else self.patch_size

    def _default_patch_size(self) -> int | None:
        return 256 if self.data_path.name.lower() == "jl1-cd" else None

    def _prepare_pair(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label: np.ndarray | None,
        patch_name: str,
        source_name: str = "",
        patch_y: int = 0,
        patch_x: int = 0,
        patch_size: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, PairQualityReport]:
        b_shape_original = _shape_text(img_b)
        label_shape_original = _shape_text(label)
        b_resized = img_b.shape[:2] != img_a.shape[:2]
        label_resized = label is not None and label.shape[:2] != img_a.shape[:2]

        if b_resized:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
        if label_resized and label is not None:
            label = cv2.resize(label, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

        shift_x, shift_y, shift_magnitude, shift_response = _estimate_translation(img_a, img_b)
        warnings: list[str] = []
        if b_resized:
            warnings.append("B resized to A shape")
        if label_resized:
            warnings.append("label resized to A shape")

        alignment_applied = False
        alignment_shift_x = 0.0
        alignment_shift_y = 0.0
        estimated_reliable = shift_response >= self.min_alignment_response
        if not estimated_reliable:
            warnings.append(f"phase correlation unreliable: response {shift_response:.3f} below {self.min_alignment_response:.3f}")
        elif shift_magnitude > self.alignment_warning_px:
            warnings.append(f"estimated shift {shift_magnitude:.2f}px exceeds {self.alignment_warning_px:.2f}px")
            can_align = (
                self.auto_align_b
                and shift_magnitude <= self.max_alignment_shift_px
            )
            if can_align:
                img_b = _align_translation(img_b, shift_x, shift_y, interpolation=cv2.INTER_LINEAR)
                if label is not None and self.align_label:
                    label = _align_translation(label, shift_x, shift_y, interpolation=cv2.INTER_NEAREST)
                alignment_applied = True
                alignment_shift_x = -shift_x
                alignment_shift_y = -shift_y
            elif self.auto_align_b:
                if shift_magnitude > self.max_alignment_shift_px:
                    warnings.append(f"alignment skipped: shift {shift_magnitude:.2f}px exceeds max {self.max_alignment_shift_px:.2f}px")

        residual_x, residual_y, residual_magnitude, residual_response = _estimate_translation(img_a, img_b)
        residual_reliable = residual_response >= self.min_alignment_response
        if alignment_applied and not residual_reliable:
            warnings.append(f"residual phase correlation unreliable: response {residual_response:.3f}")
        elif alignment_applied and residual_magnitude > self.alignment_warning_px:
            warnings.append(f"residual shift {residual_magnitude:.2f}px after alignment")

        report = PairQualityReport(
            patch_name=patch_name,
            a_shape=_shape_text(img_a),
            b_shape_original=b_shape_original,
            label_shape_original=label_shape_original,
            b_resized=bool(b_resized),
            label_resized=bool(label_resized),
            estimated_shift_x=shift_x,
            estimated_shift_y=shift_y,
            estimated_shift_magnitude=shift_magnitude,
            estimated_shift_response=shift_response,
            estimated_shift_reliable=bool(estimated_reliable),
            alignment_applied=alignment_applied,
            alignment_shift_x=alignment_shift_x,
            alignment_shift_y=alignment_shift_y,
            residual_shift_x=residual_x,
            residual_shift_y=residual_y,
            residual_shift_magnitude=residual_magnitude,
            residual_shift_response=residual_response,
            residual_shift_reliable=bool(residual_reliable),
            warning="; ".join(warnings),
            source_name=source_name,
            patch_y=int(patch_y),
            patch_x=int(patch_x),
            patch_size=int(patch_size),
        )
        return img_a, img_b, label, report

    def _iter_patches(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label: np.ndarray | None,
        file_name: str,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, str, int, int, int]]:
        patch_size = self.patch_size
        if not patch_size or (img_a.shape[0] <= patch_size and img_a.shape[1] <= patch_size):
            return [(img_a, img_b, label, file_name, 0, 0, 0)]

        stem = Path(file_name).stem
        suffix = Path(file_name).suffix or ".png"
        patches = []
        for y in _positions(img_a.shape[0], patch_size, self.patch_stride):
            for x in _positions(img_a.shape[1], patch_size, self.patch_stride):
                y2 = min(y + patch_size, img_a.shape[0])
                x2 = min(x + patch_size, img_a.shape[1])
                patch_a = img_a[y:y2, x:x2]
                patch_b = img_b[y:y2, x:x2]
                patch_label = label[y:y2, x:x2] if label is not None else None
                patch_name = f"{stem}_y{y}_x{x}{suffix}"
                patches.append((patch_a, patch_b, patch_label, patch_name, y, x, patch_size))
        return patches

    def load_pair(
        self,
        a_path: Path,
        b_path: Path,
        label_path: Path | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        img_a, img_b, label, _ = self.load_pair_with_report(a_path, b_path, label_path)
        return img_a, img_b, label

    def load_pair_with_report(
        self,
        a_path: Path,
        b_path: Path,
        label_path: Path | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, PairQualityReport | None]:
        img_a = read_image(a_path, cv2.IMREAD_COLOR)
        img_b = read_image(b_path, cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            return None, None, None, None
        label = read_image(label_path, cv2.IMREAD_GRAYSCALE) if label_path is not None and label_path.exists() else None
        img_a, img_b, label, report = self._prepare_pair(img_a, img_b, label, a_path.name, source_name=a_path.name)
        return img_a, img_b, label, report

    def load_split(self, split: str = "test", max_pairs: int | None = None) -> list[dict]:
        split_path = self.data_path / split
        a_dir = split_path / "A"
        b_dir = split_path / "B"
        label_dir = split_path / "label"
        if not a_dir.is_dir() or not b_dir.is_dir():
            return []

        pairs: list[dict] = []
        for a_path in sorted(a_dir.glob("*.png")):
            b_path = b_dir / a_path.name
            if not b_path.exists():
                continue
            label_path = label_dir / a_path.name if label_dir.is_dir() else None
            img_a = read_image(a_path, cv2.IMREAD_COLOR)
            img_b = read_image(b_path, cv2.IMREAD_COLOR)
            if img_a is None or img_b is None:
                continue
            label = read_image(label_path, cv2.IMREAD_GRAYSCALE) if label_path is not None and label_path.exists() else None

            # Bring B/label to A shape before cutting patches so all patches share coordinates.
            if img_b.shape[:2] != img_a.shape[:2]:
                img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
            if label is not None and label.shape[:2] != img_a.shape[:2]:
                label = cv2.resize(label, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

            for patch_a, patch_b, patch_label, patch_name, y, x, patch_size in self._iter_patches(img_a, img_b, label, a_path.name):
                prepared_a, prepared_b, prepared_label, report = self._prepare_pair(
                    patch_a,
                    patch_b,
                    patch_label,
                    patch_name=patch_name,
                    source_name=a_path.name,
                    patch_y=y,
                    patch_x=x,
                    patch_size=patch_size,
                )
                pairs.append(
                    {
                        "img_a": prepared_a,
                        "img_b": prepared_b,
                        "label": prepared_label,
                        "name": patch_name,
                        "quality_report": report.to_dict(),
                    }
                )
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs

        return pairs
