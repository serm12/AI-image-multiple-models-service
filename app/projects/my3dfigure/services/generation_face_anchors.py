"""Build My3dFigure-only face identity anchors without altering the upload master."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np


_FACE_ANCHOR_MARGIN = 0.45


def _read_source_image(source_path: str):
    """Read Unicode Windows paths without changing the uploaded source file."""
    try:
        encoded = np.fromfile(source_path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except OSError:
        return None


def create_face_identity_anchors(
    source_path: str,
    face_boxes: Iterable[tuple[float, float, float, float]],
    output_dir: str,
) -> list[str]:
    """Write padded face crops beside a task's source image.

    The original upload is untouched and remains the first reference image.
    Each crop contains only the accepted face plus enough surrounding hair and
    head context to stabilize identity; it is never used as clothing evidence.
    """
    # OpenCV imread is not reliable with Unicode Windows paths. Read bytes
    # first so Chinese customer filenames use the same safe path as detection.
    image = _read_source_image(source_path)
    if image is None:
        return []
    height, width = image.shape[:2]
    paths: list[str] = []
    for index, raw_box in enumerate(face_boxes, start=1):
        x, y, box_width, box_height = raw_box
        padding = max(box_width, box_height) * _FACE_ANCHOR_MARGIN
        left = max(0, int(round(x - padding)))
        top = max(0, int(round(y - padding)))
        right = min(width, int(round(x + box_width + padding)))
        bottom = min(height, int(round(y + box_height + padding)))
        if right <= left or bottom <= top:
            continue
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        path = os.path.join(output_dir, f"identity_anchor_{index}.png")
        if not cv2.imwrite(path, crop):
            continue
        paths.append(path)
    return paths


def create_pet_face_identity_anchor(
    source_path: str,
    face_box: tuple[float, float, float, float],
    output_dir: str,
) -> str | None:
    """Write exactly one validated pet-face identity anchor beside its task.

    The full upload remains the first reference image and supplies pose, body,
    coat and visible accessories.  This tight crop only authorizes the one pet
    whose face passed the local validation pipeline.
    """
    image = _read_source_image(source_path)
    if image is None:
        return None
    try:
        x, y, width, height = (float(value) for value in face_box)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    image_height, image_width = image.shape[:2]
    left, top = max(0, int(round(x))), max(0, int(round(y)))
    right = min(image_width, int(round(x + width)))
    bottom = min(image_height, int(round(y + height)))
    if right - left < 8 or bottom - top < 8:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    path = os.path.join(output_dir, "pet_identity_anchor.png")
    return path if cv2.imwrite(path, crop) else None
