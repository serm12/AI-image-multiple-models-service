"""Discard flat graphic face overlays next to a photographic SCRFD anchor."""
from __future__ import annotations

import cv2
import numpy as np


def _monochrome_ink_structure(patch: np.ndarray) -> bool:
    """Read connected flat ink/paper regions, not photographic grayscale.

    A real black-and-white face still has continuous skin shading. A pasted
    black/white graphic instead contains large connected nearly uniform white
    and black regions with comparatively little midtone area. JPEG ringing
    and the visible facial details must not count as a flat background.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    spread = (cv2.dilate(patch, kernel).astype(np.int16)
              - cv2.erode(patch, kernel).astype(np.int16))
    flat = spread <= 6
    white = ((patch >= 240) & flat).astype(np.uint8)
    dark = ((patch <= 25) & flat).astype(np.uint8)
    white_area, dark_area = float(white.mean()), float(dark.mean())
    midtones = float(((patch >= 40) & (patch <= 230)).mean())
    if white_area < 0.11 or midtones >= 0.55:
        return False

    def connected_fraction(mask):
        _count, _labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return float(max(stats[1:, cv2.CC_STAT_AREA], default=0)) / mask.size

    white_component, dark_component = connected_fraction(white), connected_fraction(dark)
    # Require a broad flat paper field or a predominantly two-tone design.
    # Saturated photographic highlights alone remain insufficient evidence.
    paper_or_ink = (dark_area >= 0.25
                    or (white_area >= 0.20 and dark_component >= 0.5 * dark_area)
                    or (dark_area >= 0.15 and midtones < 0.36)
                    or (white_area >= 0.15 and dark_area >= 0.10
                        and midtones < 0.40))
    return paper_or_ink and white_component >= 0.07 and dark_component >= 0.02


def filter_candidates(image: np.ndarray, candidates: list[dict]) -> list[dict]:
    if len(candidates) < 2:
        return candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def pixels(candidate):
        x, y, width, height = map(int, candidate["box"])
        ys = slice(max(0, y), min(image.shape[0], y + height))
        xs = slice(max(0, x), min(image.shape[1], x + width))
        return image[ys, xs], gray[ys, xs]

    anchors = []
    for candidate in candidates:
        _roi, patch = pixels(candidate)
        if (patch.size and min(candidate["box"][2:]) >= 120
                and max(candidate["score"], min(candidate.get("source_candidate", {}).get("confirm", [0]))) >= 0.82
                and float((patch > 240).mean()) < 0.20):
            anchors.append(candidate)
    if not anchors:
        return candidates

    kept = []
    for candidate in candidates:
        if any(candidate is anchor for anchor in anchors):
            kept.append(candidate)
            continue
        roi, patch = pixels(candidate)
        if not patch.size:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        low, _middle, high = np.percentile(patch, (10, 50, 90))
        white = float((patch > 240).mean())
        dark = float((patch < 40).mean())
        gray_fraction = float((roi.max(axis=2).astype(float)-roi.min(axis=2) <= 10).mean())
        ink_structure = (
            gray_fraction > 0.75 and white >= 0.1 and high - low >= 140 and dark >= 0.025
            and _monochrome_ink_structure(patch)
        )
        ink_on_paper = (white >= 0.25 and high - low >= 140 and dark >= 0.025
                        and (gray_fraction <= 0.75 or ink_structure))
        flat_colored_art = (
            float(np.median(hsv[:, :, 1])) >= 80
            and float(hsv[:, :, 0].std()) < 5
            and high - low < 40
        )
        if not (ink_on_paper or ink_structure or flat_colored_art):
            kept.append(candidate)
    return kept
