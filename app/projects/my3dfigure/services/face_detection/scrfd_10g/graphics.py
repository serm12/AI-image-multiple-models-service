"""Discard flat graphic face overlays next to a photographic SCRFD anchor."""
from __future__ import annotations

import cv2
import numpy as np


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
                and candidate["score"] >= 0.82
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
        ink_on_paper = white >= 0.25 and high - low >= 140 and dark >= 0.025
        flat_colored_art = (
            float(np.median(hsv[:, :, 1])) >= 80
            and float(hsv[:, :, 0].std()) < 5
            and high - low < 40
        )
        if not (ink_on_paper or flat_colored_art):
            kept.append(candidate)
    return kept
