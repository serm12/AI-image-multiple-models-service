"""Source-pixel morphology supporting uncertain SCRFD face proposals."""
from __future__ import annotations

import cv2
import numpy as np


def broad_dark_nose(image: np.ndarray, candidate: dict) -> bool:
    """Find a broad compact dark nasal region, rather than fine nostrils.

    The region is normalized by eye separation, so its measurements do not
    depend on image resolution or head tilt. This is supporting evidence only:
    callers must also require weak local SCRFD confirmation. Beard, shadow,
    dark skin or a dark pixel alone must never establish a pet identity.
    """
    details = candidate["features"]
    points = np.asarray(candidate["points"], dtype=np.float32)
    span = float(np.linalg.norm(points[1] - points[0]))
    if (span < 10 or details["eye_span"] < 0.25
            or not 0.5 < details["mouth_depth"] < 1.8
            or abs(details["nose_side"]) > 0.4):
        return False
    middle = points[:2].mean(axis=0)
    horizontal = (points[1] - points[0]) / span
    down = np.array([-horizontal[1], horizontal[0]])
    if np.dot(points[3] - middle, down) < 0:
        down = -down
    normalized_span = 96
    matrix = np.array([
        [*horizontal, -np.dot(horizontal, middle)],
        [*down, -np.dot(down, middle)],
    ]) * normalized_span / span
    matrix[:, 2] += np.array([96, 24])
    view = cv2.warpAffine(image, matrix, (192, 192))
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    top = round(24 + 0.18 * normalized_span)
    bottom = round(24 + 0.82 * details["mouth_depth"] * normalized_span)
    region = gray[top:bottom, 38:154]
    threshold = min(65.0, float(np.percentile(region, 65)) * 0.52)
    mask = (region < threshold).astype(np.uint8)
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for index in range(1, count):
        x, _y, width, height, area = map(int, stats[index])
        center_x = centers[index][0]
        if x == 0 or x + width >= region.shape[1] or not 25 < center_x < 91:
            continue
        if (width > 0.5 * normalized_span and height > 0.17 * normalized_span
                and area > 0.09 * normalized_span ** 2
                and area / max(1, width * height) > 0.65):
            return True
    return False
