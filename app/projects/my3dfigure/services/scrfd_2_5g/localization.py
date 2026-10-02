"""Coordinate-only refinement of weak compact profile-face proposals."""
from __future__ import annotations

import cv2
import numpy as np

from . import evidence


def refine(image: np.ndarray, candidate: dict, detector) -> list[float]:
    """Use three agreeing native-orientation contexts to localize a profile.

    This runs after acceptance, counting and recovery. It cannot admit or
    remove a face. Compact weak profiles with narrow projected eyes can have
    a box on the ear or a box that truncates the nose/chin. Strong existing
    boxes and elongated boundary proposals retain their original coordinates.
    """
    original = candidate["box"]
    x, y, width, height = original
    if not (0.5 <= candidate["score"] < 0.7
            and candidate["features"]["eye_span"] < 0.2
            and min(candidate["confirm"]) < 0.7
            and 0.65 <= height / max(width, 1) <= 2.2):
        return original
    local_boxes = []
    scores = []
    for factor in (1.6, 2.3, 3.2):
        side = max(64, round(max(width, height) * factor))
        center = (x + width / 2, y + height / 2)
        matrix = np.array([[1.0, 0.0, side / 2 - center[0]],
                           [0.0, 1.0, side / 2 - center[1]]])
        view = cv2.warpAffine(image, matrix, (side, side), borderMode=cv2.BORDER_CONSTANT)
        boxes, _points = detector.detect(view)
        matches = []
        for left, top, right, bottom, score in boxes:
            mapped = [float(left - matrix[0, 2]), float(top - matrix[1, 2]),
                      float(right - left), float(bottom - top)]
            if score >= 0.55 and evidence._overlap_smaller(mapped, original) >= 0.4:
                matches.append((float(score), mapped))
        if not matches:
            return original
        score, box = max(matches, key=lambda item: item[0])
        scores.append(score)
        local_boxes.append(box)
    if max(scores) < 0.65:
        return original
    for index, a in enumerate(local_boxes):
        for b in local_boxes[index + 1:]:
            intersection = evidence._overlap_smaller(a, b) * min(a[2] * a[3], b[2] * b[3])
            if intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection) < 0.65:
                return original
    corners = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in local_boxes])
    left, top, right, bottom = np.median(corners, axis=0)
    # A small margin covers the visible contour just outside predicted points;
    # apply it only to this independently confirmed localization.
    pad_x, pad_y = 0.05 * (right - left), 0.05 * (bottom - top)
    refined = [float(left - pad_x), float(top - pad_y),
               float(right - left + 2 * pad_x), float(bottom - top + 2 * pad_y)]
    if (refined[2] * refined[3] > 2 * width * height
            or abs((left + right) / 2 - center[0]) > 0.6 * width
            or abs((top + bottom) / 2 - center[1]) > 0.6 * height):
        return original
    return refined
