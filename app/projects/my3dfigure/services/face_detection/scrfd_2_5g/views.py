"""SCRFD-only upright view search for rotated source photos."""
from __future__ import annotations

import cv2
import numpy as np

from . import evidence, quality


def _map_points(points, angle: int, width: int, height: int):
    if angle == 90:
        return np.column_stack([points[:, 1], height - 1 - points[:, 0]])
    if angle == 180:
        return np.column_stack([width - 1 - points[:, 0],
                                height - 1 - points[:, 1]])
    return np.column_stack([width - 1 - points[:, 1], points[:, 0]])


def recover(image: np.ndarray, detector, present: list[list[float]]) -> list[list[float]]:
    """Accept only independently confirmed faces from orthogonal views."""
    height, width = image.shape[:2]
    recovered = []
    for angle in (90, 270, 180):
        if angle == 180 and (present or recovered):
            break
        view = cv2.rotate(image, {
            90: cv2.ROTATE_90_CLOCKWISE,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
        }[angle])
        detections, keypoints = detector.detect(view)
        for box, points in zip(detections, keypoints):
            score = float(box[4])
            if score < 0.65:
                continue
            corners = np.array([
                [box[0], box[1]], [box[2], box[1]],
                [box[0], box[3]], [box[2], box[3]],
            ])
            mapped = _map_points(corners, angle, width, height)
            low, high = mapped.min(axis=0), mapped.max(axis=0)
            candidate = {
                "box": [*map(float, low), *map(float, high - low)],
                "points": _map_points(np.asarray(points), angle, width, height),
                "score": score,
                "confirm": [1.0, 1.0],
                "views": [f"whole:{angle}"],
            }
            if any(evidence._overlap_smaller(candidate["box"], other) > 0.35
                   for other in present + recovered):
                continue
            candidate["features"] = quality.features(image, candidate)
            if candidate["features"]["visible"] == 0:
                continue
            candidate["confirm"] = evidence.confirm(image, candidate, detector)
            if min(candidate["confirm"]) < 0.7 or quality.quality(candidate):
                continue
            # A rotated recovery needs credible upright landmark proportions;
            # narrow predicted eyes with a distant mouth can be a muzzle.
            details = candidate["features"]
            if (details["eye_span"] < 0.2 and details["mouth_depth"] > 3
                    and min(candidate["confirm"]) < 0.8):
                continue
            if evidence.inconsistent_texture(image, candidate, detector):
                continue
            x, y, face_width, face_height = candidate["box"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2 = min(float(width), x + face_width)
            y2 = min(float(height), y + face_height)
            if x2 > x1 and y2 > y1:
                recovered.append([x1, y1, x2 - x1, y2 - y1])
    return recovered
