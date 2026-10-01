"""Recheck weak partial and small SCRFD proposals on source pixels."""
from __future__ import annotations

import numpy as np

from . import evidence, quality


def recover(image: np.ndarray, detector, present: list[list[float]]) -> list[list[float]]:
    height, width = image.shape[:2]
    detections, keypoints = detector.detect(image, det_thresh=0.15)
    recovered = []
    for (left, top, right, bottom, score), points in zip(detections, keypoints):
        if not (0.15 <= score < 0.5):
            continue
        boundary = left < 0 or top < 0 or right > width or bottom > height
        if not boundary and len(present) < 2:
            continue
        candidate = {
            "box": [float(left), float(top), float(right - left), float(bottom - top)],
            "points": points,
            "score": float(score),
            "confirm": [1.0, 1.0],
            "views": ["whole:0"],
        }
        if any(evidence._overlap_smaller(candidate["box"], other) > 0.35
               for other in present + recovered):
            continue
        candidate["features"] = quality.features(image, candidate)
        details = candidate["features"]
        if details["visible"] == 0:
            continue
        initial_quality = quality.quality(candidate)
        if boundary and initial_quality:
            continue
        if not boundary and initial_quality != "FACE_TOO_SMALL":
            continue
        candidate["confirm"] = evidence.confirm(
            image, candidate, detector, collect_boxes=boundary
        )
        reason = quality.quality(candidate)
        if boundary:
            compact_partial = (
                score >= 0.15 and min(candidate["confirm"]) >= 0.5
                and details["visible"] >= 0.75 and details["side"] >= 40
                and details["eye_span"] >= 0.15
            )
            large_partial = (
                score >= 0.25 and min(candidate["confirm"]) >= 0.5
                and details["visible"] >= 0.45 and details["side"] >= 100
                and details["eye_span"] >= 0.25
            )
            if reason or not (compact_partial or large_partial):
                continue
        else:
            readable_small = (
                reason == "FACE_TOO_SMALL" and score >= 0.17
                and min(candidate["confirm"]) >= 0.7
                and details["visible"] >= 0.8 and details["side"] >= 30
                and details["native_lap"] >= 100
                and 0.2 <= details["eye_span"] <= 0.5
            )
            if not readable_small:
                continue
        if evidence.inconsistent_texture(image, candidate, detector):
            continue
        if boundary:
            left, top, face_width, face_height = evidence.partial_box(candidate)
            right, bottom = left + face_width, top + face_height
        x1, y1 = max(0.0, float(left)), max(0.0, float(top))
        x2, y2 = min(float(width), float(right)), min(float(height), float(bottom))
        if x2 > x1 and y2 > y1:
            recovered.append([x1, y1, x2 - x1, y2 - y1])
    return recovered
