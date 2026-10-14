"""SCRFD-only evidence for unusable faces when no face can be counted."""
from __future__ import annotations

import cv2
import numpy as np

from . import quality


def central_tiny_face(image: np.ndarray, detector) -> bool:
    """A native-pixel center view can expose faces too small in the full frame."""
    height, width = image.shape[:2]
    if min(height, width) < 400:
        return False
    left, top = (width - 400) // 2, (height - 400) // 2
    detections, _ = detector.detect(image[top:top + 400, left:left + 400])
    return any(
        score >= 0.65 and 5 <= min(right - x, bottom - y) < 20
        for x, y, right, bottom, score in detections
    )


def washed_out_proposal(image: np.ndarray, detector) -> bool:
    """Use weak SCRFD evidence only to diagnose a clipped, unusable face."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 170 or float(np.mean(gray >= 245)) < 0.08:
        return False
    detections, keypoints = detector.detect(image, det_thresh=0.15)
    for (left, top, right, bottom, score), points in zip(detections, keypoints):
        if score < 0.15 or min(right - left, bottom - top) < 100:
            continue
        candidate = {
            "box": [float(left), float(top), float(right - left), float(bottom - top)],
            "points": points,
            "score": float(score),
            "confirm": [0.0, 0.0],
            "views": ["whole:0"],
        }
        details = quality.features(image, candidate)
        if details["visible"] and details["p50"] >= 245 and details["white"] > 0.5:
            return True
    return False


def clipped_face_silhouettes(image: np.ndarray) -> bool:
    """Recognize severe clipping from bright head-shaped regions and hair."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    white_fraction = float(np.mean(gray >= 245))
    if (white_fraction >= 0.6 and float(np.mean(gray)) >= 200
            and float(np.mean(gray[round(height * 0.7):] < 160)) >= 0.3):
        return True
    if white_fraction < 0.15:
        return False
    _, _, components, _ = cv2.connectedComponentsWithStats((gray >= 245).astype(np.uint8))
    heads = []
    for left, top, box_width, box_height, area in components[1:]:
        fraction = area / (width * height)
        if not (0.04 <= fraction <= 0.18
                and 0.5 <= box_width / max(box_height, 1) <= 1.0
                and 0.25 <= box_height / height <= 0.65
                and 0.1 <= top / height <= 0.5
                and 0.1 <= left / width <= 0.8):
            continue
        hair = gray[max(0, top - round(box_height * 0.1)):top,
                    left:left + box_width]
        if hair.size and float(np.mean(hair < 150)) >= 0.3:
            heads.append((left + box_width / 2) / width)
    return any(abs(first - second) >= 0.15
               for index, first in enumerate(heads)
               for second in heads[index + 1:])
