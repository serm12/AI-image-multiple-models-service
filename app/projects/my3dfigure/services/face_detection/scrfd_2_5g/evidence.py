"""Local SCRFD confirmation on the submitted image's own pixels."""
from __future__ import annotations

import cv2
import numpy as np

from . import appearance


def _overlap_smaller(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, min(a[2] * a[3], b[2] * b[3]))


def same_location(a, b) -> bool:
    intersection = _overlap_smaller(a,b) * min(a[2]*a[3],b[2]*b[3])
    union = a[2]*a[3]+b[2]*b[3]-intersection
    center_a=np.array(a[:2])+np.array(a[2:])/2
    center_b=np.array(b[:2])+np.array(b[2:])/2
    return (intersection / max(union,1) >= .25
            and np.linalg.norm(center_a-center_b) <= .5*max(a[2:]))


def confirm(image: np.ndarray, candidate: dict, detector, *, collect_boxes=False) -> list[float]:
    """Reobserve a proposed face in two upright local contexts."""
    x, y, width, height = candidate["box"]
    eyes = np.asarray(candidate["points"][:2], dtype=float)
    cache = getattr(detector, "evidence_cache", None)
    key = None
    if cache is not None:
        geometry = (tuple(map(float, candidate["box"])), tuple(map(float, eyes.ravel())))
        key = detector.evidence_key("confirm", image, geometry)
        cached = cache.get(key)
        if cached is not None:
            scores, local_boxes = cached
            if collect_boxes:
                candidate["local_boxes"] = [list(box) for box in local_boxes]
            return list(scores)
    delta = eyes[1] - eyes[0]
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    scores = []
    local_boxes = []
    for factor in (1.6, 2.3):
        side = max(64, round(max(width, height) * factor))
        center = (x + width / 2, y + height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix[:, 2] += np.array([side / 2, side / 2]) - np.array(center)
        view = cv2.warpAffine(image, matrix, (side, side), borderMode=cv2.BORDER_CONSTANT)
        boxes, _points = detector.detect(view)
        inverse = cv2.invertAffineTransform(matrix)
        found = []
        matched = []
        for left, top, right, bottom, score in boxes:
            corners = np.array([
                [left, top, 1], [right, top, 1],
                [left, bottom, 1], [right, bottom, 1],
            ]) @ inverse.T
            low, high = corners.min(axis=0), corners.max(axis=0)
            mapped = [*low, *(high - low)]
            if _overlap_smaller(mapped, candidate["box"]) >= 0.4:
                found.append(float(score))
                matched.append((float(score), mapped))
        scores.append(max(found, default=0.0))
        if matched:
            local_boxes.append(max(matched, key=lambda item: item[0])[1])
    if key is not None:
        cache[key] = (tuple(scores), tuple(tuple(box) for box in local_boxes))
    if collect_boxes:
        candidate["local_boxes"] = local_boxes
    return scores


def inconsistent_texture(image: np.ndarray, candidate: dict, detector) -> bool:
    """Require stable human-face evidence for weak, finely textured proposals.

    Fur can produce a sharp human-face proposal that depends on its context.
    Pixel sharpness alone is insufficient: require a large confidence change
    between the local views, or broad nasal morphology with weak confirmation.
    This uses only SCRFD, without a species classifier.
    """
    if not (0.5 <= candidate["score"] < 0.9
            and candidate["features"]["lap"] >= 60):
        return False
    if candidate["confirm"] == [1.0, 1.0]:
        candidate["confirm"] = confirm(image, candidate, detector)
    tight, wide = candidate["confirm"]
    # A large colorless obstruction also destabilizes local detection. Leave
    # its usability to quality analysis instead of inferring a furry muzzle.
    unstable_context = (tight < 0.7 and wide - tight > 0.12
                        and candidate["features"]["gray_fraction"] < 0.4)
    nasal_evidence = (min(tight, wide) < 0.8
                      and (appearance.broad_dark_nose(image, candidate)
                           or appearance.compact_dark_muzzle(image, candidate)))
    return unstable_context or nasal_evidence


def partial_box(candidate: dict) -> list[float]:
    """Refine a truncated weak box only when both local views agree."""
    original = candidate["box"]
    boxes = candidate.get("local_boxes", [])
    if len(boxes) != 2 or min(candidate["confirm"]) < 0.5:
        return original
    if _overlap_smaller(boxes[0], boxes[1]) < 0.8:
        return original
    if not all(b[2] >= 1.15 * original[2] and b[3] >= 1.15 * original[3]
               and b[2] * b[3] <= 3 * original[2] * original[3] for b in boxes):
        return original
    corners = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
    x1, y1, x2, y2 = np.median(corners, axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
