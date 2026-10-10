"""SCRFD-only source-pixel face quality evidence.

This module is independent of the YuNet and BlazeFace runtime pipelines.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from contextvars import ContextVar

import cv2
import numpy as np


# The context and all cached statistics belong to this analysis request only.
_SOURCE_FEATURES = ContextVar('source_features', default=None)


def begin_source_features(image):
    return _SOURCE_FEATURES.set((image, (image.dtype.str, image.shape, image.strides), OrderedDict()))


def end_source_features(token):
    _SOURCE_FEATURES.reset(token)


def features(image, candidate):
    context = _SOURCE_FEATURES.get()
    if (context is None or image is not context[0] or image.flags.writeable
            or (image.dtype.str, image.shape, image.strides) != context[1]):
        return _features_uncached(image, candidate)
    box = candidate['box']
    points = np.asarray(candidate['points'])
    if (len(box) != 4 or points.shape != (5, 2)
            or not all(type(v) is float for v in box) or points.dtype.hasobject
            or points.dtype.fields is not None or points.dtype.kind not in 'fiu'):
        return _features_uncached(image, candidate)
    key = (tuple(v.hex() for v in box), points.dtype.str, points.shape, points.tobytes())
    cache = context[2]
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return copy.deepcopy(cached)
    result = _features_uncached(image, candidate)
    # Fixed-schema source features have bounded size; never store view frames,
    # caller candidates or mutable returned dictionaries in the memo.
    if len(cache) >= 256:
        cache.popitem(last=False)
    cache[key] = copy.deepcopy(result)
    return result


# Geometry-only plans for the three fixed quality patches; no image data.
_PERCENTILE_PLANS = {}
for _shape, _qs in (((160, 160), (10, 50, 90)), ((32, 32), (5, 10, 90)), ((83, 122), (90,))):
    _index = (int(np.prod(_shape)) - 1) * (np.asarray(_qs, dtype=np.float64) / 100)
    _lower = np.floor(_index).astype(np.intp)
    _upper = np.minimum(_lower + 1, int(np.prod(_shape)) - 1)
    _PERCENTILE_PLANS[(_shape, _qs)] = (_lower, _upper, _index - _lower)
del _shape, _qs, _index, _lower, _upper


def _pixel_percentile(image, q):
    qs = (q,) if isinstance(q, (int, float)) else tuple(q)
    plan = _PERCENTILE_PLANS.get((image.shape, qs))
    if type(image) is not np.ndarray or image.dtype != np.uint8 or plan is None:
        return np.percentile(image, q)
    # At most 25,600 integer pixels: float32 histogram counts are exact.
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumulative = histogram.cumsum(dtype=np.int64)
    lower, upper, gamma = plan
    a = np.searchsorted(cumulative, lower, side='right').astype(np.float64)
    b = np.searchsorted(cumulative, upper, side='right').astype(np.float64)
    delta = b - a
    values = np.where(gamma >= .5, b - delta * (1 - gamma), a + delta * gamma)
    return values[0] if isinstance(q, (int, float)) else values


def _channel_range(roi):
    """Exact uint8 channel extrema, without per-pixel NumPy reductions."""
    if roi.dtype == np.uint8 and roi.ndim == 3 and roi.shape[2] == 3:
        blue, green, red = cv2.split(roi)
        high = cv2.max(cv2.max(blue, green), red)
        low = cv2.min(cv2.min(blue, green), red)
        # high >= low: saturated subtraction equals uint8 subtraction exactly.
        return cv2.subtract(high, low)
    return (roi.max(axis=2) - roi.min(axis=2) if roi.dtype == np.uint8
            else roi.max(axis=2).astype(float) - roi.min(axis=2))


def _features_uncached(image, candidate):
    h, w = image.shape[:2]
    x, y, bw, bh = candidate['box']
    left, top = (min(w, max(0, int(x))), min(h, max(0, int(y))))
    right, bottom = (max(0, min(w, int(np.ceil(x + bw)))), max(0, min(h, int(np.ceil(y + bh)))))
    roi = image[top:bottom, left:right]
    if roi.size == 0:
        return {'visible': 0}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA if min(gray.shape) > 160 else cv2.INTER_CUBIC)
    p = _pixel_percentile(norm, [10, 50, 90])
    core = norm[29:112, 19:141]
    energies = []
    gray_float = gray.astype(np.float32)
    for axis in (0, 1):
        energy = np.abs(np.diff(gray_float, axis=axis)).sum(axis=1 - axis)
        energies.append(float(np.sort(energy)[-max(1, len(energy) // 10):].sum() / max(float(energy.sum()), 1)))
    points = np.array(candidate['points'])
    eyes = points[:2]
    span = np.linalg.norm(eyes[1] - eyes[0])
    unit = (eyes[1] - eyes[0]) / max(span, 1)
    down = np.array([-unit[1], unit[0]])
    if np.dot(points[3] - eyes.mean(axis=0), down) < 0:
        down = -down
    patches = []
    patch_gradients = []
    landmark_patches = []
    landmark_outside = [not (0 <= px < w and 0 <= py < h) for px, py in points[:5]]
    needs_partial_landmarks = (
        x < 0 or y < 0 or x + bw > w or y + bh > h or any(landmark_outside)
    )
    for index, (px, py) in enumerate(points[:5]):
        if index == 4 and not needs_partial_landmarks:
            break
        radius = max(3, round(min(bw, bh) * 0.09))
        patch = image[min(h, max(0, round(py) - radius)):max(0, min(h, round(py) + radius)), min(w, max(0, round(px) - radius)):max(0, min(w, round(px) + radius))]
        if not patch.size:
            metrics = [0.0, 0.0, 0.0]
            if len(landmark_patches) < 4:
                patches.append(metrics)
                patch_gradients.append([0.0, 0.0])
            landmark_patches.append(metrics)
            continue
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patch_levels = _pixel_percentile(patch, [5, 10, 90])
        metrics = [float(patch_levels[0]), float(patch_levels[2] - patch_levels[1]), float(cv2.Laplacian(patch, cv2.CV_64F).var())]
        landmark_patches.append(metrics)
        if len(patches) < 4:
            patches.append(metrics)
            patch_gradients.append([float(cv2.Sobel(patch, cv2.CV_64F, 1, 0).var()), float(cv2.Sobel(patch, cv2.CV_64F, 0, 1).var())])
    color_range = _channel_range(roi)
    return {'visible': float(gray.size / max(1, bw * bh)), 'side': float(min(bw, bh)), 'extent_fraction': float(bw * bh / max(1, w * h)), 'landmark_outside': landmark_outside, 'p10': float(p[0]), 'p50': float(p[1]), 'p90': float(p[2]), 'lap': float(cv2.Laplacian(norm, cv2.CV_64F).var()), 'native_lap': float(cv2.Laplacian(gray, cv2.CV_64F).var()), 'sx': float(cv2.Sobel(core, cv2.CV_64F, 1, 0).var()), 'sy': float(cv2.Sobel(core, cv2.CV_64F, 0, 1).var()), 'white': float((norm >= 245).mean()), 'eyes': patches, 'patch_gradients': patch_gradients, 'landmark_patches': landmark_patches, 'gray_fraction': float((color_range <= 10).mean()), 'core_p90': float(_pixel_percentile(core, 90)), 'core_white': float((core >= 245).mean()), 'block': energies, 'eye_span': float(span / max(bw, bh, 1)), 'mouth_depth': float(np.dot(points[3] - eyes.mean(axis=0), down) / max(span, 1)), 'nose_side': float(np.dot(points[2] - eyes.mean(axis=0), unit) / max(span, 1)), 'points_inside': sum((0 <= px < w and 0 <= py < h for px, py in points[:4]))}

def quality(c):
    """Reject strong source-pixel defects while preserving readable dark faces.

    Normalized detail, bilateral eye patches and core brightness are combined:
    a single low sharpness value must not reject an otherwise readable face.
    These are heuristic quality checks, not a learned occlusion classifier.
    """
    f = c['features']
    if f['visible'] < 0.4:
        return 'FACE_INCOMPLETE'
    if f['visible'] < 0.75 and f['points_inside'] < 3 and f['side'] < 120 and c['score'] < 0.6 and max(c.get('confirm',[0])) == 0 and len(set(c['views'])) < 5:
        return 'FACE_INCOMPLETE'
    detailed_small = (f['side'] >= 20 and ((c['score'] >= 0.7 and min(c.get('confirm', [0])) >= 0.6) or (c['score'] >= 0.65 and min(c.get('confirm', [0])) >= 0.7))
                       and f['lap'] >= 3
                       and min(p[1] for p in f['eyes']) >= 30
                       and min(p[2] for p in f['eyes'][:2]) >= 40)
    if f['side'] < 29 and not detailed_small and (not (f['side'] >= 24 and c['score'] >= 0.8 and (min(c.get('confirm', [0])) >= 0.6))):
        return 'FACE_TOO_SMALL'
    # Small faces need real pixel detail; enlargement cannot recover it.
    visible_eye_detail = (min(p[1] for p in f['eyes'][:2]) >= 40
                          and min(p[2] for p in f['eyes'][:2]) >= 50)
    readable_small = (f['side'] >= 29 and c['score'] >= 0.7
                      and max(c.get('confirm', [0])) >= 0.6
                      and visible_eye_detail)
    if f['side'] < 40 and f['lap'] < 3 and c['score'] < 0.8 and not readable_small:
        return 'FACE_TOO_SMALL'
    if f['side'] < 40 and f['lap'] < 4 and c['score'] < 0.65 and not visible_eye_detail:
        return 'FACE_TOO_SMALL'
    if (f['side'] < 40 and c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.6
            and not (len(set(c['views'])) >= 3 and max(c.get('confirm',[0])) >= 0.5
                     and min(p[1] for p in f['eyes']) >= 30)):
        return 'FACE_TOO_SMALL'
    if (f['side'] < 100 and f['p50'] < 60 and f['lap'] < 10
            and c['score'] < 0.8 and max(c.get('confirm', [0])) < 0.8
            and min(p[1] for p in f['eyes']) < 15
            and min(p[2] for p in f['eyes'][:2]) < 12):
        return 'FACE_BLURRY'
    # A weak small proposal needs visible detail around both nose and mouth,
    # not just the exposed eyes above a dark obstruction.
    if (f['side'] < 100 and f['p50'] < 60 and f['lap'] < 10
            and c['score'] < 0.8 and max(c.get('confirm', [0])) < 0.8
            and min(p[0] for p in f['eyes'][:2]) >= 20
            and ((f['eyes'][2][0] < 20 and f['eyes'][3][0] < 15 and f['eyes'][3][2] < 20)
                 or (min(c.get('confirm',[0])) < 0.5 and f['eyes'][2][0] <= 25
                     and f['eyes'][3][0] < 20 and f['eyes'][3][2] < 40))):
        return 'FACE_OCCLUDED'
    if f['core_p90'] < 24:
        return 'FACE_TOO_DARK'
    if (f['side'] < 100 and f['p50'] < 35 and f['lap'] < 8
            and c['score'] < 0.65 and max(c.get('confirm',[0])) < 0.8):
        return 'FACE_BLURRY'
    if (c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.7
            and f['side'] >= 120 and min(p[0] for p in f['eyes'][:2]) > 90
            and min(p[1] for p in f['eyes'][:2]) < 25 and f['eyes'][2][0] > 90):
        return 'FACE_OCCLUDED'
    if (f['side'] < 100 and f['p50'] < 45 and f['lap'] < 3
            and c['score'] < 0.8 and max(c.get('confirm',[0])) < 0.8):
        return 'FACE_BLURRY'
    if (f['side'] < 150 and f['p50'] < 25 and f['lap'] < 30
            and c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.7
            and abs(f['nose_side']) < 0.5):
        return 'FACE_TOO_DARK'
    stable_dark_detail = (c['score'] >= 0.8 and max(c.get('confirm', [0])) >= 0.65
                          and len(set(c['views'])) >= 2 and f['native_lap'] >= 2.5)
    readable_dark = f['side'] >= 120 and f['p90'] >= 24 and (f['p90'] - f['p10'] >= 18) and (f['lap'] >= (2.5 if stable_dark_detail else 3)) and (f['p50'] >= 12 or max((e[1] for e in f['eyes'][:2])) >= 12)
    if f['p90'] < 45:
        if not readable_dark:
            return 'FACE_TOO_DARK'
        return None
    if f['p10'] > 240:
        return 'FACE_OVEREXPOSED'
    eyes = f['eyes'][:2]
    if (80 <= f['side'] < 150 and c['score'] < .8 and f['lap'] < 4
            and min(p[1] for p in eyes) < 25 and min(p[2] for p in eyes) < 3
            and min(c.get('confirm',[0])) < .85):
        return 'FACE_BLURRY'
    # Readability should not disappear merely because a face occupies more
    # pixels after an image export. Require balanced, visible eye detail.
    readable_soft = (c['score'] >= 0.65 and len(set(c['views'])) >= 3
                     and f['lap'] >= 2.5
                     and min(p[1] for p in eyes) >= 25
                     and min(p[1] for p in eyes) >= 0.6 * max(p[1] for p in eyes)
                     and max(p[2] for p in eyes) >= 4
                     and min(p[1] for p in f['eyes']) >= (18 if abs(f['nose_side']) >= 0.5 and max(c.get('confirm',[0])) >= 0.5 else 25))
    if not readable_soft and f['side'] >= 150 and f['lap'] < 6 and (f['sx'] < 450) and (f['sy'] < 450) and (max((p[2] for p in eyes)) < 15) and (max((p[1] for p in eyes)) < 60):
        return 'FACE_BLURRY'
    if not readable_soft and f['side'] >= 150 and f['visible'] >= 0.85 and (max((p[2] for p in f['eyes'])) < 10) and (max((p[1] for p in f['eyes'])) < 55):
        return 'FACE_BLURRY'
    if 45 <= f['side'] < 80 and f['p50'] < 45 and (f['lap'] < 6) and (min((p[1] for p in eyes)) < 20):
        return 'FACE_BLURRY'
    gradients = f.get('patch_gradients', [])
    if not readable_soft and gradients and f['side'] >= 250 and (max((p[2] for p in f['eyes'])) < 100):
        if max(sum((x < 0.3 * y for x, y in gradients)), sum((y < 0.3 * x for x, y in gradients))) >= 3:
            return 'FACE_BLURRY'
    if f['core_white'] > 0.5 and min((e[0] for e in eyes)) > 180:
        return 'FACE_OVEREXPOSED'
    if f['lap'] < 1.8 or (f['lap'] < 6 and max((e[2] for e in eyes)) < 8 and (min((e[1] for e in eyes)) < 20)):
        return 'FACE_BLURRY'
    if not readable_soft and f['side'] >= 200 and f['sx'] < 450 and (f['sy'] < 1000) and (max((e[1] for e in eyes)) < 45):
        return 'FACE_BLURRY'
    if f['side'] >= 80 and min(f['block']) > 0.3 and (max((e[2] for e in eyes)) < 200):
        return 'FACE_UNRECOGNIZABLE'
    if f['side'] >= 150 and f['lap'] >= 300 and (min((e[0] for e in eyes)) > 60) and (max((e[0] for e in eyes)) > 90) and (max((e[1] for e in eyes)) < 100) and (f['side'] >= 250 or f['sx'] > 3000):
        return 'FACE_OCCLUDED'
    return None


def asymmetric_blur(candidate):
    """Detect a flat or directionally smeared eye despite a strong face box."""
    f = candidate["features"]
    eyes = f["eyes"][:2]
    score = candidate["score"]
    flat_eye = (
        score >= 0.8 and f["side"] >= 100 and f["lap"] < 10
        and min(p[1] for p in eyes) < 12
        and max(p[1] for p in eyes) > 50
        and max(p[2] for p in eyes) < 15
    )
    gradients = f["patch_gradients"][:2]
    directional_smear = (
        score >= 0.85 and f["side"] >= 200 and f["lap"] < 30
        and min(g[0] for g in gradients) < 0.06 * max(g[0] for g in gradients)
        and min(p[1] for p in eyes) < 0.45 * max(p[1] for p in eyes)
        and min(p[2] for p in eyes) < 30
    )
    return flat_eye or directional_smear


def blocky_eye_pixels(candidate):
    """Reject hard video blocks masquerading as sharp eye details."""
    f = candidate["features"]
    eyes = f["eyes"][:2]
    return (
        f["side"] < 120 and max(f["block"]) > 0.3
        and max(p[0] for p in eyes) < 5
        and min(p[1] for p in eyes) > 180
        and min(p[2] for p in eyes) > 500
    )


def partial_landmark_state(image_shape, candidate):
    """Return whether a truncated weak face has reliable visible landmarks.

    SCRFD's five points remain useful when the full face proposal is too weak
    to count. Local source pixels reject tiny texture boxes and landmarks that
    land on flat/background regions.
    """
    height, width = image_shape[:2]
    x, y, box_width, box_height = candidate["box"]
    truncated = (
        x < 0 or y < 0 or x + box_width > width or y + box_height > height
        or any(px < 0 or py < 0 or px >= width or py >= height
               for px, py in candidate["points"][:5])
    )
    details = candidate.get("features") or {}
    side = details.get("side", min(box_width, box_height))
    score = candidate.get("score", 0.0)
    if not truncated or side < 80 or score < 0.05:
        return False, False

    # A slightly virtual box edge does not truncate the facial landmarks.
    # Keep ordinary, well-contained faces in a broader scene on their
    # existing quality path; diagnose materially cropped proposals below.
    if (not any(details.get("landmark_outside", []))
            and details.get("visible", 0) >= .85
            and details.get("extent_fraction", 1) < .4):
        return False, False

    landmarks = np.asarray(candidate["points"][:5], dtype=float)
    metrics = details.get("landmark_patches", [])
    if len(metrics) < 5:
        return False, False

    eye_margin = .25 * float(np.linalg.norm(landmarks[1] - landmarks[0]))
    effective_outside = list(details.get("landmark_outside", [False] * 5))
    visible_support = []
    fragment_support = []
    for index, ((px, py), (low, contrast, sharpness)) in enumerate(zip(landmarks, metrics)):
        if not (1.0 <= px < width - 1.0 and 1.0 <= py < height - 1.0):
            visible_support.append(False)
            fragment_support.append(False)
            continue
        if index < 2:
            if not (eye_margin <= px < width - eye_margin):
                effective_outside[index] = True
                visible_support.append(False)
                fragment_support.append(low >= 35 and contrast >= 40 and sharpness >= 80)
                continue
            support = low >= 35 and contrast >= 40 and sharpness >= 80
        else:
            support = low >= 25 and contrast >= 20 and sharpness >= 10
            fragment_support.append(support)
            if index == 2 and not (.035 * side <= px < width - .035 * side):
                effective_outside[index] = True
                support = False
        visible_support.append(support)
        if index < 2:
            fragment_support.append(support)
    details["partial_landmark_support"] = visible_support
    details["landmark_outside"] = effective_outside
    details["partial_fragment_support"] = fragment_support

    eye_visible = any(visible_support[:2])
    both_eyes_visible = all(visible_support[:2])
    nose_visible = visible_support[2]
    mouth_visible = any(visible_support[3:5])
    if not eye_visible and side < 120:
        return any(fragment_support), False
    high_quality_context = (side >= 120 and details.get("native_lap", 0) >= 100)
    enough_to_use = high_quality_context and (
        (score >= 0.5 and eye_visible and nose_visible and mouth_visible
         and details.get("visible", 0) >= 0.35
         and (details.get("visible", 0) >= .4 or details.get("points_inside", 0) >= 3))
        or (score >= 0.5 and nose_visible
            and not any(effective_outside[:2])
            and all(p[0] >= 35 and p[1] >= 40 and p[2] >= 50 for p in metrics[:2])
            and details.get("visible", 0) >= 0.65)
        or (score >= 0.28 and eye_visible and mouth_visible
            and (details.get("visible", 0) >= 0.45
                 or (details.get("extent_fraction", 1) < .5
                     and details.get("visible", 0) >= .4)))
        or (score >= 0.6 and side >= 200 and eye_visible and mouth_visible
            and details.get("visible", 0) >= 0.4
            and details.get("native_lap", 0) >= 200)
    )
    # The visible nasal bridge may stop just before the model's nose-tip
    # point. Require both whole eyes and real texture at that boundary.
    nose_patch = metrics[2]
    bridge_visible = (both_eyes_visible and score >= .65 and side >= 120
        and details.get("visible", 0) >= .5 and high_quality_context
        and landmarks[2, 1] >= height
        and landmarks[2, 1] - height <= .09 * side
        and 0 <= landmarks[2, 0] < width
        and nose_patch[0] >= 25 and nose_patch[1] >= 25 and nose_patch[2] >= 5)
    return any(fragment_support), enough_to_use or bridge_visible

def should_mark_partial_face_incomplete(candidate):
    """Require a recognizable but materially truncated feature pattern."""
    details = candidate.get("features") or {}
    support = details.get("partial_fragment_support", details.get("partial_landmark_support", []))
    if len(support) < 5:
        return False
    # Stable native eye detail can survive a cropped lower face even when
    # the model projects its nose and mouth below the image boundary.
    patches = details.get("landmark_patches", [])
    if (candidate.get("score", 0) >= .5
            and min(candidate.get("confirm", [0])) >= .7
            and details.get("points_inside", 0) >= 2
            and all(details.get("landmark_outside", [False] * 5)[3:5])
            and details.get("native_lap", 0) >= 20
            and len(patches) >= 2
            and all(p[0] >= 35 and p[1] >= 40 and p[2] >= 50 for p in patches[:2])):
        return False
    eye_count = sum(bool(value) for value in support[:2])
    nose = bool(support[2])
    mouth_count = sum(bool(value) for value in support[3:5])
    extent = details.get("extent_fraction", 0)
    visible = details.get("visible", 0)
    score = candidate.get("score", 0)
    outside = details.get("landmark_outside", [False] * 5)
    other_feature_visible = bool(nose or mouth_count)
    visible_eye_fragment = (
        eye_count > 0 and any(outside[:2]) and extent >= 0.09 and score >= 0.05
        and (score >= 0.12 or nose or (extent <= 0.2 and details.get("native_lap", 0) >= 100))
    )
    dominant_fragment = extent >= 1.0 and eye_count >= 2 and any(outside) and score >= 0.08
    upper_face_fragment = (
        eye_count > 0 and 0.1 <= score < 0.6 and extent >= 0.4
        and (all(outside[2:5]) or (nose and mouth_count == 0 and all(outside[3:5])))
    )
    low_score_crop = (
        score >= 0.05 and visible < 0.32 and mouth_count > 0
        and (nose or extent >= 0.75)
    )
    compact_nose_mouth = (
        nose and mouth_count > 0 and score >= 0.09
        and extent <= 1.0 and visible < 0.4
    )
    clearer_crop = (
        nose and mouth_count > 0 and score >= 0.11
        and extent <= 2.0 and visible >= 0.4
    )
    lower_feature_crop = (
        eye_count == 0 and all(outside[:2]) and other_feature_visible
        and extent >= 0.15
        and (score >= 0.18 or low_score_crop or compact_nose_mouth or clearer_crop)
    )
    collapsed_partial = (extent >= .4 and score >= .3
        and details.get("eye_span", 1) < .15 and details.get("mouth_depth", 0) > 2.8
        and not nose and eye_count > 0 and mouth_count > 0)
    return visible_eye_fragment or dominant_fragment or upper_face_fragment or lower_feature_crop or collapsed_partial
