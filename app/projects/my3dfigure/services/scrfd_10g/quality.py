"""SCRFD-only source-pixel face quality evidence.

This module is independent of the YuNet and BlazeFace runtime pipelines.
"""
from __future__ import annotations

import cv2
import numpy as np

def features(image, candidate):
    h, w = image.shape[:2]
    x, y, bw, bh = candidate['box']
    left, top = (min(w, max(0, int(x))), min(h, max(0, int(y))))
    right, bottom = (max(0, min(w, int(np.ceil(x + bw)))), max(0, min(h, int(np.ceil(y + bh)))))
    roi = image[top:bottom, left:right]
    if roi.size == 0:
        return {'visible': 0}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA if min(gray.shape) > 160 else cv2.INTER_CUBIC)
    p = np.percentile(norm, [10, 50, 90])
    core = norm[29:112, 19:141]
    energies = []
    for axis in (0, 1):
        energy = np.abs(np.diff(gray.astype(np.float32), axis=axis)).sum(axis=1 - axis)
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
    for px, py in points[:4]:
        radius = max(3, round(min(bw, bh) * 0.09))
        patch = image[min(h, max(0, round(py) - radius)):max(0, min(h, round(py) + radius)), min(w, max(0, round(px) - radius)):max(0, min(w, round(px) + radius))]
        if not patch.size:
            patches.append([0.0, 0.0, 0.0])
            patch_gradients.append([0.0, 0.0])
            continue
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patches.append([float(np.percentile(patch, 5)), float(np.percentile(patch, 90) - np.percentile(patch, 10)), float(cv2.Laplacian(patch, cv2.CV_64F).var())])
        patch_gradients.append([float(cv2.Sobel(patch, cv2.CV_64F, 1, 0).var()), float(cv2.Sobel(patch, cv2.CV_64F, 0, 1).var())])
    return {'visible': float(gray.size / max(1, bw * bh)), 'side': float(min(bw, bh)), 'p10': float(p[0]), 'p50': float(p[1]), 'p90': float(p[2]), 'lap': float(cv2.Laplacian(norm, cv2.CV_64F).var()), 'native_lap': float(cv2.Laplacian(gray, cv2.CV_64F).var()), 'sx': float(cv2.Sobel(core, cv2.CV_64F, 1, 0).var()), 'sy': float(cv2.Sobel(core, cv2.CV_64F, 0, 1).var()), 'white': float((norm >= 245).mean()), 'eyes': patches, 'patch_gradients': patch_gradients, 'gray_fraction': float((roi.max(axis=2).astype(float) - roi.min(axis=2) <= 10).mean()), 'core_p90': float(np.percentile(core, 90)), 'core_white': float((core >= 245).mean()), 'block': energies, 'eye_span': float(span / max(bw, bh, 1)), 'mouth_depth': float(np.dot(points[3] - eyes.mean(axis=0), down) / max(span, 1)), 'nose_side': float(np.dot(points[2] - eyes.mean(axis=0), unit) / max(span, 1)), 'points_inside': sum((0 <= px < w and 0 <= py < h for px, py in points[:4]))}

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
