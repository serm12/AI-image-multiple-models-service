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


def refine_after_prune(image: np.ndarray, candidate: dict, detector) -> list[float]:
    """Localize a surviving weak profile using a shared context consensus.

    The caller limits this fallback to primary boxes that survived the final
    pruning unchanged. Three contexts must share one agreeing bridge frame,
    readable overlap and stable centers; this changes coordinates only.
    """
    original = candidate['box']
    x, y, width, height = original
    if not (0.5 <= candidate['score'] < 0.7
            and candidate['features']['eye_span'] < 0.2
            and min(candidate['confirm']) < 0.7
            and 0.65 <= height / max(width, 1) <= 2.2):
        return original
    center = np.array([x + width/2, y + height/2])
    frames, scores = [], []
    for factor in (1.6, 2.3, 3.2):
        side = max(64, round(max(width, height)*factor))
        matrix = np.array([[1., 0., side/2-center[0]], [0., 1., side/2-center[1]]])
        view = cv2.warpAffine(image, matrix, (side, side), borderMode=cv2.BORDER_CONSTANT)
        boxes, _points = detector.detect(view)
        matches = []
        for left, top, right, bottom, score in boxes:
            mapped = [float(left-matrix[0, 2]), float(top-matrix[1, 2]),
                      float(right-left), float(bottom-top)]
            if score >= .55 and evidence._overlap_smaller(mapped, original) >= .4:
                matches.append((float(score), mapped))
        if not matches:
            return original
        score, box = max(matches, key=lambda item: item[0])
        scores.append(score)
        frames.append(box)
    if max(scores) < .65:
        return original
    connected = [[False]*3 for _ in range(3)]
    for i, a in enumerate(frames):
        connected[i][i] = True
        for j in range(i+1, 3):
            b = frames[j]
            overlap = evidence._overlap_smaller(a, b)
            ca = np.array(a[:2])+np.array(a[2:])/2
            cb = np.array(b[:2])+np.array(b[2:])/2
            if overlap < .9 or np.linalg.norm(ca-cb) > .15*min(*a[2:], *b[2:]):
                return original
            inter = overlap*min(a[2]*a[3], b[2]*b[3])
            connected[i][j] = connected[j][i] = inter/max(1, a[2]*a[3]+b[2]*b[3]-inter) >= .65
    if not any(all(row) for row in connected):
        return original
    corners = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in frames])
    left, top, right, bottom = np.median(corners, axis=0)
    pad_x, pad_y = .05*(right-left), .05*(bottom-top)
    refined = [float(left-pad_x), float(top-pad_y),
               float(right-left+2*pad_x), float(bottom-top+2*pad_y)]
    if (refined[2]*refined[3] > 2*width*height
            or abs((left+right)/2-center[0]) > .6*width
            or abs((top+bottom)/2-center[1]) > .6*height):
        return original
    return refined



import cv2
import numpy as np
from . import evidence


def _observations(image, box, points, detector, angle, threshold):
    x, y, width, height = box
    center = np.array([x + width / 2, y + height / 2])
    observations = []
    for factor in (1.6, 2.3, 3.2):
        side = max(64, round(max(width, height) * factor))
        matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
        matrix[:, 2] += side / 2 - center
        view = cv2.warpAffine(image, matrix, (side, side))
        inverse = cv2.invertAffineTransform(matrix)
        boxes, landmarks = detector.detect(view, input_size=(640, 640), det_thresh=threshold)
        matches = []
        for b, p in zip(boxes, landmarks):
            mapped = np.column_stack([p, np.ones(5)]) @ inverse.T
            if np.max(np.linalg.norm(mapped - points, axis=1)) > .24 * min(width, height):
                continue
            if np.linalg.norm(mapped[:3].mean(0) - points[:3].mean(0)) > .16 * min(width, height):
                continue
            corners = np.array([[b[0], b[1], 1], [b[2], b[1], 1],
                                [b[0], b[3], 1], [b[2], b[3], 1]]) @ inverse.T
            low, high = corners.min(0), corners.max(0)
            mapped_box = [*low, *(high - low)]
            if evidence._overlap_smaller(mapped_box, box) < .7:
                continue
            # Rotating a full rectangle includes empty corner triangles. An
            # ellipse supported by the upright detector rectangle gives the
            # axis-aligned extent of its rounded face contour instead.
            radii = np.array([b[2] - b[0], b[3] - b[1]]) / 2
            extent = np.sqrt((inverse[:, :2] ** 2) @ (radii ** 2))
            midpoint = inverse @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.])
            contour = np.concatenate([midpoint - extent, midpoint + extent])
            matches.append((float(b[4]), np.concatenate([low, high]), contour, mapped))
        if matches:
            observations.append(max(matches, key=lambda row: row[0]))
    return observations


def refine_contour(image, box, candidate, detector):
    """Reposition the existing face only; no quality or admission decisions."""
    if candidate['score'] >= .9:
        return box
    width, height = box[2:]
    points = np.asarray(candidate['points'], dtype=float)
    view = candidate.get('views', ['whole:0'])[0].split(':')
    angle = float(view[3]) if view[0] == 'proposal' else float(view[1]) if view[0] == 'local' else 0.
    rotated_envelope = 25 <= abs(angle) <= 65
    features = candidate['features']
    # Native frontal frames have no rotated-corner inflation. Restrict new
    # contour reobservation to foreshortened profiles or rotated proposals.
    profile = features['eye_span'] <= .4 and abs(features['nose_side']) >= .3
    if not (profile or rotated_envelope):
        return box
    original = np.array([box[0], box[1], box[0] + width, box[1] + height])
    native = _observations(image, box, points, detector, 0., .4)
    if len(native) >= 2 and max(row[0] for row in native) >= .5:
        frames = np.array([row[1] for row in native])
        frames[:, :2] = np.maximum(frames[:, :2], [0., 0.])
        frames[:, 2:] = np.minimum(frames[:, 2:], [float(image.shape[1]), float(image.shape[0])])
        limits = np.array([width, height, width, height])
        shifts = frames - original
        consistent = ((shifts.min(0) > .07 * limits) | (shifts.max(0) < -.07 * limits))
        stable = np.ptp(frames, axis=0) < .12 * limits
        adjusted = np.where(consistent & stable, np.median(frames, axis=0), original)
        if np.max(np.abs(adjusted - original) / limits) <= .22 and np.any(adjusted != original):
            return [float(adjusted[0]), float(adjusted[1]),
                    float(adjusted[2] - adjusted[0]), float(adjusted[3] - adjusted[1])]
        if len(native) == 3:
            return box
    if not rotated_envelope:
        return box
    rotated = _observations(image, box, points, detector, angle, .4)
    if len(rotated) != 3 or max(row[0] for row in rotated) < .5:
        return box
    contours = np.array([row[2] for row in rotated])
    contours[:, :2] = np.maximum(contours[:, :2], [0., 0.])
    contours[:, 2:] = np.minimum(contours[:, 2:], [float(image.shape[1]), float(image.shape[0])])
    limits = np.array([width, height, width, height])
    if np.any(np.ptp(contours, axis=0) > .15 * limits):
        return box
    tight = np.median(contours, axis=0)
    # Do not exclude observed eyes/nose/mouth when shrinking a rounded frame.
    all_points = np.concatenate([row[3] for row in rotated])
    tight[:2] = np.minimum(tight[:2], all_points.min(0) - .03 * min(width, height))
    tight[2:] = np.maximum(tight[2:], all_points.max(0) + .03 * min(width, height))
    # A side-view nose protrudes beyond an oval cheek contour. Protect its
    # tip separately without restoring the empty rotated hair corners.
    noses = np.array([row[3][2] for row in rotated])
    tight[:2] = np.minimum(tight[:2], noses.min(0) - .12 * min(width, height))
    tight[2:] = np.maximum(tight[2:], noses.max(0) + .12 * min(width, height))
    # Compare actual image-space edges, not virtual padding outside the input.
    tight[:2] = np.maximum(tight[:2], [0., 0.])
    tight[2:] = np.minimum(tight[2:], [float(image.shape[1]), float(image.shape[0])])
    if np.any(np.abs(tight - original) > .25 * limits):
        return box
    return [float(tight[0]), float(tight[1]), float(tight[2] - tight[0]), float(tight[3] - tight[1])]


def refine_final_boxes(image, boxes, candidates, detector):
    """Preserve face order and reject ambiguous associations to nearby people."""
    output = []
    for box in boxes:
        center = np.array(box[:2]) + np.array(box[2:]) / 2
        matches = []
        for selected in candidates:
            candidate = selected['source_candidate']
            other = candidate['box']
            distance = np.linalg.norm(center - np.array(other[:2]) - np.array(other[2:]) / 2)
            if distance <= .25 * min(box[2:]) and evidence._overlap_smaller(box, other) >= .8:
                matches.append(candidate)
        refined = refine_contour(image, box, matches[0], detector) if len(matches) == 1 else box
        x, y, width, height = refined
        left, top = max(0., x), max(0., y)
        right, bottom = min(float(image.shape[1]), x + width), min(float(image.shape[0]), y + height)
        output.append([left, top, right - left, bottom - top] if refined != box and right > left and bottom > top else box)
    return output
