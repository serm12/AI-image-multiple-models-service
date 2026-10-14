"""Image-only v34 refinement, isolated copy for pre-install evaluation.

This file is intentionally importable without patching module globals. The
caller supplies the request-local edge-review record from the unified policy.
"""

from __future__ import annotations

import statistics

import cv2
import numpy as np


def _box(value):
    return tuple(map(int, value))


def _native_detail_is_below(policy, image, box, sharp_floor, contrast_floor):
    sharp, contrast = policy._native_detail(image, box)
    return sharp < sharp_floor and contrast < contrast_floor


def _refine_boxes(detector, policy, image, result):
    old_boxes = [_box(value) for value in result.get("usable_face_boxes", [])]
    boxes = list(old_boxes)
    trace = list(policy.TRACE)
    work = detector._request_work.get() or {}
    known = work.get("consistency_accepted_candidates", {})
    source_selections = {_box(row["source_box"]): row for row in trace
                         if row.get("stage") == "proposal-selected"}
    box_selections = {_box(row["box"]): row for row in trace
                      if row.get("stage") == "proposal-selected"}
    evidence = {_box(row["box"]): row for row in trace
                if row.get("stage") == "proposal-evidence"}
    guards = {_box(item["box"]): item["reason"] for row in trace
              if row.get("stage") == "feature-guard-rejected-additions"
              for item in row.get("items", [])}
    initial_row = next((row for row in trace if row.get("stage") == "complete-input"), {})
    initial_scores = {_box(item["box"]): float(item.get("score") or 0)
                      for item in initial_row.get("boxes", [])}

    # A repeated upright face can narrowly miss an independent mirror gate.
    for rejection in trace:
        if (rejection.get("stage") != "proposal-rejected"
                or rejection.get("reason") != "mirror_floor"):
            continue
        source = _box(rejection["box"])
        choice, proof = source_selections.get(source, {}), evidence.get(source, {})
        box = _box(rejection.get("chosen", ()))
        if (len(box) != 4 or choice.get("angle") != 0
                or float(proof.get("score", 0)) < .85
                or float(rejection.get("score", 0)) < .84
                or float(rejection.get("mirror_floor", 1))
                   - float(rejection.get("mirror_score", 0)) > .025
                or min(box[2:]) < 70
                or int(proof.get("strong", 0)) < 5
                or int(proof.get("contexts", 0)) < 3
                or len(proof.get("sizes", [])) < 2
                or any(choice.get(key) is not None for key in ("quality", "occlusion", "core"))
                or any(detector._box_overlap_over_smaller(box, other) >= .45 for other in boxes)):
            continue
        boxes.append(box)

    # The edge reviewer already checked geometry across orientations. Its
    # additional candidates need a real source-pixel quality check here.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for candidate in work.get("refinement_edge_profiles", []):
        box = _box(candidate["box"])
        if (guards.get(box) != "small_candidate_without_repeatability_record"
                or box in boxes or min(box[2:]) < 80
                or float(candidate.get("score", 0)) < .82
                or candidate.get("raw_face") is None):
            continue
        face = np.asarray(candidate["raw_face"])
        eyes = face[4:8].reshape(2, 2)
        eye_span = float(np.linalg.norm(eyes[0] - eyes[1])) / min(box[2:])
        if (eye_span < .08
                or detector._get_face_quality_issue(gray, box) is not None
                or detector._get_face_occlusion_issue(gray, box, face) == "FACE_OCCLUDED"
                or detector._native_extreme_motion_blur(image, box)
                or any(detector._box_overlap_over_smaller(box, other) >= .45 for other in boxes)):
            continue
        boxes.append(box)

    # Apply a symmetric native-detail gate to already retained and newly
    # localized tiny collage faces; detector score alone can match ink.
    boxes = [box for box in boxes if not (
        min(box[2:]) < 50
        and _native_detail_is_below(policy, image, box, 20, 40)
    )]
    for row in trace:
        if row.get("stage") != "feature-guard-rejected-additions":
            continue
        for item in row.get("items", []):
            if item.get("reason") != "insufficient_native_feature_detail":
                continue
            box = _box(item["box"])
            choice = box_selections.get(box, {})
            proof = evidence.get(_box(choice.get("source_box", ())), {})
            sharp, contrast = policy._native_detail(image, box)
            if (box in boxes or min(box[2:]) < 50
                    or float(choice.get("score", 0)) < .85
                    or int(proof.get("strong", 0)) < 6
                    or int(proof.get("contexts", 0)) < 3
                    or sharp < 25 or contrast < 40
                    or any(choice.get(key) is not None for key in ("quality", "core", "occlusion"))
                    or any(detector._box_overlap_over_smaller(box, other) >= .45 for other in boxes)):
                continue
            boxes.append(box)

    # Dense-scene weak evidence should not be grandfathered merely because
    # it arrived from the ordinary first pass.
    if len(boxes) >= 10:
        kept = []
        for box in boxes:
            score = float(known.get(box, {}).get("score") or initial_scores.get(box) or 0)
            if (min(box[2:]) >= 70 and 0 < score < .92
                    and _native_detail_is_below(policy, image, box, 60, 100)):
                continue
            kept.append(box)
        boxes = kept or boxes

    # For a weak outlier in a strong group, require two independent defects:
    # native occlusion/face failure and unstable low-score eye geometry.
    # Previously recovered edge profiles lack an ordinary proposal record and
    # must not be discarded by this gate.
    if len(boxes) >= 4:
        scores = {box: float(known.get(box, {}).get("score")
                             or initial_scores.get(box)
                             or box_selections.get(box, {}).get("score") or 0)
                  for box in boxes}
        if statistics.median(scores.values()) >= .92:
            kept = []
            for box in boxes:
                score = scores[box]
                if ((box not in initial_scores and box not in box_selections)
                        or not (.82 <= score < .89)):
                    kept.append(box)
                    continue
                candidate = known.get(box)
                if candidate is None or candidate.get("raw_face") is None:
                    kept.append(box)
                    continue
                native_occlusion = detector._get_face_occlusion_issue(
                    gray, box, candidate["raw_face"])
                if native_occlusion not in {"NO_FACE", "FACE_OCCLUDED"}:
                    kept.append(box)
                    continue
                observations = policy.context_evidence(detector, image, candidate)
                if not observations:
                    kept.append(box)
                    continue
                strongest = max(observations, key=lambda item: float(item["score"]))
                face = np.asarray(strongest["face"])
                eyes = face[4:8].reshape(2, 2)
                eye_span = float(np.linalg.norm(eyes[0] - eyes[1])) / min(box[2:])
                if float(strongest["score"]) < .89 and eye_span < .10:
                    continue
                kept.append(box)
            boxes = kept or boxes

    if boxes == old_boxes:
        return result
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in boxes]
    revised["face_count"] = revised["usable_face_count"] = len(boxes)
    revised["face"] = list(max(boxes, key=lambda box: box[2] * box[3]))
    return detector._with_usable_face_count(revised)


def _rescue_edge_eye_crop(detector, policy, image, native_baseline, result, locale):
    """Do not call a sharp, already localized edge-eye crop motion blur.

    This uses only current-image landmarks and source pixels. It is not an
    original-image lookup, filename exception, or cropped-upload branch.
    """
    if (result.get("usable_face_count") != 0
            or result.get("code") != "FACE_BLURRY"
            or native_baseline.get("code") != "FACE_BLURRY"):
        return result
    boxes = native_baseline.get("usable_face_boxes") or []
    if len(boxes) != 1:
        return result
    box = _box(boxes[0])
    work = detector._request_work.get() or {}
    candidate = work.get("consistency_accepted_candidates", {}).get(box)
    if candidate is None or candidate.get("raw_face") is None:
        return result
    score = float(candidate.get("score") or 0)
    if not (.85 <= score < .90) or candidate.get("quality_issue") is not None:
        return result
    height, width = image.shape[:2]
    if (min(box[2:]) < 200
            or box[2] * box[3] / float(width * height) < .10
            or not detector._frame_edges_touched(box, width, height)[3]):
        return result
    face = np.asarray(candidate["raw_face"])
    if face.size < 15 or not np.all(np.isfinite(face[:15])):
        return result
    eyes = face[4:8].reshape(2, 2)
    nose = face[8:10]
    if (not np.all((eyes[:, 0] >= 0) & (eyes[:, 0] < width)
                   & (eyes[:, 1] >= 0) & (eyes[:, 1] < height))
            or not (height < float(nose[1]) <= height + .15 * box[3])):
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (detector._get_face_quality_issue(gray, box) is not None
            or detector._native_face_core_issue(image, box, score) is not None
            or not detector._native_extreme_motion_blur(image, box)
            or detector._source_edge_crop_verdict(image, box) != "readable_bottom_crop"
            or not policy._eye_features_resolved(gray, box, face, strict=True)):
        return result
    revised = dict(result)
    revised.update(valid=True, usable_face_boxes=[list(box)], face=list(box),
                   face_count=1, usable_face_count=1,
                   message=detector.get_message("FACE_DETECTION_PASSED", locale))
    revised.pop("code", None)
    revised.pop("confirmed_quality_issues", None)
    return detector._with_usable_face_count(revised)


def refine(detector, policy, image_path, native_baseline, reviewed, locale=None):
    if reviewed.get("code") in {"IMAGE_READ_ERROR", "FACE_DETECTION_FAILED"}:
        return reviewed
    image = detector._read_image(image_path)
    if image is None:
        return reviewed
    result = _refine_boxes(detector, policy, image, reviewed)
    return _rescue_edge_eye_crop(detector, policy, image, native_baseline, result, locale)
