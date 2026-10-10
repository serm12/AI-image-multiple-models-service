"""Independent SCRFD-2.5G-KPS detection, source quality and local evidence."""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from app.projects.my3dfigure.core.i18n import (
    get_double_face_error_message,
    get_message,
    get_single_face_error_message,
)
from . import fast_inference, diagnosis, evidence, graphics, localization, quality, recheck, runtime, views, weak_edges

_LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()
_TECHNICAL_CODES = frozenset({"IMAGE_READ_ERROR", "FACE_DETECTION_FAILED"})
MODEL_PATH = (
    Path(__file__).resolve().parents[5]
    / "models"
    / "my3dfigure"
    / "scrfd_2_5g_kps"
    / "det_2.5g.onnx"
)
MODEL_SHA256 = "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9"


# Private to 2.5G: share CPU cores between overlapping requests without
# changing tensors, weights, thresholds or the detection/quality sequence.
_REQUEST_LOCK = threading.Lock()
_ACTIVE_REQUESTS = 0


def _begin_request():
    global _ACTIVE_REQUESTS
    with _REQUEST_LOCK:
        _ACTIVE_REQUESTS += 1


def _end_request():
    global _ACTIVE_REQUESTS
    with _REQUEST_LOCK:
        _ACTIVE_REQUESTS -= 1


def _session_for_load(input_size):
    with _REQUEST_LOCK:
        threads = 4 if _ACTIVE_REQUESTS <= 1 else 2
    return _native_detector(threads)._session_for_input_size(input_size)


def _detector():
    active = getattr(_THREAD_LOCAL, "request_detector", None)
    if active is not None:
        return active
    return _native_detector(4)


def _native_detector(intra_threads):
    cache_name = "parallel_detector" if intra_threads == 2 else "detector"
    detector = getattr(_THREAD_LOCAL, cache_name, None)
    if detector is not None:
        return detector
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"SCRFD-2.5G-KPS model missing: {MODEL_PATH}")
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("SCRFD-2.5G-KPS model does not match the official tested weight")
    from insightface.model_zoo import get_model
    import onnxruntime as ort

    options=ort.SessionOptions()
    options.intra_op_num_threads=intra_threads
    options.inter_op_num_threads=1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    detector = get_model(str(MODEL_PATH), providers=["CPUExecutionProvider"],sess_options=options)
    if detector is None or not detector.use_kps:
        raise RuntimeError("SCRFD-2.5G-KPS detector or keypoints unavailable")
    detector.prepare(ctx_id=-1)
    fast_inference.install(detector)
    detector._my3dfigure_session_for_size = _session_for_load
    setattr(_THREAD_LOCAL, cache_name, detector)
    return detector


def _read_image(image_path: str):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, ValueError, cv2.error):
        return None


def analyze_human_faces(image_path: str, locale: str | None = None) -> dict:
    """Run SCRFD before applying a target face count."""
    image = _read_image(image_path)
    if image is None:
        return {
            "valid": False,
            "code": "IMAGE_READ_ERROR",
            "message": get_message("IMAGE_READ_ERROR", locale),
        }
    feature_token = None
    request_active = False
    try:
        _begin_request()
        request_active = True
        _THREAD_LOCAL.request_detector = runtime.RequestDetector(_detector())
        source_height, source_width = image.shape[:2]
        if max(source_height, source_width) > 2000:
            scale = 2000 / max(source_height, source_width)
            image = cv2.resize(
                image,
                (round(source_width * scale), round(source_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        image = _THREAD_LOCAL.request_detector.register_source(image)
        feature_token = quality.begin_source_features(image)
        detections, keypoints = _detector().detect(image)
        height, width = image.shape[:2]
        factor_x, factor_y = source_width / width, source_height / height
        selected = []
        issues = []
        issue_support = []
        incomplete_boxes = []
        strongly_turned_away = False
        strongly_motion_blurred = False
        for (left, top, right, bottom, score), points in zip(detections, keypoints):
            # Very weak boundary proposals can have collapsed eye and mouth
            # landmarks. Their large boxes are unstable after JPEG resizing.
            side = max(1.0, min(float(right - left), float(bottom - top)))
            eye_span = math.dist(points[0], points[1]) / side
            mouth_span = math.dist(points[3], points[4]) / side
            beyond_image = left < 0 or top < 0 or right > width or bottom > height
            if (beyond_image and score < 0.55
                    and eye_span < 0.04 and mouth_span < 0.04):
                continue
            candidate = {
                "box": [float(left), float(top), float(right - left), float(bottom - top)],
                "points": points,
                "score": float(score),
                "confirm": [1.0, 1.0],
                "views": ["whole:0"],
            }
            candidate["features"] = quality.features(image, candidate)
            details = candidate["features"]
            reason = quality.quality(candidate)
            if score < 0.8 or reason is not None:
                candidate["confirm"] = evidence.confirm(image, candidate, _detector())
                reason = quality.quality(candidate)
            has_visible_landmark, sufficiently_visible = quality.partial_landmark_state(
                image.shape, candidate
            )
            if (has_visible_landmark and not sufficiently_visible
                    and reason in (None, "FACE_INCOMPLETE")
                    and quality.should_mark_partial_face_incomplete(candidate)):
                issues.append("FACE_INCOMPLETE")
                issue_support.append(max(candidate["confirm"]))
                incomplete_boxes.append(list(candidate["box"]))
                continue
            if (score >= 0.5 and details["side"] >= 140
                    and details["eye_span"] < 0.08
                    and (abs(details["nose_side"]) > 2
                         or details["mouth_depth"] > 5)
                    and max(candidate["confirm"]) < 0.1):
                strongly_turned_away = True
            if (score >= 0.5 and details["side"] >= 100
                    and max(candidate["confirm"]) < 0.1 and reason is None):
                x1, y1 = max(0, round(left)), max(0, round(top))
                x2, y2 = min(width, round(right)), min(height, round(bottom))
                if x2 > x1 and y2 > y1:
                    crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    horizontal = float(np.mean(np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0))))
                    vertical = float(np.mean(np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1))))
                    if min(horizontal, vertical) > 3 and (
                        max(horizontal, vertical) / min(horizontal, vertical) > 3
                    ):
                        strongly_motion_blurred = True
            if reason == "FACE_BLURRY" and (
                min(candidate["confirm"]) >= 0.87
                or (score >= 0.875 and min(candidate["confirm"]) >= 0.85)
            ):
                reason = None
            if details["visible"] == 0:
                continue
            if (reason == "FACE_BLURRY" and 0.75 <= score < 0.82
                    and min(candidate["confirm"]) >= 0.75
                    and 40 <= details["side"] < 80
                    and details["native_lap"] >= 50):
                reason = None
            if (reason == "FACE_TOO_SMALL" and score >= 0.6
                    and min(candidate["confirm"]) >= 0.75
                    and details["side"] >= 24
                    and details["native_lap"] >= 100
                    and 0.2 <= details["eye_span"] <= 0.55
                    and details["points_inside"] >= 4):
                reason = None
            if reason == "FACE_TOO_SMALL" and recheck.compact_visible_face(candidate):
                reason = None
            if reason == "FACE_BLURRY" and recheck.stable_soft_details(candidate):
                reason = None
            weak_landmarks = (
                score < 0.8 and min(candidate["confirm"]) < 0.7
                and details["mouth_depth"] > 3
                and (details["eye_span"] < 0.08
                     or abs(details["nose_side"]) > 2)
            )
            eye_patches = details["eyes"][:2]
            one_eye_flat = (
                score < 0.8 and min(candidate["confirm"]) < 0.8
                and min(patch[1] for patch in eye_patches) < 8
                and max(patch[1] for patch in eye_patches) >= 40
                and min(patch[2] for patch in eye_patches) < 5
            )
            if weak_landmarks or (one_eye_flat and not sufficiently_visible):
                if reason is not None:
                    issues.append(reason)
                    issue_support.append(max(candidate["confirm"]))
                continue
            weak_native_detail = (
                details["side"] >= 80 and details["lap"] < 6
                and details["native_lap"] < 10
                and min(candidate["confirm"]) < 0.7 and score < 0.75
            )
            if weak_native_detail:
                continue
            if quality.unreliable_detail(candidate):
                continue
            if quality.asymmetric_blur(candidate):
                issues.append("FACE_BLURRY")
                issue_support.append(max(candidate["confirm"]))
                continue
            if quality.blocky_eye_pixels(candidate):
                continue
            if score < 0.55 and min(candidate["confirm"]) < 0.5:
                continue
            if score < 0.75 and max(candidate["confirm"]) < 0.5:
                continue
            if reason is not None:
                issues.append(reason)
                issue_support.append(max(candidate["confirm"]))
                continue
            if evidence.inconsistent_texture(image, candidate, _detector()):
                continue
            x1 = max(0.0, min(float(width), float(left)))
            y1 = max(0.0, min(float(height), float(top)))
            x2 = max(x1, min(float(width), float(right)))
            y2 = max(y1, min(float(height), float(bottom)))
            selected.append({
                "box": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "source_candidate": candidate,
            })
        strong_primary = [c['box'] for c in selected if c['score'] >= .8]
        selected = [c for c in selected
                    if not recheck.unsupported_native_texture(c['source_candidate'])
                    and (c['score'] >= .8 or c['source_candidate']['features']['side'] >= 120
                         or recheck._linked_recovery_confirmation(
                        image, c['source_candidate'], _detector(), strong_primary))]
        accepted = graphics.filter_candidates(image, selected)
        if incomplete_boxes:
            # A second, tighter proposal for the same clipped face can pass
            # the ordinary quality gate. Do not let it restore a face already
            # rejected as incomplete from the primary detector pass.
            accepted = [
                candidate for candidate in accepted
                if not any(
                    evidence._overlap_smaller(candidate["box"], incomplete) >= 0.45
                    for incomplete in incomplete_boxes
                )
            ]
        working_boxes = [candidate["box"] for candidate in accepted]
        primary_count = len(working_boxes)
        accepted_primary_boxes = [list(box) for box in working_boxes]
        if primary_count <= 1:
            recovered_views = views.recover(image, _detector(), working_boxes)
            if primary_count == 0 and incomplete_boxes:
                recovered_views = [
                    box for box in recovered_views
                    if not any(evidence._overlap_smaller(box, incomplete) >= 0.25
                               for incomplete in incomplete_boxes)
                ]
            working_boxes.extend(recovered_views)
        if primary_count >= 2 or (primary_count == 0 and not working_boxes):
            working_boxes.extend(weak_edges.recover(
                image, _detector(), working_boxes,
                issues=issues, issue_support=issue_support,
                issue_boxes=incomplete_boxes,
            ))
        reassess = recheck.supports_reassessment(image, _detector(), working_boxes)
        new_faces = []
        profile_source_pool = []
        if reassess:
            new_faces = recheck.recover(
                image, _detector(), working_boxes, accepted, collect_candidates=True,
                source_pool=profile_source_pool
            )
            working_boxes.extend(candidate["box"] for candidate in new_faces)
        # Localize only after all quality, counting and recovery decisions;
        # altered frame coordinates must not influence any face's acceptance.
        for index, candidate in enumerate(accepted):
            original = candidate["source_candidate"]["box"]
            refined = localization.refine(image, candidate["source_candidate"], _detector())
            if refined != original:
                x, y, face_width, face_height = refined
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + face_width), min(float(height), y + face_height)
                if x2 > x1 and y2 > y1:
                    working_boxes[index] = [x1, y1, x2 - x1, y2 - y1]
        if reassess:
            working_boxes = recheck.prune(image, _detector(), working_boxes, new_faces)
            supplemental = recheck.supplement_pruned_profiles(
                image, _detector(), working_boxes, new_faces, profile_source_pool, accepted
            )
            working_boxes.extend(candidate['box'] for candidate in supplemental)
            new_faces.extend(supplemental)
            missing_profiles = recheck.supplement_missing_profiles(
                image, _detector(), working_boxes, profile_source_pool, accepted
            )
            working_boxes.extend(candidate['box'] for candidate in missing_profiles)
            new_faces.extend(missing_profiles)
            boundary_faces = recheck.supplement_top_boundary_faces(
                image, _detector(), working_boxes, profile_source_pool, accepted
            )
            working_boxes.extend(candidate['box'] for candidate in boundary_faces)
            new_faces.extend(boundary_faces)
        if primary_count == 0 and incomplete_boxes:
            # A noisy border fragment must not suppress a separately
            # quality-validated recheck face in the same image.
            validated_recheck_box_ids = {id(candidate["box"]) for candidate in new_faces}
            working_boxes = [
                box for box in working_boxes
                if not any(
                    evidence._overlap_smaller(box, incomplete)
                    >= (0.45 if id(box) in validated_recheck_box_ids else 0.25)
                    for incomplete in incomplete_boxes
                )
            ]
        for candidate in new_faces:
            index = next((i for i, box in enumerate(working_boxes)
                          if box is candidate["box"]), None)
            if index is None:
                continue
            original = candidate["source_candidate"]["box"]
            refined = recheck.refine_recovered_box(image, candidate["source_candidate"], _detector())
            if refined != original:
                x, y, face_width, face_height = refined
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + face_width), min(float(height), y + face_height)
                if x2 > x1 and y2 > y1:
                    working_boxes[index] = [x1, y1, x2 - x1, y2 - y1]
        # Coordinate tightening must not choose a different representative
        # person merely because the relative box areas have changed.
        representative_index = max(
            range(len(working_boxes)),
            key=lambda index: working_boxes[index][2] * working_boxes[index][3],
            default=None,
        )
        working_boxes = localization.refine_final_boxes(
            image, working_boxes, accepted + new_faces, _detector()
        )
        boxes = [[x * factor_x, y * factor_y, box_width * factor_x,
                  box_height * factor_y] for x, y, box_width, box_height in working_boxes]
        if boxes:
            return {
                "valid": True,
                "usable_face_count": len(boxes),
                "usable_face_boxes": boxes,
                "face": boxes[representative_index],
                "img_size": (source_width, source_height),
            }
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        too_dark_to_diagnose = (
            float(np.mean(gray)) < 65
            and float(np.mean(gray < 35)) > 0.4
            and all(support < 0.6 for support in issue_support)
        )
        tiny_face = (
            not len(detections) and not issues
            and diagnosis.central_tiny_face(image, _detector())
        )
        washed_out_face = (
            not (strongly_turned_away or strongly_motion_blurred or tiny_face)
            and (diagnosis.washed_out_proposal(image, _detector())
                 or diagnosis.clipped_face_silhouettes(image))
        )
        weak_defect = (
            diagnosis.weak_defect_reason(image, _detector())
            if not (strongly_turned_away or strongly_motion_blurred or tiny_face
                    or washed_out_face or too_dark_to_diagnose or issues)
            else None
        )
        return {
            "valid": False,
            "usable_face_count": 0,
            "usable_face_boxes": [],
            "code": (
                "FACE_NOT_FRONTAL" if strongly_turned_away
                else "FACE_BLURRY" if strongly_motion_blurred
                else "FACE_TOO_SMALL" if tiny_face
                else "FACE_OVEREXPOSED" if washed_out_face
                else weak_defect if weak_defect is not None
                else "NO_FACE" if too_dark_to_diagnose
                else issues[0] if len(set(issues)) == 1 else "NO_FACE"
            ),
        }
    except Exception:
        _LOGGER.exception("SCRFD-2.5G-KPS face analysis failed")
        return {
            "valid": False,
            "code": "FACE_DETECTION_FAILED",
            "message": get_message("FACE_DETECTION_FAILED", locale),
        }
    finally:
        try:
            if feature_token is not None:
                quality.end_source_features(feature_token)
        finally:
            _THREAD_LOCAL.request_detector = None
            if request_active:
                _end_request()


def evaluate_human_face_analysis(
    analysis: dict, expected_face_count: int = 1, locale: str | None = None
) -> dict:
    result = dict(analysis)
    if result.get("code") in _TECHNICAL_CODES:
        result.pop("usable_face_count", None)
        return result
    count = result.get("usable_face_count")
    if type(count) is not int or count < 0:
        return {
            "valid": False,
            "code": "FACE_DETECTION_FAILED",
            "message": get_message("FACE_DETECTION_FAILED", locale),
        }
    expected = 2 if expected_face_count == 2 else 1
    if count == expected:
        result.pop("code", None)
        result["valid"] = True
        result["message"] = get_message("FACE_DETECTION_PASSED", locale)
        return result
    result["valid"] = False
    result["code"] = (
        "MULTIPLE_FACES" if count and expected == 1
        else "FACE_COUNT_MISMATCH" if count
        else result.get("code") or "NO_FACE"
    )
    formatter = get_single_face_error_message if expected == 1 else get_double_face_error_message
    result["message"] = formatter(result["code"], locale, face_count=count)
    return result


def contains_human(
    image_path: str,
    locale: str | None = None,
    validation_profile: str = "strict",
    expected_face_count: int = 1,
) -> dict:
    return evaluate_human_face_analysis(
        analyze_human_faces(image_path, locale), expected_face_count, locale
    )


def get_usable_face_reference_boxes(
    image_path: str, expected_face_count: int
) -> list[tuple[float, float, float, float]]:
    expected = 2 if expected_face_count == 2 else 1
    result = contains_human(image_path, expected_face_count=expected)
    if not result.get("valid") or result.get("usable_face_count") != expected:
        return []
    boxes = result.get("usable_face_boxes") or []
    if len(boxes) != expected:
        return []
    return [tuple(map(float, box)) for box in boxes]
