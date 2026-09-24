import os
import threading
import hashlib
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.projects.my3dfigure.core.i18n import get_message, get_single_face_error_message, get_double_face_error_message
from app.projects.my3dfigure.core.config import project_getenv

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency guard
    cv2 = None


MIN_SECONDARY_FACE_SIDE = 80
# A clear face in a full-body or distant street photo can occupy only a few
# pixels of the complete frame.  The relaxed storefront profile accepts a
# single detected face down to 0.2% of the frame while the strict profile
# keeps the original close-up requirement.  The existing 60px detector floor
# still prevents sub-resolution texture noise from becoming a face.
MIN_FACE_RATIO_RELAXED = 0.002
MIN_FACE_DETAIL_SIDE = 80
# A 2000px frontend master can reduce a credible small face by a few pixels.
# Keep the local crop confirmation available down to 50px; the crop must still
# pass native-quality and same-face YuNet confirmation below.
MIN_FACE_CROP_CONFIRM_SIDE = 50
NORMALIZED_FACE_QUALITY_SIZE = 160
MIN_FACE_SHARPNESS = 6.0
# YuNet is the sole human-face detector for every validation profile.  Do not
# mix detector outputs: a second detector can turn one real face into a false
# multi-person rejection after JPEG export or crop resizing.
YUNET_MODEL_PATH = Path(
    project_getenv(
        "FACE_DETECTION_YUNET_MODEL",
        Path(__file__).resolve().parents[4]
        / "models"
        / "face_detection_yunet"
        / "face_detection_yunet_2026may.onnx",
    )
)
YUNET_SCORE_THRESHOLD = 0.82
# A frontend 2000px master can lower the score of a small but genuine face.
# This lower floor is usable only with native recovery plus same-face crop
# confirmation; it is not a general YuNet acceptance threshold.
YUNET_SMALL_FACE_RECOVERY_SCORE_THRESHOLD = 0.75
# A candidate can fall just below the main acceptance score in a difficult
# source image. It remains evidence only until the shared contextual checks
# confirm it; weak texture must never become a usable face by itself.
YUNET_CANDIDATE_SCORE_THRESHOLD = 0.45
# Secondary candidates use a stricter contextual floor than the ordinary
# detector threshold to prevent texture or clothing from inflating the count.
YUNET_SECONDARY_FACE_SCORE_THRESHOLD = 0.88
YUNET_DIAGNOSTIC_SCORE_THRESHOLD = 0.10
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
# A 28px short side is below the reliable-detail boundary on the frontend's
# 2000px master. Keep this as a strict lower bound so the diagnostic path can
# truthfully report FACE_TOO_SMALL instead of accepting a borderline face.
YUNET_MIN_FACE_SIDE = 29
YUNET_MIN_FACE_RATIO = 0.0004
YUNET_QUARTER_TURN_ANGLES = (90, 180, 270)
YUNET_ROTATION_FALLBACK_ANGLES = (15, -15, 30, -30)
YUNET_COMBINED_ROTATION_ANGLES = (15, -15)
_yunet_thread_local = threading.local()

# YuNet locates human faces very well, but a centred dog face can occasionally
# look face-like enough to pass it. Keep a compact local ImageNet classifier as
# a second check for the candidate crop; it is only used to reject confident
# cat/dog predictions and never replaces face detection.
PET_CLASSIFIER_MODEL_PATH = Path(
    project_getenv(
        "FACE_DETECTION_PET_CLASSIFIER_MODEL",
        Path(__file__).resolve().parents[4]
        / "models"
        / "image_classification"
        / "mobilenetv2-12.onnx",
    )
)
PET_CLASSIFIER_INPUT_SIZE = (224, 224)
PET_CLASSIFIER_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
PET_CLASSIFIER_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_DOG_CLASS_IDS = frozenset(range(151, 269))
IMAGENET_CAT_CLASS_IDS = frozenset(range(281, 286))
IMAGENET_PET_CLASS_IDS = IMAGENET_DOG_CLASS_IDS | IMAGENET_CAT_CLASS_IDS
PET_CLASSIFICATION_MIN_CONFIDENCE = 0.6
_pet_classifier_thread_local = threading.local()




_request_work = ContextVar("my3dfigure_face_request_work", default=None)


def _array_cache_key(array):
    contiguous = np.ascontiguousarray(array)
    return (contiguous.shape, contiguous.dtype.str,
            hashlib.sha256(memoryview(contiguous).cast("B")).digest())


_orientation_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="my3d-face-view")


def _detect_orientation(view):
    detector = _get_yunet_detector((view.shape[1], view.shape[0]))
    return detector.detect(view)[1]


def _detect_yunet_orientations(resized):
    """Run the same views concurrently, consuming outputs in original order."""
    views = [(angle, resized if angle == 0 else _quarter_turn_with_matrix(resized, angle)[0])
             for angle in (0, *YUNET_QUARTER_TURN_ANGLES)]
    if _request_work.get() is None:
        for angle, view in views:
            yield angle, view, _detect_orientation(view)
        return
    futures = [_orientation_workers.submit(_detect_orientation, view) for _, view in views]
    try:
        for (angle, view), future in zip(views, futures):
            yield angle, view, future.result()
    finally:
        for future in futures:
            future.cancel()


def _pet_classifier_logits(classifier, blob):
    work = _request_work.get()
    cache = work.setdefault("pet_logits", {}) if work is not None else None
    key = (str(PET_CLASSIFIER_MODEL_PATH), _array_cache_key(blob)) if cache is not None else None
    if cache is not None and key in cache:
        return cache[key].copy()
    classifier.setInput(blob)
    logits = classifier.forward().reshape(-1)
    if cache is not None and len(cache) < 256:
        cache[key] = logits.copy()
    return logits


def _read_image(image_path: str):
    """Read images from paths containing non-ASCII characters on Windows."""
    work = _request_work.get()
    key = os.fspath(image_path)
    if work is not None and key in work.setdefault("images", {}):
        return work["images"][key].copy()
    image_bytes = np.fromfile(image_path, dtype=np.uint8)
    if image_bytes.size == 0:
        return None
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if work is not None and image is not None and len(work["images"]) < 1:
        work["images"][key] = image.copy()
    return image


def _get_yunet_detector(
    input_size: tuple[int, int],
    score_threshold: float = YUNET_SCORE_THRESHOLD,
):
    """Return one mutable FaceDetectorYN instance per worker thread."""
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("当前 OpenCV 版本不支持 FaceDetectorYN。")
    if not YUNET_MODEL_PATH.is_file():
        raise RuntimeError(f"YuNet 模型不存在: {YUNET_MODEL_PATH}")

    detectors = getattr(_yunet_thread_local, "detectors", None)
    if detectors is None:
        detectors = {}
        _yunet_thread_local.detectors = detectors
    detector_key = round(float(score_threshold), 4)
    detector = detectors.get(detector_key)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL_PATH),
            "",
            input_size,
            score_threshold,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
        detectors[detector_key] = detector
    else:
        detector.setInputSize(input_size)
    return detector


def _get_pet_classifier():
    """Return one MobileNet classifier per worker thread when available."""
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python-headless，无法执行宠物分类。")
    if not PET_CLASSIFIER_MODEL_PATH.is_file():
        raise RuntimeError(f"宠物分类模型不存在: {PET_CLASSIFIER_MODEL_PATH}")

    classifier = getattr(_pet_classifier_thread_local, "classifier", None)
    if classifier is None:
        classifier = cv2.dnn.readNetFromONNX(str(PET_CLASSIFIER_MODEL_PATH))
        _pet_classifier_thread_local.classifier = classifier
    return classifier


def _is_confident_pet_image(color_image, face_box=None) -> bool:
    """Return True only for a confident cat/dog classification."""
    if color_image is None or color_image.ndim != 3:
        return False

    try:
        crop = color_image
        if face_box is not None:
            x, y, width, height = map(int, face_box)
            padding_x = max(1, round(width * 0.35))
            padding_y = max(1, round(height * 0.35))
            left = max(0, x - padding_x)
            top = max(0, y - padding_y)
            right = min(color_image.shape[1], x + width + padding_x)
            bottom = min(color_image.shape[0], y + height + padding_y)
            crop = color_image[top:bottom, left:right]
        if crop.size == 0:
            return False

        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1 / 255.0,
            size=PET_CLASSIFIER_INPUT_SIZE,
            swapRB=True,
            crop=True,
        )
        blob = (blob - PET_CLASSIFIER_MEAN) / PET_CLASSIFIER_STD
        classifier = _get_pet_classifier()
        logits = _pet_classifier_logits(classifier, blob)
        if logits.size <= max(IMAGENET_PET_CLASS_IDS) or not np.all(np.isfinite(logits)):
            return False
        stabilized = logits - np.max(logits)
        probabilities = np.exp(stabilized)
        probabilities /= max(float(np.sum(probabilities)), 1e-9)
        pet_confidence = float(np.sum(probabilities[list(IMAGENET_PET_CLASS_IDS)]))
        if pet_confidence < PET_CLASSIFICATION_MIN_CONFIDENCE:
            return False
        if face_box is None:
            return True

        # The padded context may include a pet held next to a real human
        # face. Only discard this YuNet candidate when the face box itself
        # also looks like a pet. Keep the context check for animal ears/fur,
        # and avoid an extra inference for normal non-pet candidates.
        face_crop = color_image[
            max(0, y):min(color_image.shape[0], y + height),
            max(0, x):min(color_image.shape[1], x + width),
        ]
        return _is_confident_pet_image(face_crop)
    except Exception:
        # This is a precision enhancement. A missing/corrupt optional model
        # must not turn a valid person upload into a failed request.
        return False


def _filter_pet_face_candidates(color_image, candidates):
    """Discard YuNet false positives that MobileNet confidently identifies as pets."""
    strong_group_count = sum(
        candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
        for candidate in candidates
    )
    filtered = []
    for candidate in candidates:
        # The compact classifier is overconfident on a small, dim human face.
        # In an already established group, a repeated native recovery box is
        # still human-face evidence; do not let the optional pet check erase
        # the fourth/fifth member of a real group.
        group_recovery = (
            candidate.get("native_tile_recovery")
            and candidate.get("score", 0.0) >= .60
            and strong_group_count >= 3
        )
        if group_recovery or not _is_confident_pet_image(color_image, candidate["box"]):
            filtered.append(candidate)
    return _filter_embedded_graphic_candidates(color_image, filtered)


def _filter_embedded_graphic_candidates(image, candidates):
    """Do not treat small graphic overlays beside a photographic portrait as people.

    Standalone illustrated portraits retain their existing policy. This check
    requires a separate photographic anchor and actual flat-ink/paper evidence;
    position near the bottom of a picture is not evidence of a sticker.
    """
    if len(candidates) < 2:
        return candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def pixels(c):
        x, y, w, h = map(int, c["box"])
        return image[max(0, y):y+h, max(0, x):x+w], gray[max(0, y):y+h, max(0, x):x+w]

    anchors = []
    for c in candidates:
        roi, g = pixels(c)
        if (g.size and min(c["box"][2:]) >= 120 and c["score"] >= .82
                and float((g > 240).mean()) < .20):
            anchors.append(c)
    if not anchors:
        return candidates
    kept = []
    for c in candidates:
        if any(c is anchor for anchor in anchors):
            kept.append(c)
            continue
        roi, g = pixels(c)
        if not g.size:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        low, median, high = np.percentile(g, (10, 50, 90))
        white = float((g > 240).mean())
        dark = float((g < 40).mean())
        ink_on_paper = white >= .25 and high-low >= 140 and dark >= .025
        flat_colored_line_art = (
            float(np.median(hsv[:, :, 1])) >= 80
            and float(hsv[:, :, 0].std()) < 5
            and high-low < 40
        )
        if not (ink_on_paper or flat_colored_line_art):
            kept.append(c)
    return kept


def _non_human_face_result(locale: str | None) -> dict:
    return {
        "valid": False,
        "code": "NON_HUMAN_FACE",
        "message": get_message("NO_FACE", locale),
    }


def _face_has_sufficient_native_detail(image_path, result):
    """A small frame fraction need not mean too few usable face pixels.

    Evaluate the uploaded image, not YuNet's enlarged inference canvas.
    This exception never bypasses counting or native face-quality checks.
    """
    face = result.get("face")
    score = result.get("face_score", 0.0)
    recovered_small_face = (
        result.get("native_recovery")
        and score >= YUNET_SMALL_FACE_RECOVERY_SCORE_THRESHOLD
    )
    # A high-confidence face can lose one or two pixels when the browser
    # creates the 2000px upload master.  Keep the original detector minimum,
    # but require an independent crop confirmation before allowing it through.
    if (face and YUNET_MIN_FACE_SIDE <= min(face[2:]) < MIN_FACE_CROP_CONFIRM_SIDE
            and score >= .88):
        image = _read_image(image_path)
        if image is not None and not any(_frame_edges_touched(face, image.shape[1], image.shape[0])):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if _get_face_quality_issue(gray, face) is None and _confirm_native_face_crop(
                    image, face, min_score=.88, require_local_quality=False):
                return True
    if (not face or min(face[2:]) < MIN_FACE_CROP_CONFIRM_SIDE
            or (score < YUNET_SCORE_THRESHOLD and not recovered_small_face)):
        return False
    try:
        image = _read_image(image_path)
    except (OSError, ValueError):
        return False
    if image is None:
        return False
    if _is_face_box_cut_by_frame(face, image.shape[1], image.shape[0]):
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (_get_face_quality_issue(gray, face) is not None
            or _is_extreme_motion_blur(image_path, face)):
        return False
    if (min(face[2:]) >= MIN_FACE_DETAIL_SIDE and score >= 0.88):
        return True
    # A credible YuNet face can straddle the single-face detail floor after a
    # bounded 2000px export. Re-detect the same native crop in its original
    # local scale; this preserves the normal global size threshold and cannot
    # accept scenery, a different face, or a covered/profile face.
    return _confirm_native_face_crop(
        image, face, min_score=max(.84, min(.88, float(score))), require_local_quality=False,
    )


def _has_pixelated_portrait_evidence(image, face):
    """Diagnostic-only evidence: large frontal layout plus block-grid edges.

    Never accepts a face. Both axes must concentrate their edge energy in
    a small fraction of scanlines, as severely pixelated portraits do.
    """
    h, w = image.shape[:2]
    if (float(face[14]) < 0.10 or face[2] * face[3] / float(h * w) < 0.10
            or not _has_low_confidence_frontal_geometry(face)):
        return False
    x, y, width, height = map(int, face[:4])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[
        max(0, y):min(gray.shape[0], y + height),
        max(0, x):min(gray.shape[1], x + width),
    ].astype(np.float32)
    if gray.size == 0:
        return False
    for axis in (0, 1):
        energy = np.abs(np.diff(gray, axis=axis)).sum(axis=1-axis)
        if energy.size == 0 or float(energy.sum()) <= 0:
            return False
        concentrated = np.sort(energy)[-max(1, energy.size // 10):].sum() / energy.sum()
        # Severe mosaics present in two stable forms: a regular coarse grid
        # (nearly all of the energy falls on its block boundaries), or a
        # resized/pixelated portrait whose block boundaries are broadly but
        # still noticeably concentrated.  Natural texture is close to a
        # uniform distribution, while a readable cropped portrait tends to
        # have one dominant contour.  Require both axes to fall in one of the
        # two block-like bands; the face-shaped candidate gate above prevents
        # this diagnostic from treating scenery as a face.
        block_like = (
            0.28 <= concentrated <= 0.40
            or 0.55 <= concentrated <= 0.75
            or concentrated >= 0.85
        )
        if not block_like:
            return False
    return True


def _has_overexposed_face_evidence(gray_image, face_box) -> bool:
    """Recognize a blown-out face when YuNet only leaves a weak head box.

    This is diagnostic-only.  Hair and clothing may keep the whole box's low
    percentile dark, so look for a substantial clipped highlight population
    rather than requiring every pixel to be bright.
    """
    x, y, width, height = map(int, face_box)
    roi = gray_image[y:y + height, x:x + width]
    if roi.size == 0 or min(width, height) < 80:
        return False
    return (
        float(np.percentile(roi, 90)) >= 248
        and float(np.mean(roi >= 245)) >= 0.14
    )


def _diagnose_yunet_failure(
    detection_image,
    scale: float,
    image_width: int,
    image_height: int,
    locale: str | None,
):
    """Return a more useful error without weakening the acceptance pass."""
    detection_height, detection_width = detection_image.shape[:2]
    detection_area = detection_width * detection_height
    detector = _get_yunet_detector(
        (detection_width, detection_height),
        YUNET_DIAGNOSTIC_SCORE_THRESHOLD,
    )
    _retval, detected = detector.detect(detection_image)
    if detected is None:
        return None

    gray_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    candidates = []
    quality_evidence = []
    orientation_evidence = []
    edge_crop_evidence = []
    excluded_pet = False
    for face in detected:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        x, y, width, height = map(float, face[:4])
        score = float(face[14])
        ratio = (width * height) / float(max(detection_area, 1))
        box = (
            max(0, int(round(x))),
            max(0, int(round(y))),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )
        if score >= 0.25 and _is_confident_pet_image(detection_image, box):
            excluded_pet = True
            continue
        # The diagnostic pass is allowed to explain a failure, never to
        # accept it.  Keep quality/orientation evidence from a large,
        # face-shaped weak candidate before the normal acceptance gates
        # discard it.
        if (
            score >= 0.12
            and min(width, height) >= 120
            and ratio >= 0.01
            and _has_low_confidence_frontal_geometry(face)
        ):
            weak_issue = _get_face_quality_issue(gray_image, box)
            if weak_issue in {"FACE_TOO_DARK", "FACE_OVEREXPOSED"}:
                quality_evidence.append((score, ratio, weak_issue))
            elif _has_overexposed_face_evidence(gray_image, box):
                quality_evidence.append((score, ratio, "FACE_OVEREXPOSED"))
        if (
            score >= 0.30
            and min(width, height) >= 80
            and ratio >= 0.008
            and _is_yunet_profile(face)
        ):
            if _is_face_box_cut_by_frame(box, detection_width, detection_height):
                # A weak profile candidate that touches the frame is more
                # truthfully explained as an incomplete face than as a
                # non-frontal face.  This is diagnostic evidence only.
                edge_crop_evidence.append((score, ratio))
            else:
                orientation_evidence.append((score, ratio))
        # Very weak texture matches are still treated as no face. The lower
        # threshold is for diagnosis, not for turning scenery into a face.
        # This lower-confidence pass never accepts a face.  It only recognizes
        # a large portrait-like region well enough to explain why its features
        # are unusable.  The size/area floor prevents scenery texture from
        # becoming a spurious quality diagnosis.
        # The old 120px / 2% gate discarded the strongest partial-face
        # candidate after a browser resize, leaving a weaker texture match
        # to choose the message. Smaller candidates need stronger evidence.
        minimum_side, minimum_ratio = (48, 0.003) if score >= 0.40 else (120, 0.02)
        if score < 0.25 and _has_pixelated_portrait_evidence(detection_image, face):
            return {"valid": False, "code": "FACE_UNRECOGNIZABLE",
                    "message": get_message("FACE_UNRECOGNIZABLE", locale)}
        if score < 0.25 or min(width, height) < minimum_side or ratio < minimum_ratio:
            continue
        visual_quality_issue = _get_face_quality_issue(gray_image, box)
        occlusion_issue = (
            None
            if visual_quality_issue is not None
            else _get_face_occlusion_issue(gray_image, box, face)
        )
        candidates.append(
            {
                "score": score,
                "ratio": ratio,
                "raw_face": face,
                "box": _map_face_to_original(face, scale, image_width, image_height),
                "detection_box": box,
                "edge_touched": any(_frame_edges_touched(
                    box, detection_width, detection_height,
                )),
                "edge_crop_unusable": _is_unusable_edge_cropped_face(
                    # A low-threshold candidate may only use the landmark crop
                    # rule when it has credible face geometry. This separates a
                    # real small screenshot or blurred portrait from a weak
                    # edge texture, without weakening normal acceptance.
                    face, box, detection_width, detection_height,
                    max(score, .68)
                    if score >= .45 and (
                        _has_low_confidence_frontal_geometry(face)
                        or _is_yunet_profile(face)
                    )
                    else score,
                ),
                "visual_quality_issue": visual_quality_issue,
                "occlusion_issue": occlusion_issue,
                "quality_issue": visual_quality_issue or occlusion_issue,
            }
        )
    if not candidates:
        # Weak quality/orientation evidence may explain a failure only when
        # the scan found no stronger candidate.  It must never override a
        # credible face candidate or turn a usable face into a failure.
        if quality_evidence:
            _score, _ratio, code = max(quality_evidence)
            return {"valid": False, "code": code, "message": get_message(code, locale)}
        if edge_crop_evidence:
            return {
                "valid": False,
                "code": "FACE_INCOMPLETE",
                "message": get_message("FACE_INCOMPLETE", locale),
            }
        if orientation_evidence:
            return {
                "valid": False,
                "code": "FACE_NOT_FRONTAL",
                "message": get_message("FACE_NOT_FRONTAL", locale),
            }
        return None

    if excluded_pet and all(c["score"] < .50 and c["quality_issue"] is None for c in candidates):
        return {"valid": False, "code": "NO_FACE", "message": get_message("NO_FACE", locale)}
    best = max(candidates, key=lambda candidate: (candidate["score"], candidate["ratio"]))
    # This is a diagnostic-only candidate: it may explain a failed upload but
    # must not turn scenery at the frame edge into a "cropped face" message.
    # Require both a moderate YuNet score and a recognisable face landmark
    # layout before reporting FACE_INCOMPLETE.  The normal scan still owns
    # genuine cropped-face detection; this guard only limits the weak fallback.
    edge_crop_has_face_evidence = (
        best["edge_crop_unusable"]
        and best["score"] >= .45
        and (
            _has_low_confidence_frontal_geometry(best["raw_face"])
            or _is_yunet_profile(best["raw_face"])
            or _is_face_box_cut_by_frame(
                best["detection_box"], detection_width, detection_height,
            )
        )
    )
    if best["edge_crop_unusable"] and not edge_crop_has_face_evidence:
        return {"valid": False, "code": "NO_FACE", "message": get_message("NO_FACE", locale)}
    # A small screenshot or compressed crop can retain a recognisable face
    # even though interpolating it to YuNet's working size makes the tiny eye
    # patches look soft. Accept only a single, frontal, landmark-consistent
    # candidate above this floor; normal blurry uploads remain rejected.
    if (
        best["score"] >= 0.50
        and best["visual_quality_issue"] is None
        and (
            best["occlusion_issue"] == "FACE_BLURRY"
            or (
                best["occlusion_issue"] == "FACE_OCCLUDED"
                and best["score"] >= .55
                and best["edge_touched"]
            )
        )
        and _has_low_confidence_frontal_geometry(best["raw_face"])
        and min(best["box"][2], best["box"][3]) >= 72
        and not best["edge_crop_unusable"]
    ):
        return {
            "valid": True,
            "message": get_message("FACE_DETECTION_PASSED", locale),
            "face": best["box"],
            "img_size": (int(image_width), int(image_height)),
            "face_count": 1,
            "face_score": float(best["score"]),
        }
    # Once a stronger candidate has ruled out acceptance, use the weak
    # diagnostic evidence to explain that same failed upload.  A truncated
    # candidate remains a crop problem regardless of its lighting, so never
    # let a quality label replace FACE_INCOMPLETE.
    if not best["edge_crop_unusable"]:
        if quality_evidence:
            _score, _ratio, code = max(quality_evidence)
            return {"valid": False, "code": code, "message": get_message(code, locale)}
        if orientation_evidence:
            return {
                "valid": False,
                "code": "FACE_NOT_FRONTAL",
                "message": get_message("FACE_NOT_FRONTAL", locale),
            }
    if edge_crop_has_face_evidence:
        code = "FACE_INCOMPLETE"
    elif (
        best["score"] >= 0.50
        and (
            _is_yunet_profile(best["raw_face"])
            or (
                best["quality_issue"] in {None, "NO_FACE", "FACE_UNRECOGNIZABLE"}
                and (best["score"] >= 0.75 or not _has_occlusion_suspect_geometry(best["raw_face"]))
            )
        )
    ):
        # A large diagnostic YuNet candidate at this confidence is evidence
        # that a head is present, but not that its identity is usable.  This
        # separates far-turned profiles from a true back-of-head/no-face
        # result without lowering the acceptance threshold or adding another
        # detector.  Below this floor, the candidate is too weak to make an
        # orientation claim.
        code = "FACE_NOT_FRONTAL"
    elif (
        best["score"] >= 0.40
        and best["quality_issue"] in {None, "NO_FACE", "FACE_UNRECOGNIZABLE"}
        and _has_occlusion_suspect_geometry(best["raw_face"])
    ):
        raw = best["raw_face"]
        eye_y = float(raw[5] + raw[7]) / 2
        code = ("NO_FACE" if (eye_y - raw[1]) / raw[3] > .50
                and abs(float(raw[6] - raw[4])) / raw[2] < .20
                else "FACE_OCCLUDED")
    elif best["score"] < 0.25 and best["quality_issue"]:
        # Smooth hair, clothing and walls also have low edge detail. A weak
        # texture match cannot establish that a *face* is blurry or covered.
        code = "NO_FACE"
    elif best["quality_issue"]:
        # A tiny, low-confidence texture candidate cannot reliably establish
        # that a face is covered. Keep the generic result for this evidence,
        # rather than changing an existing no-face outcome into a specific
        # occlusion claim.
        code = (
            "NO_FACE"
            if (
                best["score"] < .60
                and best["ratio"] < .02
            )
            else best["quality_issue"]
        )
    elif (
        best["score"] < 0.62
        and best["quality_issue"] is None
        and _has_diagnostic_directional_motion_blur(
            detection_image, best["detection_box"],
        )
    ):
        code = "FACE_BLURRY"
    elif min(best["box"][2], best["box"][3]) < 28 or best["ratio"] < 0.0004:
        code = "FACE_TOO_SMALL"
    elif best["score"] < 0.62:
        # Below the acceptance score, YuNet can place a face-shaped box on a
        # back of head, hair or an ear.  Only retain an "unrecognizable face"
        # diagnosis when its landmark layout still resembles a frontal face;
        # otherwise the truthful result is that no usable face is visible.
        code = (
            "FACE_UNRECOGNIZABLE"
            if _has_low_confidence_frontal_geometry(best["raw_face"])
            else "NO_FACE"
        )
    else:
        code = "FACE_NOT_FRONTAL"

    result = {
        "valid": False,
        "code": code,
        "message": get_message(
            code,
            locale,
            face_percent=f"{best['ratio'] * 100:.1f}",
        ) if code == "FACE_TOO_SMALL" else get_message(code, locale),
    }
    if code == "FACE_TOO_SMALL":
        result["face"] = best["box"]
        result["img_size"] = (int(image_width), int(image_height))
    return result


def _diagnose_tiny_yunet_face(image, locale):
    """Recover evidence of a distant face solely to explain a rejection.

    The dynamic-input pass inspects the source pixels directly. A tiny
    detection is never accepted or added to the person count; this helper only
    explains a rejection and still requires the normal acceptance score.
    """
    height, width = image.shape[:2]
    scale = 1.0
    _, faces = _get_yunet_detector((width, height), .60).detect(image)
    if faces is None:
        return None
    tiny = []
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)) or float(face[14]) < .60:
            continue
        box = _map_face_to_original(face, scale, width, height)
        ratio = box[2] * box[3] / float(width * height)
        # A source face is below the detector's usable resolution when either
        # its projected side or its projected area is below the floor. Requiring
        # both turns borderline rounding into an inaccurate "no face" message.
        if ratio < YUNET_MIN_FACE_RATIO or min(box[2:]) < 20:
            tiny.append((float(face[14]), box, ratio))
    if not tiny:
        return None
    _, box, ratio = max(tiny)
    return {"valid": False, "code": "FACE_TOO_SMALL",
            "message": get_message("FACE_TOO_SMALL", locale, face_percent=f"{ratio * 100:.3f}"),
            "face": box, "img_size": (width, height)}


def _diagnose_no_face_quality_issue(image_path: str, color_image, locale: str | None):
    """Distinguish an unusable portrait from a genuinely face-free image.

    YuNet intentionally has a high acceptance threshold.  A severely
    overexposed or pixelated portrait can therefore produce no YuNet box even
    though a person is plainly present.  Do not lower that acceptance
    threshold; return a truthful failure reason instead.
    """
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    low, middle, high = np.percentile(gray_image, (10, 50, 90))
    # Require both a predominantly clipped image and some retained dark
    # content (hair/clothes).  This avoids describing a blank white upload as
    # an overexposed portrait.
    if low >= 120 and middle >= 245 and high >= 252:
        return {
            "valid": False,
            "code": "FACE_OVEREXPOSED",
            "message": get_message("FACE_OVEREXPOSED", locale),
        }

    # A small clipped face can be the only detector evidence in an otherwise
    # mixed-light image, so the global white-pixel rule above is intentionally
    # conservative.  Require a matching weak face-shaped box as well; this
    # avoids relabelling ordinary bright backgrounds as overexposure.
    if middle >= 80 and high >= 245:
        _retval, faces = _get_yunet_detector(
            (color_image.shape[1], color_image.shape[0]), .55,
        ).detect(color_image)
        if faces is not None:
            for face in faces:
                if len(face) < 15 or not np.all(np.isfinite(face)) or float(face[14]) < .55:
                    continue
                box = tuple(map(int, face[:4]))
                if min(box[2:]) > 40:
                    continue
                local_issue = _get_face_quality_issue(gray_image, box)
                if local_issue in {"FACE_BLURRY", "FACE_OVEREXPOSED"}:
                    return {
                        "valid": False,
                        "code": "FACE_OVEREXPOSED",
                        "message": get_message("FACE_OVEREXPOSED", locale),
                    }

    return None


def _diagnose_native_low_quality_face(image, locale):
    """Explain a failed native scan from one substantial weak face candidate."""
    detector = _get_yunet_detector((image.shape[1], image.shape[0]), .10)
    _retval, faces = detector.detect(image)
    if faces is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best = None
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        box = tuple(map(int, face[:4]))
        if score < .20 or min(box[2:]) < 100 or any(_frame_edges_touched(box, image.shape[1], image.shape[0])):
            continue
        quality = _get_face_quality_issue(gray, box)
        core = _native_face_core_issue(image, box, score)
        occlusion = _get_face_occlusion_issue(gray, box, face)
        issue = quality or core or occlusion
        if issue not in {"FACE_BLURRY", "FACE_OCCLUDED"}:
            continue
        if best is None or score > best[0]:
            best = (score, issue)
    if best is None:
        return None
    return {
        "valid": False,
        "code": best[1],
        "message": get_message(best[1], locale),
    }


def _native_weak_face_signals(image):
    """Return bounded native evidence for final diagnostic prioritization."""
    if image is None:
        return []
    detector = _get_yunet_detector((image.shape[1], image.shape[0]), .10)
    _retval, faces = detector.detect(image)
    if faces is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signals = []
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        box = tuple(map(int, face[:4]))
        signals.append({
            "score": float(face[14]),
            "box": box,
            "quality": _get_face_quality_issue(gray, box),
            "occlusion": _get_face_occlusion_issue(gray, box, face),
            "core": _native_face_core_issue(image, box, float(face[14])),
        })
    return signals


def _map_face_to_original(face, scale: float, image_width: int, image_height: int):
    """Map a YuNet bounding box from canonical pixels to upload pixels."""
    x, y, width, height = (float(value) / max(scale, 1e-9) for value in face[:4])
    left = max(0, min(int(round(x)), image_width - 1))
    top = max(0, min(int(round(y)), image_height - 1))
    right = max(left + 1, min(int(round(x + width)), image_width))
    bottom = max(top + 1, min(int(round(y + height)), image_height))
    return left, top, right - left, bottom - top


def _map_rotated_box_to_unrotated(
    face,
    inverse_matrix,
    image_width: int,
    image_height: int,
):
    """Map a YuNet box on a rotated canvas back to the canonical image."""
    x, y, width, height = map(float, face[:4])
    corners = np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, inverse_matrix).reshape(-1, 2)
    left = max(0.0, min(float(np.min(mapped[:, 0])), float(image_width - 1)))
    top = max(0.0, min(float(np.min(mapped[:, 1])), float(image_height - 1)))
    right = max(left + 1.0, min(float(np.max(mapped[:, 0])), float(image_width)))
    bottom = max(top + 1.0, min(float(np.max(mapped[:, 1])), float(image_height)))
    return np.array([left, top, right - left, bottom - top], dtype=np.float32)


def _map_quarter_turn_box_to_unrotated(
    face,
    angle: int,
    image_width: int,
    image_height: int,
):
    """Map a cv2.rotate 90/180/270-degree box to canonical coordinates."""
    x, y, width, height = map(float, face[:4])
    if angle == 90:  # ROTATE_90_CLOCKWISE
        left, top, mapped_width, mapped_height = y, image_height - (x + width), height, width
    elif angle == 180:
        left, top, mapped_width, mapped_height = (
            image_width - (x + width),
            image_height - (y + height),
            width,
            height,
        )
    else:  # ROTATE_90_COUNTERCLOCKWISE (270 degrees clockwise)
        left, top, mapped_width, mapped_height = image_width - (y + height), x, height, width
    return np.array([left, top, mapped_width, mapped_height], dtype=np.float32)


def _quarter_turn_with_matrix(image, angle: int):
    """Rotate a canonical image and return the affine map to its source."""
    image_height, image_width = image.shape[:2]
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        matrix = np.array([[0.0, -1.0, image_height - 1.0], [1.0, 0.0, 0.0]])
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        matrix = np.array(
            [[-1.0, 0.0, image_width - 1.0], [0.0, -1.0, image_height - 1.0]]
        )
    else:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, image_width - 1.0]])
    return rotated, matrix


def _compose_affine(first, second):
    """Compose two 2x3 affine transforms in application order."""
    first_h = np.vstack([first, [0.0, 0.0, 1.0]])
    second_h = np.vstack([second, [0.0, 0.0, 1.0]])
    return (first_h @ second_h)[:2]


def _filter_neck_satellite_candidates(candidates):
    """Remove a tightly-scoped YuNet false positive beneath one larger face.

    A strongly turned close-up can make the jaw/neck/collar boundary resemble a
    second small face.  Do not use a blanket score or size cutoff: real small
    faces must still count.  Only suppress a candidate when it is much smaller
    than, horizontally under, and immediately below the same dominant face.
    """
    if len(candidates) < 2:
        return list(candidates)

    retained = []
    for candidate in candidates:
        x, y, width, height = map(float, candidate["box"])
        area = width * height
        center_x = x + width / 2.0
        is_neck_satellite = False
        for anchor in candidates:
            if anchor is candidate:
                continue
            anchor_x, anchor_y, anchor_width, anchor_height = map(float, anchor["box"])
            anchor_area = anchor_width * anchor_height
            if anchor_area < area * 12:
                continue
            # The false box starts around the lower jaw/neck, remains inside
            # the wider face's horizontal projection, and is far smaller than
            # that face.  A separately positioned second person does not meet
            # this conjunction.
            starts_at_neck = anchor_y + anchor_height * 0.78 <= y <= anchor_y + anchor_height * 1.35
            contained_horizontally = anchor_x - anchor_width * 0.12 <= center_x <= anchor_x + anchor_width * 1.12
            much_smaller = width <= anchor_width * 0.45 and height <= anchor_height * 0.35
            not_stronger = float(candidate.get("score", 0.0)) <= float(anchor.get("score", 0.0)) + 0.02
            if starts_at_neck and contained_horizontally and much_smaller and not_stronger:
                is_neck_satellite = True
                break
        if not is_neck_satellite:
            retained.append(candidate)
    return retained


def _confirm_native_face_crop(
    image,
    box,
    min_score: float = 0.92,
    require_local_quality: bool = True,
):
    """Confirm the same face in its original context at native resolution.

    This local check is only for coverage inferred after a small face is found.
    It does not change the normal eye-coverage rule or accept a different face.
    """
    x, y, w, h = map(int, box)
    left, top = max(0, x - w // 2), max(0, y - h // 2)
    crop = image[top:min(image.shape[0], y + h + h // 2),
                 left:min(image.shape[1], x + w + w // 2)]
    if crop.size == 0:
        return False
    expected = (x - left, y - top, w, h)
    scale = 1.0
    view = crop
    _, faces = _get_yunet_detector(
        (view.shape[1], view.shape[0]), min(YUNET_SCORE_THRESHOLD, min_score)
    ).detect(view)
    if faces is None:
        return False
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    for face in faces:
        if (len(face) < 15 or not np.all(np.isfinite(face)) or float(face[14]) < min_score
                or _is_yunet_profile(face)):
            continue
        local_box = _map_face_to_original(face, 1, view.shape[1], view.shape[0])
        native_box = _map_face_to_original(face, scale, crop.shape[1], crop.shape[0])
        if (_box_iou(native_box, expected) >= .5
                and (not require_local_quality
                     or _get_face_quality_issue(gray, local_box) is None)):
            return True
    return False


def _native_tile_origins(image):
    """Return overlapping source windows for native-resolution recovery.

    These are crops of the original pixels, not resized copies.  A crop gives
    YuNet a smaller dynamic input canvas while preserving every source pixel,
    which is useful for a small face that is underrepresented in a large
    scene.  The overlap lets us require repeatable evidence and avoid adding
    a tile-edge texture as a person.
    """
    height, width = image.shape[:2]
    if min(height, width) < 320:
        return []
    tile_width = max(1, int(round(width * .72)))
    tile_height = max(1, int(round(height * .72)))
    x_positions = tuple(dict.fromkeys((0, max(0, width - tile_width))))
    y_positions = tuple(dict.fromkeys((0, max(0, height - tile_height))))
    if len(x_positions) == 1 and len(y_positions) == 1:
        return []
    return [
        (x, y, image[y:y + tile_height, x:x + tile_width])
        for y in y_positions
        for x in x_positions
    ]


def _recover_native_candidate_crops(image, primary_candidates):
    """Recheck weak native candidates in padded source crops.

    A dynamic detector still sees a very small face in the context of the
    complete upload.  A padded crop gives that same source face a smaller
    native canvas without inventing pixels or resampling the upload.  The crop
    pass is only allowed to promote a detector match that returns to the same
    source location with the normal confidence and quality gates.
    """
    height, width = image.shape[:2]
    detector = _get_yunet_detector((width, height), YUNET_DIAGNOSTIC_SCORE_THRESHOLD)
    _retval, faces = detector.detect(image)
    if faces is None:
        return primary_candidates
    recovered = list(primary_candidates)
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    primary_usable = [
        candidate for candidate in primary_candidates
        if candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
    ]
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        weak_score = float(face[14])
        if weak_score < .20:
            continue
        x, y, face_width, face_height = map(int, face[:4])
        if min(face_width, face_height) < 18:
            continue
        source_box = (x, y, face_width, face_height)
        edge_ok = not _is_unusable_edge_cropped_face(
            face, source_box, width, height, weak_score,
        )
        group_recovery = False
        # A second-person recovery must have enough native pixels to support
        # the same quality gates as the primary face.  Very small weak boxes
        # are commonly repeated texture or a partial facial pattern; allowing
        # them here turns a valid upload into a false multi-face
        # result.  The normal detector/tile paths still handle genuinely small
        # faces through their own recurrence checks.
        if primary_usable and weak_score >= .60 and min(face_width, face_height) >= 180:
            source_issue = _get_face_quality_issue(source_gray, source_box)
            source_occlusion = _get_face_occlusion_issue(
                source_gray, source_box, face,
            )
            source_profile = _is_yunet_profile(face)
            edge_ok = edge_ok or (source_profile and weak_score >= .70)
            for anchor in primary_usable:
                anchor_width, anchor_height = anchor["box"][2:]
                similar_size = (
                    min(face_width, face_height) >= min(anchor_width, anchor_height) * .55
                    and max(face_width, face_height) <= max(anchor_width, anchor_height) * 1.8
                )
                distinct = (
                    _box_iou(source_box, anchor["box"]) < .35
                    and np.hypot(
                        x + face_width / 2.0 - (anchor["box"][0] + anchor_width / 2.0),
                        y + face_height / 2.0 - (anchor["box"][1] + anchor_height / 2.0),
                    ) > max(face_width, face_height) * .55
                )
                if (
                    source_issue is None
                    and source_occlusion in {None, "FACE_BLURRY", "NO_FACE"}
                    and edge_ok
                    and similar_size
                    and distinct
                ):
                    group_recovery = True
                    break
        pad_x = max(16, int(round(face_width * 1.0)))
        pad_y = max(16, int(round(face_height * 1.0)))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(width, x + face_width + pad_x)
        bottom = min(height, y + face_height + pad_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        crop_detector = _get_yunet_detector(
            (crop.shape[1], crop.shape[0]), .45,
        )
        _crop_retval, crop_faces = crop_detector.detect(crop)
        if crop_faces is None:
            continue
        expected_center = (x + face_width / 2.0, y + face_height / 2.0)
        best = None
        for crop_face in crop_faces:
            if len(crop_face) < 15 or not np.all(np.isfinite(crop_face)):
                continue
            score = float(crop_face[14])
            # A local crop is a recovery path, not a second detector.  Keep
            # the promotion floor above the weak diagnostic floor so a hand,
            # pet or background texture cannot become a new usable person.
            crop_floor = .55 if group_recovery else .88
            if score < crop_floor:
                continue
            local_box = _map_face_to_original(
                crop_face, 1.0, crop.shape[1], crop.shape[0],
            )
            confirmed_box = (
                local_box[0] + left, local_box[1] + top,
                local_box[2], local_box[3],
            )
            center = (
                confirmed_box[0] + confirmed_box[2] / 2.0,
                confirmed_box[1] + confirmed_box[3] / 2.0,
            )
            if np.hypot(center[0] - expected_center[0], center[1] - expected_center[1]) > max(face_width, face_height) * 1.5:
                continue
            center_distance = float(np.hypot(
                center[0] - expected_center[0], center[1] - expected_center[1],
            ))
            if (
                best is None
                or center_distance < best[4] * .80
                or (center_distance <= max(face_width, face_height) * .45
                    and score > best[0])
            ):
                best = (score, crop_face, confirmed_box, local_box, center_distance)
        if best is None:
            continue
        score, crop_face, confirmed_box, local_box, _center_distance = best
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        local_quality = _get_face_quality_issue(gray, local_box)
        if local_quality is not None:
            continue
        local_occlusion = _get_face_occlusion_issue(gray, local_box, crop_face)
        if local_occlusion is not None and not (
            group_recovery and local_occlusion in {"FACE_BLURRY", "NO_FACE"}
        ):
            continue
        if any(
            _box_iou(confirmed_box, existing["box"]) >= .25
            or _box_overlap_over_smaller(confirmed_box, existing["box"]) >= .55
            for existing in recovered
        ):
            continue
        # If the full native scan already established a readable face, a
        # local crop may only corroborate that face.  It must not introduce a
        # second person from a weakly repeated texture in a portrait
        # scene.  Group completion remains available through the repeated
        # native-tile path below, which has its own recurrence gate.
        if primary_usable and not group_recovery:
            continue
        recovered.append({
            "box": confirmed_box,
            "ratio": confirmed_box[2] * confirmed_box[3] / float(max(width * height, 1)),
            "score": score,
            "quality_issue": None,
            "raw_face": crop_face,
            "native_crop_recovery": True,
            # A grouped crop may touch one source edge while the facial
            # features themselves remain in frame.  The source verdict is
            # the authoritative crop check; do not reject that candidate
            # merely because its local recovery score is below the ordinary
            # single-face .88 floor.
            "native_edge_readable": bool(
                score >= .88
                or (
                    score >= .75
                    and max(confirmed_box[2:]) >= 300
                )
                or not _is_unusable_edge_cropped_face(
                    crop_face, local_box, crop.shape[1], crop.shape[0], score,
                )
            ),
        })
    return _filter_neck_satellite_candidates(recovered)


def _recover_native_tile_faces(image, primary_candidates):
    """Recover clear faces from overlapping native source crops.

    A candidate must either recur in at least two overlapping windows or be a
    very strong, non-edge detection. This keeps the native-input path from
    counting one-off background texture while recovering small/distant faces
    without resampling the upload.
    """
    tile_candidates = []
    established_primary_count = sum(
        candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
        for candidate in _filter_pet_face_candidates(image, primary_candidates)
    )
    tile_recovery_floor = .60 if established_primary_count >= 3 else .78
    for left, top, tile in _native_tile_origins(image):
        # The ordinary native pass intentionally keeps the production floor at
        # 0.82.  Recovery needs the detector's weak output as well: a face can
        # be repeatably visible in source crops at ~0.60 even when the full
        # canvas suppresses it.  Weak output is accepted here only after
        # repeated overlapping native windows and the same geometry/quality
        # gates used by the normal candidate pipeline.
        view_height, view_width = tile.shape[:2]
        detector = _get_yunet_detector((view_width, view_height), YUNET_CANDIDATE_SCORE_THRESHOLD)
        _retval, faces = detector.detect(tile)
        if faces is None:
            continue
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        for face in faces:
            if len(face) < 15 or not np.all(np.isfinite(face)):
                continue
            score = float(face[14])
            # Repetition is necessary but not sufficient: a stable animal
            # eye, hand or UI detail can repeat in overlapping windows too.
            # Keep a bounded native recovery floor below the main threshold,
            # while rejecting the weakest diagnostic-only tile matches.
            if score < max(YUNET_CANDIDATE_SCORE_THRESHOLD, tile_recovery_floor):
                continue
            x, y, width, height = map(float, face[:4])
            small_group_face = (
                score >= YUNET_SCORE_THRESHOLD
                and min(width, height) >= 18
                and max(width, height) <= 90
            )
            if (
                (min(width, height) < YUNET_MIN_FACE_SIDE and not small_group_face)
                or width * height / float(max(view_width * view_height, 1)) < YUNET_MIN_FACE_RATIO
            ):
                continue
            box = _map_face_to_original(face, 1.0, view_width, view_height)
            issue = _get_face_quality_issue(gray, box)
            # Occlusion-suspect landmark geometry is not rejected at this
            # stage.  It is only a weak geometric hint; repeated clean native
            # crop evidence below can distinguish a real small face from a
            # one-off texture candidate.  Profiles remain excluded because a
            # side-looking fragment needs the established profile path.
            if issue is not None or _is_yunet_profile(face):
                continue
            candidate = {
                "box": box,
                "ratio": box[2] * box[3] / float(max(image.shape[1] * image.shape[0], 1)),
                "score": score,
                "quality_issue": None,
                "raw_face": face,
            }
            x, y, width, height = candidate["box"]
            mapped = {
                **candidate,
                "box": (x + left, y + top, width, height),
                "native_tile_recovery": True,
            }
            tile_candidates.append(mapped)

    if not tile_candidates:
        return primary_candidates
    strong_primary = [
        candidate for candidate in [
            *primary_candidates,
            *tile_candidates,
        ]
        if candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
    ]
    clusters = []
    for candidate in tile_candidates:
        for cluster in clusters:
            if (_box_iou(candidate["box"], cluster["anchor"]["box"]) >= .25
                    or _box_overlap_over_smaller(
                        candidate["box"], cluster["anchor"]["box"],
                    ) >= .55):
                cluster["items"].append(candidate)
                if candidate.get("score", 0.0) > cluster["anchor"].get("score", 0.0):
                    cluster["anchor"] = candidate
                break
        else:
            clusters.append({"anchor": candidate, "items": [candidate]})

    augmented = list(primary_candidates)
    image_height, image_width = image.shape[:2]
    for cluster in clusters:
        candidate = cluster["anchor"]
        repeated = len(cluster["items"]) >= 2
        x, y, width, height = candidate["box"]
        touches_source_edge = any(_frame_edges_touched(
            candidate["box"], image_width, image_height,
        ))
        small_recovery = (
            min(width, height) >= 18
            and max(width, height) <= 90
            # A one-off small tile match is too easily a hand, animal eye or
            # background detail.  Keep the native-tile path dynamic, but use
            # the normal YuNet confidence floor for a non-repeated match.
            and candidate.get("score", 0.0) >= tile_recovery_floor
        )
        if not repeated and not small_recovery:
            continue
        # For a medium/large box, repetition alone is not enough: hands,
        # props, and background contours can be detected consistently in two
        # overlapping windows.  A real secondary face at this size should
        # still clear a useful confidence margin, while low-score distant
        # faces are handled by the native crop-recovery path with explicit
        # source-location corroboration.
        if repeated and max(width, height) > 90 and candidate.get("score", 0.0) < .85:
            continue
        low_score_large = candidate.get("score", 0.0) < .82 and max(width, height) > 90
        if low_score_large and (not strong_primary or candidate.get("score", 0.0) < .78):
            continue
        if (
            candidate.get("score", 0.0) < .82
            and len(strong_primary) >= 2
            and min(width, height) < 90
            and not any(
                np.hypot(
                    candidate["box"][0] + candidate["box"][2] / 2
                    - (anchor["box"][0] + anchor["box"][2] / 2),
                    candidate["box"][1] + candidate["box"][3] / 2
                    - (anchor["box"][1] + anchor["box"][3] / 2),
                ) <= max(250, max(anchor["box"][2:]) * 3)
                for anchor in strong_primary
            )
        ):
            continue
        if candidate.get("score", 0.0) < .55 and touches_source_edge:
            continue
        if any(
            _box_iou(candidate["box"], existing["box"]) >= .25
            or _box_overlap_over_smaller(candidate["box"], existing["box"]) >= .55
            for existing in augmented
        ):
            continue
        augmented.append(candidate)
    return _filter_neck_satellite_candidates(augmented)


def _scan_yunet_native_orientations(
    image,
    *,
    allow_native_tiles=False,
):
    """Run dynamic-input YuNet on the native image dimensions.

    The model evaluates all four orientations on the source pixels, then
    chooses one view; it never sums repeated detections across resized views.
    Coordinates and quality checks therefore remain in source-image pixels.
    """
    image_height, image_width = image.shape[:2]
    image_area = max(image_width * image_height, 1)
    upright_views = {}
    all_views = []
    occluded_views = []
    # FaceDetectorYN receives the original array below; the key is retained
    # only for request-local upright-view bookkeeping.
    native_key = max(image_width, image_height, 1)
    for native_edge in (native_key,):
        source_view = image
        height, width = source_view.shape[:2]
        views = []
        upright_views[native_edge] = []
        for angle, view, faces in _detect_yunet_orientations(source_view):
            if faces is None:
                continue
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            candidates = []
            for face in faces:
                if len(face) < 15 or not np.all(np.isfinite(face)):
                    continue
                x, y, w, h = map(float, face[:4])
                score = float(face[14])
                if (
                    score < YUNET_CANDIDATE_SCORE_THRESHOLD
                    or min(w, h) < YUNET_MIN_FACE_SIDE
                    or w * h / float(width * height) < YUNET_MIN_FACE_RATIO
                ):
                    continue
                # Clip the full box, not just its origin, before assessing quality.
                box = _map_face_to_original(face, 1.0, view.shape[1], view.shape[0])
                issue = _get_face_quality_issue(gray, box)
                edge_readable = (
                    any(_frame_edges_touched(box, view.shape[1], view.shape[0]))
                    and not _is_unusable_edge_cropped_face(
                        face, box, view.shape[1], view.shape[0], score,
                    )
                )
                if (issue is None and score < YUNET_SECONDARY_FACE_SCORE_THRESHOLD and edge_readable
                        and _has_two_in_frame_eye_landmarks(face, view.shape[1], view.shape[0])
                        and _get_face_occlusion_issue(gray, box, face) == "FACE_OCCLUDED"):
                    # A side crop that appears to contain two eyes but whose
                    # eye patches are covered is an insufficient fragment,
                    # not a usable one-eye portrait.
                    issue = "FACE_INCOMPLETE"
                elif issue is None and (
                    (angle != 0 and any(_frame_edges_touched(
                        box, view.shape[1], view.shape[0],
                    )[1::2]))
                    or _is_unusable_edge_cropped_face(
                        face, box, view.shape[1], view.shape[0], score,
                    )
                ):
                    issue = "FACE_INCOMPLETE"
                elif (
                    issue is None
                    and score < YUNET_SCORE_THRESHOLD
                    and _is_yunet_profile(face)
                    and not any(_frame_edges_touched(box, view.shape[1], view.shape[0]))
                ):
                    # A face-shaped box on an ear/back-of-head can clear
                    # YuNet's score threshold.  It is evidence of a person,
                    # never a usable frontal face for identity generation.
                    issue = "FACE_NOT_FRONTAL"
                elif issue is None and score < YUNET_SECONDARY_FACE_SCORE_THRESHOLD and not (
                    any(_frame_edges_touched(box, view.shape[1], view.shape[0]))
                    and not _is_unusable_edge_cropped_face(
                        face, box, view.shape[1], view.shape[0], score,
                    )
                ):
                    # Geometry alone is not enough to call a real, dim side
                    # face covered: both eye patches must independently show
                    # the coverage evidence.  This still rejects hands whose
                    # landmarks mimic a face, but retains a visible face that
                    # merely has a compressed eye span.
                    occlusion_issue = _get_face_occlusion_issue(gray, box, face)
                    issue = (
                        "FACE_OCCLUDED"
                        if (_has_occlusion_suspect_geometry(face)
                            and occlusion_issue == "FACE_OCCLUDED")
                        else occlusion_issue
                    )
                mapped = face if angle == 0 else _map_quarter_turn_box_to_unrotated(
                    face, angle, width, height
                )
                original_box = _map_face_to_original(mapped, 1.0, image_width, image_height)
                candidate = {
                    "box": original_box,
                    "ratio": original_box[2] * original_box[3] / float(image_area),
                    "score": score,
                    "quality_issue": issue,
                    "raw_face": face,
                }
                if angle == 0:
                    upright_views[native_edge].append(candidate)
                if score >= YUNET_SCORE_THRESHOLD:
                    candidates.append(candidate)
            candidates = _filter_neck_satellite_candidates(candidates)
            recognizable = [c for c in candidates if c["quality_issue"] is None]
            if recognizable:
                views.append((
                    len(recognizable),
                    max(c["score"] for c in recognizable),
                    candidates,
                ))
            elif candidates and all(c["quality_issue"] == "FACE_OCCLUDED" for c in candidates):
                # Preserve strong, corroborated coverage evidence so the
                # later diagnostic fallback cannot rediscover a looser box
                # and turn an eye-covered portrait into an accepted face.
                occluded_views.append((
                    len(candidates),
                    max(c["score"] for c in candidates),
                    candidates,
                ))
        all_views.extend(views)
    work = _request_work.get()
    if work is not None and work.get("capture_upright_group_views", True):
        work.setdefault("upright_group_views", {})[id(image)] = (image, upright_views)
    if not all_views:
        # Preserve quality evidence even when no orientation produced a
        # usable candidate; the diagnostic path will explain the failure.
        selected = (
            max(occluded_views, key=lambda item: (item[0], item[1]))[2]
            if occluded_views else []
        )
    else:
        # Select one orientation only (never union boxes across rotations) and
        # prioritize its recognized-face count so duplicate views cannot
        # inflate the group count.
        selected = max(all_views, key=lambda item: (item[0], item[1]))[2]
    if allow_native_tiles:
        selected = _recover_native_candidate_crops(image, selected)
        selected = _recover_native_tile_faces(image, selected)
    # Screenshots and phone captures can contain tiny portrait thumbnails,
    # stickers or control-strip imagery below an otherwise clear group.  When
    # two or more substantial faces are already established, a tiny candidate
    # at the extreme bottom that is far from every anchor is not allowed to
    # inflate the group count.  Keep this geometric guard narrow so genuinely
    # distributed group faces elsewhere in the frame remain countable.
    strong = [
        candidate for candidate in selected
        if candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
        and max(candidate["box"][2:]) >= 70
    ]
    if len(strong) >= 2:
        image_height = image.shape[0]
        anchors = [
            (candidate["box"][0] + candidate["box"][2] / 2.0,
             candidate["box"][1] + candidate["box"][3] / 2.0,
             max(candidate["box"][2:]))
            for candidate in strong
        ]
        kept = []
        for candidate in selected:
            x, y, width, height = candidate["box"]
            center = (x + width / 2.0, y + height / 2.0)
            tiny_bottom = (
                max(width, height) <= 70
                and y + height >= image_height * .82
            )
            far_from_anchors = not any(
                np.hypot(center[0] - ax, center[1] - ay)
                <= max(250.0, anchor_size * 3.0)
                for ax, ay, anchor_size in anchors
            )
            x0, y0, w0, h0 = map(int, candidate["box"])
            tiny_roi = cv2.cvtColor(image[y0:y0+h0, x0:x0+w0], cv2.COLOR_BGR2GRAY)
            thumbnail_readable = (
                candidate["score"] >= .84 and min(w0, h0) >= 35
                and tiny_roi.size
                and np.percentile(tiny_roi, 90)-np.percentile(tiny_roi, 10) >= 35
                and cv2.Laplacian(cv2.resize(tiny_roi, (160, 160)), cv2.CV_64F).var() >= 4
            )
            if tiny_bottom and far_from_anchors and not thumbnail_readable:
                continue
            kept.append(candidate)
        selected = kept
    # The detector score alone is not enough for a secondary person.  Run the
    # same native-pixel facial-core check used by roll recovery before the
    # shared usable-face count is computed.  This removes clearly blurred
    # background people while leaving high-confidence/clear candidates and
    # all established quality labels unchanged.
    for candidate in selected:
        if candidate.get("quality_issue") is not None or candidate.get("score", 0.0) >= .88:
            continue
        native_issue = _native_face_core_issue(
            image, candidate["box"], candidate.get("score", 0.0),
        )
        if native_issue == "FACE_BLURRY":
            candidate["quality_issue"] = native_issue
    return selected


def _confirm_additional_roll_face(view, candidate):
    """Require stronger, spatially matching evidence for a new weak face.

    Hands can survive one rotation's detector threshold. A contextual crop
    must confirm the same location, not a different person elsewhere.
    """
    if candidate["score"] >= YUNET_SECONDARY_FACE_SCORE_THRESHOLD:
        return True
    x, y, w, h = candidate["box"]
    left, top = max(0, x-w), max(0, y-h)
    crop = view[top:min(view.shape[0], y+2*h), left:min(view.shape[1], x+2*w)]
    expected = (x-left, y-top, w, h)
    return any(c["quality_issue"] is None
               and c["score"] >= YUNET_SECONDARY_FACE_SCORE_THRESHOLD
               and _box_iou(c["box"], expected) >= 0.25
               for c in _filter_pet_face_candidates(crop, _scan_yunet_native_orientations(crop)))



def _same_clear_upright_group(reference, observed):
    """One-to-one agreement, including quality and ambiguous extra candidates."""
    if len(reference) < 2 or len(observed) != len(reference):
        return False
    unmatched = list(reference)
    for candidate in observed:
        if (candidate.get("quality_issue") is not None
                or candidate.get("score", 0) < .90
                or candidate.get("raw_face") is None
                or _has_significant_eye_line_roll(candidate.get("raw_face"))):
            return False
        matches = [i for i, anchor in enumerate(unmatched)
                   if _box_iou(candidate["box"], anchor["box"]) >= .50]
        if len(matches) != 1:
            return False
        unmatched.pop(matches[0])
    return not unmatched


def _stable_upright_group_needs_no_roll(image, candidates):
    """Skip roll only after repeated native-source agreement.

    Uses request-local evidence; absent evidence always keeps the full scan.
    The returned candidates, thresholds and public result are never rewritten.
    """
    work = _request_work.get()
    if work is not None:
        work["capture_upright_group_views"] = False
    if work is None or not _same_clear_upright_group(candidates, candidates):
        return False
    evidence = work.get("upright_group_views", {}).get(id(image))
    if evidence is None or evidence[0] is not image:
        return False
    views = evidence[1]
    # Any extra or low-confidence native-source candidate blocks the shortcut.
    for observed in views.values():
        if any(not any(_box_iou(c["box"], a["box"]) >= .50 for a in candidates)
               or c.get("quality_issue") is not None for c in observed):
            return False
    if sum(_same_clear_upright_group(candidates, view) for view in views.values()) < 2:
        return False
    # Preserve the chance to find an additional distant face before stopping.
    work["capture_upright_group_views"] = True
    try:
        _scan_yunet_native_orientations(image)
    finally:
        work["capture_upright_group_views"] = False
    verified = work.get("upright_group_views", {}).get(id(image))
    native_key = max(image.shape[:2], default=1)
    if verified is None or not _same_clear_upright_group(candidates, verified[1].get(native_key, [])):
        return False
    work["stable_group_roll_skipped"] = True
    return True


def _roll_view_confirms_group(source, found, inverse, width, height):
    """Independent rotated view must reproduce every clear source face."""
    if len(source) < 2 or len(found) != len(source):
        return False
    if any(c.get("quality_issue") is not None or c.get("score", 0) < .85 for c in source):
        return False
    unmatched = list(source)
    for candidate in found:
        if candidate.get("quality_issue") is not None or candidate.get("score", 0) < .85:
            return False
        box = _map_rotated_box(candidate["box"], inverse, width, height)
        matches = [i for i, anchor in enumerate(unmatched) if _box_iou(box, anchor["box"]) >= .50]
        if len(matches) != 1:
            return False
        unmatched.pop(matches[0])
    return not unmatched


def _recover_roll_candidates(image, candidates):
    """Recheck camera roll for uncertain faces and existing groups.

    Keep the normal YuNet confidence/quality gates. Select one view, never
    union boxes across rotations. A confident upright single face bypasses
    this pass. Weak or rolled faces are rechecked; established group scans
    keep their count-first view selection.
    """
    if _stable_upright_group_needs_no_roll(image, candidates):
        return candidates
    usable = [c for c in candidates if c["quality_issue"] is None]
    # Once the native source scan has already established two or more usable
    # faces, a roll view must not replace that count with a different subset.
    # Group completion is handled by the native source-window pass before this
    # function is called.
    if len(usable) >= 2:
        return _refine_weak_group_faces(image, candidates)
    if len(usable) == 1 and usable[0]["score"] >= .85 and not any(_has_significant_eye_line_roll(c.get("raw_face")) or (c.get("raw_face") is not None and _is_yunet_profile(c["raw_face"])) for c in candidates):
        # Keep the inexpensive early return for ordinary clear single-face
        # photos. A separate .50+ upright hint away from that face is enough
        # to justify the bounded roll check for a possible tilted inset face.
        probe, probe_scale = image, 1.0
        _retval, hints = _get_yunet_detector(
            (probe.shape[1], probe.shape[0]), .50,
        ).detect(probe)
        has_distinct_hint = False
        for hint in ([] if hints is None else hints):
            if (len(hint) < 15 or not np.all(np.isfinite(hint))
                    or float(hint[14]) < .50
                    or min(hint[2:4]) < YUNET_MIN_FACE_SIDE):
                continue
            hint_box = _map_face_to_original(
                hint, probe_scale, image.shape[1], image.shape[0],
            )
            if (hint_box[2] * hint_box[3] / float(max(image.shape[1] * image.shape[0], 1)) >= .003
                    and _box_iou(hint_box, usable[0]["box"]) < .25):
                has_distinct_hint = True
                break
        if not has_distinct_hint:
            return candidates
    # An uncertain small primary is not a reason to skip checking a separate
    # tilted face. The ordinary confident-single fast path above still applies.
    height, width = image.shape[:2]
    native_issues = [(c, _native_face_core_issue(image, c["box"], c["score"])) for c in candidates] if len(usable) <= 1 else []
    weak_native_failures = False
    if not usable:
        probe, scale = image, 1.0
        _, hints = _get_yunet_detector((probe.shape[1], probe.shape[0]), 0.50).detect(probe)
        if hints is None or not any(
            len(hint) >= 15 and np.all(np.isfinite(hint))
            and float(hint[14]) >= 0.50
            and min(hint[2:4]) / scale >= YUNET_MIN_FACE_SIDE
            for hint in hints
        ):
            return candidates
        # A weak detection is not a usable face. Only consider early stopping
        # when every hint has a native, rotation-invariant detail failure.
        weak_native_failures = not candidates and all(
            len(hint) >= 15 and np.all(np.isfinite(hint))
            and not _has_significant_eye_line_roll(hint)
            and _native_face_core_issue(image, _map_face_to_original(
                hint, scale, width, height), float(hint[14]))
                in {"FACE_TOO_DARK", "FACE_BLURRY"}
            for hint in hints
        ) and not _has_scene_camera_roll(image)
    empty_roll_views = 0
    diagnostic_candidates = []
    roll_extra_candidates = []
    best = candidates
    best_count = len(usable)
    best_score = max((c["score"] for c in usable), default=0)
    roll_group_confirmations = 0
    for angle in (-15, 15, -30, 30, -45, 45):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        view = cv2.warpAffine(image, matrix, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        found = _filter_pet_face_candidates(
            view, _scan_yunet_native_orientations(view),
        )
        if angle in (-15, 15) and _roll_view_confirms_group(
                candidates, found, cv2.invertAffineTransform(matrix), width, height):
            roll_group_confirmations += 1
        if angle == 15 and roll_group_confirmations == 2 and best is candidates:
            work = _request_work.get()
            if work is not None:
                work["roll_group_converged"] = True
            return candidates
        if weak_native_failures and angle in (-15, 15) and not found:
            empty_roll_views += 1
        if angle == 15 and empty_roll_views == 2 and best is candidates:
            # A separate full-resolution source check must also find no
            # candidates; any new evidence retains the original full sweep.
            verified = _filter_pet_face_candidates(
                image, _scan_yunet_native_orientations(image))
            if not verified:
                work = _request_work.get()
                if work is not None:
                    work["weak_quality_converged"] = True
                return candidates
        readable = [c for c in found if c["quality_issue"] is None]
        # Preserve independent high-confidence extra-face evidence before a
        # rotated view is skipped for not improving the main-face score. A
        # tilted inset may be the only face seen in one roll view.
        inverse = cv2.invertAffineTransform(matrix)
        if len(usable) == 1:
            for candidate in readable:
                box = _map_rotated_box(candidate["box"], inverse, width, height)
                if (
                    candidate["score"] >= YUNET_SCORE_THRESHOLD
                    and min(box[2:]) >= MIN_SECONDARY_FACE_SIDE
                    and not any(_frame_edges_touched(box, width, height))
                    and not any(_box_iou(box, anchor["box"]) >= 0.05 for anchor in candidates)
                    and _confirm_additional_roll_face(view, candidate)
                ):
                    roll_extra_candidates.append((angle, {
                        **candidate, "box": box,
                        "ratio": box[2] * box[3] / float(width * height),
                    }))
        if len(readable) < best_count or (len(readable) == best_count and (len(usable) >= 2 or max((c["score"] for c in readable),default=0) <= best_score)):
            continue
        mapped = []
        for candidate in found:
            box = _map_rotated_box(candidate["box"], inverse, width, height)
            if (candidate["quality_issue"] is None
                    and any(_frame_edges_touched(box, width, height)[1::2])):
                # Rotating a partial face can make its edge contact disappear
                # in the temporary canvas. Keep the original-frame crop
                # rejection when bringing that candidate back.
                candidate = {**candidate, "quality_issue": "FACE_INCOMPLETE"}
            # Rotation expands axis-aligned boxes into hair/background. Keep
            # native quality evidence from the same established face region.
            for anchor, native_issue in native_issues:
                if native_issue and _box_iou(box, anchor["box"]) >= .25:
                    candidate = {**candidate, "quality_issue": native_issue}
                    break
            # Do not replace a readable upright face with a larger rotated
            # box that only gains confidence by absorbing blurred background.
            # This is checked on native pixels, so it cannot reject a true
            # tilted inset whose own face detail is still readable.
            if len(usable) == 1 and candidate["quality_issue"] is None:
                rotated_native_issue = _native_face_core_issue(
                    image, box, candidate["score"],
                )
                if rotated_native_issue is not None:
                    candidate = {**candidate, "quality_issue": rotated_native_issue}
            # A rotation-only addition to a single-face result needs enough
            # native pixels; enlarging a small head cannot establish identity.
            if (len(usable) == 1 and min(box[2:]) < MIN_SECONDARY_FACE_SIDE
                    and not any(_box_iou(box, anchor["box"]) >= 0.25 for anchor in candidates)):
                continue
            is_new_candidate = (
                candidate["quality_issue"] is None
                and not any(
                    _box_iou(box, anchor["box"]) >= 0.05
                    for anchor in (usable if len(usable) >= 2 else candidates)
                )
            )
            # A tilted inset or second portrait can evade the upright scan.
            # Preserve it only as evidence here; it is promoted below only if
            # two independent roll views reproduce the same in-frame face.
            if (
                len(usable) == 1
                and is_new_candidate
                and candidate["score"] >= YUNET_SCORE_THRESHOLD
                and not any(_frame_edges_touched(box, width, height))
                and _confirm_additional_roll_face(view, candidate)
            ):
                roll_extra_candidates.append((angle, {
                    **candidate, "box": box,
                    "ratio": box[2] * box[3] / float(width * height),
                }))
            if is_new_candidate and not _confirm_additional_roll_face(view, candidate):
                # Keep native blur evidence for the failure explanation only.
                # An unconfirmed rotation must never contribute a usable face.
                if not usable and _native_extreme_motion_blur(image, box):
                    diagnostic_candidates.append({**candidate, "box": box,
                        "quality_issue": "FACE_BLURRY", "diagnostic_only": True})
                continue
            mapped.append({**candidate, "box": box, "native_roll_confirmed": True,
                           "ratio": box[2] * box[3] / float(width * height)})
        confirmed_count = len([c for c in mapped if c["quality_issue"] is None])
        if confirmed_count < best_count or (confirmed_count == best_count and (len(usable) >= 2 or max((c["score"] for c in mapped if c["quality_issue"] is None),default=0) <= best_score)):
            continue
        best, best_count = mapped, confirmed_count
        best_score = max((c["score"] for c in mapped if c["quality_issue"] is None),default=0)
        # An empty primary scan can stop once a strong contextual candidate is
        # confirmed; no need to pay for every remaining angle.
        if not usable and best_score >= .92:
            break
    if len(usable) == 1:
        clusters = []
        for angle, candidate in roll_extra_candidates:
            for cluster in clusters:
                if _box_iou(candidate["box"], cluster["anchor"]["box"]) >= .50:
                    cluster["items"].append((angle, candidate))
                    if candidate["score"] > cluster["anchor"]["score"]:
                        cluster["anchor"] = candidate
                    break
            else:
                clusters.append({"anchor": candidate, "items": [(angle, candidate)]})
        for cluster in clusters:
            if len({angle for angle, _candidate in cluster["items"]}) >= 2:
                return _filter_neck_satellite_candidates([*candidates, cluster["anchor"]])
        # A roll view is still useful for confirming a distinct second face,
        # but it must not replace an already readable upright face merely
        # because its larger rotated box scores a little higher.  That box can
        # include background and make native blur validation less reliable.
        return candidates
    return best or diagnostic_candidates


def _context_face_confirmations(image, source_box, angles, min_score, required_angles):
    """Corroborate one location in three distinct native-pixel contexts.

    Context windows never add source detail. Every observation must pass the
    existing quality/occlusion gates, and all landmarks must map back inside
    the real upload rather than artificial rotation padding. Repetitions are
    evidence for one face, never extra faces in the usable count.
    """
    height, width = image.shape[:2]
    x, y, box_width, box_height = source_box
    contexts = set()
    confirmations = []
    for padding in (.5, 1.0, 1.5):
        left = max(0, round(x - box_width * padding))
        top = max(0, round(y - box_height * padding))
        right = min(width, round(x + box_width * (1 + padding)))
        bottom = min(height, round(y + box_height * (1 + padding)))
        bounds = (left, top, right, bottom)
        if bounds in contexts or right <= left or bottom <= top:
            return None
        contexts.add(bounds)
        crop = image[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        observations = []
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1)
            view = cv2.warpAffine(crop, matrix, (crop_width, crop_height)) if angle else crop
            inverse = cv2.invertAffineTransform(matrix)
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            _, faces = _get_yunet_detector((crop_width, crop_height), min_score).detect(view)
            for face in (() if faces is None else faces):
                if len(face) < 15 or not np.all(np.isfinite(face)) or float(face[14]) < min_score:
                    continue
                box = _map_face_to_original(face, 1, crop_width, crop_height)
                mapped = _map_rotated_box(box, inverse, crop_width, crop_height)
                mapped = (mapped[0] + left, mapped[1] + top, mapped[2], mapped[3])
                score = float(face[14])
                if _box_iou(source_box, mapped) < .35:
                    continue
                if (
                    _get_face_quality_issue(gray, box) is not None
                    or _get_face_occlusion_issue(gray, box, face) is not None
                    or _is_unusable_edge_cropped_face(face, box, crop_width, crop_height, score)
                    or _native_face_core_issue(image, mapped, score) is not None
                ):
                    continue
                points = cv2.transform(
                    np.asarray(face[4:14], dtype=np.float32).reshape(1, 5, 2), inverse,
                )[0] + np.asarray((left, top))
                if not np.all(
                    (points[:, 0] >= left + 2) & (points[:, 0] < right - 2)
                    & (points[:, 1] >= top + 2) & (points[:, 1] < bottom - 2)
                ):
                    continue
                observations.append({
                    "angle": angle, "box": mapped, "score": score,
                    "source_landmarks": points,
                })
        if len({observation["angle"] for observation in observations}) < required_angles:
            return None
        confirmations.append(observations)
    best = max((c for context in confirmations for c in context), key=lambda c: c["score"])
    # Require the requested number of views in every context to agree with
    # the same final location, not merely with a larger initial hint box.
    if not all(len({c["angle"] for c in context if _box_iou(c["box"], best["box"]) >= .50})
               >= required_angles for context in confirmations):
        return None
    return best


def _retain_context_confirmed_primary(image, before_roll, after_roll):
    """Do not demote an established primary merely because another face appears.

    Keep the weak-secondary false-positive guard. Its exemption requires the
    original primary to recur in three unrotated contexts at the ordinary
    secondary confidence floor, with all normal quality checks intact.
    """
    usable = [c for c in before_roll if c["quality_issue"] is None]
    if len(usable) != 1 or len(after_roll) <= len(before_roll):
        return after_roll
    anchor = usable[0]
    if (anchor.get("raw_face") is None or anchor["score"] < YUNET_SECONDARY_FACE_SCORE_THRESHOLD
            or not any(c is anchor for c in after_roll)
            or _native_face_core_issue(image, anchor["box"], anchor["score"]) is not None):
        return after_roll
    if _context_face_confirmations(image, anchor["box"], (0,), YUNET_SECONDARY_FACE_SCORE_THRESHOLD, 1):
        anchor["native_context_confirmed_primary"] = True
    return after_roll


def _recover_context_confirmed_group_faces(image, candidates):
    """Challenge a settled group only when a distinct native hint remains.

    A weak full-frame hint is not a face count. It must recur above .90 in
    two different roll views in each of three source contexts. This bounded
    challenge avoids a full-scene rotation sweep for ordinary group photos.
    """
    usable = [c for c in candidates if c["quality_issue"] is None
              and c["score"] >= YUNET_SCORE_THRESHOLD]
    if len(usable) < 2:
        return candidates
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, hints = _get_yunet_detector((width, height), .50).detect(image)
    recovered = list(candidates)
    for face in (() if hints is None else hints):
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        box = _map_face_to_original(face, 1, width, height)
        if not .50 <= score < YUNET_SCORE_THRESHOLD or min(box[2:]) < 32:
            continue
        if any(_box_iou(box, c["box"]) >= .20 or _box_overlap_over_smaller(box, c["box"]) >= .55
               for c in recovered):
            continue
        if any(_frame_edges_touched(box, width, height)):
            continue
        if (_get_face_quality_issue(gray, box) is not None
                or _native_face_core_issue(image, box, score) is not None
                or _get_face_occlusion_issue(gray, box, face) is not None):
            continue
        if not any(.55 <= max(box[2:]) / max(c["box"][2:]) <= 1.8 for c in usable):
            continue
        confirmed = _context_face_confirmations(image, box, (-45, -30, -15, 15, 30, 45), .90, 2)
        if confirmed is None or _source_edge_crop_verdict(image, confirmed["box"]) == "incomplete":
            continue
        confirmed_box = confirmed["box"]
        if any(_box_iou(confirmed_box, c["box"]) >= .25
               or _box_overlap_over_smaller(confirmed_box, c["box"]) >= .55 for c in recovered):
            continue
        recovered.append({
            "box": confirmed_box,
            "ratio": confirmed_box[2] * confirmed_box[3] / float(width * height),
            "score": confirmed["score"], "raw_face": face, "quality_issue": None,
            "source_landmarks": confirmed["source_landmarks"],
            "native_roll_confirmed": True,
            "native_context_group_recovery": True,
        })
    return _filter_pet_face_candidates(image, recovered) if len(recovered) > len(candidates) else candidates


def _refine_weak_group_faces(image, candidates):
    """Reconcile uncertain group members using independent, aligned views.

    Only groups containing weak tile-only evidence need this pass. Coordinates
    and landmarks are mapped back before clustering; repetitions never add to
    the count. Every addition still has to pass the common quality checks.
    """
    usable = [c for c in candidates if c["quality_issue"] is None]
    if len(usable) < 3 or not any(c.get("native_tile_recovery") and c["score"] < .82 for c in usable):
        return candidates
    height, width = image.shape[:2]
    observations = []

    def observe(view, inverse, origin=(0, 0), tag=None):
        vh, vw = view.shape[:2]
        _, faces = _get_yunet_detector((vw, vh), .65).detect(view)
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        for face in (() if faces is None else faces):
            local_box = _map_face_to_original(face, 1, vw, vh)
            if min(local_box[2:]) < 28 or _get_face_quality_issue(gray, local_box) is not None:
                continue
            if _get_face_occlusion_issue(gray, local_box, face) is not None:
                continue
            mapped = _map_rotated_box_to_unrotated(face, inverse, vw, vh)
            mapped[0] += origin[0]
            mapped[1] += origin[1]
            box = _map_face_to_original(mapped, 1, width, height)
            points = cv2.transform(
                np.asarray(face[4:14], np.float32).reshape(-1, 1, 2), inverse,
            ).reshape(-1, 2)+np.asarray(origin)
            observations.append({
                "box": box, "ratio": box[2]*box[3]/float(width*height),
                "score": float(face[14]), "raw_face": face, "quality_issue": None,
                "source_landmarks": points, "observation": tag, "native_roll_confirmed": True,
            })

    for angle in (-45, -30, -15, 15, 30, 45):
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        observe(cv2.warpAffine(image, matrix, (width, height)),
                cv2.invertAffineTransform(matrix), tag=("full", angle))
    # Weak/distant hints get a small native window; this changes context, not
    # source detail. It prevents accepting a back-of-head tile from repetition
    # alone and recovers a genuine face suppressed by the full canvas.
    hints = list(usable)
    for c in observations:
        if not any(_box_iou(c["box"], old["box"]) >= .25 for old in hints):
            hints.append(c)
    for c in hints:
        if c["score"] >= .88:
            continue
        x, y, w, h = c["box"]
        for pad in (.7, 1.3, 2.0):
            left = max(0, int(x-pad*w))
            top = max(0, int(y-pad*h))
            crop = image[top:min(height, int(y+(1+pad)*h)), left:min(width, int(x+(1+pad)*w))]
            ch, cw = crop.shape[:2]
            for angle in (-45, -30, -15, 0, 15, 30, 45):
                matrix = cv2.getRotationMatrix2D((cw/2, ch/2), angle, 1)
                observe(cv2.warpAffine(crop, matrix, (cw, ch)), cv2.invertAffineTransform(matrix),
                        (left, top), ("crop", left, top, pad, angle))
    clusters = []
    for c in sorted(observations, key=lambda c: c["score"], reverse=True):
        for cluster in clusters:
            if (_box_iou(c["box"], cluster[0]["box"]) >= .25
                    or _box_overlap_over_smaller(c["box"], cluster[0]["box"]) >= .55):
                cluster.append(c)
                break
        else:
            clusters.append([c])
    confirmed = []
    for cluster in clusters:
        best = cluster[0]
        raw = best["raw_face"]
        eye_span = abs(float(raw[6]-raw[4]))/max(float(raw[2]), 1)
        mouth_span = abs(float(raw[12]-raw[10]))/max(float(raw[2]), 1)
        if eye_span < .15 or mouth_span < .08:
            continue
        floor = .88 if _has_occlusion_suspect_geometry(raw) else .84
        if best["score"] < floor or len({c["observation"] for c in cluster if c["score"] >= floor}) < 2:
            continue
        if _source_edge_crop_verdict(image, best["box"]) == "incomplete":
            continue
        confirmed.append(best)
    # A face behind another head may retain a good outline while its lower
    # identity features are hidden. Corroborate actual landmark overlap, not
    # the number of people or the mere proximity of two boxes.
    visible = []
    for c in confirmed:
        nose = c["source_landmarks"][2]
        mouth = c["source_landmarks"][3:].mean(axis=0)
        hidden = False
        for other in confirmed:
            if other is c:
                continue
            ox, oy, ow, oh = other["box"]
            if (c["box"][1] < oy and ox <= mouth[0] <= ox+ow
                    and oy-oh*.22 <= mouth[1] <= oy+oh*.55
                    and ox <= nose[0] <= ox+ow and oy-oh*.22 <= nose[1]):
                hidden = True
                break
        if not hidden:
            visible.append(c)
    # Keep well-established source faces when rotations provide no replacement.
    # Weak tile-only candidates, however, need the positive corroboration above.
    result = list(visible)
    for c in candidates:
        if c["quality_issue"] is not None:
            result.append(c)
        elif c["score"] >= .82 and not any(_box_iou(c["box"], v["box"]) >= .25 for v in confirmed):
            result.append(c)
    return _recover_group_edge_profiles(image, result)


def _recover_group_edge_profiles(image, candidates):
    """Confirm a cropped side profile from visible source landmarks.

    Padding never supplies facial evidence. Detector-only magnification is
    bounded to the edge window; quality and landmark visibility use source
    coordinates/pixels. A nose outside the original image is not accepted.
    """
    height, width = image.shape[:2]
    hints = []
    for angle in (0, 90, 180, 270):
        view = image if not angle else _quarter_turn_with_matrix(image, angle)[0]
        _, faces = _get_yunet_detector((view.shape[1], view.shape[0]), .45).detect(view)
        for f in (() if faces is None else faces):
            b = _map_face_to_original(
                f if not angle else _map_quarter_turn_box_to_unrotated(f, angle, width, height),
                1, width, height,
            )
            if ((b[0] <= 8 or b[0]+b[2] >= width-8) and min(b[2:]) >= 50
                    and not any(_box_iou(b, c["box"]) >= .25 for c in candidates)):
                hints.append(b)
    if not hints:
        return candidates
    found = []
    for side in {"left" if b[0] <= 8 else "right" for b in hints}:
        for top in (0, height//4, height//2):
            left = 0 if side == "left" else width-width//3
            crop = image[top:min(height, top+height//2), left:left+width//3]
            if not any(top <= b[1]+b[3]/2 <= top+crop.shape[0] for b in hints):
                continue
            scale = 3.0
            pad = 100
            view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            view = cv2.copyMakeBorder(view, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            vh, vw = view.shape[:2]
            for angle in (-60, -45, -30, -15, 15, 30, 45, 60):
                matrix = cv2.getRotationMatrix2D((vw/2, vh/2), angle, 1)
                inverse = cv2.invertAffineTransform(matrix)
                rotated = cv2.warpAffine(view, matrix, (vw, vh))
                _, faces = _get_yunet_detector((vw, vh), .65).detect(rotated)
                for f in (() if faces is None else faces):
                    points = (cv2.transform(
                        np.asarray(f[4:14], np.float32).reshape(-1, 1, 2), inverse,
                    ).reshape(-1, 2)-pad)/scale+np.array((left, top))
                    if not (np.all(points[:, 0] >= 8) and np.all(points[:, 0] <= width-8)
                            and np.all(points[:, 1] >= 0) and np.all(points[:, 1] < height)):
                        continue
                    b = _map_rotated_box_to_unrotated(f, inverse, vw, vh)
                    b[:2] = (b[:2]-pad)/scale+np.array((left, top))
                    b[2:] /= scale
                    box = _map_face_to_original(b, 1, width, height)
                    if not any(_box_iou(box, hint) >= .15 for hint in hints):
                        continue
                    if any(_box_iou(box, c["box"]) >= .25 for c in candidates):
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if _get_face_quality_issue(gray, box) is not None:
                        continue
                    found.append({
                        "box": box, "ratio": box[2]*box[3]/float(width*height), "score": float(f[14]),
                        "raw_face": f, "quality_issue": None, "native_edge_readable": True,
                        "native_roll_confirmed": True, "profile": _is_yunet_profile(f), "angle": angle,
                    })
    result = list(candidates)
    for best in sorted(found, key=lambda c: c["score"], reverse=True):
        matches = [c for c in found if _box_iou(c["box"], best["box"]) >= .4]
        if (best["score"] >= .70 and any(c["profile"] for c in matches)
                and len({c["angle"] for c in matches}) >= 2):
            if not any(_box_iou(best["box"], c["box"]) >= .25 for c in result):
                result.append(best)
    return result


def _drop_unconfirmed_weak_secondary_faces(candidates):
    """Do not count one-off weak boxes beside an established clear face.

    Native YuNet can return a plausible-looking box for a hand, clothing
    detail, or a compressed background pattern.  A weak box is retained when
    the native crop/tile recovery path independently corroborated it; those
    markers are the same source-pixel evidence used for genuine distant
    people.  Without that corroboration, a weak secondary must not turn a
    valid upload into a false multi-face result.
    """
    clear_anchors = [
        candidate for candidate in candidates
        if candidate.get("quality_issue") is None
        and candidate.get("score", 0.0) >= .90
    ]
    # Large collages can contain many clear anchors plus a couple of weak
    # tile-only detections at the bottom/edge.  Once the scene already has a
    # substantial confirmed group, those weak tile boxes are not additional
    # usable people.  The lower-cardinality group rules below remain intact.
    if len(clear_anchors) >= 8:
        return [
            candidate for candidate in candidates
            if not (
                candidate.get("native_tile_recovery")
                and candidate.get("score", 0.0) < .84
            )
        ]
    if not clear_anchors:
        strongest = max(
            (candidate for candidate in candidates if candidate.get("quality_issue") is None),
            key=lambda candidate: candidate.get("score", 0.0),
            default=None,
        )
        if strongest is not None and strongest.get("score", 0.0) >= .88:
            return [
                candidate for candidate in candidates
                if not (
                    candidate.get("native_crop_recovery")
                    and candidate.get("score", 0.0) < .70
                )
            ]
        return candidates
    # Beside one clear anchor, a small sub-.90 secondary is more often a
    # background portrait/texture than a second usable
    # face.  Keep the stronger multi-face evidence (multiple clear anchors or
    # independently recovered candidates) unchanged.
    if len(clear_anchors) == 1 and len(candidates) == 2:
        anchor = clear_anchors[0]
        kept = [
            candidate for candidate in candidates
            if not (
                candidate is not anchor
                and candidate.get("quality_issue") is None
                and candidate.get("score", 0.0) < .90
                and max(candidate.get("box", (0, 0, 0, 0))[2:]) <= 90
                and not (
                    candidate.get("native_crop_recovery")
                    or candidate.get("native_tile_recovery")
                    or (
                        candidate.get("score", 0.0) >= YUNET_SCORE_THRESHOLD
                        and candidate.get("raw_face") is not None
                        and _is_yunet_profile(candidate["raw_face"])
                    )
                    or candidate.get("native_structured_recovery")
                    or candidate.get("native_repeated_small_recovery")
                    or candidate.get("native_roll_confirmed")
                    or candidate.get("native_context_confirmed_primary")
                )
            )
        ]
        candidates = kept
        clear_anchors = [
            candidate for candidate in candidates
            if candidate.get("quality_issue") is None
            and candidate.get("score", 0.0) >= .90
        ]
    kept = []
    for candidate in candidates:
        score = candidate.get("score", 0.0)
        recovered = bool(
            candidate.get("native_crop_recovery")
            or candidate.get("native_tile_recovery")
            or candidate.get("native_structured_recovery")
            or candidate.get("native_repeated_small_recovery")
            or candidate.get("native_roll_confirmed")
            or candidate.get("native_context_confirmed_primary")
        )
        tile_only_recovery = bool(
            candidate.get("native_tile_recovery")
            and not candidate.get("native_crop_recovery")
        )
        profile_secondary = bool(
            score >= YUNET_SCORE_THRESHOLD
            and candidate.get("raw_face") is not None
            and _is_yunet_profile(candidate["raw_face"])
        )
        if score < .84 and not recovered and not profile_secondary:
            continue
        if tile_only_recovery and score < .90 and max(candidate.get("box", (0, 0, 0, 0))[2:]) > 90:
            continue
        if tile_only_recovery and score < .90 and max(candidate.get("box", (0, 0, 0, 0))[2:]) <= 90:
            candidate_top = candidate.get("box", (0, 0, 0, 0))[1]
            if candidate_top + candidate.get("box", (0, 0, 0, 0))[3] < min(
                anchor["box"][1] for anchor in clear_anchors
            ):
                continue
        kept.append(candidate)
    return kept


def _recover_structured_low_confidence_faces(image, candidates):
    """Recover a clearly structured small face missed by the strict scan.

    This is deliberately narrower than lowering YuNet's acceptance threshold:
    the candidate must be a non-edge, frontal landmark layout with a clean
    native crop, and it must not overlap an already detected face.  It is
    intended for small secondary faces in group photos, not as a general
    low-confidence acceptance path.
    """
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = _get_yunet_detector((width, height), 0.40)
    _retval, detected = detector.detect(image)
    if detected is None:
        return candidates
    recovered = list(candidates)
    for face in detected:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        if not (0.40 <= score < YUNET_SCORE_THRESHOLD):
            continue
        x, y, face_width, face_height = map(float, face[:4])
        if min(face_width, face_height) < 35 or max(face_width, face_height) > 60:
            continue
        ratio = (face_width * face_height) / float(max(width * height, 1))
        frontal = _has_low_confidence_frontal_geometry(face)
        profile = _is_yunet_profile(face)
        if ratio < 0.002 or not (
            (frontal and score >= 0.40)
            or (profile and score >= 0.70)
        ):
            continue
        box = (
            max(0, int(round(x))),
            max(0, int(round(y))),
            max(1, int(round(face_width))),
            max(1, int(round(face_height))),
        )
        if any(_box_iou(box, item.get("box", (0, 0, 0, 0))) >= 0.25 for item in recovered):
            continue
        if any(_frame_edges_touched(box, width, height)):
            continue
        if _get_face_quality_issue(gray_image, box) is not None:
            continue
        if _get_face_occlusion_issue(gray_image, box, face) is not None:
            continue
        recovered.append({
            "box": box,
            "ratio": ratio,
            "score": score,
            "quality_issue": None,
            "raw_face": face,
            "native_structured_recovery": True,
        })
    return recovered


def _rescue_dominant_occluded_face(image, candidates):
    """Clear a false occlusion label when one box dominates the scene.

    On strongly rotated full-body uploads YuNet can emit a large face box and
    a much smaller landmark-shaped companion along the same subject.  If both
    are labelled occluded, treating both as independent failed people loses
    the otherwise readable dominant face.  Require a large area gap, native
    facial detail, and no frame truncation before keeping only the dominant
    source candidate.
    """
    occluded = [
        candidate for candidate in candidates
        if candidate.get("quality_issue") == "FACE_OCCLUDED"
    ]
    if len(occluded) == 1:
        dominant = occluded[0]
        score = dominant.get("score", 0.0)
        if (
            score >= .84
            and _native_face_core_issue(image, dominant["box"], score) is None
            and not _is_face_box_cut_by_frame(
                dominant["box"], image.shape[1], image.shape[0],
            )
        ):
            return [{**dominant, "quality_issue": None}]
        return candidates
    if len(occluded) < 2:
        return candidates
    dominant = max(occluded, key=lambda candidate: candidate.get("box", (0, 0, 0, 0))[2] * candidate.get("box", (0, 0, 0, 0))[3])
    dominant_area = dominant["box"][2] * dominant["box"][3]
    companion_area = max(
        candidate["box"][2] * candidate["box"][3]
        for candidate in occluded
        if candidate is not dominant
    )
    score = dominant.get("score", 0.0)
    if (
        score >= .84
        and dominant_area >= companion_area * 2.5
        and _native_face_core_issue(image, dominant["box"], score) is None
        and not _is_face_box_cut_by_frame(
            dominant["box"], image.shape[1], image.shape[0],
        )
    ):
        return [{**dominant, "quality_issue": None}]
    return candidates


def _recover_native_soft_face(image, candidates):
    """Recover a large, readable face that native YuNet scored weakly.

    Some high-resolution AI portraits have a large but low-contrast face that
    the dynamic detector scores below the ordinary acceptance floor.  Native
    quarter-turn evidence can still localize it without resizing the upload.
    Only one substantial candidate is promoted, and it must pass the same
    source-pixel detail and frame checks used by normal candidates.
    """
    # Once the normal scan located a face-shaped region but rejected it for a
    # specific quality reason, a weaker rotated match must not erase that
    # explanation or turn an occluded face into a pass.
    if candidates:
        return candidates
    if _is_confident_pet_image(image):
        return candidates
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best = None
    soft_candidates = []
    for angle in (0, *YUNET_QUARTER_TURN_ANGLES):
        view, turn_matrix = (
            (image, None)
            if angle == 0 else _quarter_turn_with_matrix(image, angle)
        )
        detector = _get_yunet_detector(
            (view.shape[1], view.shape[0]), .10,
        )
        _retval, faces = detector.detect(view)
        if faces is None:
            continue
        inverse = None if angle == 0 else cv2.invertAffineTransform(turn_matrix)
        for face in faces:
            if len(face) < 15 or not np.all(np.isfinite(face)):
                continue
            score = float(face[14])
            if score < .60:
                continue
            box = tuple(map(int, face[:4])) if angle == 0 else _map_rotated_box(
                face[:4], inverse, width, height,
            )
            if _is_confident_pet_image(image, box):
                continue
            if angle == 90 and not _is_yunet_profile(face):
                # A weak quarter-turn match is especially prone to reading a
                # body/scene contour as a face. Strong normal detections still
                # come through the main orientation path.
                continue
            edge_readable = any(_frame_edges_touched(box, width, height))
            high_score_soft_face = score >= .88 and min(box[2:]) >= 100
            if min(box[2:]) < 180 and not high_score_soft_face:
                continue
            if edge_readable and not (
                _is_yunet_profile(face)
                and score >= .70
                and _is_unusable_edge_cropped_face(
                    face, box, width, height, score,
                )
            ):
                continue
            quality_issue = _get_face_quality_issue(gray, box)
            if quality_issue is not None and not (
                high_score_soft_face and quality_issue == "FACE_BLURRY"
            ):
                continue
            occlusion = _get_face_occlusion_issue(gray, box, face)
            if occlusion is not None and score >= .75 and not (
                high_score_soft_face and occlusion == "FACE_BLURRY"
            ):
                continue
            if _native_face_core_issue(image, box, score) is not None:
                continue
            if _is_yunet_profile(face) and score < .70:
                continue
            candidate = {
                "box": box,
                "ratio": box[2] * box[3] / float(max(width * height, 1)),
                "score": score,
                "quality_issue": None,
                "raw_face": face,
                "native_soft_recovery": True,
                "native_edge_readable": bool(edge_readable),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
            soft_candidates.append(candidate)
    soft_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    if soft_candidates and soft_candidates[0]["score"] >= .70:
        group = [soft_candidates[0]]
        for candidate in soft_candidates[1:]:
            if all(_box_iou(candidate["box"], existing["box"]) < .25 for existing in group):
                group.append(candidate)
            if len(group) == 2:
                return group
    return [best] if best is not None else candidates


def _crop_modified_same_face(first, second) -> bool:
    """Match two candidate dictionaries without changing normal deduplication."""
    return (
        _box_iou(first["box"], second["box"]) >= .25
        or _box_overlap_over_smaller(first["box"], second["box"]) >= .55
    )


def _same_crop_modified_soft_location(first, second) -> bool:
    """Collapse weak rotated localizations of one soft face in an edited crop."""
    if _crop_modified_same_face(first, second):
        return True
    first_box, second_box = first["box"], second["box"]
    first_center = np.asarray((
        first_box[0] + first_box[2] / 2.0,
        first_box[1] + first_box[3] / 2.0,
    ))
    second_center = np.asarray((
        second_box[0] + second_box[2] / 2.0,
        second_box[1] + second_box[3] / 2.0,
    ))
    largest_diagonal = max(
        float(np.hypot(first_box[2], first_box[3])),
        float(np.hypot(second_box[2], second_box[3])),
    )
    return float(np.linalg.norm(first_center - second_center)) <= largest_diagonal * .65


def _crop_modified_soft_context_evidence(image, source_box):
    """Require real-pixel corroboration before a soft second face can count.

    This is available only for a request whose final Cropper state differs from
    the complete source.  Rotated padding is inference-only: each mapped
    landmark must remain inside the unrotated source context.
    """
    height, width = image.shape[:2]
    x, y, box_width, box_height = map(int, source_box)
    evidence = {
        "contexts": [], "direct_contexts": 0, "direct_score": 0.0,
        "strict_secondary": False, "edge_secondary": False,
    }
    bounds_seen = set()
    for padding in (.5, 1.0, 1.5):
        left = max(0, round(x - box_width * padding))
        top = max(0, round(y - box_height * padding))
        right = min(width, round(x + box_width * (1 + padding)))
        bottom = min(height, round(y + box_height * (1 + padding)))
        bounds = (left, top, right, bottom)
        if bounds in bounds_seen or right <= left + 4 or bottom <= top + 4:
            return evidence
        bounds_seen.add(bounds)
        crop = image[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        clean_matches = []
        for angle in (0, -15, 15):
            matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1)
            view = crop if angle == 0 else cv2.warpAffine(
                crop, matrix, (crop_width, crop_height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            inverse = cv2.invertAffineTransform(matrix)
            _retval, faces = _get_yunet_detector(
                (crop_width, crop_height), .45,
            ).detect(view)
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            for face in (() if faces is None else faces):
                if len(face) < 15 or not np.all(np.isfinite(face)):
                    continue
                score = float(face[14])
                if score < .45:
                    continue
                local = _map_face_to_original(face, 1, crop_width, crop_height)
                mapped = _map_rotated_box(local, inverse, crop_width, crop_height)
                if mapped is None:
                    continue
                mapped = (mapped[0] + left, mapped[1] + top, mapped[2], mapped[3])
                iou = _box_iou(mapped, source_box)
                overlap = _box_overlap_over_smaller(mapped, source_box)
                if iou < .20 and overlap < .55:
                    continue
                points = cv2.transform(
                    np.asarray(face[4:14], dtype=np.float32).reshape(1, 5, 2), inverse,
                )[0] + np.asarray((left, top))
                if not np.all(
                    (points[:, 0] >= left + 2) & (points[:, 0] < right - 2)
                    & (points[:, 1] >= top + 2) & (points[:, 1] < bottom - 2)
                ):
                    continue
                if (
                    _get_face_quality_issue(gray, local) is not None
                    or _get_face_occlusion_issue(gray, local, face) is not None
                    or _is_unusable_edge_cropped_face(
                        face, local, crop_width, crop_height, score,
                    )
                    or _native_face_core_issue(image, mapped, score) is not None
                ):
                    continue
                clean_matches.append({
                    "angle": angle, "score": score, "iou": iou,
                    "overlap": overlap, "box": mapped,
                })
        strong_angles = {
            item["angle"] for item in clean_matches if item["score"] >= .65
        }
        direct = [
            item for item in clean_matches
            if item["score"] >= .60 and (item["iou"] >= .50 or item["overlap"] >= .80)
        ]
        if direct:
            evidence["direct_contexts"] += 1
            evidence["direct_score"] += max(item["score"] for item in direct)
        evidence["contexts"].append({
            "strong_angles": strong_angles,
            "edge_clean": any(item["score"] >= .55 for item in clean_matches),
        })
    evidence["strict_secondary"] = len(evidence["contexts"]) == 3 and all(
        len(context["strong_angles"]) >= 2 for context in evidence["contexts"]
    )
    evidence["edge_secondary"] = len(evidence["contexts"]) == 3 and all(
        context["edge_clean"] for context in evidence["contexts"]
    )
    return evidence


def _collect_crop_modified_readable_soft_edge_proposals(image):
    """Return source-edge soft candidates that already pass normal soft guards."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    proposals = []
    for angle in (0, *YUNET_QUARTER_TURN_ANGLES):
        view, matrix = (
            (image, None) if angle == 0 else _quarter_turn_with_matrix(image, angle)
        )
        inverse = None if matrix is None else cv2.invertAffineTransform(matrix)
        _retval, faces = _get_yunet_detector(
            (view.shape[1], view.shape[0]), .10,
        ).detect(view)
        view_gray = gray if angle == 0 else cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        for face in (() if faces is None else faces):
            if len(face) < 15 or not np.all(np.isfinite(face)):
                continue
            score = float(face[14])
            if score < .60:
                continue
            box = tuple(map(int, face[:4])) if angle == 0 else _map_rotated_box(
                face[:4], inverse, width, height,
            )
            if box is None or not any(_frame_edges_touched(box, width, height)):
                continue
            high_score_soft_face = score >= .88 and min(box[2:]) >= 100
            if min(box[2:]) < 180 and not high_score_soft_face:
                continue
            if angle == 90 and not _is_yunet_profile(face):
                continue
            if _is_confident_pet_image(image, box):
                continue
            quality_issue = _get_face_quality_issue(
                view_gray, tuple(map(int, face[:4])),
            )
            if quality_issue is not None and not (
                high_score_soft_face and quality_issue == "FACE_BLURRY"
            ):
                continue
            occlusion = _get_face_occlusion_issue(
                view_gray, tuple(map(int, face[:4])), face,
            )
            if occlusion is not None and score >= .75 and not (
                high_score_soft_face and occlusion == "FACE_BLURRY"
            ):
                continue
            if _native_face_core_issue(image, box, score) is not None:
                continue
            if _is_yunet_profile(face) and score < .70:
                continue
            if _source_edge_crop_verdict(image, box) != "readable":
                continue
            candidate = {
                "box": box,
                "ratio": box[2] * box[3] / float(max(width * height, 1)),
                "score": score,
                "quality_issue": None,
                "raw_face": face,
                "native_soft_recovery": True,
                "native_edge_readable": True,
                "crop_modified_soft_edge_proposal": True,
            }
            if not any(_crop_modified_same_face(candidate, existing) for existing in proposals):
                proposals.append(candidate)
    return proposals


def _stabilize_crop_modified_soft_group(image, recovered):
    """Keep one soft face and add a second only with independent source proof."""
    if (
        len(recovered) <= 1
        or not all(candidate.get("native_soft_recovery") for candidate in recovered)
    ):
        return recovered
    # `_recover_native_soft_face` is shared with raw uploads for compatibility.
    # Its historic edge handling cannot become a crop-only primary: repeat the
    # authoritative source-frame check here, without changing that raw path.
    normal = [
        candidate for candidate in recovered
        if _source_edge_crop_verdict(image, candidate["box"]) != "incomplete"
    ]
    if not normal:
        return []
    proposed = [*normal]
    for item in _collect_crop_modified_readable_soft_edge_proposals(image):
        if not any(_crop_modified_same_face(item, prior) for prior in proposed):
            proposed.append(item)
    evidence = {
        id(item): _crop_modified_soft_context_evidence(image, item["box"])
        for item in proposed
    }
    primary = max(
        normal,
        key=lambda item: (
            evidence[id(item)]["direct_contexts"],
            evidence[id(item)]["direct_score"],
            item.get("score", 0.0),
        ),
    )
    secondary = []
    for item in proposed:
        if item is primary or _same_crop_modified_soft_location(item, primary):
            continue
        proof = evidence[id(item)]
        if proof["strict_secondary"] or (
            item.get("crop_modified_soft_edge_proposal") and proof["edge_secondary"]
        ):
            secondary.append(item)
    if not secondary:
        return [primary]
    extra = max(
        secondary,
        key=lambda item: (
            bool(item.get("crop_modified_soft_edge_proposal")),
            evidence[id(item)]["direct_contexts"],
            evidence[id(item)]["direct_score"],
            item.get("score", 0.0),
        ),
    )
    return [primary, extra]


_CROP_MODIFIED_NORMAL_SCALES = (1.5, 2.0, 3.0)
_CROP_MODIFIED_FIRST_HINT_MAX_PIXELS = 1_000_000
_CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE = 1280
_CROP_MODIFIED_MAX_FIRST_SCALE_TILES = 16
_CROP_MODIFIED_MAX_HINTS_PER_VIEW = 24
_CROP_MODIFIED_MAX_FIRST_SCALE_SEEDS = 12
_CROP_MODIFIED_MAX_NATIVE_GUIDE_SEEDS = 4
# The two seed classes share this fixed budget.  Native hints only locate a
# local 1.5x scan; they are never scale evidence themselves.
_CROP_MODIFIED_MAX_SCALED_HINTS = (
    _CROP_MODIFIED_MAX_FIRST_SCALE_SEEDS
    + _CROP_MODIFIED_MAX_NATIVE_GUIDE_SEEDS
)
_CROP_MODIFIED_MAX_CONTEXT_CONFIRMATIONS = 8
_CROP_MODIFIED_MAX_CONTEXT_FACES_PER_VIEW = 16
# A 25px source candidate can land on the ordinary 29px floor only after
# independent contextual localization.  This is a provisional localization
# tolerance, never a final usable-face floor; do not open a wider band.
_CROP_MODIFIED_MIN_SCALED_RECOVERY_SIDE = 25
_CROP_MODIFIED_MIN_SMALL_FACE_SCALE_SCORE = .88


def _map_crop_modified_scaled_source_face(
    face,
    scale,
    left,
    top,
    image_width,
    image_height,
):
    """Map one locally scaled YuNet result into uploaded-pixel coordinates."""
    source_face = np.asarray(face, dtype=np.float32).copy()
    source_face[:14] /= scale
    source_face[0] += left
    source_face[1] += top
    landmarks = source_face[4:14].reshape(5, 2)
    landmarks += np.asarray((left, top), dtype=np.float32)
    source_face[4:14] = landmarks.reshape(-1)
    return source_face, _map_face_to_original(
        source_face, 1, image_width, image_height,
    )


def _crop_modified_scaled_context(image, source_box, *, max_source_side=None):
    """Take a bounded real-pixel context around a first-scale face hint."""
    height, width = image.shape[:2]
    x, y, box_width, box_height = map(float, source_box)
    # This is the largest context used by the subsequent three-context proof.
    # Keeping it square avoids a thin source crop turning into a distorted
    # detector canvas, while preserving the complete hint and its surroundings.
    face_side = int(np.ceil(max(box_width, box_height)))
    if max_source_side is not None and face_side > max_source_side:
        # The candidate itself cannot fit in the bounded real-pixel view.  Do
        # not replace it with an artificial global/downsampled confirmation.
        return image[:0, :0], 0, 0
    side = max(160, int(round(max(box_width, box_height) * 4)))
    if max_source_side is not None:
        # Crop the surrounding context, not the candidate.  This retains a
        # true 2x/3x observation for large faces instead of silently skipping
        # it just because the former 4-face-width window exceeded 1280px.
        side = max(face_side, min(side, int(max_source_side)))
    crop_width = min(width, side)
    crop_height = min(height, side)
    center_x = x + box_width / 2
    center_y = y + box_height / 2
    left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
    return image[top:top + crop_height, left:left + crop_width], left, top


def _crop_modified_first_scale_tiles(image):
    """Yield bounded, overlapping source tiles for a 1.5x hint scan."""
    height, width = image.shape[:2]
    scale = _CROP_MODIFIED_NORMAL_SCALES[0]
    source_side = max(
        160,
        int(np.sqrt(_CROP_MODIFIED_FIRST_HINT_MAX_PIXELS) // scale),
    )
    tile_width = min(width, source_side)
    tile_height = min(height, source_side)
    overlap = max(48, int(round(source_side * .32)))

    def starts(length, tile):
        if length <= tile:
            return [0]
        span = length - tile
        steps = max(1, int(np.ceil(span / max(1, tile - overlap))))
        return [round(span * index / steps) for index in range(steps + 1)]

    top_starts = starts(height, tile_height)
    left_starts = starts(width, tile_width)
    if len(top_starts) * len(left_starts) > _CROP_MODIFIED_MAX_FIRST_SCALE_TILES:
        # Direct callers can bypass the browser's 2000px master limit. Keep
        # this optional recovery bounded there as well, sampling the whole
        # source extent rather than concentrating all tiles in one corner.
        top_count = min(4, len(top_starts))
        left_count = min(4, len(left_starts))
        top_starts = [
            top_starts[round(index * (len(top_starts) - 1) / max(top_count - 1, 1))]
            for index in range(top_count)
        ]
        left_starts = [
            left_starts[round(index * (len(left_starts) - 1) / max(left_count - 1, 1))]
            for index in range(left_count)
        ]
    for top in top_starts:
        for left in left_starts:
            yield image[top:top + tile_height, left:left + tile_width], left, top


def _crop_modified_normal_scaled_proposals(image, settled):
    """Find only ordinary-quality scaled hints that recur at all three scales."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clusters = []

    def record_support(face, scale, left=0, top=0, *, allow_new, anchor_box=None):
        if len(face) < 15 or not np.all(np.isfinite(face)):
            return False
        score = float(face[14])
        if score < .45:
            return False
        source_face, box = _map_crop_modified_scaled_source_face(
            face, scale, left, top, width, height,
        )
        if min(box[2:]) < 18:
            return None
        # A native .45 hint is only permitted to guide a local scale scan.  It
        # cannot cause an unrelated detection in that scan to become evidence.
        if anchor_box is not None and not _crop_modified_same_face(
            {"box": box}, {"box": anchor_box},
        ):
            return None
        support = {
            "scale": scale,
            "score": score,
            "box": box,
            "face": source_face,
        }
        for cluster in clusters:
            if _crop_modified_same_face({"box": box}, {"box": cluster["box"]}):
                cluster["supports"].append(support)
                if score > cluster["score"]:
                    cluster.update(score=score, box=box, face=source_face)
                return cluster
        if allow_new:
            cluster = {
                "score": score,
                "box": box,
                "face": source_face,
                "supports": [support],
            }
            clusters.append(cluster)
            return cluster
        return None

    def ranked_hint_faces(faces, scale):
        """Bound a low-threshold detector view before any clustering work."""
        minimum_local_side = 18 * scale
        hints = [
            face for face in (() if faces is None else faces)
            if (
                len(face) >= 15
                and np.all(np.isfinite(face))
                and float(face[14]) >= .45
                and min(face[2:4]) >= minimum_local_side
            )
        ]
        return sorted(
            hints,
            key=lambda face: (
                float(face[14]), min(face[2:4]), float(face[2] * face[3]),
            ),
            reverse=True,
        )[:_CROP_MODIFIED_MAX_HINTS_PER_VIEW]

    # Preserve the 1.5x first-tier evidence. Low-area (including narrow, tall)
    # crops keep the original complete view; larger masters are split into
    # overlapping real-pixel tiles before scaling. A face seen only after
    # enlargement can still enter the recovery without an unbounded canvas.
    first_scale = _CROP_MODIFIED_NORMAL_SCALES[0]
    scaled_pixels = int(round(width * first_scale)) * int(round(height * first_scale))
    first_contexts = (
        [(image, 0, 0)]
        if scaled_pixels <= _CROP_MODIFIED_FIRST_HINT_MAX_PIXELS
        else _crop_modified_first_scale_tiles(image)
    )
    for crop, left, top in first_contexts:
        if not crop.size:
            continue
        view = cv2.resize(
            crop, None, fx=first_scale, fy=first_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        try:
            _retval, faces = _get_yunet_detector(
                (view.shape[1], view.shape[0]), .45,
            ).detect(view)
            for face in ranked_hint_faces(faces, first_scale):
                record_support(face, first_scale, left, top, allow_new=True)
        finally:
            del view

    # A recovery probe must never fan a texture-heavy crop into an unbounded
    # number of 2x/3x DNN calls. The normal shared scan still sees every face;
    # this cap only ranks optional, first-tier hints by strongest source proof.
    first_scale_clusters = [
        cluster for cluster in clusters
        if not any(_crop_modified_same_face(cluster, item) for item in settled)
    ]
    first_scale_clusters.sort(
        key=lambda item: (
            item["score"], min(item["box"][2:]), item["box"][2] * item["box"][3],
        ),
        reverse=True,
    )
    seed_clusters = first_scale_clusters[:_CROP_MODIFIED_MAX_FIRST_SCALE_SEEDS]

    # A full native .45 scan restores a small amount of context-sensitive
    # discovery that tiled 1.5x scans can lose at tile seams.  It is a bounded
    # *position guide* only: every accepted guide must first produce a matching
    # actual 1.5x local detection, then independently pass 2x/3x and the
    # source-context proof below.
    native_guides = []
    try:
        _retval, native_faces = _get_yunet_detector(
            (width, height), .45,
        ).detect(image)
    except Exception:
        native_faces = None
    for face in ranked_hint_faces(native_faces, 1.0):
        native_box = _map_face_to_original(face, 1, width, height)
        guide = {"box": native_box}
        if (
            min(native_box[2:]) < 18
            or any(_crop_modified_same_face(guide, item) for item in settled)
            # A lower-ranked first-tier cluster is deliberately still eligible
            # here: it must be seen again in the guide's own local 1.5x view
            # before it can consume one of the four supplemental slots.
            or any(_crop_modified_same_face(guide, item) for item in seed_clusters)
            or any(_crop_modified_same_face(guide, item) for item in native_guides)
        ):
            continue
        native_guides.append(guide)
        if len(native_guides) >= _CROP_MODIFIED_MAX_NATIVE_GUIDE_SEEDS:
            break

    # Convert a native guide into a real first-tier observation before it can
    # enter the fixed second/third-scale budget.
    supplemental_clusters = []
    for guide in native_guides:
        first_scale = _CROP_MODIFIED_NORMAL_SCALES[0]
        crop, left, top = _crop_modified_scaled_context(
            image,
            guide["box"],
            max_source_side=int(_CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE // first_scale),
        )
        if (
            not crop.size
            or max(crop.shape[:2]) * first_scale > _CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE
        ):
            continue
        view = cv2.resize(
            crop, None, fx=first_scale, fy=first_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        try:
            _retval, faces = _get_yunet_detector(
                (view.shape[1], view.shape[0]), .45,
            ).detect(view)
            for face in ranked_hint_faces(faces, first_scale):
                cluster = record_support(
                    face,
                    first_scale,
                    left,
                    top,
                    allow_new=True,
                    anchor_box=guide["box"],
                )
                if (
                    cluster is not None
                    and not any(cluster is existing for existing in seed_clusters)
                    and not any(cluster is existing for existing in supplemental_clusters)
                ):
                    supplemental_clusters.append(cluster)
        finally:
            del view

    # Keep the combined recovery bounded even when a detector view returns
    # several duplicate localizations for the same guide.
    seed_clusters.extend(
        supplemental_clusters[:_CROP_MODIFIED_MAX_NATIVE_GUIDE_SEEDS],
    )
    seed_clusters = seed_clusters[:_CROP_MODIFIED_MAX_SCALED_HINTS]
    if not seed_clusters:
        return []

    # Scale only real-pixel source windows around the bounded 1.5x hints. The
    # 2x/3x tiers can corroborate an existing cluster, never create a new one.
    for scale in _CROP_MODIFIED_NORMAL_SCALES[1:]:
        for cluster in seed_clusters:
            source_box = cluster["box"]
            crop, left, top = _crop_modified_scaled_context(
                image,
                source_box,
                max_source_side=int(_CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE // scale),
            )
            if (
                not crop.size
                or max(crop.shape[:2]) * scale > _CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE
            ):
                # A truly large candidate already has full-image evidence; do
                # not make a global-scale fallback exceed the normal upload
                # detector's maximum input side merely to corroborate it.
                continue
            view = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,
            )
            try:
                _retval, faces = _get_yunet_detector(
                    (view.shape[1], view.shape[0]), .45,
                ).detect(view)
                for face in ranked_hint_faces(faces, scale):
                    # Later tiers may corroborate first-scale clusters only.
                    record_support(
                        face,
                        scale,
                        left,
                        top,
                        allow_new=False,
                        anchor_box=source_box,
                    )
            finally:
                del view

    proposals = []
    for cluster in clusters:
        box = cluster["box"]
        strongest_scale_scores = {}
        for scale in _CROP_MODIFIED_NORMAL_SCALES:
            scores = [
                item["score"]
                for item in cluster["supports"]
                if item["scale"] == scale
            ]
            if scores:
                strongest_scale_scores[scale] = max(scores)
        # The standard 29px floor remains authoritative for ordinary scans.
        # An edited crop may nevertheless preserve a 25–28px face that was
        # already readable in the original.  Admit that narrow band only when
        # every independently scaled real-pixel observation is very strong;
        # it must still clear the strict three-context confirmation below.
        strong_small_face = (
            min(box[2:]) >= _CROP_MODIFIED_MIN_SCALED_RECOVERY_SIDE
            and len(strongest_scale_scores) == len(_CROP_MODIFIED_NORMAL_SCALES)
            and min(strongest_scale_scores.values())
            >= _CROP_MODIFIED_MIN_SMALL_FACE_SCALE_SCORE
        )
        if (
            {item["scale"] for item in cluster["supports"]} != set(_CROP_MODIFIED_NORMAL_SCALES)
            or cluster["score"] < .82
            or (
                min(box[2:]) < YUNET_MIN_FACE_SIDE
                and not strong_small_face
            )
            or any(_crop_modified_same_face({"box": box}, item) for item in settled)
            or _get_face_quality_issue(gray, box) is not None
            or _get_face_occlusion_issue(gray, box, cluster["face"]) is not None
            or _native_face_core_issue(image, box, cluster["score"]) is not None
            or _source_edge_crop_verdict(image, box) == "incomplete"
        ):
            continue
        proposals.append(cluster)
    return sorted(proposals, key=lambda item: item["score"], reverse=True)


def _map_crop_modified_context_box(face, inverse, scale, width, height, left, top):
    """Map a scaled, rotated context detection back to uploaded pixels."""
    local = _map_face_to_original(face, 1, width, height)
    unrotated = _map_rotated_box(local, inverse, width, height)
    if unrotated is None:
        return None
    x, y, box_width, box_height = unrotated
    return (
        int(round(x / scale)) + left,
        int(round(y / scale)) + top,
        max(1, int(round(box_width / scale))),
        max(1, int(round(box_height / scale))),
    )


def _crop_modified_face_uses_only_real_context_pixels(face, valid_mask):
    """Reject a rotated-context result that reaches synthetic border pixels."""
    if len(face) < 15:
        return False
    mask_height, mask_width = valid_mask.shape[:2]
    x, y, box_width, box_height = map(float, face[:4])
    # Include the detector rectangle as well as every landmark.  The one-pixel
    # interior offset avoids treating a valid right/bottom canvas coordinate as
    # a synthetic rotation border solely because OpenCV boxes are half-open.
    points = np.vstack((
        np.asarray((
            (x + 1, y + 1),
            (x + box_width - 1, y + 1),
            (x + box_width - 1, y + box_height - 1),
            (x + 1, y + box_height - 1),
            (x + box_width / 2, y + box_height / 2),
        ), dtype=np.float32),
        np.asarray(face[4:14], dtype=np.float32).reshape(5, 2),
    ))
    if not np.all(np.isfinite(points)):
        return False
    for point_x, point_y in points:
        pixel_x, pixel_y = int(round(point_x)), int(round(point_y))
        if (
            pixel_x < 1 or pixel_x >= mask_width - 1
            or pixel_y < 1 or pixel_y >= mask_height - 1
            or not np.all(valid_mask[
                pixel_y - 1:pixel_y + 2,
                pixel_x - 1:pixel_x + 2,
            ])
        ):
            return False
    return True


def _confirm_crop_modified_normal_source_context(image, source_box):
    """Confirm an ordinary scaled hint from three real source contexts only."""
    height, width = image.shape[:2]
    x, y, box_width, box_height = map(int, source_box)
    contexts, observations_by_context = [], []
    for padding in (.5, 1.0, 1.5):
        left = max(0, round(x - box_width * padding))
        top = max(0, round(y - box_height * padding))
        right = min(width, round(x + box_width * (1 + padding)))
        bottom = min(height, round(y + box_height * (1 + padding)))
        bounds = (left, top, right, bottom)
        if bounds in contexts or right <= left or bottom <= top:
            return None
        contexts.append(bounds)
        crop = image[top:bottom, left:right]
        crop_long_side = max(crop.shape[:2])
        # Retain the normal small-face enlargement, but cap large source
        # contexts before any rotated DNN view is allocated.  `inference_scale`
        # may legitimately be below one here and every map below divides by it.
        inference_scale = min(
            min(3.0, max(1.0, 360.0 / crop_long_side)),
            _CROP_MODIFIED_CONTEXT_MAX_VIEW_SIDE / crop_long_side,
        )
        base = (
            cv2.resize(crop, None, fx=inference_scale, fy=inference_scale,
                       interpolation=cv2.INTER_LINEAR)
            if inference_scale != 1.0 else crop
        )
        base_height, base_width = base.shape[:2]
        base_real_mask = np.full((base_height, base_width), 255, dtype=np.uint8)
        accepted = []
        for angle in (0, -15, 15):
            matrix = cv2.getRotationMatrix2D((base_width / 2, base_height / 2), angle, 1)
            if angle == 0:
                view, valid_mask = base, base_real_mask
            else:
                view = cv2.warpAffine(
                    base, matrix, (base_width, base_height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                valid_mask = cv2.warpAffine(
                    base_real_mask,
                    matrix,
                    (base_width, base_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            inverse = cv2.invertAffineTransform(matrix)
            _retval, faces = _get_yunet_detector(
                (base_width, base_height), .45,
            ).detect(view)
            # The recovery must stay bounded even on a texture-heavy crop.
            # Quality/core/edge checks are expensive, so rank the only faces
            # that could possibly satisfy the .82 confirmation threshold.
            context_faces = sorted(
                (
                    face for face in (() if faces is None else faces)
                    if (
                        len(face) >= 15
                        and np.all(np.isfinite(face))
                        and float(face[14]) >= .82
                    )
                ),
                key=lambda face: (
                    float(face[14]), min(face[2:4]), float(face[2] * face[3]),
                ),
                reverse=True,
            )[:_CROP_MODIFIED_MAX_CONTEXT_FACES_PER_VIEW]
            view_gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            for face in context_faces:
                score = float(face[14])
                if not _crop_modified_face_uses_only_real_context_pixels(
                    face, valid_mask,
                ):
                    continue
                mapped = _map_crop_modified_context_box(
                    face, inverse, inference_scale,
                    base_width, base_height, left, top,
                )
                if mapped is None:
                    continue
                if (
                    _box_iou(mapped, source_box) < .20
                    and np.hypot(
                        mapped[0] + mapped[2] / 2 - (x + box_width / 2),
                        mapped[1] + mapped[3] / 2 - (y + box_height / 2),
                    ) > max(box_width, box_height) * .65
                    ):
                    continue
                local = _map_face_to_original(face, 1, base_width, base_height)
                quality_issue = _get_face_quality_issue(view_gray, local)
                occlusion_issue = _get_face_occlusion_issue(
                    view_gray, local, face,
                )
                # The final candidate is still checked against its unscaled
                # source pixels below.  A very strong context view can be
                # marked blurry merely because its tight detector rectangle
                # differs by a few resampling pixels; do not discard it until
                # the final source-pixel guard has made that decision.
                allow_context_blur = score >= .90
                if (
                    (quality_issue is not None and not (
                        allow_context_blur and quality_issue == "FACE_BLURRY"
                    ))
                    or (occlusion_issue is not None and not (
                        allow_context_blur and occlusion_issue == "FACE_BLURRY"
                    ))
                    or _source_edge_crop_verdict(image, mapped) == "incomplete"
                    or _native_face_core_issue(image, mapped, score) is not None
                ):
                    continue
                points = cv2.transform(
                    np.asarray(face[4:14], np.float32).reshape(-1, 1, 2), inverse,
                ).reshape(-1, 2) / inference_scale + np.asarray((left, top))
                if not np.all(
                    (points[:, 0] >= left + .5) & (points[:, 0] < right - .5)
                    & (points[:, 1] >= top + .5) & (points[:, 1] < bottom - .5)
                ):
                    continue
                source_face = np.asarray(face, dtype=np.float32).copy()
                source_face[:4] = np.asarray(mapped, dtype=np.float32)
                source_face[4:14] = points.reshape(-1)
                source_face[14] = score
                accepted.append({
                    "box": mapped,
                    "score": score,
                    "angle": angle,
                    "source_landmarks": points,
                    "source_face": source_face,
                })
        if not accepted:
            return None
        observations_by_context.append(accepted)

    source_matches = []
    for group in observations_by_context:
        matching = [
            item for item in group
            if _box_iou(item["box"], source_box) >= .30
            or _box_overlap_over_smaller(item["box"], source_box) >= .55
        ]
        if len({item["angle"] for item in matching}) < 2:
            return None
        source_matches.extend(matching)
    best = max(source_matches, key=lambda item: item["score"])
    if not all(
        any(
            _box_iou(item["box"], best["box"]) >= .30
            or _box_overlap_over_smaller(item["box"], best["box"]) >= .55
            for item in group
        )
        for group in observations_by_context
    ):
        return None
    return best


def _append_crop_modified_normal_scaled_context_faces(image, settled):
    """Append only independently confirmed ordinary faces to an edited crop."""
    recovered = list(settled)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for proposal in _crop_modified_normal_scaled_proposals(
        image, recovered,
    )[:_CROP_MODIFIED_MAX_CONTEXT_CONFIRMATIONS]:
        confirmed = _confirm_crop_modified_normal_source_context(image, proposal["box"])
        if confirmed is None:
            continue
        # The three-context result corroborates identity/location; use the
        # most faithful *strictly valid* source localization as the final box.
        # A tightly resized context can move a rectangle by a few pixels and
        # trigger a spurious blur flag, while the 1.5/2/3 proposal is already
        # independently verified at original coordinates.
        final_options = (
            (proposal["box"], proposal["face"], proposal["score"]),
            (confirmed["box"], confirmed["source_face"], confirmed["score"]),
        )
        selected = None
        for box, source_face, source_score in final_options:
            if (
                min(box[2:]) < YUNET_MIN_FACE_SIDE
                or box[2] * box[3] / float(max(width * height, 1)) < YUNET_MIN_FACE_RATIO
                or any(_crop_modified_same_face({"box": box}, item) for item in recovered)
                or _get_face_quality_issue(gray, box) is not None
                or _get_face_occlusion_issue(gray, box, source_face) is not None
                or _native_face_core_issue(image, box, source_score) is not None
                or _source_edge_crop_verdict(image, box) == "incomplete"
            ):
                continue
            selected = (box, source_face, source_score)
            break
        if selected is None:
            continue
        box, source_face, source_score = selected
        candidate = {
            "box": box,
            "ratio": box[2] * box[3] / float(max(image.shape[0] * image.shape[1], 1)),
            "score": max(proposal["score"], confirmed["score"], source_score),
            "quality_issue": None,
            "raw_face": source_face,
            "source_landmarks": np.asarray(source_face[4:14], dtype=np.float32).reshape(5, 2),
            "crop_modified_scaled_context_confirmed": True,
        }
        # Only test the new candidate against the current group. Existing
        # accepted faces remain untouched even if this changes graphic context.
        filtered = _filter_pet_face_candidates(image, [*recovered, candidate])
        if any(item is candidate for item in filtered):
            recovered.append(candidate)
    return recovered


def _count_recheck_preserves_existing_faces(result, checked) -> bool:
    """Require a crop-only count recheck to retain every established face.

    The optional recovery may ask the ordinary one-face verifier to complete a
    partially recovered group.  A larger count is not sufficient evidence if
    the corrected view has swapped out one of the faces already established on
    the actual upload.  Match each original box to a distinct verified box so
    the recheck stays genuinely add-only.
    """
    existing_boxes = result.get("usable_face_boxes") or ([result.get("face")] if result.get("face") else [])
    checked_boxes = checked.get("usable_face_boxes") or ([checked.get("face")] if checked.get("face") else [])
    if not existing_boxes or len(checked_boxes) < len(existing_boxes):
        return False

    match_for_checked = [-1] * len(checked_boxes)

    def same_face(first, second):
        return (
            _box_iou(first, second) >= .25
            or _box_overlap_over_smaller(first, second) >= .55
        )

    def assign(existing_index, seen):
        for checked_index, checked_box in enumerate(checked_boxes):
            if checked_index in seen or not same_face(
                existing_boxes[existing_index], checked_box,
            ):
                continue
            seen.add(checked_index)
            prior = match_for_checked[checked_index]
            if prior < 0 or assign(prior, seen):
                match_for_checked[checked_index] = existing_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(existing_boxes)))


def _verify_face_count(
    image_path,
    result,
    locale,
    *,
    allow_crop_recovery_count_recheck: bool = False,
):
    """Challenge a one-candidate pass without weakening face thresholds.

    Count within one corrected view only. Two strong faces plus a match to
    the original face are required; never sum detections across rotations.
    Ordinary close portraits and multi-face results bypass this. A real
    crop-only recovery may request one add-only recheck after it promoted an
    original one-face result to a partial group; it cannot lower or replace
    that group with weaker evidence.
    """
    face = result.get("face")
    size = result.get("img_size")
    if (
        not result.get("valid")
        or not face
        or not size
        or (
            result.get("face_count") != 1
            and not (
                allow_crop_recovery_count_recheck
                and result.get("face_count", 0) > 1
            )
        )
    ):
        return result
    # Older callers/tests may provide only the public face box.  Without a
    # detector score there is no safe basis for the close-pair challenge; a
    # clearly large face can return unchanged without touching the image.
    if (result.get("face_score") is None
            and face[2] * face[3] / float(max(size[0] * size[1], 1)) >= .02):
        return result
    try:
        image = _read_image(image_path)
    except (OSError, ValueError):
        return result
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        return result
    height, width = image.shape[:2]
    # Close profiles can merge into one localization. A second box is only a
    # candidate: count it after the same quality review, never via a pre-built
    # MULTIPLE_FACES error that the product layer could mistake for usable faces.
    close_faces = None
    view, scale = image, 1.0
    if result.get("face_score", 0.0) < .90:
        _retval, close_faces = _get_yunet_detector(
            (view.shape[1], view.shape[0]), .45,
        ).detect(view)
    close_pair = []
    for candidate in ([] if close_faces is None else close_faces):
        if len(candidate) < 15 or not np.all(np.isfinite(candidate)):
            continue
        score = float(candidate[14])
        box = _map_face_to_original(candidate, scale, width, height)
        if min(box[2:]) < MIN_SECONDARY_FACE_SIDE or any(_frame_edges_touched(box, width, height)):
            continue
        close_pair.append((score, candidate, box))
    if len(close_pair) >= 2:
        profiles = [item for item in close_pair if item[0] >= .85 and _is_yunet_profile(item[1])]
        # Native dynamic input is more sensitive to scene texture at the old
        # .45 probe threshold.  Keep the close-pair rejection tied to the
        # ordinary YuNet acceptance floor so a single portrait cannot gain a
        # second person from a weak native-only texture candidate.
        neighbours = [item for item in close_pair if item[0] >= YUNET_SCORE_THRESHOLD]
        if profiles and any(_box_iou(profile[2], neighbour[2]) < .25 for profile in profiles for neighbour in neighbours):
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            reviewed=[]
            for score,raw,box in close_pair:
                if score < YUNET_SCORE_THRESHOLD:
                    continue
                issue=_get_face_quality_issue(gray,box)
                if issue is None and score < YUNET_SECONDARY_FACE_SCORE_THRESHOLD:
                    issue=_get_face_occlusion_issue(gray,box,raw)
                reviewed.append({"box":box,"ratio":box[2]*box[3]/float(width*height),
                                 "score":score,"raw_face":raw,"quality_issue":issue})
            checked=_yunet_candidate_result(_filter_pet_face_candidates(image,reviewed),width,height,locale,image_path=image_path)
            if (
                checked
                and checked.get("face_count", 0) > result.get("face_count", 0)
                and (
                    not allow_crop_recovery_count_recheck
                    or _count_recheck_preserves_existing_faces(result, checked)
                )
            ):
                return checked
    if face[2] * face[3] / float(max(size[0] * size[1], 1)) >= 0.02:
        return result
    # Camera roll can hide extra faces even when one face was already found.
    for angle in (-30, 30, 0):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        view = image if angle == 0 else cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        candidates = _filter_pet_face_candidates(
            view, _scan_yunet_native_orientations(view),
        )
        recognizable = [c for c in candidates if c["quality_issue"] is None]
        strong = [c for c in recognizable if c["score"] >= 0.88]
        if len(strong) < 2:
            continue
        inverse = cv2.invertAffineTransform(matrix)
        if not any(_box_iou(
            face, c["box"] if angle == 0 else _map_rotated_box(c["box"], inverse, width, height),
        ) >= 0.25 for c in recognizable):
            continue
        mapped=[{**c,"box":c["box"] if angle==0 else _map_rotated_box(c["box"],inverse,width,height)} for c in recognizable]
        checked=_yunet_candidate_result(mapped,width,height,locale,image_path=image_path)
        if (
            checked
            and checked.get("face_count", 0) > result.get("face_count", 0)
            and (
                not allow_crop_recovery_count_recheck
                or _count_recheck_preserves_existing_faces(result, checked)
            )
        ):
            return checked
    return result


def _native_face_core_issue(image, box, score=0):
    """Check central facial detail at native resolution, excluding hair/background.

    Downscaling a candidate below the small-face floor must not bypass quality.
    This is conservative evidence of lost detail, not a face/person detector.
    """
    x, y, w, h = map(int, box)
    if min(w, h) < 100:
        return None
    roi = image[max(0, y + int(h * .18)):min(image.shape[0], y + int(h * .70)),
                max(0, x + int(w * .12)):min(image.shape[1], x + int(w * .88))]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    core = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
    if _has_readable_dark_detail(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box):
        core = np.clip(core.astype(np.float32)*4, 0, 255).astype(np.uint8)
    if np.percentile(core, 90) < 45 and not _has_readable_dark_detail(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box):
        return "FACE_TOO_DARK"
    dx = float(cv2.Sobel(core, cv2.CV_64F, 1, 0).var())
    dy = float(cv2.Sobel(core, cv2.CV_64F, 0, 1).var())
    if (score < YUNET_SECONDARY_FACE_SCORE_THRESHOLD and dx < 100 and dy < 100) or (score < .90 and dx < 250 and dy < 500 and dy > dx * 1.8):
        return "FACE_BLURRY"
    return None


def _has_readable_soft_face(image, candidate):
    """Retain a soft, strongly localized face with balanced facial edge detail."""
    if candidate["score"] < .92 or min(candidate["box"][2:]) < 200:
        return False
    x, y, w, h = map(int, candidate["box"])
    crop = image[y+int(h*.18):y+int(h*.70), x+int(w*.12):x+int(w*.88)]
    if crop.size == 0:
        return False
    core = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (160,160), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(core, (10,90))
    dx = float(cv2.Sobel(core, cv2.CV_64F, 1,0).var())
    dy = float(cv2.Sobel(core, cv2.CV_64F, 0,1).var())
    return (high >= 45 and low <= 240 and min(dx,dy) >= 150
            and max(dx,dy) < min(dx,dy)*1.5)


def _yunet_candidate_result(
    candidates,
    image_width,
    image_height,
    locale,
    image_path: str | None = None,
):
    if image_path and candidates:
        native_image = _read_image(image_path)
        if native_image is not None:
            for candidate in candidates:
                source_crop_verdict = (
                    _source_edge_crop_verdict(native_image, candidate["box"])
                    if candidate["score"] < .90 or candidate["quality_issue"] == "FACE_OCCLUDED"
                    else None
                )
                if source_crop_verdict == "incomplete":
                    if not (
                        candidate.get("native_edge_readable")
                    ):
                        candidate["quality_issue"] = "FACE_INCOMPLETE"
                elif (
                    candidate["quality_issue"] == "FACE_OCCLUDED"
                    and source_crop_verdict == "readable_bottom_crop"
                ):
                    # Bottom-cropped portraits can make the occlusion scan
                    # mistake the missing mouth/chin for a covered face. A
                    # direct source-frame check with two eyes and a complete
                    # nose is stronger evidence and keeps this permitted crop.
                    candidate["quality_issue"] = None
                if candidate["quality_issue"] is None:
                    if (
                        source_crop_verdict == "incomplete"
                        and not (
                            candidate.get("native_edge_readable")
                        )
                    ):
                        candidate["quality_issue"] = "FACE_INCOMPLETE"
                    else:
                        native_issue = _native_face_core_issue(
                            native_image, candidate["box"], candidate["score"],
                        )
                        # A strong, fully in-frame YuNet face just below .90
                        # can have directional texture that trips the native
                        # blur heuristic without actually losing identity
                        # detail.  Keep the ordinary quality gate authoritative
                        # for these candidates; retain the core rejection for
                        # weaker secondary faces and genuinely small evidence.
                        if (
                            native_issue == "FACE_BLURRY"
                            and candidate["score"] >= .88
                            and not _is_face_box_cut_by_frame(
                                candidate["box"], native_image.shape[1], native_image.shape[0],
                            )
                        ):
                            native_issue = None
                        candidate["quality_issue"] = native_issue
                elif (
                    candidate["quality_issue"] in {"FACE_BLURRY", "FACE_OCCLUDED"}
                    and candidate["score"] >= .90
                    and source_crop_verdict != "incomplete"
                    and _native_face_core_issue(native_image, candidate["box"], candidate["score"]) is None
                ):
                    # Native dynamic input can make a clear face inherit a
                    # stricter eye-patch/blur label than the former canonical
                    # view.  Keep the existing source-frame crop guard, but
                    # trust a strong, otherwise detailed candidate when the
                    # native facial core has no independent failure.
                    candidate["quality_issue"] = None
                elif candidate["quality_issue"] == "FACE_BLURRY" and _has_readable_soft_face(native_image, candidate):
                    candidate["quality_issue"] = None
    recognizable = [c for c in candidates if c["quality_issue"] is None]
    face_count = len(recognizable)
    if face_count == 0:
        return None
    best = max(recognizable, key=lambda candidate: candidate["ratio"])
    result = {
        "valid": True,
        "message": get_message("FACE_DETECTION_PASSED", locale),
        "face": best["box"],
        # Keep the already accepted YuNet boxes as private generation evidence.
        # This does not alter acceptance, count, or public detection semantics.
        "usable_face_boxes": [candidate["box"] for candidate in recognizable],
        "img_size": (int(image_width), int(image_height)),
        "face_count": face_count,
        "face_score": float(best["score"]),
    }
    if (
        best.get("native_tile_recovery")
        or best.get("native_crop_recovery")
        or best.get("native_soft_recovery")
    ):
        result["native_recovery"] = True
    if best["ratio"] < MIN_FACE_RATIO_RELAXED:
        result["face_ratio"] = best["ratio"]
        result["face_small"] = True
    return result


def _find_secondary_border_fragment(image, accepted_face):
    """Find a second fragment only when the accepted face is also truncated.

    A normal face detector can recognize only one of two equally cropped edge
    faces depending on scan rotation.  Never let that asymmetry turn a group
    upload into a valid single-face upload.  This is intentionally limited
    to boxes that extend materially beyond the original frame, not ordinary
    portraits positioned close to an edge.
    """
    # A separately cropped face cannot invalidate a complete usable face.
    # Keep the ambiguity guard only when the accepted face is itself truncated.
    if accepted_face and not _is_face_box_cut_by_frame(
        accepted_face, image.shape[1], image.shape[0],
    ):
        return None
    resized, scale = image, 1.0
    height, width = resized.shape[:2]
    detector = _get_yunet_detector((width, height), 0.50)
    _retval, faces = detector.detect(resized)
    if faces is None:
        return None
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)) or float(face[14]) < YUNET_SCORE_THRESHOLD:
            continue
        x, y, face_width, face_height = map(float, face[:4])
        if min(face_width, face_height) < 80:
            continue
        if not (x <= -8 or x + face_width >= width + 8):
            continue
        box = _map_face_to_original(face, scale, image.shape[1], image.shape[0])
        if accepted_face and _box_iou(box, accepted_face) >= 0.20:
            continue
        return box
    return None


def _has_scene_camera_roll(image):
    """Strong structural edges can reveal camera roll when eye landmarks fail.

    Smooth texture first so fabric/finger detail cannot trigger this retry.
    This only enables the existing rotated YuNet checks; it never accepts a face.
    """
    scale = min(1.0, 640 / max(image.shape[:2]))
    view = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    gray = cv2.GaussianBlur(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), (7, 7), 2)
    edge = max(view.shape[:2])
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 100), 1, np.pi / 180, 65,
                            minLineLength=edge * .25, maxLineGap=12)
    if lines is None:
        return False
    total = tilted = longest_tilted = 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = (float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) + 45) % 90 - 45
        total += length
        if abs(angle) >= 15:
            tilted += length
            longest_tilted = max(longest_tilted, length)
    return longest_tilted >= edge * .4 and tilted >= total * .35


def _contains_human_yunet(
    image_path: str,
    locale: str | None = None,
    *,
    crop_modified: bool = False,
) -> dict:
    """Validate a storefront upload with YuNet at native dynamic resolution."""
    image = _read_image(image_path)
    if image is None:
        return {
            "valid": False,
            "code": "IMAGE_READ_ERROR",
            "message": get_message("IMAGE_READ_ERROR", locale),
        }

    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Complete face counting must happen before any side-profile rescue.  A
    # profile in a group photo must never short-circuit the shared
    # requirement.
    raw_candidates = _scan_yunet_native_orientations(
        image,
        allow_native_tiles=True,
    )
    raw_candidates = _recover_structured_low_confidence_faces(
        image, raw_candidates,
    )
    raw_candidates = _recover_repeated_small_face_candidates(
        image, raw_candidates,
    )
    raw_candidates = _recover_readable_dark_candidates(image, raw_candidates)
    candidates = _filter_pet_face_candidates(image, raw_candidates)
    if not (
        candidates and all(candidate["quality_issue"] == "FACE_OCCLUDED" for candidate in candidates)
        and not any(_has_significant_eye_line_roll(candidate.get("raw_face")) for candidate in candidates)
        and not _has_scene_camera_roll(image)
    ):
        before_roll = candidates
        candidates = _recover_roll_candidates(image, candidates)
        candidates = _retain_context_confirmed_primary(image, before_roll, candidates)
        candidates = _recover_context_confirmed_group_faces(image, candidates)
    confident_pet_image = _is_confident_pet_image(image)
    if confident_pet_image:
        candidates = [
            candidate for candidate in candidates
            if candidate.get("score", 0.0) >= .90
        ]
    candidates = _rescue_dominant_occluded_face(image, candidates)
    candidates = _drop_unconfirmed_weak_secondary_faces(candidates)
    soft_input_was_empty = not candidates
    candidates = _recover_native_soft_face(image, candidates)
    if crop_modified and soft_input_was_empty and len(candidates) > 1:
        candidates = _stabilize_crop_modified_soft_group(image, candidates)
    candidates = _recover_repeated_mild_profile_faces(
        image, candidates, confident_pet_image=confident_pet_image,
    )
    crop_recovery_needs_count_recheck = False
    if crop_modified and not confident_pet_image:
        # This is an optional crop-only recovery after the common detector has
        # already completed.  A local resize/DNN failure must not turn a
        # previously usable crop into FACE_DETECTION_FAILED.
        usable_before_crop_recovery = sum(
            candidate.get("quality_issue") is None for candidate in candidates
        )
        try:
            candidates = _append_crop_modified_normal_scaled_context_faces(
                image, candidates,
            )
        except Exception:
            pass
        else:
            # The ordinary one-face verifier can discover a complete group in
            # a corrected view.  If this optional crop-only append changes
            # that one candidate into a partial group, let that existing
            # strict verifier finish its add-only count check below.  Without
            # this marker, its normal one-face guard would exit early and the
            # append could hide a third face it did not itself recover.
            crop_recovery_needs_count_recheck = (
                usable_before_crop_recovery == 1
                and sum(
                    candidate.get("quality_issue") is None
                    for candidate in candidates
                ) > usable_before_crop_recovery
            )
    # The shared candidate scan is the only count source. Product mode is
    # applied later, after quality analysis, by contains_human().
    if raw_candidates and not candidates:
        return _non_human_face_result(locale)
    if candidates and all(c.get("diagnostic_only") for c in candidates):
        if _is_confident_pet_image(image):
            return _non_human_face_result(locale)
        # Unconfirmed roll texture cannot replace an established crop defect.
        # This refinement only explains zero usable faces; it cannot accept one.
        probe, probe_scale = image, 1.0
        crop_evidence = _diagnose_yunet_failure(
            probe, probe_scale, image_width, image_height, locale)
        if crop_evidence and not crop_evidence.get("valid") and crop_evidence.get("code") == "FACE_INCOMPLETE":
            return crop_evidence
        return {"valid": False, "code": "FACE_BLURRY", "message": get_message("FACE_BLURRY", locale)}
    result = _yunet_candidate_result(
        candidates, image_width, image_height, locale,
        image_path=image_path,
    )
    if candidates and all(candidate["quality_issue"] == "FACE_OCCLUDED" for candidate in candidates):
        # The primary native-orientation scan already found only eye-covered faces.
        # Do not let the fallback detector replace that evidence with a larger,
        # looser box and incorrectly accept the upload.
        if not any(
            candidate["score"] >= .60
            and min(candidate["box"][2], candidate["box"][3]) >= 100
            and candidate["ratio"] >= .01
            for candidate in candidates
        ):
            # A profile-only diagnostic is more specific than occlusion for
            # strongly turned faces.  The diagnostic path never accepts a
            # face; it only restores the established customer-facing reason.
            profile_diagnostic = _diagnose_yunet_failure(
                image,
                1.0,
                image_width,
                image_height,
                locale,
            )
            if profile_diagnostic and profile_diagnostic.get("code") == "FACE_NOT_FRONTAL":
                return profile_diagnostic
            return {"valid": False, "code": "NO_FACE", "message": get_message("NO_FACE", locale)}
        return {
            "valid": False,
            "code": "FACE_OCCLUDED",
            "message": get_message("FACE_OCCLUDED", locale),
        }
    if result is not None:
        if (
            crop_recovery_needs_count_recheck
            and result.get("valid")
            and result.get("face_count", 0) > 1
        ):
            # Private hand-off to the shared post-analysis verifier.  The
            # public result never exposes this implementation detail.
            result["_crop_modified_count_recheck"] = True
        if result.get("valid") and result.get("face_count") == 1:
            fragment = _find_secondary_border_fragment(image, result.get("face"))
            if fragment:
                return {
                    "valid": False,
                    "code": "FACE_INCOMPLETE",
                    "message": get_message("FACE_INCOMPLETE", locale),
                    "face": fragment,
                    "img_size": (int(image_width), int(image_height)),
                }
        return result

    detection_image, scale = image, 1.0
    detection_height, detection_width = detection_image.shape[:2]
    detection_area = detection_width * detection_height
    detector = _get_yunet_detector((detection_width, detection_height))
    _retval, detected = detector.detect(detection_image)
    detection_source = detection_image
    inverse_rotation = None
    quarter_turn = None

    # Exact native orientations failed. Retain the bounded small-angle fallback
    # for distant subjects photographed with camera roll.
    if detected is None or len(detected) == 0:
        center = (detection_width / 2.0, detection_height / 2.0)
        for angle in YUNET_ROTATION_FALLBACK_ANGLES:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                detection_image,
                rotation_matrix,
                (detection_width, detection_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            _retval, rotated_detected = detector.detect(rotated_image)
            if rotated_detected is not None and len(rotated_detected) > 0:
                detected = rotated_detected
                detection_source = rotated_image
                inverse_rotation = cv2.invertAffineTransform(rotation_matrix)
                quarter_turn = None
                break

    # A quarter-turned photo may still have the camera's original small roll.
    # Try the combination only after all cheap single transforms fail.
    if detected is None or len(detected) == 0:
        for quarter_angle in (90, 270, 180):
            quarter_image, quarter_matrix = _quarter_turn_with_matrix(
                detection_image,
                quarter_angle,
            )
            quarter_height, quarter_width = quarter_image.shape[:2]
            center = (quarter_width / 2.0, quarter_height / 2.0)
            for tilt_angle in YUNET_COMBINED_ROTATION_ANGLES:
                tilt_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)
                rotated_image = cv2.warpAffine(
                    quarter_image,
                    tilt_matrix,
                    (quarter_width, quarter_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                detector.setInputSize((quarter_width, quarter_height))
                _retval, rotated_detected = detector.detect(rotated_image)
                if rotated_detected is not None and len(rotated_detected) > 0:
                    detected = rotated_detected
                    detection_source = rotated_image
                    inverse_rotation = cv2.invertAffineTransform(
                        _compose_affine(tilt_matrix, quarter_matrix)
                    )
                    quarter_turn = None
                    break
            if detected is not None and len(detected) > 0:
                break

        detector.setInputSize((detection_width, detection_height))

    candidates = []
    if detected is not None:
        detection_gray = cv2.cvtColor(detection_source, cv2.COLOR_BGR2GRAY)
        for face in detected:
            if len(face) < 15 or not np.all(np.isfinite(face)):
                continue
            x, y, width, height = map(float, face[:4])
            score = float(face[14])
            face_ratio = (width * height) / float(max(detection_area, 1))
            if (
                score < YUNET_SCORE_THRESHOLD
                or min(width, height) < YUNET_MIN_FACE_SIDE
                or face_ratio < YUNET_MIN_FACE_RATIO
            ):
                continue

            detection_box = (
                max(0, int(round(x))),
                max(0, int(round(y))),
                max(1, int(round(width))),
                max(1, int(round(height))),
            )
            canonical_face = (
                _map_quarter_turn_box_to_unrotated(
                    face,
                    quarter_turn,
                    detection_width,
                    detection_height,
                )
                if quarter_turn is not None
                else _map_rotated_box_to_unrotated(
                    face,
                    inverse_rotation,
                    detection_width,
                    detection_height,
                )
                if inverse_rotation is not None
                else face
            )
            original_box = _map_face_to_original(
                canonical_face,
                scale,
                image_width,
                image_height,
            )
            quality_issue = _get_face_quality_issue(detection_gray, detection_box)
            if quality_issue is None and (
                (quarter_turn is not None or inverse_rotation is not None)
                and any(_frame_edges_touched(original_box, image_width, image_height)[1::2])
            ):
                # A rotated fallback can move an eye-only top/bottom crop away
                # from the working canvas boundary. Its mapped original box
                # still carries the crop evidence, so it must not become a
                # usable face through rotation alone.
                quality_issue = "FACE_INCOMPLETE"
            elif quality_issue is None and _is_unusable_edge_cropped_face(
                face, detection_box, detection_width, detection_height, score,
            ):
                quality_issue = "FACE_INCOMPLETE"
            elif quality_issue is None and score < 0.90 and not (
                any(_frame_edges_touched(detection_box, detection_width, detection_height))
                and not _is_unusable_edge_cropped_face(
                    face, detection_box, detection_width, detection_height, score,
                )
            ):
                # This is a last-resort fallback after every native orientation
                # failed.  Unlike the primary scan, it has no independent
                # clear-face confirmation, so a compressed landmark layout
                # must remain coverage evidence rather than becoming a pass.
                quality_issue = (
                    "FACE_OCCLUDED"
                    if _has_occlusion_suspect_geometry(face)
                    else _get_face_occlusion_issue(detection_gray, detection_box, face)
                )
            candidates.append(
                {
                    "box": original_box,
                    "ratio": (original_box[2] * original_box[3]) / float(max(image_area, 1)),
                    "score": score,
                    "quality_issue": quality_issue,
                }
            )

    raw_candidates = candidates
    candidates = _filter_pet_face_candidates(image, raw_candidates)
    if raw_candidates and not candidates:
        return _non_human_face_result(locale)
    result = _yunet_candidate_result(
        candidates, image_width, image_height, locale,
        image_path=image_path,
    )
    if result is not None:
        if result.get("valid") and result.get("face_count") == 1:
            fragment = _find_secondary_border_fragment(image, result.get("face"))
            if fragment:
                return {
                    "valid": False,
                    "code": "FACE_INCOMPLETE",
                    "message": get_message("FACE_INCOMPLETE", locale),
                    "face": fragment,
                    "img_size": (int(image_width), int(image_height)),
                }
        return result

    if not candidates:
        if _is_confident_pet_image(image):
            return _non_human_face_result(locale)
        tiny_diagnostic = _diagnose_tiny_yunet_face(image, locale)
        if tiny_diagnostic:
            return tiny_diagnostic
        diagnostic = _diagnose_yunet_failure(
            detection_image,
            scale,
            image_width,
            image_height,
            locale,
        )
        if diagnostic and diagnostic.get("code") == "FACE_UNRECOGNIZABLE":
            # A weak native landmark match below the diagnostic confidence
            # floor is not enough to claim that a face is present.  Keep the
            # specific unrecognizable explanation for a genuinely pixelated
            # portrait (handled by the compatibility hint below), but avoid
            # promoting ordinary scene texture into face evidence.
            native_signals = _native_weak_face_signals(image)
            if native_signals and .25 <= max(
                signal.get("score", 0.0) for signal in native_signals
            ) < .40:
                diagnostic = {
                    "valid": False,
                    "code": "NO_FACE",
                    "message": get_message("NO_FACE", locale),
                }
        if diagnostic and diagnostic.get("code") == "NO_FACE":
            # Preserve the established dark-photo reason when native YuNet
            # found a substantial dark face but the strict candidate path
            # discarded it as too weak/covered.
            native_signals = _native_weak_face_signals(image)
            dark_signal = any(
                signal.get("score", 0.0) >= .60
                and min(signal.get("box", (0, 0, 0, 0))[2:]) >= 120
                and signal.get("quality") == "FACE_TOO_DARK"
                for signal in native_signals
            )
            if dark_signal:
                diagnostic = {
                    "valid": False,
                    "code": "FACE_TOO_DARK",
                    "message": get_message("FACE_TOO_DARK", locale),
                }
        if diagnostic is None:
            # Use overlapping crops of the original pixels only for a stable
            # explanation of severely pixelated uploads. This is diagnostic
            # evidence, not an acceptance/counting path, and it must repeat
            # in at least two native tiles before restoring a specific reason.
            tile_diagnostics = []
            for _left, _top, native_tile in _native_tile_origins(image):
                tile_diagnostic = _diagnose_yunet_failure(
                    native_tile,
                    1.0,
                    native_tile.shape[1],
                    native_tile.shape[0],
                    locale,
                )
                if tile_diagnostic and tile_diagnostic.get("code") == "FACE_UNRECOGNIZABLE":
                    tile_diagnostics.append(tile_diagnostic)
            if len(tile_diagnostics) >= 2:
                diagnostic = tile_diagnostics[0]
        # No legacy 1280px diagnostic fallback: the dynamic-input YuNet
        # result and all quality diagnostics use the uploaded source pixels.
        if diagnostic:
            if diagnostic.get("code") == "FACE_OCCLUDED":
                # This branch has no native-orientation candidate. A weak
                # diagnostic landmark match can explain that a head-like
                # region failed, but cannot reliably make a specific
                # occlusion claim; retain the established generic outcome.
                diagnostic = {
                    "valid": False,
                    "code": (
                        "FACE_TOO_DARK"
                        if float(np.percentile(image_gray, 90)) < 80
                        else "NO_FACE"
                    ),
                    "message": get_message(
                        "FACE_TOO_DARK" if float(np.percentile(image_gray, 90)) < 80 else "NO_FACE",
                        locale,
                    ),
                }
            return diagnostic
        quality_diagnostic = _diagnose_no_face_quality_issue(
            image_path,
            image,
            locale,
        )
        if quality_diagnostic:
            return quality_diagnostic

    if candidates:
        best = max(candidates, key=lambda candidate: (candidate["score"], candidate["ratio"]))
        issue_code = best["quality_issue"] or "FACE_UNRECOGNIZABLE"
        return {
            "valid": False,
            "code": issue_code,
            "message": get_message(issue_code, locale),
        }

    if float(np.percentile(image_gray, 90)) < 80:
        return {
            "valid": False,
            "code": "FACE_TOO_DARK",
            "message": get_message("FACE_TOO_DARK", locale),
        }
    return {
        "valid": False,
        "code": "NO_FACE",
        "message": get_message("NO_FACE", locale),
    }


def _box_iou(first, second) -> float:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = first_w * first_h
    second_area = second_w * second_h
    return intersection / float(max(first_area + second_area - intersection, 1))


def _box_overlap_over_smaller(first, second) -> float:
    """Measure containment so differently-sized detections can be deduplicated."""
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0, right - left) * max(0, bottom - top)
    smaller_area = min(first_w * first_h, second_w * second_h)
    return intersection / float(max(smaller_area, 1))


def _map_rotated_box(box, inverse_matrix, image_width: int, image_height: int):
    """Map a detection on a rotated canvas back to original-image coordinates."""
    x, y, w, h = map(float, box)
    points = np.array(
        [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
        dtype=np.float32,
    )
    mapped = cv2.transform(points, inverse_matrix)[0]
    left = max(0, int(np.floor(mapped[:, 0].min())))
    top = max(0, int(np.floor(mapped[:, 1].min())))
    right = min(image_width, int(np.ceil(mapped[:, 0].max())))
    bottom = min(image_height, int(np.ceil(mapped[:, 1].max())))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top




def _has_readable_dark_detail(gray_image, face_box):
    """Low exposure alone is not identity loss; require source contrast/detail.

    This only relaxes lighting for an already localized face. It neither finds
    a face nor accepts unconfirmed candidates or brightness-amplified texture.
    """
    x, y, w, h = map(int, face_box)
    roi = gray_image[max(0, y):min(gray_image.shape[0], y+h), max(0, x):min(gray_image.shape[1], x+w)]
    # At very low exposure, a small blurred background face can have the same
    # contrast/edge energy as a readable larger face. Keep the exposure-only
    # exception for substantial source faces; normal-light small faces retain
    # their independent detail/crop-confirmation path.
    if roi.size == 0 or min(roi.shape) < 120:
        return False
    low, median, high = np.percentile(roi, (10, 50, 90))
    if not (12 <= median and 24 <= high < 45 and high-low >= 18):
        return False
    normalized = cv2.resize(roi, (160, 160), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(normalized, cv2.CV_64F).var()) >= 3.0


def _recover_readable_dark_candidates(image, candidates):
    """Corroborate visible underexposed faces, without modifying the upload."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray, 90) >= 80:
        return candidates
    height, width = gray.shape
    seeds = []
    for angle in (0, 90, 180, 270):
        view = image if not angle else _quarter_turn_with_matrix(image, angle)[0]
        _, detected = _get_yunet_detector((view.shape[1], view.shape[0]), .65).detect(view)
        for face in (() if detected is None else detected):
            box = _map_face_to_original(face if not angle else _map_quarter_turn_box_to_unrotated(face, angle, width, height), 1, width, height)
            if _has_readable_dark_detail(gray, box):
                seeds.append(box)
    if not seeds:
        return candidates
    recovered = list(candidates)
    for gain in (4.0, 8.0):
        enhanced = np.clip(image.astype(np.float32)*gain, 0, 255).astype(np.uint8)
        enhanced_candidates=[]
        for angle in (0,90,180,270):
            view=enhanced if not angle else _quarter_turn_with_matrix(enhanced,angle)[0]
            _, faces=_get_yunet_detector((view.shape[1],view.shape[0]),.82).detect(view)
            for face in (() if faces is None else faces):
                local_box=_map_face_to_original(face,1,view.shape[1],view.shape[0])
                if _is_unusable_edge_cropped_face(face,local_box,view.shape[1],view.shape[0],float(face[14])):
                    continue
                box=_map_face_to_original(face if not angle else _map_quarter_turn_box_to_unrotated(face,angle,width,height),1,width,height)
                enhanced_candidates.append({"box":box,"ratio":box[2]*box[3]/float(width*height),"score":float(face[14]),"quality_issue":None,"raw_face":face})
        for c in enhanced_candidates:
            if not any(_box_iou(c["box"], box) >= .35 for box in seeds):
                continue
            if not _has_readable_dark_detail(gray, c["box"]):
                continue
            matches = [i for i, old in enumerate(recovered) if _box_iou(old["box"], c["box"]) >= .25]
            candidate = {**c, "quality_issue": None, "native_dark_recovery": True, "native_edge_readable": True}
            if matches:
                i = matches[0]
                if recovered[i]["quality_issue"] is not None:
                    recovered[i] = candidate
            else:
                recovered.append(candidate)
    return recovered


def _recover_repeated_small_face_candidates(image, candidates):
    """Recover a small face only when several native context views agree.

    The browser's 2000px export can push a legitimate, already-localized
    thumbnail face below the ordinary YuNet acceptance score.  This is not a
    lower global threshold: the same source-pixel face must recur in three
    differently padded native crops, clear the normal quality/occlusion gates,
    and remain fully inside the real image frame.  Crops change only detector
    context and never synthesize detail or add a second detector.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = _get_yunet_detector((width, height), YUNET_CANDIDATE_SCORE_THRESHOLD)
    _retval, detected = detector.detect(image)
    if detected is None:
        return candidates

    recovered = list(candidates)
    for face in detected:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        x, y, face_width, face_height = map(float, face[:4])
        # This path is intentionally limited to a sub-threshold, but still
        # meaningful, compact face.  Larger weak faces retain the existing
        # quality and roll-recovery paths; smaller boxes cannot expose enough
        # source pixels after master-image export.
        if not (
            .75 <= score < YUNET_SCORE_THRESHOLD
            and 30 <= min(face_width, face_height) <= 45
            and max(face_width, face_height) <= 60
        ):
            continue
        box = _map_face_to_original(face, 1.0, width, height)
        if any(_frame_edges_touched(box, width, height)):
            continue
        if any(
            _box_iou(box, candidate.get("box", (0, 0, 0, 0))) >= .25
            or _box_overlap_over_smaller(box, candidate.get("box", (0, 0, 0, 0))) >= .55
            for candidate in recovered
        ):
            continue
        if (
            _get_face_quality_issue(gray, box) is not None
            or _get_face_occlusion_issue(gray, box, face) is not None
            or _native_face_core_issue(image, box, score) is not None
        ):
            continue

        confirmations = []
        for padding in (.5, 1.0, 1.5):
            left = max(0, int(x - padding * face_width))
            top = max(0, int(y - padding * face_height))
            right = min(width, int(x + (1 + padding) * face_width))
            bottom = min(height, int(y + (1 + padding) * face_height))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            crop_detector = _get_yunet_detector(
                (crop.shape[1], crop.shape[0]), YUNET_CANDIDATE_SCORE_THRESHOLD,
            )
            _crop_retval, crop_faces = crop_detector.detect(crop)
            matches = []
            for crop_face in (() if crop_faces is None else crop_faces):
                if len(crop_face) < 15 or not np.all(np.isfinite(crop_face)):
                    continue
                crop_score = float(crop_face[14])
                if crop_score < .70:
                    continue
                local_box = _map_face_to_original(
                    crop_face, 1.0, crop.shape[1], crop.shape[0],
                )
                mapped_box = (
                    local_box[0] + left,
                    local_box[1] + top,
                    local_box[2],
                    local_box[3],
                )
                if _box_iou(mapped_box, box) < .30:
                    continue
                if _get_face_quality_issue(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), local_box,
                ) is not None:
                    continue
                if _get_face_occlusion_issue(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), local_box, crop_face,
                ) is not None:
                    continue
                matches.append((crop_score, crop_face, mapped_box))
            if matches:
                confirmations.append(max(matches, key=lambda item: item[0]))

        # Three independent context windows all have to point back to the
        # same small source region.  A hand, texture or UI glyph does not get
        # promoted merely by reappearing in one enlarged local crop.
        if len(confirmations) < 3:
            continue
        best_score, best_face, best_box = max(confirmations, key=lambda item: item[0])
        recovered.append({
            "box": best_box,
            "ratio": best_box[2] * best_box[3] / float(max(width * height, 1)),
            "score": best_score,
            "quality_issue": None,
            "raw_face": best_face,
            "native_repeated_small_recovery": True,
        })
    return recovered


def _recover_repeated_mild_profile_faces(
    image, candidates, *, confident_pet_image=False,
):
    """Complete a clear two-face group with a corroborated compact profile.

    This deliberately runs after the roll pass.  Adding a weak third face
    earlier can make the roll pass treat a still-incomplete group as settled
    and lose a different, roll-dependent member.  It is not a lower global
    YuNet threshold: a near-threshold upright candidate has to recur at the
    same source location in three differently padded native-pixel contexts,
    with two strong local detections and repeated profile geometry.
    """
    if confident_pet_image:
        return candidates
    usable = [
        candidate for candidate in candidates
        if candidate.get("quality_issue") is None
    ]
    # Keep this final completion path narrower than normal group detection.
    # It only resolves a compact mild profile beside two independently clear
    # people; it never creates a second face for an otherwise single portrait.
    if len(usable) != 2 or sum(
        candidate.get("score", 0.0) >= .90 for candidate in usable
    ) != 2:
        return candidates

    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = _get_yunet_detector(
        (width, height), YUNET_CANDIDATE_SCORE_THRESHOLD,
    )
    _retval, detected = detector.detect(image)
    if detected is None:
        return candidates

    for face in detected:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        x, y, face_width, face_height = map(float, face[:4])
        if not (
            YUNET_SCORE_THRESHOLD - .02 <= score < YUNET_SCORE_THRESHOLD
            and 35 <= min(face_width, face_height) <= 45
            and max(face_width, face_height) <= 60
            and _has_mild_yunet_profile_geometry(face)
        ):
            continue
        box = _map_face_to_original(face, 1.0, width, height)
        ratio = box[2] * box[3] / float(max(width * height, 1))
        if min(box[2:]) < YUNET_MIN_FACE_SIDE or ratio < YUNET_MIN_FACE_RATIO:
            continue
        if any(_frame_edges_touched(box, width, height)):
            continue
        if any(
            _box_iou(box, candidate.get("box", (0, 0, 0, 0))) >= .25
            or _box_overlap_over_smaller(
                box, candidate.get("box", (0, 0, 0, 0)),
            ) >= .55
            for candidate in candidates
        ):
            continue
        if (
            _get_face_quality_issue(gray_image, box) is not None
            or _get_face_occlusion_issue(gray_image, box, face) is not None
            or _native_face_core_issue(image, box, score) is not None
        ):
            continue

        confirmations = []
        for padding in (.5, 1.0, 1.5):
            left = max(0, int(x - padding * face_width))
            top = max(0, int(y - padding * face_height))
            right = min(width, int(x + (1 + padding) * face_width))
            bottom = min(height, int(y + (1 + padding) * face_height))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_detector = _get_yunet_detector(
                (crop.shape[1], crop.shape[0]),
                YUNET_CANDIDATE_SCORE_THRESHOLD,
            )
            _crop_retval, crop_faces = crop_detector.detect(crop)
            matches = []
            for crop_face in (() if crop_faces is None else crop_faces):
                if len(crop_face) < 15 or not np.all(np.isfinite(crop_face)):
                    continue
                crop_score = float(crop_face[14])
                if crop_score < .65:
                    continue
                local_box = _map_face_to_original(
                    crop_face, 1.0, crop.shape[1], crop.shape[0],
                )
                mapped_box = (
                    local_box[0] + left,
                    local_box[1] + top,
                    local_box[2],
                    local_box[3],
                )
                if _box_iou(mapped_box, box) < .30:
                    continue
                if (
                    _get_face_quality_issue(crop_gray, local_box) is not None
                    or _get_face_occlusion_issue(
                        crop_gray, local_box, crop_face,
                    ) is not None
                ):
                    continue
                matches.append((crop_score, crop_face, mapped_box))
            if matches:
                confirmations.append(max(matches, key=lambda item: item[0]))

        # All three context windows must corroborate the source location; two
        # of them must clear the normal production floor and two must retain
        # profile geometry.  This rejects repeated texture/hand false
        # positives without widening the ordinary candidate threshold.
        if (
            len(confirmations) != 3
            or sum(item[0] >= YUNET_SCORE_THRESHOLD for item in confirmations) < 2
            or sum(_is_yunet_profile(item[1]) for item in confirmations) < 2
        ):
            continue
        if any(
            _box_iou(first[2], second[2]) < .45
            and _box_overlap_over_smaller(first[2], second[2]) < .70
            for index, first in enumerate(confirmations)
            for second in confirmations[index + 1:]
        ):
            continue
        _best_score, _best_face, best_box = max(
            confirmations, key=lambda item: item[0],
        )
        best_ratio = best_box[2] * best_box[3] / float(max(width * height, 1))
        if (
            min(best_box[2:]) < YUNET_MIN_FACE_SIDE
            or best_ratio < YUNET_MIN_FACE_RATIO
        ):
            continue
        recovered = {
            # The local windows only corroborate the source candidate. Keep
            # the original-image box, score and landmarks so all ordinary
            # crop/quality guards still assess the real upload coordinates.
            "box": box,
            "ratio": ratio,
            "score": score,
            "quality_issue": None,
            "raw_face": face,
            "native_repeated_mild_profile_recovery": True,
        }
        combined = _filter_pet_face_candidates(image, [*candidates, recovered])
        if any(candidate is recovered for candidate in combined):
            return [*candidates, recovered]
    return candidates


def _get_face_quality_issue(gray_image, face_box):
    """Return a specific, conservative quality error for a detected face."""
    x, y, w, h = map(int, face_box)
    face_roi = gray_image[y:y + h, x:x + w]
    if face_roi.size == 0:
        return "FACE_INCOMPLETE"

    normalized = cv2.resize(
        face_roi,
        (NORMALIZED_FACE_QUALITY_SIZE, NORMALIZED_FACE_QUALITY_SIZE),
        interpolation=(
            cv2.INTER_AREA
            if min(face_roi.shape[:2]) > NORMALIZED_FACE_QUALITY_SIZE
            else cv2.INTER_CUBIC
        ),
    )
    low, high = np.percentile(normalized, (10, 90))
    if _has_readable_dark_detail(gray_image, face_box):
        normalized = np.clip(normalized.astype(np.float32)*4, 0, 255).astype(np.uint8)
    if high < 45 and not _has_readable_dark_detail(gray_image, face_box):
        return "FACE_TOO_DARK"
    if low > 240:
        return "FACE_OVEREXPOSED"
    # Small, resized faces naturally have a lower Laplacian variance than a
    # large face from the same photo. Keep the strict blur floor for faces
    # with enough native pixels, but use a source-size floor for small faces
    # that already passed the visible-eye test.
    sharpness_floor = MIN_FACE_SHARPNESS if min(w, h) >= 100 else MIN_FACE_SHARPNESS * 0.30
    if float(cv2.Laplacian(normalized, cv2.CV_64F).var()) < sharpness_floor:
        return "FACE_BLURRY"
    return None


def _is_extreme_motion_blur(image_path: str, face_box) -> bool:
    """Path adapter for the existing native motion-detail check."""
    if not face_box or min(int(face_box[2]), int(face_box[3])) < 250:
        return False
    return _native_extreme_motion_blur(_read_image(image_path), face_box)


def _native_extreme_motion_blur(image, face_box) -> bool:
    """Detect a large, horizontally smeared face that YuNet can still box.

    A detector score alone is not evidence that facial features are usable.
    In particular, strong horizontal camera motion can retain the head outline
    while removing every readable eye, nose and mouth detail.  Keep this
    deliberately narrow: normal soft portraits and readable side profiles
    retain an eye-cascade match and do not enter this branch.
    """
    if not face_box or min(int(face_box[2]), int(face_box[3])) < 250:
        return False
    if image is None:
        return False
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if _has_readable_dark_detail(gray_image, face_box):
        gray_image = np.clip(gray_image.astype(np.float32)*4,0,255).astype(np.uint8)
    x, y, width, height = map(int, face_box)
    # Examine the central facial area rather than hair, shoulders, or the
    # image background; those sharp edges otherwise conceal motion blur.
    left = max(0, x + int(width * 0.12))
    right = min(gray_image.shape[1], x + int(width * 0.88))
    top = max(0, y + int(height * 0.18))
    bottom = min(gray_image.shape[0], y + int(height * 0.70))
    face_core = gray_image[top:bottom, left:right]
    if face_core.size == 0:
        return False
    normalized = cv2.resize(face_core, (160, 160), interpolation=cv2.INTER_AREA)
    horizontal_detail = float(cv2.Sobel(normalized, cv2.CV_64F, 1, 0).var())
    vertical_detail = float(cv2.Sobel(normalized, cv2.CV_64F, 0, 1).var())
    return horizontal_detail < 450 and vertical_detail < 1000


def _has_diagnostic_directional_motion_blur(image, face_box) -> bool:
    """Recognize strong one-axis motion blur in a diagnostic-only face hint.

    This never accepts a face. It runs only after normal YuNet detection
    failed, and requires a sizable low-confidence, landmark-consistent hint.
    The directional check catches a horizontally smeared portrait whose
    vertical contours remain visible, a case the ordinary low-detail floor
    intentionally does not classify from weak evidence.
    """
    if not face_box or min(int(face_box[2]), int(face_box[3])) < 160:
        return False
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, width, height = map(int, face_box)
    core = gray_image[
        max(0, y + int(height * .18)):min(gray_image.shape[0], y + int(height * .70)),
        max(0, x + int(width * .12)):min(gray_image.shape[1], x + int(width * .88)),
    ]
    if core.size == 0:
        return False
    normalized = cv2.resize(core, (160, 160), interpolation=cv2.INTER_AREA)
    horizontal_detail = float(cv2.Sobel(normalized, cv2.CV_64F, 1, 0).var())
    vertical_detail = float(cv2.Sobel(normalized, cv2.CV_64F, 0, 1).var())
    return horizontal_detail < 1500 and vertical_detail > max(horizontal_detail * 2.5, 1000)


def _get_face_occlusion_issue(gray_image, face_box, yunet_face=None):
    """Use YuNet eye landmarks, not a second face detector, for occlusion.

    The detector supplies the two eye locations.  We only reject a large face
    when those landmarks are absent or geometrically impossible; this keeps
    readable side faces valid and avoids Haar's texture false positives.
    """
    x, y, width, height = map(int, face_box)
    if _has_readable_dark_detail(gray_image, face_box):
        gray_image = np.clip(gray_image.astype(np.float32)*4, 0, 255).astype(np.uint8)
    if yunet_face is not None and 35 <= min(width,height) < 80 and float(yunet_face[14]) < .88 and not _is_yunet_profile(yunet_face):
        points=np.asarray(yunet_face[4:14]).reshape(5,2)
        eye_span=float(np.linalg.norm(points[1]-points[0]))/max(width,1)
        mouth_span=float(np.linalg.norm(points[4]-points[3]))/max(width,1)
        if eye_span < .18 and mouth_span < .15:
            return "FACE_OCCLUDED"
        contrasts=[]
        radius=max(2,round(min(width,height)*.10))
        for px,py in points:
            px,py=round(float(px)),round(float(py))
            patch=gray_image[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
            contrasts.append(float(np.percentile(patch,90)-np.percentile(patch,10)) if patch.size else 0)
        if .30 <= eye_span <= .65 and min(contrasts[:2]) < 18 and min(contrasts[3:]) < 10:
            return "FACE_OCCLUDED"
    if min(width, height) < 80:
        return None
    if yunet_face is None or len(yunet_face) < 15:
        return "FACE_OCCLUDED"
    eyes = np.asarray(yunet_face[4:8], dtype=np.float32).reshape(2, 2)
    if not np.all(np.isfinite(eyes)):
        return "NO_FACE"
    normalized = (eyes - np.array((x, y), dtype=np.float32)) / np.array(
        (max(width, 1), max(height, 1)), dtype=np.float32
    )
    eye_distance = abs(float(normalized[1, 0] - normalized[0, 0]))
    eye_height_difference = abs(float(normalized[1, 1] - normalized[0, 1]))
    eyes_in_upper_face = bool(
        np.all(normalized[:, 0] >= 0.03)
        and np.all(normalized[:, 0] <= 0.97)
        and np.all(normalized[:, 1] >= 0.05)
        and np.all(normalized[:, 1] <= 0.70)
    )
    if not eyes_in_upper_face or not (0.10 <= eye_distance <= 0.82) or eye_height_difference > 0.35:
        # A fabricated landmark layout on hair, an ear or the back of a head
        # is not evidence of an occluded face.
        return "NO_FACE"
    # YuNet may interpolate landmarks through hands, masks or motion blur.
    # Inspect only tiny patches centred on those YuNet landmarks: visible eyes
    # normally retain dark iris/lash pixels, while a hand-covered eye region
    # is uniformly skin-toned; strong motion blur loses local edge detail.
    patch_radius = max(8, int(round(min(width, height) * 0.14)))
    eye_metrics = []
    for eye_x, eye_y in eyes:
        left = max(0, int(round(eye_x)) - patch_radius)
        right = min(gray_image.shape[1], int(round(eye_x)) + patch_radius)
        top = max(0, int(round(eye_y)) - patch_radius)
        bottom = min(gray_image.shape[0], int(round(eye_y)) + patch_radius)
        patch = gray_image[top:bottom, left:right]
        if patch.size == 0:
            return "FACE_OCCLUDED"
        eye_metrics.append((
            float(np.percentile(patch, 5)),
            float(np.percentile(patch, 25)),
            float(cv2.Laplacian(patch, cv2.CV_64F).var()),
        ))
    if all(detail < 12.0 for _darkest, _lower_quartile, detail in eye_metrics):
        return "FACE_BLURRY"
    darkest_values = [darkest for darkest, _lower_quartile, _detail in eye_metrics]
    if max(darkest_values) > 90.0 and min(darkest_values) > 60.0:
        return "FACE_OCCLUDED"
    if all(lower_quartile > 105.0 for _darkest, lower_quartile, _detail in eye_metrics):
        return "FACE_OCCLUDED"
    return None


def _is_face_box_cut_by_frame(face_box, image_width: int, image_height: int) -> bool:
    """Return whether a face is severely truncated at two or more boundaries.

    A readable portrait can be intentionally cropped along one image edge.
    Treating every one-edge crop as incomplete rejects otherwise usable
    references, while two touching edges are strong evidence of a fragment.
    """
    x, y, width, height = map(int, face_box)
    margin_x = max(8, int(round(image_width * 0.01)))
    margin_y = max(8, int(round(image_height * 0.01)))
    touched_edges = sum(
        (
            x <= margin_x,
            y <= margin_y,
            x + width >= image_width - margin_x,
            y + height >= image_height - margin_y,
        )
    )
    # A narrow strip at one edge is not a usable portrait even when it only
    # reaches a single boundary.  Keep moderate side crops valid, but reject
    # a fragment that retains less than roughly two-fifths of a face's normal
    # width-to-height extent.
    visible_aspect_ratio = min(width, height) / float(max(width, height, 1))
    return touched_edges >= 2 or (
        touched_edges >= 1 and visible_aspect_ratio < 0.48
    )


def _frame_edges_touched(face_box, image_width: int, image_height: int) -> tuple[bool, bool, bool, bool]:
    """Return left, top, right and bottom contact for a YuNet face box."""
    x, y, width, height = map(float, face_box[:4])
    margin_x = max(8.0, image_width * .01)
    margin_y = max(8.0, image_height * .01)
    return (
        x <= margin_x,
        y <= margin_y,
        x + width >= image_width - margin_x,
        y + height >= image_height - margin_y,
    )


def _has_complete_in_frame_edge_profile(face, face_box, image_width, image_height, score):
    """Distinguish an intact side face near the frame from a real side crop.

    The ordinary edge guard deliberately uses a safety band. A complete
    profile inside that band has a compressed eye span, which is not missing
    facial evidence. Only strictly interior source boxes and landmarks may
    use this exception; genuinely truncated boxes retain the existing rules.
    """
    if len(face) < 15 or not np.all(np.isfinite(face[:15])):
        return False
    left, top, right, bottom = _frame_edges_touched(face_box, image_width, image_height)
    if top or bottom or left == right or score < YUNET_SCORE_THRESHOLD:
        return False
    x, y, width, height = map(float, face[:4])
    if (min(width, height) < YUNET_MIN_FACE_SIDE or x < 0 or y < 0
            or x + width > image_width or y + height > image_height
            or not _is_yunet_profile(face)):
        return False
    points = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    margin = max(3.0, width * .05)
    if not (np.all(points[:, 0] >= margin) and np.all(points[:, 0] <= image_width - margin)
            and np.all(points[:, 1] >= 3) and np.all(points[:, 1] <= image_height - 3)):
        return False
    right_eye, left_eye, nose, mouth_left, mouth_right = points
    eye_span = float(np.linalg.norm(left_eye - right_eye)) / width
    mouth_span = float(np.linalg.norm(mouth_right - mouth_left)) / width
    return bool(
        .12 <= eye_span < .35 and .10 <= mouth_span <= .55
        and abs(float(left_eye[1] - right_eye[1])) <= height * .15
        and nose[1] - (left_eye[1] + right_eye[1]) / 2 >= height * .08
        and (mouth_left[1] + mouth_right[1]) / 2 - nose[1] >= height * .08
        and (not left or max(float(mouth_left[0]), float(mouth_right[0])) >= width * .32)
        and (not right or min(float(mouth_left[0]), float(mouth_right[0])) <= image_width - width * .32)
    )


def _is_unusable_edge_cropped_face(face, face_box, image_width: int, image_height: int, score: float) -> bool:
    """Judge a frame crop from visible facial landmarks, not box contact alone.

    YuNet's box often extends past an image edge for an otherwise readable
    side portrait.  A box touching the frame therefore is only a cue to check
    its landmarks.  Side crops remain usable when one eye, the nose and a
    mouth corner are still supported by a confident detection.  Vertical
    crops need both eyes and a nose tip inside the picture; an eye-only strip
    cannot establish a recognisable face.  This is deliberately a crop rule:
    blur, lighting and occlusion stay with their existing quality checks.
    """
    left, top, right, bottom = _frame_edges_touched(
        face_box, image_width, image_height,
    )
    if not any((left, top, right, bottom)):
        return False
    if len(face) < 15 or not np.all(np.isfinite(face[:15])):
        return True

    _x, _y, box_width, box_height = map(float, face[:4])
    if box_width <= 0 or box_height <= 0 or score < .68:
        return True
    landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    right_eye, left_eye, nose, mouth_left, mouth_right = landmarks

    def inside(point, *, x_tolerance=0.0, y_tolerance=0.0):
        return (
            -x_tolerance <= float(point[0]) <= image_width + x_tolerance
            and -y_tolerance <= float(point[1]) <= image_height + y_tolerance
        )

    # A side crop may also cut off the chin at the lower edge. In that case,
    # one clear eye, the nose and the mouth are sufficient; requiring two eyes
    # solely because the chin touches the frame would reject a readable half
    # portrait. Keep the same stricter mouth-margin rule for every side crop.
    horizontal_tolerance = box_width * .30
    visible_eyes = sum(inside(eye, x_tolerance=horizontal_tolerance) for eye in (right_eye, left_eye))
    mouth_inner_margin = box_width * .32
    visible_mouth = any(
        inside(mouth, x_tolerance=horizontal_tolerance)
        and (not left or float(mouth[0]) >= mouth_inner_margin)
        and (not right or float(mouth[0]) <= image_width - mouth_inner_margin)
        for mouth in (mouth_left, mouth_right)
    )
    eye_span = float(np.linalg.norm(left_eye-right_eye)) / box_width
    horizontal_readable = (
        visible_eyes >= 1
        and eye_span >= .35
        and inside(nose, x_tolerance=horizontal_tolerance)
        and visible_mouth
    )
    # A diagonal corner crop can retain most of the face while the detector's
    # estimated nose/mouth extends slightly outside the frame. Both eye sites
    # and at least 75% of the inferred face must remain supported by the image.
    if (bottom and (left or right) and score >= .85 and min(box_width, box_height) >= 100):
        visible_fraction = face_box[2]*face_box[3] / max(box_width*box_height, 1)
        eyes_present = all(inside(eye, x_tolerance=box_width*.05) for eye in (right_eye, left_eye))
        if (visible_fraction >= .75 and eyes_present and eye_span >= .30
                and inside(nose, y_tolerance=box_height*.10)
                and any(inside(mouth, y_tolerance=box_height*.10) for mouth in (mouth_left,mouth_right))):
            return False

    # A top/bottom crop overrides the side exception. Both eyes and the nose
    # tip must genuinely remain in frame, with a normal eye-to-nose distance.
    # For a bottom crop, leave enough pixels below the nose that the model did
    # not merely extrapolate the tip at the frame boundary. This rejects an
    # eye-and-bridge strip while retaining a full nose when the mouth is cut.
    if top or bottom:
        interior_y = max(3.0, box_height * .02)
        if not (
            inside(right_eye, y_tolerance=-interior_y)
            and inside(left_eye, y_tolerance=-interior_y)
            and inside(nose, y_tolerance=-interior_y)
        ):
            return not horizontal_readable if (left or right) else True
        eye_mid_y = float((right_eye[1] + left_eye[1]) / 2.0)
        # A top crop commonly removes only hair/forehead.  The eyes, nose and
        # mouth can still be fully readable even when the inferred box starts
        # just inside the top border; use the normal depth floor for bottom
        # crops and a slightly more tolerant one for this top-only case.
        nose_depth_floor = .10 if top and not bottom else .15
        nose_has_depth = float(nose[1] - eye_mid_y) >= box_height * nose_depth_floor
        nose_has_bottom_margin = (
            not bottom or float(image_height - nose[1]) >= box_height * .08
        )
        vertical_readable = nose_has_depth and nose_has_bottom_margin
        if left or right:
            return not (vertical_readable or horizontal_readable)
        return not vertical_readable

    # For a horizontal crop one landmark can fall just beyond the inferred
    # detector box.  Keep a small tolerance only on the cropped axis; this
    # covers an actual nose at the edge without turning an eye-only sliver
    # into a recognisable person.
    # The inferred box can extend roughly a third of a face-width past a real
    # side crop. Keep that tolerance bounded by the .68 confidence floor
    # above; weaker eye-only fragments still remain incomplete.
    return not (horizontal_readable or _has_complete_in_frame_edge_profile(
        face, face_box, image_width, image_height, score,
    ))


def _source_edge_crop_verdict(image, candidate_box) -> str | None:
    """Confirm a source-edge crop after a rotated/native candidate passes.

    Rotation can move a source edge away from the detector canvas. Re-read a
    matching unrotated face and use it only when its own source box reaches the
    frame: reject a proven incomplete crop, and identify a readable bottom crop
    whose missing mouth/chin must not be called occlusion. A missing or weak
    confirmation leaves the established candidate untouched.
    """
    image_height, image_width = image.shape[:2]
    view, scale = image, 1.0
    view_height, view_width = view.shape[:2]
    work = _request_work.get()
    cache = work.setdefault("source_edge_detections", {}) if work is not None else None
    cache_key = (id(image), image.shape, image.strides)
    detections = cache.get(cache_key) if cache is not None else None
    if detections is None:
        try:
            _retval, faces = _get_yunet_detector(
                (view_width, view_height), YUNET_DIAGNOSTIC_SCORE_THRESHOLD,
            ).detect(view)
        except Exception:
            return None
        detections = []
        for face in (() if faces is None else faces):
            if len(face) < 15 or not np.all(np.isfinite(face)):
                continue
            score = float(face[14])
            if score < .50:
                continue
            x, y, width, height = map(float, face[:4])
            if min(width, height) < YUNET_MIN_FACE_SIDE:
                continue
            source_box = _map_face_to_original(face, scale, image_width, image_height)
            edges = _frame_edges_touched(source_box, image_width, image_height)
            if not any(edges):
                continue
            detection_box = (
                max(0, int(round(x))), max(0, int(round(y))),
                max(1, int(round(width))), max(1, int(round(height))),
            )
            detections.append((score, np.asarray(face).copy(), source_box, detection_box, edges))
        if cache is not None:
            cache[cache_key] = detections
    matches = [
        (score, face, detection_box, edges)
        for score, face, source_box, detection_box, edges in detections
        if _box_iou(source_box, candidate_box) >= .25
    ]
    if not matches:
        return None
    score, face, detection_box, edges = max(matches, key=lambda item: item[0])
    if _is_unusable_edge_cropped_face(
        # This source pass only corroborates an existing high-confidence
        # candidate. At this point .50 is sufficient landmark evidence; keep
        # the ordinary .68 candidate-acceptance floor unchanged elsewhere.
        face, detection_box, view_width, view_height, max(score, .68),
    ):
        return "incomplete"
    return "readable_bottom_crop" if edges[3] else "readable"


def _has_two_in_frame_eye_landmarks(face, image_width: int, image_height: int) -> bool:
    """Return whether both YuNet eye landmarks are actually inside the canvas."""
    if len(face) < 15:
        return False
    eyes = np.asarray(face[4:8], dtype=np.float32).reshape(2, 2)
    return bool(
        np.all(np.isfinite(eyes))
        and np.all(eyes[:, 0] >= 0)
        and np.all(eyes[:, 0] <= image_width)
        and np.all(eyes[:, 1] >= 0)
        and np.all(eyes[:, 1] <= image_height)
    )


def _is_yunet_profile(face) -> bool:
    """Identify a strong side profile from YuNet's nose landmark placement."""
    if len(face) < 15:
        return False
    x, _y, width, _height = map(float, face[:4])
    if width <= 0:
        return False
    nose_x = float(face[8])
    # In a strong profile YuNet places the nose near the outer edge of its
    # face box. A tilted frontal head keeps the nose safely within this band.
    nose_position = (nose_x - x) / width
    return nose_position < 0.12 or nose_position > 0.88


def _has_mild_yunet_profile_geometry(face) -> bool:
    """Recognize a compact near-profile with complete landmark order.

    This is intentionally not enough to accept a weak candidate by itself.
    The final recovery path additionally requires three native-context
    confirmations, two at the production confidence floor, and two strict
    profile observations.  Keeping this geometry separate lets that path
    distinguish a mild side face from a frontal candidate or arbitrary
    face-shaped texture.
    """
    if len(face) < 15 or not _has_occlusion_suspect_geometry(face):
        return False
    x, _y, width, _height = map(float, face[:4])
    if width <= 0:
        return False
    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    mouth = np.asarray(face[10:14], dtype=np.float32).reshape(2, 2)
    eye_span = abs(float(left_eye[0] - right_eye[0])) / width
    mouth_span = abs(float(mouth[1, 0] - mouth[0, 0])) / width
    nose_position = (float(face[8]) - x) / width
    return (
        .15 <= eye_span < .38
        and .10 <= mouth_span <= .55
        and (
            .12 <= nose_position <= .32
            or .68 <= nose_position <= .88
        )
    )


def _has_low_confidence_frontal_geometry(face) -> bool:
    """Identify a weak but face-like frontal landmark layout for diagnosis."""
    if len(face) < 15:
        return False
    x, y, width, height = map(float, face[:4])
    if width <= 0 or height <= 0:
        return False
    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    nose = np.asarray(face[8:10], dtype=np.float32)
    mouth = np.asarray(face[10:14], dtype=np.float32).reshape(2, 2).mean(axis=0)
    if not np.all(np.isfinite([*right_eye, *left_eye, *nose, *mouth])):
        return False
    eye_span = abs(float(left_eye[0] - right_eye[0])) / width
    eye_y_delta = abs(float(left_eye[1] - right_eye[1])) / height
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2.0)
    return (
        0.45 <= eye_span <= 0.80
        and eye_y_delta <= 0.10
        and nose[1] > eye_mid_y
        and mouth[1] > nose[1]
        and x <= right_eye[0] <= x + width
        and x <= left_eye[0] <= x + width
        and y <= right_eye[1] <= y + height
        and y <= left_eye[1] <= y + height
    )


def _has_significant_eye_line_roll(face) -> bool:
    """Keep camera-roll recovery available for tilted, apparently covered faces.

    Axis-aligned eye patches can miss visible eyes on a tilted portrait.
    Coverage is not conclusive until the existing roll scan has had a chance
    to align those eyes. Upright covered faces do not need that recovery.
    """
    if face is None or len(face) < 15:
        return False
    eyes = np.asarray(face[4:8], dtype=np.float32).reshape(2, 2)
    if not np.all(np.isfinite(eyes)):
        return False
    dx, dy = np.abs(eyes[1] - eyes[0])
    return bool(dx > 0 and np.degrees(np.arctan2(dy, dx)) >= 15.0)


def _has_occlusion_suspect_geometry(face) -> bool:
    """Recognize a partial-face layout without treating scenery as a face.

    A hidden eye or hair across one side commonly compresses YuNet's eye span
    while leaving both eye points, nose and mouth in a plausible vertical
    order. This selects an explanation only; it never accepts a candidate.
    """
    if len(face) < 15:
        return False
    x, y, width, height = map(float, face[:4])
    if width <= 0 or height <= 0:
        return False
    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    nose = np.asarray(face[8:10], dtype=np.float32)
    mouth = np.asarray(face[10:14], dtype=np.float32).reshape(2, 2).mean(axis=0)
    if not np.all(np.isfinite([*right_eye, *left_eye, *nose, *mouth])):
        return False
    eye_span = abs(float(left_eye[0] - right_eye[0])) / width
    eye_y_delta = abs(float(left_eye[1] - right_eye[1])) / height
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2.0)
    return (
        0.0 <= eye_span < 0.38
        and eye_y_delta <= 0.18
        and 0.12 <= float(nose[0] - x) / width <= 0.88
        and nose[1] > eye_mid_y
        and mouth[1] > nose[1]
        and x <= right_eye[0] <= x + width
        and x <= left_eye[0] <= x + width
        and y <= right_eye[1] <= y + height
        and y <= left_eye[1] <= y + height
    )












_HUMAN_QUALITY_CODES = frozenset({
    "FACE_TOO_SMALL", "FACE_INCOMPLETE", "FACE_TOO_DARK",
    "FACE_UNRECOGNIZABLE", "FACE_OCCLUDED", "FACE_BLURRY",
    "FACE_OVEREXPOSED", "FACE_NOT_FRONTAL", "FACE_TILTED",
})
_HUMAN_TECHNICAL_CODES = frozenset({"IMAGE_READ_ERROR", "FACE_DETECTION_FAILED"})


def _confirmed_zero_face_quality_issues(image_path: str) -> set[str]:
    """Diagnostic-only, bounded scan; never supplies or changes usable counts.

    Require bounded diagnostic confidence AND face geometry, not the permissive
    failure scanner's weak candidates. Keep independent crop/lighting evidence
    on the same face and distinct issues on different faces. Low illumination
    can itself erase edges/landmarks: never label that as additional blur,
    occlusion or pose evidence. Do not combine alternate rotations into one
    count.
    """
    image = _read_image(image_path)
    if image is None:
        return set()
    view, scale = image, 1.0
    height, width = view.shape[:2]
    _, faces = _get_yunet_detector((width, height), .60).detect(view)
    if faces is None:
        return set()
    candidates = []
    for face in faces:
        if (len(face) < 15 or not np.all(np.isfinite(face))
                or float(face[14]) < .60
                or min(face[2:4]) < YUNET_MIN_FACE_SIDE
                or float(face[2] * face[3]) / (width * height) < YUNET_MIN_FACE_RATIO):
            continue
        if not (_has_low_confidence_frontal_geometry(face)
                or (float(face[14]) < .75 and _has_occlusion_suspect_geometry(face))
                or (_is_yunet_profile(face)
                    and face[9] > (face[5] + face[7]) / 2
                    and (face[11] + face[13]) / 2 > face[9])):
            continue
        candidates.append({"box": _map_face_to_original(face, 1.0, width, height),
                           "score": float(face[14]), "raw_face": face})
    candidates = _filter_pet_face_candidates(view, _filter_neck_satellite_candidates(candidates))
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    issues = set()
    for candidate in candidates:
        box, face = candidate["box"], candidate["raw_face"]
        visual = _get_face_quality_issue(gray, box)
        if candidate["score"] < .75:
            # A dark, separately localized small face can also lack spatial
            # detail. Keep weak geometry out of all other diagnostic decisions.
            x,y,w,h=box
            roi=gray[y:y+h,x:x+w]
            if (visual == "FACE_TOO_DARK" and min(w,h)>=60
                    and (_has_low_confidence_frontal_geometry(face) or _has_occlusion_suspect_geometry(face)) and roi.size
                    and float(cv2.Laplacian(cv2.resize(roi,(160,160)),cv2.CV_64F).var()) < 3.0):
                issues.update(("FACE_TOO_DARK","FACE_BLURRY"))
            continue
        cropped = (
            _is_face_box_cut_by_frame(box, width, height)
            or _is_unusable_edge_cropped_face(
                face, box, width, height, candidate["score"],
            )
        )
        if cropped:
            issues.add("FACE_INCOMPLETE")
        if not cropped and _is_yunet_profile(face) and candidate["score"] >= .75:
            issues.add("FACE_NOT_FRONTAL")
        if visual in _HUMAN_QUALITY_CODES:
            issues.add(visual)
        core_issue = _native_face_core_issue(image, box, candidate["score"])
        if cropped and core_issue == "FACE_BLURRY":
            issues.add(core_issue)
        if visual is not None:
            continue
        if cropped:
            # Crop and occlusion are independent only when the eye-patch
            # evidence itself confirms coverage. This makes a cropped,
            # scarf-covered face a generic zero-face result without adding
            # speculative secondary reasons to an ordinary incomplete crop.
            occlusion = _get_face_occlusion_issue(gray, box, face)
            if occlusion == "FACE_OCCLUDED":
                issues.add(occlusion)
            continue
        # The appearance checks below are alternatives, not independent proof
        # of several defects caused by the same unreliable landmark layout.
        if _is_yunet_profile(face) and candidate["score"] < 0.90:
            issues.add("FACE_NOT_FRONTAL")
        else:
            occlusion = _get_face_occlusion_issue(gray, box, face)
            if occlusion in _HUMAN_QUALITY_CODES:
                issues.add(occlusion)
            elif min(box[2:]) / max(scale, 1e-9) < YUNET_MIN_FACE_SIDE:
                issues.add("FACE_TOO_SMALL")
    return issues


def _has_occluded_edge_crop_evidence(image_path: str) -> bool:
    """Confirm a substantial cropped-face hint for an established occlusion.

    This is called only after the normal detector has already confirmed
    FACE_OCCLUDED. A large, independent edge candidate then establishes the
    second crop reason without letting a weak candidate change any count.
    """
    image = _read_image(image_path)
    if image is None:
        return False
    view, _scale = image, 1.0
    height, width = view.shape[:2]
    _, faces = _get_yunet_detector((width, height), .10).detect(view)
    if faces is None:
        return False
    image_area = max(width * height, 1)
    crop_sides = set()
    for face in faces:
        if len(face) < 15 or not np.all(np.isfinite(face)):
            continue
        score = float(face[14])
        if score < .15 or min(face[2:4]) < 150:
            continue
        box = _map_face_to_original(face, 1.0, width, height)
        if (box[2] * box[3] / float(image_area) < .05
                or not any(_frame_edges_touched(box, width, height))):
            continue
        if _is_unusable_edge_cropped_face(face, box, width, height, score):
            left, _top, right, _bottom = _frame_edges_touched(box, width, height)
            if left:
                crop_sides.add("left")
            if right:
                crop_sides.add("right")
    # Require opposing independent crop hints. A single low-confidence edge
    # box is too weak to relabel an otherwise specific occlusion result.
    return crop_sides == {"left", "right"}


def _with_usable_face_count(evidence: dict) -> dict:
    """Adapt legacy scanner evidence AFTER all quality checks have completed.

    A legacy quality failure may still carry face_count=1 for its rejected
    candidate. That is not a usable face. Never infer a count from a box, nor
    turn an incomplete/failed analysis into a successful zero-face analysis.
    Explicit usable counts are authoritative and independent of the style.
    """
    result = dict(evidence)
    if result.get("code") in _HUMAN_TECHNICAL_CODES:
        result.pop("usable_face_count", None)
        result.pop("face_count", None)
        return result
    if "usable_face_count" in result:
        count = result["usable_face_count"]
    elif result.get("valid"):
        count = result.get("face_count")
    elif result.get("code") in _HUMAN_QUALITY_CODES | {"NO_FACE", "NON_HUMAN_FACE"}:
        count = 0
    else:
        count = None
    if type(count) is int and count >= 0:
        result.pop("face_count", None)
        result["usable_face_count"] = count
    else:
        result.pop("usable_face_count", None)
        result.pop("face_count", None)
    return result


def _apply_human_face_requirement(evidence: dict, expected_face_count: int, locale=None) -> dict:
    """Pure product decision: usable count first, explanation only at zero."""
    result = _with_usable_face_count(evidence)
    if result.get("code") in _HUMAN_TECHNICAL_CODES:
        result.update(valid=False, message=get_message(result["code"], locale))
        return result
    count = result.get("usable_face_count")
    if count is None:
        result.update(valid=False, code="FACE_DETECTION_FAILED",
                      message=get_message("FACE_DETECTION_FAILED", locale))
    elif count == expected_face_count:
        result.pop("code", None)
        result.update(valid=True, message=get_message("FACE_DETECTION_PASSED", locale))
    elif count > 0:
        result.update(valid=False, code="MULTIPLE_FACES" if expected_face_count == 1 else "FACE_COUNT_MISMATCH")
    else:
        code = result.get("code")
        confirmed = set(result.get("confirmed_quality_issues", ())) & _HUMAN_QUALITY_CODES
        # Repeated instances of one problem are still one reason. Orientation
        # labels describe the same family, not two independent quality defects.
        if "FACE_TILTED" in confirmed:
            confirmed.discard("FACE_TILTED")
            confirmed.add("FACE_NOT_FRONTAL")
        if len(confirmed) >= 2:
            code = "NO_FACE"
        elif code not in _HUMAN_QUALITY_CODES | {"NO_FACE", "NON_HUMAN_FACE"}:
            code = "NO_FACE"
        result.update(valid=False, code=code)
    return result


def _recover_color_cast_faces(image_path, locale):
    """Retry localization under strong channel imbalance, without changing pixels on disk."""
    try:
        image = _read_image(image_path)
    except (OSError, ValueError):
        return None
    if image is None:
        return None
    means = image.reshape(-1, 3).mean(axis=0)
    if means.max() < 60 or means.max() / max(means.min(), 1) < 2.5:
        return None
    gains = means.max() / np.maximum(means, 1)
    balanced = np.clip(image * gains, 0, 255).astype(np.uint8)
    candidates = _filter_pet_face_candidates(balanced, _scan_yunet_native_orientations(balanced))
    candidates = _recover_roll_candidates(balanced, candidates)
    # Exposure is judged on the original; balancing only restores channel contrast.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = [c for c in candidates if not c.get("diagnostic_only")
                  and _get_face_quality_issue(gray, c["box"]) not in {"FACE_TOO_DARK", "FACE_OVEREXPOSED"}]
    result = _yunet_candidate_result(candidates, image.shape[1], image.shape[0], locale)
    if result:
        result["color_detail_confirmed"] = True
    return result


def _analyze_human_faces(image_path, locale=None, *, crop_modified: bool = False):
    """Shared image evidence, independent of the requested product headcount.

    The native YuNet scan includes the same orientation recovery and quality
    diagnostics for every mode. Its observed usable count is evidence, not a
    final product rejection; only contains_human applies the requested count.
    """
    result = _contains_human_yunet(
        image_path, locale, crop_modified=crop_modified,
    )
    # `_contains_human_yunet` uses this short-lived private marker only for a
    # crop-only append that would otherwise suppress the established
    # one-face group verifier. Remove it before any public result handling.
    crop_recovery_count_recheck = bool(
        result.pop("_crop_modified_count_recheck", False)
    )
    if not result.get("valid") and result.get("code") in {"FACE_BLURRY", "FACE_UNRECOGNIZABLE", "NO_FACE"}:
        result = _recover_color_cast_faces(image_path, locale) or result
    if (
        not result.get("valid")
        and result.get("code") in {"NO_FACE", "FACE_UNRECOGNIZABLE", "FACE_NOT_FRONTAL"}
    ):
        try:
            low_quality = _diagnose_native_low_quality_face(
                _read_image(image_path), locale,
            )
        except Exception:
            low_quality = None
        if low_quality is not None:
            # Keep a confirmed viewing-angle explanation ahead of a weaker
            # occlusion hint from the same low-confidence scan.
            if (
                result.get("code") != "FACE_NOT_FRONTAL"
                or low_quality.get("code") != "FACE_OCCLUDED"
            ):
                result = low_quality
    if not result.get("valid") and result.get("code") in {
        "FACE_OCCLUDED", "FACE_UNRECOGNIZABLE", "FACE_INCOMPLETE",
    }:
        try:
            weak_signals = _native_weak_face_signals(_read_image(image_path))
        except Exception:
            weak_signals = []
        if weak_signals:
            blurry_signals = [
                signal for signal in weak_signals
                if signal.get("core") == "FACE_BLURRY"
                and signal.get("score", 0.0) < .25
            ]
            # A severely motion-blurred scene can produce a weak generic
            # candidate which the low-quality fallback mislabels as
            # occlusion. Require repeated, independently localized blur
            # evidence so a single soft or covered face is not relabeled.
            if (
                result.get("code") in {"FACE_OCCLUDED", "FACE_UNRECOGNIZABLE"}
                and len(blurry_signals) >= 2
            ):
                result = {
                    "valid": False,
                    "code": "FACE_BLURRY",
                    "message": get_message("FACE_BLURRY", locale),
                }
                weak_signals = []
            if not weak_signals:
                pass
            else:
                best_signal = max(weak_signals, key=lambda signal: signal["score"])
                box = best_signal["box"]
                if (
                    result.get("code") == "FACE_OCCLUDED"
                    and best_signal["score"] < .75
                    and min(box[2:]) < 180
                ) or (
                    result.get("code") == "FACE_INCOMPLETE"
                    and best_signal["score"] < .75
                    and min(box[2:]) < 250
                    and any(_frame_edges_touched(
                        box, _read_image(image_path).shape[1],
                        _read_image(image_path).shape[0],
                    ))
                    and best_signal.get("quality") is not None
                ):
                    result = {
                        "valid": False,
                        "code": "NO_FACE",
                        "message": get_message("NO_FACE", locale),
                    }
                elif (
                    result.get("code") == "FACE_UNRECOGNIZABLE"
                    and .40 <= best_signal["score"] < .55
                    and min(box[2:]) >= 300
                    and best_signal["quality"] is None
                    and best_signal["core"] is None
                ):
                    result = {
                        "valid": False,
                        "code": "FACE_OCCLUDED",
                        "message": get_message("FACE_OCCLUDED", locale),
                    }
    if result.get("code") == "NO_FACE":
        try:
            source_image = _read_image(image_path)
        except Exception:
            source_image = None
        if source_image is not None:
            high_light = float(np.percentile(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), 90))
            # Retain the established dark-photo diagnosis for a visibly dim
            # portrait-like upload, while leaving blank/near-black no-face
            # images as the generic outcome.
            native_dark_signal = any(
                signal.get("score", 0.0) >= .60
                and min(signal.get("box", (0, 0, 0, 0))[2:]) >= 120
                and signal.get("quality") == "FACE_TOO_DARK"
                for signal in _native_weak_face_signals(source_image)
            )
            if 40 <= high_light < 80 or (native_dark_signal and high_light < 80):
                result = {
                    "valid": False,
                    "code": "FACE_TOO_DARK",
                    "message": get_message("FACE_TOO_DARK", locale),
                }
    if result.get("valid"):
        result = _verify_face_count(
            image_path,
            result,
            locale,
            allow_crop_recovery_count_recheck=crop_recovery_count_recheck,
        )
    if result.get("valid") and result.get("face_count") == 1:
        face, size = result.get("face"), result.get("img_size")
        if face and size:
            ratio = face[2] * face[3] / float(max(size[0] * size[1], 1))
            if (ratio < MIN_FACE_RATIO_RELAXED
                    and not _face_has_sufficient_native_detail(image_path, result)):
                result = {**result, "valid": False, "code": "FACE_TOO_SMALL", "face_ratio": ratio,
                          "message": get_message("FACE_TOO_SMALL", locale, face_percent=f"{ratio * 100:.3f}")}
    if (
        result.get("valid")
        and not result.get("color_detail_confirmed")
        and not result.get("native_recovery")
        and result.get("face_score", 0.0) < 0.88
    ):
        if _is_extreme_motion_blur(image_path, result.get("face")):
            result = {**result, "valid": False, "code": "FACE_BLURRY",
                      "message": get_message("FACE_BLURRY", locale)}
    result = _with_usable_face_count(result)
    if result.get("usable_face_count") == 0 and result.get("code") in _HUMAN_QUALITY_CODES:
        try:
            confirmed = _confirmed_zero_face_quality_issues(image_path)
            # Extra diagnostics cannot replace the established specific reason
            # with unrelated evidence. Merge only if it too is confirmed.
            if (result["code"] == "FACE_OCCLUDED"
                    and _has_occluded_edge_crop_evidence(image_path)):
                result["confirmed_quality_issues"] = ["FACE_INCOMPLETE", "FACE_OCCLUDED"]
            elif result["code"] in confirmed and len(confirmed) >= 2:
                result["confirmed_quality_issues"] = sorted(confirmed)
        except Exception:
            # Optional explanation refinement must not turn a completed scan
            # into a technical failure or change acceptance/counts.
            pass
    return result


def analyze_human_faces(
    image_path: str,
    locale: str | None = None,
    *,
    crop_modified: bool = False,
) -> dict:
    """Analyze image evidence once; deliberately accepts no style or headcount.

    Single- and double-person validation, including anchor selection, consume
    this same analysis. Technical failure remains distinct from zero faces.
    """
    request_token = _request_work.set({})
    try:
        return _analyze_human_faces(
            image_path, locale, crop_modified=crop_modified,
        )
    except Exception:
        return {
            "valid": False,
            "code": "FACE_DETECTION_FAILED",
            "message": get_message("FACE_DETECTION_FAILED", locale),
        }
    finally:
        _request_work.reset(request_token)


def evaluate_human_face_analysis(analysis: dict, expected_face_count: int = 1, locale=None) -> dict:
    """Pure final product decision: no image loading, detection or re-analysis."""
    expected = 2 if expected_face_count == 2 else 1
    result = _apply_human_face_requirement(analysis, expected, locale)
    if result.get("valid") and analysis.get("usable_face_count") == expected:
        result["usable_face_boxes"] = analysis.get("usable_face_boxes", [])
    if result.get("valid") or result.get("code") in _HUMAN_TECHNICAL_CODES:
        return result
    formatter = get_single_face_error_message if expected == 1 else get_double_face_error_message
    result["message"] = formatter(result.get("code"), locale, face_count=result.get("usable_face_count", 0))
    return result


def contains_human(image_path: str, locale: str | None = None,
                   validation_profile: str = "strict", expected_face_count: int = 1,
                   *, crop_modified: bool = False) -> dict:
    """Compatibility entry point; both styles use exactly the same analyzer."""
    return evaluate_human_face_analysis(
        analyze_human_faces(image_path, locale, crop_modified=crop_modified),
        expected_face_count,
        locale,
    )


def get_usable_face_reference_boxes(
    image_path: str,
    expected_face_count: int,
    *,
    crop_modified: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Return existing accepted YuNet boxes for generation identity anchors.

    This is intentionally a read-only view of the same shared analysis used by
    validation. It never changes acceptance thresholds or counts, and returns
    no anchors unless the uploaded image has exactly the requested usable-face
    count.
    """
    expected = 2 if expected_face_count == 2 else 1
    # Use the public validation wrapper so the anchor read runs with exactly the
    # same request-local detector context and acceptance pipeline as check-photo.
    evidence = contains_human(
        image_path,
        expected_face_count=expected,
        crop_modified=crop_modified,
    )
    if evidence.get("usable_face_count") != expected:
        return []
    boxes = evidence.get("usable_face_boxes") or ([evidence["face"]] if evidence.get("face") else [])
    if len(boxes) != expected:
        return []
    return [tuple(map(float, box)) for box in boxes]
