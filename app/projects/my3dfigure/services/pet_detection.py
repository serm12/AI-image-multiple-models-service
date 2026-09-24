"""Single cat/dog upload validation for My3dFigure pet styles.

This deliberately does not participate in human-face detection.  It uses the
OpenCV Zoo YOLOX-s COCO model only to find cat/dog subjects, then applies a
per-pet image quality and local facial visibility analysis before counting.
"""

from __future__ import annotations

import threading
import logging
from pathlib import Path

import numpy as np

from app.projects.my3dfigure.core.config import project_getenv
from app.projects.my3dfigure.core.i18n import get_message
from .pet_face_visibility import analyze_pet_face

_logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency guard
    cv2 = None


PET_DETECTOR_MODEL_PATH = Path(
    project_getenv(
        "PET_DETECTION_YOLOX_MODEL",
        Path(__file__).resolve().parents[4]
        / "models"
        / "object_detection_yolox"
        / "object_detection_yolox_2022nov_int8.onnx",
    )
)
PET_DETECTION_INPUT_SIZE = (640, 640)
PET_DETECTION_CONFIDENCE = 0.40
# Extra orientations have more opportunities for false positives than one view.
PET_ROTATION_CONFIDENCE = 0.60
PET_WEAK_DETECTION_CONFIDENCE = 0.10
PET_DETECTION_NMS_THRESHOLD = 0.5
PET_MIN_IMAGE_RATIO = 0.01
PET_MIN_SHARPNESS = 25.0
PET_DARK_MEAN = 35.0
PET_BRIGHT_MEAN = 235.0
PET_EDGE_MARGIN_RATIO = 0.015
COCO_CAT_CLASS_ID = 15
COCO_DOG_CLASS_ID = 16
COCO_PERSON_CLASS_ID = 0
_SUBJECT_CLASS_NAMES = {
    COCO_PERSON_CLASS_ID: "person",
    COCO_CAT_CLASS_ID: "cat",
    COCO_DOG_CLASS_ID: "dog",
}
_pet_detector_thread_local = threading.local()


def _read_image(image_path: str):
    """Read paths containing non-ASCII characters on Windows."""
    image_bytes = np.fromfile(image_path, dtype=np.uint8)
    if image_bytes.size == 0:
        return None
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


def _get_detector():
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for pet detection")
    if not PET_DETECTOR_MODEL_PATH.is_file():
        raise RuntimeError(f"pet detector model is missing: {PET_DETECTOR_MODEL_PATH}")
    detector = getattr(_pet_detector_thread_local, "detector", None)
    if detector is None:
        detector = cv2.dnn.readNetFromONNX(str(PET_DETECTOR_MODEL_PATH))
        _pet_detector_thread_local.detector = detector
    return detector


def _yolox_anchors():
    grids, expanded_strides = [], []
    for stride in (8, 16, 32):
        height = PET_DETECTION_INPUT_SIZE[0] // stride
        width = PET_DETECTION_INPUT_SIZE[1] // stride
        xv, yv = np.meshgrid(np.arange(height), np.arange(width))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


_YOLOX_GRIDS, _YOLOX_EXPANDED_STRIDES = _yolox_anchors()


def _letterbox(image):
    target_height, target_width = PET_DETECTION_INPUT_SIZE
    ratio = min(target_height / image.shape[0], target_width / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded = np.full((target_height, target_width, 3), 114.0, dtype=np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return padded, ratio


def _detect_subjects(image, confidence: float = PET_DETECTION_CONFIDENCE) -> list[dict]:
    """Return NMS-deduplicated person/cat/dog boxes in original-image coordinates."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    padded, scale = _letterbox(rgb)
    blob = np.transpose(padded, (2, 0, 1))[np.newaxis, :, :, :]
    detector = _get_detector()
    detector.setInput(blob)
    outputs = detector.forward(detector.getUnconnectedOutLayersNames())[0]
    detections = outputs[0].astype(np.float32, copy=True)
    if detections.ndim != 2 or detections.shape[1] <= COCO_DOG_CLASS_ID + 5:
        raise RuntimeError("pet detector returned an unexpected output shape")

    detections[:, :2] = (detections[:, :2] + _YOLOX_GRIDS[0]) * _YOLOX_EXPANDED_STRIDES[0]
    detections[:, 2:4] = np.exp(detections[:, 2:4]) * _YOLOX_EXPANDED_STRIDES[0]
    boxes, scores, class_ids = [], [], []
    for class_id in _SUBJECT_CLASS_NAMES:
        class_scores = detections[:, 4] * detections[:, 5 + class_id]
        for index in np.flatnonzero(class_scores >= confidence):
            center_x, center_y, width, height = detections[index, :4]
            boxes.append([
                float((center_x - width / 2) / scale),
                float((center_y - height / 2) / scale),
                float(width / scale),
                float(height / scale),
            ])
            scores.append(float(class_scores[index]))
            class_ids.append(class_id)
    if not boxes:
        return []

    kept = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, confidence, PET_DETECTION_NMS_THRESHOLD)
    candidates = []
    image_height, image_width = image.shape[:2]
    for index in np.asarray(kept).reshape(-1):
        x, y, width, height = boxes[int(index)]
        left = max(0, round(x))
        top = max(0, round(y))
        right = min(image_width, round(x + width))
        bottom = min(image_height, round(y + height))
        if right <= left or bottom <= top:
            continue
        candidates.append({
            "box": (left, top, right - left, bottom - top),
            "score": scores[int(index)],
            "species": _SUBJECT_CLASS_NAMES[class_ids[int(index)]],
        })
    return candidates


def _box_iou(first_box, second_box) -> float:
    first_x, first_y, first_width, first_height = first_box
    second_x, second_y, second_width, second_height = second_box
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union else 0.0


def _remove_person_duplicate_pets(subjects: list[dict]) -> list[dict]:
    """Discard cat/dog boxes that are effectively a duplicate of a person box.

    YOLOX can label a full-body person as both person and dog in tilted shots.
    This keeps genuine pets held by a person: their box is much smaller and
    therefore does not have near-identical overlap with the person box.
    """
    people = [subject for subject in subjects if subject["species"] == "person"]
    return [
        subject
        for subject in subjects
        if subject["species"] not in {"cat", "dog"}
        or not any(_box_iou(subject["box"], person["box"]) >= 0.80 for person in people)
    ]


def _brighten_low_light(image):
    """Use a brightened copy only for locating a pet; quality uses the source."""
    return cv2.convertScaleAbs(image, alpha=2.2, beta=20)


def _box_from_quarter_turn(box, turns, image_width, image_height):
    """Map a box from np.rot90's counterclockwise view to source coordinates."""
    x, y, width, height = box
    if turns == 1:
        return (image_width - y - height, x, height, width)
    if turns == 2:
        return (image_width - x - width, image_height - y - height, width, height)
    if turns == 3:
        return (y, image_height - x - width, height, width)
    return box


def _detect_rotated_pets(image, source_subjects=()):
    """Rescue sideways/inverted pets, without altering the upload or its quality.

    Check all three views before counting, and merge overlapping boxes in source
    coordinates so the same pet is not counted once per orientation/species.
    Person-duplicate filtering must happen within each view before mapping.
    """
    image_height, image_width = image.shape[:2]
    candidates = []
    source_people = [subject for subject in source_subjects if subject['species'] == 'person']
    for turns in (1, 2, 3):
        rotated = np.ascontiguousarray(np.rot90(image, turns))
        subjects = _remove_person_duplicate_pets(_detect_subjects(rotated))
        for subject in subjects:
            if subject['species'] not in {'cat', 'dog'} or subject['score'] < PET_ROTATION_CONFIDENCE:
                continue
            candidates.append({
                **subject,
                'box': _box_from_quarter_turn(subject['box'], turns, image_width, image_height),
            })
    pets = []
    for candidate in sorted(candidates, key=lambda item: item['score'], reverse=True):
        # Upright human close-ups can look like dogs when turned sideways.
        # Retain the stronger source-view person evidence for substantially
        # overlapping regions. A small pet held inside a person's much larger
        # box has low IoU and is still eligible for rescue.
        if any(
            _box_iou(candidate['box'], person['box']) >= 0.80
            or (_box_iou(candidate['box'], person['box']) >= 0.25
                and person['score'] > candidate['score'])
            for person in source_people
        ):
            continue
        if not any(_box_iou(candidate['box'], pet['box']) >= PET_DETECTION_NMS_THRESHOLD for pet in pets):
            pets.append(candidate)
    return pets


def _pet_quality_issues(image, box):
    """Collect independently measurable defects without changing thresholds."""
    x, y, width, height = box
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        return {"PET_INCOMPLETE"}
    issues = set()
    image_height, image_width = image.shape[:2]
    # A body box touching the image edge does not make a clear portrait
    # unusable. Facial edge evidence is checked by analyze_pet_face instead.
    if width * height / float(image_width * image_height) < PET_MIN_IMAGE_RATIO:
        issues.add("PET_TOO_SMALL")
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(grayscale))
    if brightness < PET_DARK_MEAN:
        issues.add("PET_TOO_DARK")
    elif brightness > PET_BRIGHT_MEAN:
        issues.add("PET_OVEREXPOSED")
    # Exposure loss and very small crops can themselves erase texture. Do not
    # count the same loss of detail again as an independent blur diagnosis.
    elif "PET_TOO_SMALL" not in issues and float(cv2.Laplacian(grayscale, cv2.CV_64F).var()) < PET_MIN_SHARPNESS:
        issues.add("PET_BLURRY")
    return issues


def _same_pet_box(a, b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    overlap=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return _box_iou(a,b) >= .4 or overlap/max(1,min(aw*ah,bw*bh)) >= .7


def _recover_local_pet_candidates(image, pets, weak):
    missing=[p for p in weak if not any(_same_pet_box(p['box'],q['box']) for q in pets)]
    if not missing:return pets
    h,w=image.shape[:2];found=[]
    for fx,fy in ((0,0),(.35,0),(0,.35),(.35,.35),(.175,.175)):
        x,y=int(w*fx),int(h*fy);crop=image[y:y+int(h*.65),x:x+int(w*.65)]
        for p in _remove_person_duplicate_pets(_detect_subjects(crop)):
            if p['species'] not in {'cat','dog'}:continue
            bx,by,bw,bh=p['box'];box=(x+bx,y+by,bw,bh)
            if any(_same_pet_box(box,q['box']) for q in pets):continue
            found.append({**p,'box':box,'tile':(fx,fy)})
    for p in sorted(found,key=lambda q:q['score'],reverse=True):
        confirmations={q['tile'] for q in found if _same_pet_box(p['box'],q['box'])}
        if len(confirmations)>=2 and not any(_same_pet_box(p['box'],q['box']) for q in pets):
            pets.append({k:v for k,v in p.items() if k!='tile'})
    return pets


def _recover_clear_companion_pet(image, pets):
    """Rescue a missed companion only from repeated, local cat/dog evidence.

    This path is called only after the normal candidates produced no usable
    pet.  A newly found animal must recur in at least three vertically shifted
    local views, retain a detector score of .60 or higher, and later pass the
    unchanged source-quality and facial-visibility checks.
    """
    height, width = image.shape[:2]
    candidates = []
    for fx in (.25, .30, .35):
        for fy in (.05, .10, .15, .20, .25, .30):
            x, y = int(width * fx), int(height * fy)
            crop = image[y:y + int(height * .65), x:x + int(width * .65)]
            for subject in _remove_person_duplicate_pets(_detect_subjects(crop)):
                if subject["species"] not in {"cat", "dog"} or subject["score"] < .60:
                    continue
                box = (x + subject["box"][0], y + subject["box"][1], subject["box"][2], subject["box"][3])
                if any(_same_pet_box(box, pet["box"]) for pet in pets):
                    continue
                candidates.append({**subject, "box": box, "tile": (fx, fy)})
    recovered = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        matches = [item for item in candidates if item["species"] == candidate["species"] and _same_pet_box(item["box"], candidate["box"])]
        vertical_views = {item["tile"][1] for item in matches}
        if len(vertical_views) >= 3 and not any(_same_pet_box(candidate["box"], pet["box"]) for pet in recovered):
            recovered.append({key: value for key, value in candidate.items() if key != "tile"})
    return recovered


def _contains_single_pet(image_path: str, locale: str | None = None) -> dict:
    """Validate exactly one clear cat or dog without invoking YuNet."""
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for pet detection")
    try:
        image = _read_image(image_path)
    except (OSError, ValueError):
        return {"valid": False, "code": "IMAGE_READ_ERROR", "message": get_message("IMAGE_READ_ERROR", locale)}
    if image is None:
        return {"valid": False, "code": "IMAGE_READ_ERROR", "message": get_message("IMAGE_READ_ERROR", locale)}
    subjects = _remove_person_duplicate_pets(_detect_subjects(image))
    pets = [subject for subject in subjects if subject["species"] in {"cat", "dog"}]
    if not pets and float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) < PET_DARK_MEAN:
        subjects = _remove_person_duplicate_pets(_detect_subjects(_brighten_low_light(image)))
        pets = [subject for subject in subjects if subject["species"] in {"cat", "dog"}]
    weak_subjects = _remove_person_duplicate_pets(_detect_subjects(image, PET_WEAK_DETECTION_CONFIDENCE))
    weak_pets = [p for p in weak_subjects if p['species'] in {'cat','dog'}]
    pets = _recover_local_pet_candidates(image, pets, weak_pets)
    if not pets and not weak_pets and not subjects:
        # A bounded finer scan may identify a distant animal, but cannot
        # promote its inadequate source resolution to a usable portrait.
        h,w=image.shape[:2]
        for fy in (0,.3,.6):
            for fx in (0,.3,.6):
                x,y=int(w*fx),int(h*fy)
                for q in _remove_person_duplicate_pets(_detect_subjects(image[y:y+int(h*.4),x:x+int(w*.4)])):
                    if q['species'] not in {'cat','dog'}: continue
                    bx,by,bw,bh=q['box'];box=(x+bx,y+by,bw,bh)
                    if bw*bh/float(w*h) < PET_MIN_IMAGE_RATIO and not any(_same_pet_box(box,t['box']) for t in pets):
                        pets.append({**q,'box':box,'diagnostic_only':True})
    if not pets:
        pets = _detect_rotated_pets(image, subjects)
    # Weak boxes may establish a quality failure only after semantic pet
    # confirmation; they never increase the usable count on their own.
    for p in sorted(weak_pets,key=lambda q:q['score'],reverse=True):
        if not any(_same_pet_box(p['box'],q['box']) for q in pets):
            pets.append({**p,'diagnostic_only':True})
    if not pets:
        return {'valid':False,'code':'PET_NOT_DETECTED','message':get_message('PET_NOT_DETECTED',locale),'pet_count':0}
    unique = []
    for pet in sorted(pets,key=lambda q:q['score'],reverse=True):
        if not any(_same_pet_box(pet['box'],q['box']) for q in unique): unique.append(pet)
    analyzed = []
    for pet in unique:
        face = analyze_pet_face(image, pet['box'], pet['species'])
        if face.get('nonpet'):
            continue
        ev=face.get('visual_similarities')
        if pet.get('diagnostic_only') and pet['score'] < PET_DETECTION_CONFIDENCE and ev and ev['visible'] < ev['nonpet'] + .02:
            continue
        quality = set(_pet_quality_issues(image, pet['box']))
        face_issues = set(face['issues'])
        # Exposure and tiny size can erase texture; avoid naming the same
        # detail loss a second independent blur cause for this animal.
        if quality:
            face_issues.clear()
        quality.update(face_issues)
        if pet.get('diagnostic_only') and not quality:
            face = {**face,'usable':False}
        analyzed.append({**pet, "quality_issues": sorted(quality),
                         "usable": not quality and face is not None and face["usable"],
                         "face_analysis": face})
    usable = [pet for pet in analyzed if pet["usable"]]
    if not usable and analyzed and all(pet["face_analysis"].get("visual_similarities") for pet in analyzed):
        for pet in _recover_clear_companion_pet(image, unique):
            face = analyze_pet_face(image, pet["box"], pet["species"])
            if face.get("nonpet"):
                continue
            quality = set(_pet_quality_issues(image, pet["box"]))
            face_issues = set(face["issues"])
            if quality:
                face_issues.clear()
            quality.update(face_issues)
            analyzed.append({**pet, "quality_issues": sorted(quality),
                             "usable": not quality and face["usable"],
                             "face_analysis": face})
        usable = [pet for pet in analyzed if pet["usable"]]
    issues = sorted({issue for pet in analyzed for issue in pet["quality_issues"]})
    count = len(usable)
    result = {"valid": count == 1, "pet_count": count,
              "quality_issues": issues, "pet_analysis": analyzed,
              "image_size": (image.shape[1], image.shape[0])}
    if count == 1:
        pet = usable[0]
        result.update(message=get_message("PET_DETECTION_PASSED", locale),
                      pet=pet["box"], pet_species=pet["species"])
        anchor_box = pet.get("face_analysis", {}).get("identity_anchor_box")
        if isinstance(anchor_box, list) and len(anchor_box) == 4:
            result["pet_face_anchor_box"] = anchor_box
    else:
        code = "PET_COUNT_MISMATCH" if count > 1 else issues[0] if len(issues) == 1 and not any(not q["quality_issues"] for q in analyzed) else "PET_NOT_DETECTED"
        result.update(code=code, message=get_message(code, locale, pet_count=count))
        if len(analyzed) == 1:
            result["pet"] = analyzed[0]["box"]
    return result


def contains_single_pet(image_path: str, locale: str | None = None) -> dict:
    """Keep model/read failures separate from a completed zero-pet result."""
    try:
        return _contains_single_pet(image_path, locale)
    except Exception:
        _logger.exception("My3dFigure pet validation failed")
        return {"valid": False, "code": "PET_DETECTION_FAILED",
                "message": get_message("PET_DETECTION_FAILED", locale)}
