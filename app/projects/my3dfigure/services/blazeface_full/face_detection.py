"""Independent My3dFigure BlazeFace Full Range face analysis.

The detector uses Google's float16 Full Range model in MediaPipe IMAGE mode
with default confirmation settings and a separate low-confidence proposal task. Independent
multi-view confirmation and source-pixel quality checks qualify usable faces.
This pipeline never calls the YuNet implementation.
"""
from __future__ import annotations
import hashlib
import logging
import threading
from pathlib import Path
import cv2
import numpy as np
from . import pipeline, localization
_LOGGER = logging.getLogger(__name__)
from app.projects.my3dfigure.core.i18n import get_double_face_error_message, get_message, get_single_face_error_message
MODEL_PATH = Path(__file__).resolve().parents[5] / 'models' / 'my3dfigure' / 'blazeface_full_range' / 'blaze_face_full_range.tflite'
MODEL_SHA256 = '3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b'
_THREAD_LOCAL = threading.local()
_TECHNICAL_CODES = frozenset({'IMAGE_READ_ERROR', 'FACE_DETECTION_FAILED'})

def _detector():
    """Use a separate mutable MediaPipe task in each request worker thread."""
    instance = getattr(_THREAD_LOCAL, 'instance', None)
    if instance is not None:
        return instance
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'BlazeFace model missing: {MODEL_PATH}')
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError('BlazeFace model does not match the official tested weight')
    import mediapipe as mp
    instance = mp.tasks.vision.FaceDetector.create_from_model_path(str(MODEL_PATH))
    _THREAD_LOCAL.instance = instance
    _THREAD_LOCAL.mediapipe = mp
    return instance

def _proposal_detector():
    """Lower confidence only gathers evidence; it never bypasses quality checks."""
    instance = getattr(_THREAD_LOCAL, 'proposal_instance', None)
    if instance is None:
        _detector()
        mp = _THREAD_LOCAL.mediapipe
        options = mp.tasks.vision.FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)), min_detection_confidence=0.2)
        instance = mp.tasks.vision.FaceDetector.create_from_options(options)
        _THREAD_LOCAL.proposal_instance = instance
    return instance

def _read_image(image_path: str):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, ValueError, cv2.error):
        return None

def analyze_human_faces(image_path: str, locale: str | None=None) -> dict:
    """Determine usable faces before applying either product's target count."""
    image = _read_image(image_path)
    if image is None:
        return {'valid': False, 'code': 'IMAGE_READ_ERROR', 'message': get_message('IMAGE_READ_ERROR', locale)}
    try:
        detector = _detector()
        accepted, reason = pipeline.analyze(image, detector, _proposal_detector(), _THREAD_LOCAL.mediapipe)
        height, width = image.shape[:2]
        boxes = []
        for candidate in accepted:
            x, y, w, h = localization.face_box(candidate)
            left, top = (max(0.0, float(x)), max(0.0, float(y)))
            right, bottom = (min(float(width), float(x + w)), min(float(height), float(y + h)))
            boxes.append([left, top, right - left, bottom - top])
        if boxes:
            return {'valid': True, 'usable_face_count': len(boxes), 'usable_face_boxes': boxes, 'face': max(boxes, key=lambda box: box[2] * box[3]), 'img_size': (width, height)}
        return {'valid': False, 'usable_face_count': 0, 'usable_face_boxes': [], 'code': reason}
    except Exception:
        _LOGGER.exception('BlazeFace face analysis failed')
        return {'valid': False, 'code': 'FACE_DETECTION_FAILED', 'message': get_message('FACE_DETECTION_FAILED', locale)}

def evaluate_human_face_analysis(analysis: dict, expected_face_count: int=1, locale: str | None=None) -> dict:
    """Apply the selected product count only after BlazeFace quality analysis."""
    result = dict(analysis)
    if result.get('code') in _TECHNICAL_CODES:
        result.pop('usable_face_count', None)
        return result
    count = result.get('usable_face_count')
    if type(count) is not int or count < 0:
        return {'valid': False, 'code': 'FACE_DETECTION_FAILED', 'message': get_message('FACE_DETECTION_FAILED', locale)}
    expected = 2 if expected_face_count == 2 else 1
    if count == expected:
        result.pop('code', None)
        result['valid'] = True
        result['message'] = get_message('FACE_DETECTION_PASSED', locale)
        return result
    result['valid'] = False
    if count:
        result['code'] = 'MULTIPLE_FACES' if expected == 1 else 'FACE_COUNT_MISMATCH'
    else:
        result['code'] = result.get('code') or 'NO_FACE'
    formatter = get_single_face_error_message if expected == 1 else get_double_face_error_message
    result['message'] = formatter(result['code'], locale, face_count=count)
    return result

def contains_human(image_path: str, locale: str | None=None, validation_profile: str='strict', expected_face_count: int=1) -> dict:
    """Keep the existing My3dFigure API signature for both human styles."""
    return evaluate_human_face_analysis(analyze_human_faces(image_path, locale), expected_face_count, locale)

def get_usable_face_reference_boxes(image_path: str, expected_face_count: int) -> list[tuple[float, float, float, float]]:
    """Use the same selected pipeline for generation face anchors."""
    expected = 2 if expected_face_count == 2 else 1
    result = contains_human(image_path, expected_face_count=expected)
    if not result.get('valid') or result.get('usable_face_count') != expected:
        return []
    boxes = result.get('usable_face_boxes') or []
    if len(boxes) != expected:
        return []
    return [tuple(map(float, box)) for box in boxes]
