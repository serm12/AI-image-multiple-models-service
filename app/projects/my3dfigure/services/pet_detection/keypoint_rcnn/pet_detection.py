"""Pets-Face-Recognition v1.3 head/eye/nose detector, independent of YOLOX.

Official pet weights use ResNet50-FPN, one foreground class and three landmarks.
All post-NMS heads are inspected before applying the single usable pet rule.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from app.projects.my3dfigure.core.config import project_getenv
from app.projects.my3dfigure.core.i18n import get_message
from .visual import species_evidence
from .quality import analyze, diagnostic_issues
from .tiny import find_tiny_pet

MODEL_PATH = Path(project_getenv('PET_KEYPOINT_MODEL',
    Path(__file__).resolve().parents[6] / 'models/pet_detection/keypoint-rcnn/keypoint-rcnn-v1.3.pt'))
HEAD_CONFIDENCE = 0.70
PROPOSAL_CONFIDENCE = 0.05
# KeypointRCNN heatmap scores are raw scores, NOT probabilities or RTMPose scores.
LANDMARK_MIN_SCORE = 2.0
_model = None
_lock = threading.RLock()
_logger = logging.getLogger(__name__)


def _get_model():
    global _model
    with _lock:
        if _model is None:
            import torch
            from torchvision.models.detection import KeypointRCNN
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            from torchvision.ops import FrozenBatchNorm2d
            # Explicit FrozenBN matches the original pretrained ResNet backbone.
            backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None,
                                          norm_layer=FrozenBatchNorm2d, trainable_layers=5)
            model = KeypointRCNN(backbone, num_classes=2, num_keypoints=3,
                                 min_size=(320, 336, 352, 368, 384, 400), max_size=640,
                                 box_score_thresh=PROPOSAL_CONFIDENCE,
                                 box_detections_per_img=sys.maxsize)
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True), strict=True)
            _model = model.eval()
    return _model


def detect_heads(image, high_resolution=False):
    """Return every surviving head; no first-box selection or target-count cap."""
    import torch
    tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float()/255
    with _lock, torch.inference_mode():
        model = _get_model()
        previous = (model.transform.min_size, model.transform.max_size)
        try:
            if high_resolution:
                model.transform.min_size, model.transform.max_size = (800,), 1333
            output = model([tensor])[0]
        finally:
            model.transform.min_size, model.transform.max_size = previous
    expected = {'boxes', 'scores', 'keypoints', 'keypoints_scores'}
    if not expected.issubset(output):
        raise RuntimeError('Incomplete KeypointRCNN output')
    arrays = {key: output[key].detach().cpu().numpy() for key in expected}
    count = len(arrays['scores'])
    if arrays['boxes'].shape != (count, 4) or arrays['scores'].shape != (count,) or arrays['keypoints'].shape != (count, 3, 3) or arrays['keypoints_scores'].shape != (count, 3):
        raise RuntimeError('Unexpected KeypointRCNN output shapes')
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Nonfinite KeypointRCNN output')
    height, width = image.shape[:2]
    heads = []
    for box, score, points, point_scores in zip(arrays['boxes'], arrays['scores'], arrays['keypoints'], arrays['keypoints_scores']):
        if score < PROPOSAL_CONFIDENCE:
            continue
        left, top = np.maximum(np.floor(box[:2]).astype(int), 0)
        right, bottom = np.minimum(np.ceil(box[2:]).astype(int), (width, height))
        if right <= left or bottom <= top:
            continue
        heads.append({'box': [int(left), int(top), int(right-left), int(bottom-top)],
                      'score': float(score), 'landmarks': points[:, :2].tolist(),
                      'landmark_scores': point_scores.tolist()})
    return heads


def _measure(image, head):
    x,y,w,h=head['box']
    crop=image[y:y+h,x:x+w]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    return {**head,'evidence':species_evidence(crop), 'pixels':dict(
        mean=float(gray.mean()),std=float(gray.std()),p90=float(np.percentile(gray,90)),
        lap=float(cv2.Laplacian(gray,cv2.CV_64F).var()),bright=float(np.mean(gray>245)))}


def _analyze(image, head, context=None):
    return analyze(_measure(image,head),image.shape[:2],context)


def _contains_single_pet(image_path, locale=None):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, ValueError, cv2.error):
        image = None
    if image is None:
        return {'valid': False, 'code': 'IMAGE_READ_ERROR', 'message': get_message('IMAGE_READ_ERROR', locale)}
    context=species_evidence(image)
    heads=[_measure(image,h) for h in detect_heads(image)]
    analyzed=[p for h in heads if h['score']>=HEAD_CONFIDENCE
              and (p:=analyze(h,image.shape[:2],context)) is not None]
    usable=[p for p in analyzed if p['usable']]
    # Revisit a plausible eye/nose pair at higher spatial resolution.
    if not usable and not analyzed and any(
            h['score']>=.2 and max(h['landmark_scores'][:2])>=1 and h['landmark_scores'][2]>=2
            and max(e['visible']-e['nonpet'] for e in h['evidence'].values())>.025 for h in heads):
        high=[_measure(image,h) for h in detect_heads(image,high_resolution=True) if h['score']>=HEAD_CONFIDENCE]
        analyzed=[p for h in high if (p:=analyze(h,image.shape[:2],context)) is not None]
        usable=[p for p in analyzed if p['usable']]
    count=len(usable)
    issues=sorted({issue for pet in analyzed for issue in pet['quality_issues']})
    diagnostic=[]
    if not usable and not issues:
        diagnostic=diagnostic_issues(heads,image.shape[:2],context)
        issues=diagnostic
    unclear=any(not p['usable'] and not p['quality_issues'] for p in analyzed)
    tiny_evidence=[]
    if not usable and not analyzed and not issues:
        scene=max(context.values(),key=lambda e:e['visible'])
        if scene['visible']>.22 and scene['visible']>scene['nonpet']-.02:
            tiny_evidence=find_tiny_pet(image,detect_heads)
            if tiny_evidence:
                issues=['PET_TOO_SMALL']
                diagnostic=issues.copy()
    result = {'valid': count == 1, 'pet_count': count, 'quality_issues': issues,
              'pet_analysis': analyzed, 'image_size': (image.shape[1], image.shape[0]), 'diagnostic_issues': diagnostic,
              'tiny_pet_evidence':tiny_evidence}
    if count == 1:
        pet = usable[0]
        result.update(message=get_message('PET_DETECTION_PASSED', locale), pet=pet['box'],
                      pet_species=pet['species'], pet_face_anchor_box=pet['face_analysis']['identity_anchor_box'])
    else:
        code = 'PET_COUNT_MISMATCH' if count > 1 else issues[0] if len(issues)==1 and not unclear else 'PET_NOT_DETECTED'
        result.update(code=code, message=get_message(code, locale, pet_count=count))
        if len(analyzed)==1:
            result['pet'] = analyzed[0]['box']
    return result


def contains_single_pet(image_path: str, locale: str | None = None) -> dict:
    try:
        return _contains_single_pet(image_path, locale)
    except Exception:
        _logger.exception('My3dFigure KeypointRCNN pet validation failed')
        return {'valid': False, 'code': 'PET_DETECTION_FAILED', 'message': get_message('PET_DETECTION_FAILED', locale)}
