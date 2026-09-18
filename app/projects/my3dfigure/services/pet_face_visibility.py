"""Local cat/dog facial visibility assessment; never participates in human detection.

RTMPose provides facial landmarks, not proof of occlusion. Weak landmarks are
cross-checked against CLIP visual evidence; missing eyes alone never means an
occluder was identified. Scores are model similarities, not probabilities.
"""
from pathlib import Path
import json
import threading
import numpy as np
import cv2
from PIL import Image, ImageOps

from app.projects.my3dfigure.core.config import project_getenv

MODEL_DIRECTORY = Path(project_getenv(
    "PET_FACE_MODEL_DIR", Path(__file__).resolve().parents[4] / "models/my3dfigure/pet_face"
))
TEXT_FEATURE_DIRECTORY = MODEL_DIRECTORY / 'clip'
_local = threading.local()
_lock = threading.Lock()
_visual = None
_features = None
_labels = None


def _pose_net():
    if not hasattr(_local, 'pose'):
        _local.pose = cv2.dnn.readNetFromONNX(str(MODEL_DIRECTORY / 'rtmpose-ap10k.onnx'))
    return _local.pose


def _landmarks(crop, padding=1.25, *, with_points=False):
    height, width = crop.shape[:2]
    side = max(width, height) * padding
    transform = np.array([[256 / side, 0, 128 - width * 128 / side],
                          [0, 256 / side, 128 - height * 128 / side]], np.float32)
    view = cv2.warpAffine(crop, transform, (256, 256))
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
    net = _pose_net()
    net.setInput(normalized.transpose(2, 0, 1)[None])
    sx, sy = net.forward(net.getUnconnectedOutLayersNames())
    if sx.shape != (1, 17, 512) or sy.shape != (1, 17, 512):
        raise RuntimeError('Unexpected animal landmark model output')
    points = np.stack([sx.argmax(-1), sy.argmax(-1)], axis=-1)[0] / 2
    scores = np.minimum(sx.max(-1), sy.max(-1))[0]
    points = points / 256 * side + np.array([width, height]) / 2 - side / 2
    if not np.all(np.isfinite(scores)):
        raise RuntimeError('Nonfinite animal landmark scores')
    eyes = np.flatnonzero(scores[:2] >= .75)
    geometry = any(np.linalg.norm(points[i] - points[2]) >= min(width, height) * .025 for i in eyes)
    inside = all(0 <= points[i, 0] < width and 0 <= points[i, 1] < height for i in [2, int(np.argmax(scores[:2]))])
    strength = float(min(scores[2], max(scores[:2]))) if geometry and inside else 0.0
    result = (strength, scores[:3].tolist())
    return (*result, points[:3].tolist()) if with_points else result


def _visual_models():
    global _visual, _features, _labels
    if _visual is None:
        with _lock:
            if _visual is None:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                features = np.load(TEXT_FEATURE_DIRECTORY / 'text-features-v4.npz', allow_pickle=False)['features']
                labels = json.loads((TEXT_FEATURE_DIRECTORY / 'prompts-v4.json').read_text(encoding='utf8'))['labels']
                if features.shape != (len(labels), 512) or not np.all(np.isfinite(features)):
                    raise RuntimeError('Invalid pet visibility text features')
                session = ort.InferenceSession(str(MODEL_DIRECTORY / 'clip/vision_model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
                _features, _labels, _visual = features, labels, session
    return _visual, _features, _labels


def _visual_evidence(crop, species):
    session, features, labels = _visual_models()
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(124, 116, 104))
    tensor = np.asarray(image, np.float32) / 255
    tensor = (tensor - np.array([.48145466, .4578275, .40821073], np.float32)) / np.array([.26862954, .26130258, .27577711], np.float32)
    vector = session.run(None, {'pixel_values': tensor.transpose(2, 0, 1)[None]})[0][0]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('Invalid pet visibility image embedding')
    similarities = features @ (vector / norm)
    return {label: float(max(similarities[i] for i, item in enumerate(labels) if item == [species, label]))
            for label in ('visible', 'occluded', 'back', 'fur', 'blur', 'nonpet', 'cut', 'dark', 'bright', 'motion')}


def _decide(strength, evidence):
    """Require positive visual evidence to name a specific failure cause."""
    visible, occluded, back, fur, blur = (evidence[k] for k in ('visible', 'occluded', 'back', 'fur', 'blur'))
    if occluded >= .28 and occluded > max(visible + .015, back + .01, blur + .01):
        return False, ['PET_OCCLUDED']
    if strength < 1.0 and (fur > max(visible + .015, blur + .01)
                          or back > max(visible + .003, blur + .015)):
        return False, ['PET_FACE_UNRECOGNIZABLE']
    if strength >= 1.0 or (strength >= .75 and visible >= max(occluded, back, blur) - .01):
        return True, []
    # Insufficient face evidence is not a technical failure or proof of an
    # occluder. Exclude this candidate without fabricating a specific cause.
    return False, []



def _border_visibility_rejection(evidence):
    """Additional evidence before admitting newly eligible body-cropped pets."""
    visible = evidence['visible']
    if evidence['nonpet'] >= visible - .015:
        return []
    if evidence['back'] >= visible and evidence['back'] > evidence['blur'] + .015:
        return ['PET_FACE_UNRECOGNIZABLE']
    if evidence['blur'] > visible + .01:
        return []
    return None

def _facial_features_cut_by_frame(points, scores, box, image_shape, turns=0):
    """Require reliable eye/nose evidence at the actual image boundary.

    Body/ear/tail clipping and tight detector boxes alone are not failures.
    Missing landmarks alone cannot establish which image edge cuts the face.
    """
    x, y, width, height = box
    ih, iw = image_shape[:2]
    if x > 1 and y > 1 and x + width < iw - 1 and y + height < ih - 1:
        return False
    selected = [i for i in range(3) if scores[i] >= .75]
    if 2 not in selected or not any(i < 2 for i in selected):
        return False
    pts = np.asarray(points, dtype=float)[selected].copy()
    px, py = pts[:, 0].copy(), pts[:, 1].copy()
    if turns == 1: pts[:, 0], pts[:, 1] = width - py, px
    elif turns == 2: pts[:, 0], pts[:, 1] = width - px, height - py
    elif turns == 3: pts[:, 0], pts[:, 1] = py, height - px
    span = float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)))
    if not np.isfinite(span) or span < 3:
        return False
    # Small local margin around measured features, not an inferred full body
    # or symmetric second eye (readable profiles remain eligible).
    margin = max(1.0, span * .20)
    pts += [x, y]
    return bool(np.any(pts[:, 0] < margin) or np.any(pts[:, 0] > iw - margin)
                or np.any(pts[:, 1] < margin) or np.any(pts[:, 1] > ih - margin))


def analyze_pet_face(image, box, species):
    x, y, width, height = map(int, box)
    crop = image[y:y+height, x:x+width]
    if not crop.size or species not in {'cat','dog'}:
        raise ValueError('Invalid pet face input')
    strength, scores, points = _landmarks(crop, with_points=True)
    evidence = _visual_evidence(crop, species)
    issues = set()
    # Semantic identity is checked before assigning a quality error.
    if evidence['nonpet'] >= evidence['visible'] - .005:
        return {'usable': False, 'issues': [], 'nonpet': True, 'visual_similarities': evidence}
    ih, iw = image.shape[:2]
    at_frame = x <= iw*.015 or y <= ih*.015 or x+width >= iw*.985 or y+height >= ih*.985
    if _facial_features_cut_by_frame(points, scores, box, image.shape):
        issues.add('PET_INCOMPLETE')
    elif (x <= iw*.015 or x+width >= iw*.985 or y <= ih*.015) and strength < .75 and evidence['cut'] > evidence['visible'] + .003:
        issues.add('PET_INCOMPLETE')
    # Exposure evidence is measured locally at plausible facial landmarks,
    # not averaged over the body, floor or surrounding background.
    if min(scores[:2]) >= .35:
        pts = np.asarray(points)
        span = float(np.linalg.norm(pts[0]-pts[1]))
        if span >= 8 and span < min(width,height)*.7:
            left, top = np.floor(pts.min(0)-span*.35).astype(int)
            right, bottom = np.ceil(pts.max(0)+span*.35).astype(int)
            core = crop[max(0,top):min(height,bottom),max(0,left):min(width,right)]
            if core.size:
                gray = cv2.cvtColor(core,cv2.COLOR_BGR2GRAY)
                if float(np.mean(gray)) < 35: issues.add('PET_TOO_DARK')
                elif float(np.mean(gray>245)) > .65: issues.add('PET_OVEREXPOSED')
    if not issues and evidence['motion'] > max(evidence['visible']+.025,evidence['back']+.015,evidence['fur']+.01):
        issues.add('PET_BLURRY')
    if not issues and strength < 1.0 and evidence['fur'] > max(evidence['visible']+.015,evidence['motion']+.01):
        issues.add('PET_OCCLUDED')
    if not issues and evidence['occluded'] >= .28 and evidence['occluded'] > max(evidence['visible']+.015,evidence['back']+.01,evidence['motion']+.01):
        issues.add('PET_OCCLUDED')
    if issues:
        return {'usable':False,'issues':sorted(issues),'landmark_strength':strength,'visual_similarities':evidence}
    if at_frame:
        rejection = _border_visibility_rejection(evidence)
        if rejection is not None:
            return {'usable':False,'issues':[], 'landmark_strength':strength,'visual_similarities':evidence}
    if strength >= 1.0:
        return {'usable':True,'issues':[],'landmark_strength':strength,'visual_similarities':evidence}
    usable = strength >= .75 and evidence['visible'] > max(evidence['back']+.01,evidence['blur'],evidence['occluded'])
    # Rotation rescue requires strong measured facial landmarks and matching
    # visual evidence, never the weak .75 threshold used by the old path.
    if not usable:
        for turns in (1,2,3):
            view=np.ascontiguousarray(np.rot90(crop,turns));power,ss,pp=_landmarks(view,with_points=True)
            paired_face = min(ss[:2]) >= 1.1 and ss[2] >= 1.0
            if power < 1.2 and not paired_face: continue
            ev=_visual_evidence(view,species)
            visual_face = ev['visible'] > max(ev['back']+.003,ev['blur']-.005,ev['nonpet']+.02)
            paired_face = paired_face and ev['nonpet'] < ev['visible']-.02 and ev['motion'] < ev['visible']+.025 and ev['occluded'] < ev['visible']+.015
            if (visual_face or paired_face) and not _facial_features_cut_by_frame(pp,ss,box,image.shape,turns):
                usable=True;break
    return {'usable':bool(usable),'issues':[],'landmark_strength':strength,'visual_similarities':evidence}
