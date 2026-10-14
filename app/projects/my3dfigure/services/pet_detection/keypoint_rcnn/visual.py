"""Private CLIP species/visibility evidence; no detector fallback."""
from pathlib import Path
import json
import threading
import numpy as np
import cv2
from PIL import Image, ImageOps

from app.projects.my3dfigure.core.config import project_getenv

MODEL_DIRECTORY = Path(project_getenv(
    "PET_KEYPOINT_CLIP_MODEL_DIR", Path(__file__).resolve().parents[6] / "models/pet_detection"
))
TEXT_FEATURE_DIRECTORY = MODEL_DIRECTORY / 'clip'
_local = threading.local()
_lock = threading.Lock()
_visual = None
_features = None
_labels = None


def _visual_models():
    global _visual, _features, _labels
    if _visual is None:
        with _lock:
            if _visual is None:
                import onnxruntime as ort
                options = ort.SessionOptions()
                # Session-local threads only; never change process-wide thread settings.
                options.intra_op_num_threads = 4
                options.inter_op_num_threads = 1
                features = np.load(TEXT_FEATURE_DIRECTORY / 'text-features-v4.npz', allow_pickle=False)['features']
                labels = json.loads((TEXT_FEATURE_DIRECTORY / 'prompts-v4.json').read_text(encoding='utf8'))['labels']
                if features.shape != (len(labels), 512) or not np.all(np.isfinite(features)):
                    raise RuntimeError('Invalid pet visibility text features')
                session = ort.InferenceSession(str(MODEL_DIRECTORY / 'clip/vision_model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
                _features, _labels, _visual = features, labels, session
    return _visual, _features, _labels


def species_evidence(crop):
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
    return {species: {label: float(max(similarities[i] for i, item in enumerate(labels) if item == [species, label]))
                     for label in ('visible', 'occluded', 'back', 'fur', 'blur', 'nonpet', 'cut', 'dark', 'bright', 'motion')}
            for species in ('cat', 'dog')}
