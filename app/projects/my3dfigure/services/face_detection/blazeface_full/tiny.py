"""Collect small native-pixel faces for diagnostics only, never usable counts."""
import cv2, numpy as np
from . import evidence

def collect(image, detector, mp):
    h, w = image.shape[:2]
    if min(h, w) < 29:
        return []
    found = []
    for fraction, grid in ((0.08, 14), (0.12, 10), (0.18, 7)):
        tw, th = (round(w * fraction), round(h * fraction))
        for yi, y in enumerate(np.linspace(0, h - th, grid).astype(int)):
            for xi, x in enumerate(np.linspace(0, w - tw, grid).astype(int)):
                roi = image[y:y + th, x:x + tw]
                for c in evidence.detect_view(roi, detector, mp, np.array([[1.0, 0, x], [0, 1.0, y]]), f'tiny:{grid}:{xi}:{yi}'):
                    if min(c['box'][2:]) < 29:
                        found.append(c)
    return found
