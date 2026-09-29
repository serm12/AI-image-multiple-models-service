"""Recover strong color casts using independently normalized channels; source quality is unchanged."""
import cv2, numpy as np
from . import evidence

def collect(image, detector, mp):
    ih, iw = image.shape[:2]
    found = []
    for channel in range(3):
        gray = image[:, :, channel]
        lo, hi = np.percentile(gray, [1, 99])
        if hi - lo < 10:
            continue
        normalized = np.clip((gray.astype(float) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
        view = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        for fraction in (1.0, 0.6):
            tw, th = (round(iw * fraction), round(ih * fraction))
            for y in (0, ih - th) if fraction < 1 else (0,):
                for x in (0, iw - tw) if fraction < 1 else (0,):
                    roi = view[y:y + th, x:x + tw]
                    side = round(max(tw, th) * 1.3)
                    scale = min(1.0, 640 / side)
                    for angle in range(0, 360, 30):
                        m = cv2.getRotationMatrix2D((tw / 2, th / 2), angle, scale)
                        m[:, 2] += np.array([side * scale / 2] * 2) - np.array([tw / 2, th / 2])
                        transformed = cv2.warpAffine(roi, m, (round(side * scale), round(side * scale)))
                        inv = cv2.invertAffineTransform(m)
                        inv[:, 2] += np.array([x, y])
                        for c in evidence.detect_view(transformed, detector, mp, inv, f'channel:{channel}:{fraction}:{x}:{y}:{angle}'):
                            found.append(c)
    return found
