"""Re-detect local candidates and retain the actual confirmed boxes, never a nearby seed box."""
import cv2
import numpy as np
from . import evidence, probe

def collect(image, candidates, det, mp):
    found = []
    for j, c in enumerate(candidates):
        if max(c['confirm']) < 0.5 or c['score'] < 0.25:
            continue
        x, y, w, h = c['box']
        points = np.array(c['points'])
        eye = points[1] - points[0]
        angle = np.degrees(np.arctan2(eye[1], eye[0]))
        for factor in (1.6, 2.3):
            side = max(64, round(max(w, h) * factor))
            scale = min(1.0, 768 / side)
            center = (x + w / 2, y + h / 2)
            m = cv2.getRotationMatrix2D(center, float(angle), scale)
            m[:, 2] += np.array([side * scale / 2] * 2) - np.array(center)
            view = cv2.warpAffine(image, m, (round(side * scale), round(side * scale)))
            for f in evidence.detect_view(view, det, mp, cv2.invertAffineTransform(m), f'canonical:{j}:{factor}'):
                if evidence.overlap(f['box'], c['box']) >= 0.3:
                    found.append(f)
    groups = probe.merge(found)
    for c in groups:
        c['features'] = evidence.features(image, c)
        c['confirm'] = evidence.confirm(image, c, det, mp)
    return groups
