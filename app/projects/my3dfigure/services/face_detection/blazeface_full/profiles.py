"""Mirror and rotate uncertain local views; map eye and ear order back to source coordinates."""
import cv2
import numpy as np
from . import evidence, probe

def collect(image, candidates, det, mp):
    ih, iw = image.shape[:2]
    found = []
    for j, c in enumerate(candidates):
        x, y, w, h = c['box']
        edge = x < 0 or y < 0 or x + w > iw or (y + h > ih)
        if max(c['confirm']) >= 0.6 or min(w, h) < 45:
            continue
        if not (c['score'] >= 0.3 and len(set(c['views'])) >= 2 or (c['score'] >= 0.25 and edge and (min(w, h) >= min(iw, ih) * 0.07))):
            continue
        eye = np.array(c['points'][1]) - np.array(c['points'][0])
        angle = np.degrees(np.arctan2(eye[1], eye[0]))
        for factor in (1.3, 2.0):
            side = max(64, round(max(w, h) * factor))
            scale = min(1.0, 512 / side)
            center = (x + w / 2, y + h / 2)
            for offset in (-60, -30, 0, 30, 60, 180):
                m = cv2.getRotationMatrix2D(center, float(angle + offset), scale)
                m[:, 2] += np.array([side * scale / 2] * 2) - np.array(center)
                view = cv2.warpAffine(image, m, (round(side * scale), round(side * scale)))
                inv = cv2.invertAffineTransform(m)
                flip = np.array([[-1.0, 0, view.shape[1] - 1], [0, 1.0, 0], [0, 0, 1.0]])
                mapped = np.vstack([inv, [0, 0, 1]]) @ flip
                for f in evidence.detect_view(cv2.flip(view, 1), det, mp, mapped[:2], f'profile:{j}:{factor}:{offset}'):
                    if evidence.overlap(f['box'], c['box']) >= 0.3:
                        f['points'] = [f['points'][k] for k in [1, 0, 2, 3, 5, 4]]
                        found.append(f)
    groups = probe.merge(found)
    for c in groups:
        c['features'] = evidence.features(image, c)
        c['confirm'] = evidence.confirm(image, c, det, mp)
    return groups
