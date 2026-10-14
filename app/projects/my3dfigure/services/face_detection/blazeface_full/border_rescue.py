"""Confirm difficult border profiles across local contexts and photometric views."""
import cv2, numpy as np
from . import evidence, probe

def collect(image, candidates, det, mp):
    ih, iw = image.shape[:2]
    found = []
    seeds = [c for c in candidates if c['score'] >= 0.25 and max(c['confirm']) < 0.5 and min(c['box'][2:]) >= 32 and (c['box'][0] < max(0.05 * iw,0.5*c['box'][2]) or c['box'][0] + c['box'][2] > iw-max(0.05 * iw,0.5*c['box'][2]))]
    for j, c in enumerate(seeds):
        x, y, w, h = c['box']
        center = (x + w / 2, y + h / 2)
        for factor in (1.3, 1.8, 2.5):
            side = max(64, round(max(w, h) * factor))
            scale = min(1.0, 512 / side)
            for angle in range(-90, 91, 30):
                m = cv2.getRotationMatrix2D(center, angle, scale)
                m[:, 2] += np.array([side * scale / 2] * 2) - np.array(center)
                v = cv2.warpAffine(image, m, (round(side * scale), round(side * scale)), borderMode=cv2.BORDER_REPLICATE)
                inv = cv2.invertAffineTransform(m)
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
                for kind, view in [('rgb', v), ('gray', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), ('contrast', cv2.convertScaleAbs(v, alpha=1.8, beta=-30))]:
                    for flip in (False, True):
                        matrix = inv if not flip else (np.vstack([inv, [0, 0, 1]]) @ np.array([[-1.0, 0, view.shape[1] - 1], [0, 1.0, 0], [0, 0, 1.0]]))[:2]
                        for f in evidence.detect_view(cv2.flip(view, 1) if flip else view, det, mp, matrix, f'border:{j}:{factor}:{angle}:{kind}:{flip}'):
                            if flip:
                                f['points'] = [f['points'][k] for k in [1, 0, 2, 3, 5, 4]]
                            if evidence.overlap(c['box'], f['box']) > 0.4:
                                found.append(f)
    groups = probe.merge(found)
    for j, c in enumerate(groups):
        if min(c['box'][2:]) < 45:
            continue
        x, y, w, h = c['box']
        center = (x + w / 2, y + h / 2)
        if not (x < w or x + w > iw-w):
            continue
        for factor in (1.6, 2.0, 2.4, 2.8, 3.2):
            side = max(64, round(max(w, h) * factor))
            scale = min(1.0, 512 / side)
            for angle in range(-75, 76, 15):
                m = cv2.getRotationMatrix2D(center, angle, scale)
                m[:, 2] += np.array([side * scale / 2] * 2) - np.array(center)
                v = cv2.warpAffine(image, m, (round(side * scale), round(side * scale)), borderMode=cv2.BORDER_REPLICATE)
                inv = cv2.invertAffineTransform(m)
                for alpha in (1.4, 1.8, 2.2):
                    view = cv2.convertScaleAbs(v, alpha=alpha, beta=-30)
                    for flip in (False, True):
                        matrix = inv if not flip else (np.vstack([inv, [0, 0, 1]]) @ np.array([[-1.0, 0, view.shape[1] - 1], [0, 1.0, 0], [0, 0, 1.0]]))[:2]
                        for f in evidence.detect_view(cv2.flip(view, 1) if flip else view, det, mp, matrix, f'border-refine:{j}:{factor}:{angle}:{alpha}:{flip}'):
                            if flip:
                                f['points'] = [f['points'][k] for k in [1, 0, 2, 3, 5, 4]]
                            if evidence.overlap(c['box'], f['box']) > 0.5:
                                found.append(f)
    return probe.merge(found)
