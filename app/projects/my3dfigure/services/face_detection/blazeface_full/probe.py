"""Pixel-only supplementary BlazeFace evidence, for local confirmation."""
import cv2
import numpy as np
from . import evidence

def augmented(image, detector, mp):
    h, w = image.shape[:2]
    output = []
    contexts = [(0, 0, w, h, 'full')]
    for fraction, grid in ((0.55, 3), (0.28, 5)):
        tw, th = (round(w * fraction), round(h * fraction))
        if min(tw, th) < 50:
            continue
        for j, y in enumerate(np.linspace(0, h - th, grid).astype(int)):
            for i, x in enumerate(np.linspace(0, w - tw, grid).astype(int)):
                contexts.append((x, y, tw, th, f'{grid}:{i}:{j}'))
    for x, y, tw, th, tag in contexts:
        roi = image[y:y + th, x:x + tw]
        side = round(max(tw, th) * 1.35)
        scale = min(1.0, 768 / side)
        for angle in (-45, 0, 45, 90, 180, 270) if tag != 'full' else (-60, -30, 0, 30, 60, 90, 180, 270):
            m = cv2.getRotationMatrix2D((tw / 2, th / 2), angle, scale)
            m[:, 2] += np.array([side * scale / 2] * 2) - np.array([tw / 2, th / 2])
            view = cv2.warpAffine(roi, m, (round(side * scale), round(side * scale)))
            inv = cv2.invertAffineTransform(m)
            inv[:, 2] += np.array([x, y])
            found = evidence.detect_view(view, detector, mp, inv, f'extra:{tag}:{angle}')
            output.extend(found)
        if tag == 'full':
            for kind, view in [('gray', cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)), ('contrast', cv2.convertScaleAbs(image, alpha=3))]:
                for angle in (0, 45, 90, 180, 270):
                    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
                    m[:, 2] += np.array([side * scale / 2] * 2) - np.array([w / 2, h / 2])
                    transformed = cv2.warpAffine(view, m, (round(side * scale), round(side * scale)))
                    output.extend(evidence.detect_view(transformed, detector, mp, cv2.invertAffineTransform(m), f'photo:{kind}:{angle}'))
    return output

def merge(candidates):
    groups = []
    for c in sorted(candidates, key=lambda c: c['score'], reverse=True):
        match = None
        for g in groups:
            distance = np.mean(np.linalg.norm(np.array(c['points'][:4]) - np.array(g['points'][:4]), axis=1))
            side = min(*c['box'][2:], *g['box'][2:])
            if distance < side * 0.25 or (evidence.overlap(c['box'], g['box']) > 0.5 and distance < side * 0.4):
                match = g
                break
        if match is None:
            groups.append(c)
        else:
            match['views'].extend(c['views'])
    return groups
