"""Collect border candidates with replicated padding; quality always uses original pixels."""
import cv2, numpy as np
from . import evidence

def collect(image, detector, mp):
    ih, iw = image.shape[:2]
    found = []
    for fraction in (0.25, 0.4, 0.6):
        tw, th = (round(iw * fraction), round(ih * fraction))
        for y in np.linspace(0, ih - th, 4).astype(int):
            for x in (0, iw - tw):
                roi = image[y:y + th, x:x + tw]
                side = round(max(tw, th) * 1.7)
                scale = min(1.0, 640 / side)
                for angle in (-60, -30, 0, 30, 60, 90, 180, 270):
                    m = cv2.getRotationMatrix2D((tw / 2, th / 2), angle, scale)
                    m[:, 2] += np.array([side * scale / 2] * 2) - np.array([tw / 2, th / 2])
                    view = cv2.warpAffine(roi, m, (round(side * scale), round(side * scale)), borderMode=cv2.BORDER_REPLICATE)
                    inv = cv2.invertAffineTransform(m)
                    inv[:, 2] += np.array([x, y])
                    for flip in (False, True):
                        matrix = inv
                        if flip:
                            matrix = (np.vstack([inv, [0, 0, 1]]) @ np.array([[-1.0, 0, view.shape[1] - 1], [0, 1.0, 0], [0, 0, 1.0]]))[:2]
                        for c in evidence.detect_view(cv2.flip(view, 1) if flip else view, detector, mp, matrix, f'edge:{fraction}:{x}:{y}:{angle}:{flip}'):
                            if flip:
                                c['points'] = [c['points'][k] for k in (1, 0, 2, 3, 5, 4)]
                            bx, by, bw, bh = c['box']
                            if bx < iw * 0.12 or bx + bw > iw * 0.88:
                                found.append(c)
    return found
