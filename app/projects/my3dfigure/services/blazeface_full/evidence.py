"""Independent BlazeFace view collection and source-pixel quality evidence.

Every input follows this same bounded view schedule. Model proposals are mapped
back to the submitted image; padded/rotated pixels never supply quality evidence.
No request mode, image filename, regression label, or YuNet output is consulted.
"""
from __future__ import annotations
import cv2
import numpy as np

def overlap(a, b):
    x, y = (max(a[0], b[0]), max(a[1], b[1]))
    area = max(0, min(a[0] + a[2], b[0] + b[2]) - x) * max(0, min(a[1] + a[3], b[1] + b[3]) - y)
    return area / max(1, min(a[2] * a[3], b[2] * b[3]))

def detect_view(image, detector, mp, matrix, tag):
    h, w = image.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    view = cv2.resize(image, (round(w * scale), round(h * scale))) if scale < 1 else image
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    candidates = []
    for det in result.detections:
        r = det.bounding_box
        x, y, bw, bh = (r.origin_x / scale, r.origin_y / scale, r.width / scale, r.height / scale)
        if bw <= 0 or bh <= 0 or len(det.keypoints) != 6:
            continue
        corners = np.array([[x, y, 1], [x + bw, y, 1], [x, y + bh, 1], [x + bw, y + bh, 1]]) @ matrix.T
        lo, hi = (corners.min(axis=0), corners.max(axis=0))
        points = np.array([[p.x * w, p.y * h, 1] for p in det.keypoints]) @ matrix.T
        if not np.isfinite(corners).all() or not np.isfinite(points).all():
            continue
        candidates.append({'box': [*lo, *hi - lo], 'points': points.tolist(), 'score': float(det.categories[0].score), 'views': [tag]})
    return candidates

def collect(image, detector, mp):
    h, w = image.shape[:2]
    found = detect_view(image, detector, mp, np.array([[1.0, 0, 0], [0, 1.0, 0]]), 'whole:0')
    for k in (1, 2, 3):
        view = np.ascontiguousarray(np.rot90(image, k))
        matrix = {1: [[0, -1, w - 1], [1, 0, 0]], 2: [[-1, 0, w - 1], [0, -1, h - 1]], 3: [[0, 1, 0], [-1, 0, h - 1]]}[k]
        found.extend(detect_view(view, detector, mp, np.array(matrix, dtype=float), f'whole:{k * 90}'))
    grids = [(0.62, 2), (0.4, 3)]
    if not found or max((min(c['box'][2:]) for c in found)) < min(w, h) * 0.2:
        grids.append((0.24, 5))
    for fraction, grid in grids:
        tw, th = (max(32, round(w * fraction)), max(32, round(h * fraction)))
        if min(w, h) < 180:
            continue
        for yi, y in enumerate(np.linspace(0, max(0, h - th), grid).astype(int)):
            for xi, x in enumerate(np.linspace(0, max(0, w - tw), grid).astype(int)):
                found.extend(detect_view(image[y:y + th, x:x + tw], detector, mp, np.array([[1.0, 0, x], [0, 1.0, y]]), f'tile:{grid}:{xi}:{yi}'))
    if not any((c['score'] >= 0.8 for c in found)):
        for angle in (-45, -25, 25, 45):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            side = (round(abs(w * m[0, 0]) + abs(h * m[0, 1])), round(abs(h * m[0, 0]) + abs(w * m[0, 1])))
            m[:, 2] += np.array(side) / 2 - np.array([w, h]) / 2
            view = cv2.warpAffine(image, m, side, borderMode=cv2.BORDER_CONSTANT)
            found.extend(detect_view(view, detector, mp, cv2.invertAffineTransform(m), f'roll:{angle}'))
    groups = []
    for candidate in sorted(found, key=lambda c: c['score'], reverse=True):

        def same_face(c):
            distance = float(np.mean(np.linalg.norm(np.array(c['points'][:4]) - np.array(candidate['points'][:4]), axis=1)))
            side = min(*c['box'][2:], *candidate['box'][2:])
            return distance < side * 0.25 or (overlap(c['box'], candidate['box']) >= 0.45 and distance < side * 0.5)
        match = next((c for c in groups if same_face(c)), None)
        if match is None:
            groups.append(candidate)
        else:
            match['views'].extend(candidate['views'])
    for c in groups:
        c['features'] = features(image, c)
        c['confirm'] = confirm(image, c, detector, mp)
    return groups

def collect_bands(image, detector, mp):
    """Late supplemental views for extreme aspect ratios, preserving base seeds."""
    h,w=image.shape[:2]
    if max(h,w)<=8*min(h,w):
        return []
    tall=h>w
    long_side,short_side=(h,w) if tall else (w,h)
    found=[]
    for factor in (1.5,2.0):
        band=min(long_side,round(short_side*factor))
        steps=max(2,round((long_side-band)/max(1,band*.5))+1)
        for index,start in enumerate(np.linspace(0,long_side-band,steps).astype(int)):
            roi=image[start:start+band,:] if tall else image[:,start:start+band]
            matrix=np.array([[1.,0.,0 if tall else start],[0.,1.,start if tall else 0]])
            found.extend(detect_view(roi,detector,mp,matrix,f'band:{factor}:{index}'))
    return found

def features(image, candidate):
    h, w = image.shape[:2]
    x, y, bw, bh = candidate['box']
    left, top = (min(w, max(0, int(x))), min(h, max(0, int(y))))
    right, bottom = (max(0, min(w, int(np.ceil(x + bw)))), max(0, min(h, int(np.ceil(y + bh)))))
    roi = image[top:bottom, left:right]
    if roi.size == 0:
        return {'visible': 0}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA if min(gray.shape) > 160 else cv2.INTER_CUBIC)
    p = np.percentile(norm, [10, 50, 90])
    core = norm[29:112, 19:141]
    energies = []
    for axis in (0, 1):
        energy = np.abs(np.diff(gray.astype(np.float32), axis=axis)).sum(axis=1 - axis)
        energies.append(float(np.sort(energy)[-max(1, len(energy) // 10):].sum() / max(float(energy.sum()), 1)))
    points = np.array(candidate['points'])
    eyes = points[:2]
    span = np.linalg.norm(eyes[1] - eyes[0])
    unit = (eyes[1] - eyes[0]) / max(span, 1)
    down = np.array([-unit[1], unit[0]])
    if np.dot(points[3] - eyes.mean(axis=0), down) < 0:
        down = -down
    patches = []
    patch_gradients = []
    for px, py in points[:4]:
        radius = max(3, round(min(bw, bh) * 0.09))
        patch = image[max(0, round(py) - radius):min(h, round(py) + radius), max(0, round(px) - radius):min(w, round(px) + radius)]
        if not patch.size:
            patches.append([0.0, 0.0, 0.0])
            patch_gradients.append([0.0, 0.0])
            continue
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patches.append([float(np.percentile(patch, 5)), float(np.percentile(patch, 90) - np.percentile(patch, 10)), float(cv2.Laplacian(patch, cv2.CV_64F).var())])
        patch_gradients.append([float(cv2.Sobel(patch, cv2.CV_64F, 1, 0).var()), float(cv2.Sobel(patch, cv2.CV_64F, 0, 1).var())])
    return {'visible': float(gray.size / max(1, bw * bh)), 'side': float(min(bw, bh)), 'p10': float(p[0]), 'p50': float(p[1]), 'p90': float(p[2]), 'lap': float(cv2.Laplacian(norm, cv2.CV_64F).var()), 'native_lap': float(cv2.Laplacian(gray, cv2.CV_64F).var()), 'sx': float(cv2.Sobel(core, cv2.CV_64F, 1, 0).var()), 'sy': float(cv2.Sobel(core, cv2.CV_64F, 0, 1).var()), 'white': float((norm >= 245).mean()), 'eyes': patches, 'patch_gradients': patch_gradients, 'gray_fraction': float((roi.max(axis=2).astype(float) - roi.min(axis=2) <= 10).mean()), 'core_p90': float(np.percentile(core, 90)), 'core_white': float((core >= 245).mean()), 'block': energies, 'eye_span': float(span / max(bw, bh, 1)), 'mouth_depth': float(np.dot(points[3] - eyes.mean(axis=0), down) / max(span, 1)), 'nose_side': float(np.dot(points[2] - eyes.mean(axis=0), unit) / max(span, 1)), 'points_inside': sum((0 <= px < w and 0 <= py < h for px, py in points[:4]))}

def confirm(image, candidate, detector, mp):
    """Reobserve the same face in upright local contexts, not unrelated boxes."""
    x, y, w, h = candidate['box']
    points = np.array(candidate['points'])
    eye = points[1] - points[0]
    angle = np.degrees(np.arctan2(eye[1], eye[0]))
    scores = []
    for factor in (1.6, 2.3):
        side = max(64, round(max(w, h) * factor))
        center = (x + w / 2, y + h / 2)
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        m[:, 2] += np.array([side / 2, side / 2]) - np.array(center)
        view = cv2.warpAffine(image, m, (side, side), borderMode=cv2.BORDER_CONSTANT)
        detections = detect_view(view, detector, mp, cv2.invertAffineTransform(m), 'confirm')
        matches = [c for c in detections if overlap(c['box'], candidate['box']) >= 0.5 and np.mean(np.linalg.norm(np.array(c['points'][:4]) - np.array(candidate['points'][:4]), axis=1)) < 0.35 * min(*c['box'][2:], *candidate['box'][2:])]
        scores.append(max((c['score'] for c in matches), default=0.0))
    return scores

def quality(c):
    """Reject strong source-pixel defects while preserving readable dark faces.

    Normalized detail, bilateral eye patches and core brightness are combined:
    a single low sharpness value must not reject an otherwise readable face.
    These are heuristic quality checks, not a learned occlusion classifier.
    """
    f = c['features']
    if f['visible'] < 0.4:
        return 'FACE_INCOMPLETE'
    if f['visible'] < 0.75 and f['points_inside'] < 3 and f['side'] < 120 and c['score'] < 0.6 and max(c.get('confirm',[0])) == 0 and len(set(c['views'])) < 5:
        return 'FACE_INCOMPLETE'
    detailed_small = (f['side'] >= 20 and ((c['score'] >= 0.7 and min(c.get('confirm', [0])) >= 0.6) or (c['score'] >= 0.65 and min(c.get('confirm', [0])) >= 0.7))
                       and f['lap'] >= 3
                       and min(p[1] for p in f['eyes']) >= 30
                       and min(p[2] for p in f['eyes'][:2]) >= 40)
    if f['side'] < 29 and not detailed_small and (not (f['side'] >= 24 and c['score'] >= 0.8 and (min(c.get('confirm', [0])) >= 0.6))):
        return 'FACE_TOO_SMALL'
    # Small faces need real pixel detail; enlargement cannot recover it.
    visible_eye_detail = (min(p[1] for p in f['eyes'][:2]) >= 40
                          and min(p[2] for p in f['eyes'][:2]) >= 50)
    readable_small = (f['side'] >= 29 and c['score'] >= 0.7
                      and max(c.get('confirm', [0])) >= 0.6
                      and visible_eye_detail)
    if f['side'] < 40 and f['lap'] < 3 and c['score'] < 0.8 and not readable_small:
        return 'FACE_TOO_SMALL'
    if f['side'] < 40 and f['lap'] < 4 and c['score'] < 0.65 and not visible_eye_detail:
        return 'FACE_TOO_SMALL'
    if (f['side'] < 40 and c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.6
            and not (len(set(c['views'])) >= 3 and max(c.get('confirm',[0])) >= 0.5
                     and min(p[1] for p in f['eyes']) >= 30)):
        return 'FACE_TOO_SMALL'
    if (f['side'] < 100 and f['p50'] < 60 and f['lap'] < 10
            and c['score'] < 0.8 and max(c.get('confirm', [0])) < 0.8
            and min(p[1] for p in f['eyes']) < 15
            and min(p[2] for p in f['eyes'][:2]) < 12):
        return 'FACE_BLURRY'
    # A weak small proposal needs visible detail around both nose and mouth,
    # not just the exposed eyes above a dark obstruction.
    if (f['side'] < 100 and f['p50'] < 60 and f['lap'] < 10
            and c['score'] < 0.8 and max(c.get('confirm', [0])) < 0.8
            and min(p[0] for p in f['eyes'][:2]) >= 20
            and ((f['eyes'][2][0] < 20 and f['eyes'][3][0] < 15 and f['eyes'][3][2] < 20)
                 or (min(c.get('confirm',[0])) < 0.5 and f['eyes'][2][0] <= 25
                     and f['eyes'][3][0] < 20 and f['eyes'][3][2] < 40))):
        return 'FACE_OCCLUDED'
    if f['core_p90'] < 24:
        return 'FACE_TOO_DARK'
    if (f['side'] < 100 and f['p50'] < 35 and f['lap'] < 8
            and c['score'] < 0.65 and max(c.get('confirm',[0])) < 0.8):
        return 'FACE_BLURRY'
    if (c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.7
            and f['side'] >= 120 and min(p[0] for p in f['eyes'][:2]) > 90
            and min(p[1] for p in f['eyes'][:2]) < 25 and f['eyes'][2][0] > 90):
        return 'FACE_OCCLUDED'
    if (f['side'] < 100 and f['p50'] < 45 and f['lap'] < 3
            and c['score'] < 0.8 and max(c.get('confirm',[0])) < 0.8):
        return 'FACE_BLURRY'
    if (f['side'] < 150 and f['p50'] < 25 and f['lap'] < 30
            and c['score'] < 0.7 and max(c.get('confirm',[0])) < 0.7
            and abs(f['nose_side']) < 0.5):
        return 'FACE_TOO_DARK'
    stable_dark_detail = (c['score'] >= 0.8 and max(c.get('confirm', [0])) >= 0.65
                          and len(set(c['views'])) >= 2 and f['native_lap'] >= 2.5)
    readable_dark = f['side'] >= 120 and f['p90'] >= 24 and (f['p90'] - f['p10'] >= 18) and (f['lap'] >= (2.5 if stable_dark_detail else 3)) and (f['p50'] >= 12 or max((e[1] for e in f['eyes'][:2])) >= 12)
    if f['p90'] < 45:
        if not readable_dark:
            return 'FACE_TOO_DARK'
        return None
    if f['p10'] > 240:
        return 'FACE_OVEREXPOSED'
    eyes = f['eyes'][:2]
    # Readability should not disappear merely because a face occupies more
    # pixels after an image export. Require balanced, visible eye detail.
    readable_soft = (c['score'] >= 0.65 and len(set(c['views'])) >= 3
                     and f['lap'] >= 2.5
                     and min(p[1] for p in eyes) >= 25
                     and min(p[1] for p in eyes) >= 0.6 * max(p[1] for p in eyes)
                     and max(p[2] for p in eyes) >= 4
                     and min(p[1] for p in f['eyes']) >= (18 if abs(f['nose_side']) >= 0.5 and max(c.get('confirm',[0])) >= 0.5 else 25))
    if not readable_soft and f['side'] >= 150 and f['lap'] < 6 and (f['sx'] < 450) and (f['sy'] < 450) and (max((p[2] for p in eyes)) < 15) and (max((p[1] for p in eyes)) < 60):
        return 'FACE_BLURRY'
    if not readable_soft and f['side'] >= 150 and f['visible'] >= 0.85 and (max((p[2] for p in f['eyes'])) < 10) and (max((p[1] for p in f['eyes'])) < 55):
        return 'FACE_BLURRY'
    if 45 <= f['side'] < 80 and f['p50'] < 45 and (f['lap'] < 6) and (min((p[1] for p in eyes)) < 20):
        return 'FACE_BLURRY'
    gradients = f.get('patch_gradients', [])
    if not readable_soft and gradients and f['side'] >= 250 and (max((p[2] for p in f['eyes'])) < 100):
        if max(sum((x < 0.3 * y for x, y in gradients)), sum((y < 0.3 * x for x, y in gradients))) >= 3:
            return 'FACE_BLURRY'
    if f['core_white'] > 0.5 and min((e[0] for e in eyes)) > 180:
        return 'FACE_OVEREXPOSED'
    if f['lap'] < 1.8 or (f['lap'] < 6 and max((e[2] for e in eyes)) < 8 and (min((e[1] for e in eyes)) < 20)):
        return 'FACE_BLURRY'
    if not readable_soft and f['side'] >= 200 and f['sx'] < 450 and (f['sy'] < 1000) and (max((e[1] for e in eyes)) < 45):
        return 'FACE_BLURRY'
    if f['side'] >= 80 and min(f['block']) > 0.3 and (max((e[2] for e in eyes)) < 200):
        return 'FACE_UNRECOGNIZABLE'
    if f['side'] >= 150 and f['lap'] >= 300 and (min((e[0] for e in eyes)) > 60) and (max((e[0] for e in eyes)) > 90) and (max((e[1] for e in eyes)) < 100) and (f['side'] >= 250 or f['sx'] > 3000):
        return 'FACE_OCCLUDED'
    return None

def decide(candidates):
    accepted = []
    issues = []
    for c in candidates:
        if c['features']['visible'] < 0.4:
            issues.append('FACE_INCOMPLETE')
            continue
        views = set(c['views'])
        confirmed = max(c.get('confirm', [0])) >= 0.6
        strong = c['score'] >= 0.8 and 'whole:0' in views
        original_views = {v for v in views if v.startswith(('whole:', 'tile:', 'roll:'))}
        primary = c is candidates[0] and c['score'] >= 0.55 and (len(original_views) >= 2)
        profile_confirmed = len(views) >= 3 and max(c.get('confirm', [0])) >= 0.5
        if not primary and (not profile_confirmed) and (c['features']['eye_span'] < 0.1 or abs(c['features']['nose_side']) > 1):
            continue
        if c['features']['eye_span'] < 0.08 and max(c.get('confirm', [0])) < 0.6:
            continue
        supplemental = c['score'] >= 0.5 and len(views) >= 6 and (max(c.get('confirm', [0])) >= 0.5)
        readable_pair = original_views and c['score'] >= 0.6 and (len(views) >= 2) and (min((p[1] for p in c['features']['eyes'][:2])) >= 50 or c['score'] >= 0.7) and (max(c.get('confirm', [0])) >= 0.5 or (c['score'] >= 0.65 and len(views) >= 3))
        if not (confirmed or strong or primary or supplemental or readable_pair or (c['score'] >= 0.65 and len(views) >= 3 and (max(c.get('confirm', [0])) >= 0.5))):
            continue
        q = quality(c)
        if q:
            if q == 'FACE_BLURRY' and max(c.get('confirm', [0])) == 0 and c['score'] < 0.8 and len(views) < 5:
                continue
            issues.append(q)
        elif not any((np.mean(np.linalg.norm(np.array(c['points'][:4]) - np.array(a['points'][:4]), axis=1)) < 0.25 * min(*c['box'][2:], *a['box'][2:]) for a in accepted)):
            accepted.append(c)
    reason = issues[0] if issues and len(set(issues)) == 1 else 'NO_FACE'
    return (accepted, reason)
