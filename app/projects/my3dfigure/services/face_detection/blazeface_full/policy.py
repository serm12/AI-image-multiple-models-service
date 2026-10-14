"""Qualify source-pixel face evidence before applying a product target count."""
import numpy as np
import math
from . import evidence

def diagnosis(row, reason):
    area = row['shape'][0] * row['shape'][1]
    candidates = row.get('diagnostics', [])
    if any((0.5 <= c['score'] < 0.8 and c['features'].get('eyes') and max((p[0] for p in c['features'].get('eyes', [])[:2])) < 15 and (max((p[1] for p in c['features']['eyes'][:2])) < 12) for c in row['candidates'])):
        return 'NO_FACE'
    if reason != 'NO_FACE':
        return reason
    for c in candidates:
        f = c['features']
        ratio = c['box'][2] * c['box'][3] / area
        if c['score'] >= 0.3 and len(set(c['views'])) >= 2 and (ratio >= 0.01) and (f.get('white', 0) > 0.35) and (f.get('p90', 0) >= 248):
            return 'FACE_OVEREXPOSED'
    for c in candidates:
        f = c['features']
        ratio = c['box'][2] * c['box'][3] / area
        if c['score'] >= 0.6 and len(set(c['views'])) >= 6 and (ratio >= 0.008) and (f.get('side', 0) >= 80):
            if max(f['block']) > 0.3 and 10 <= min((e[1] for e in f['eyes'][:2])) and max((e[1] for e in f['eyes'][:2])) < 40 and (min((e[2] for e in f['eyes'][:2])) < 30):
                return 'FACE_BLURRY'
    profiles = [c for c in candidates if c['score'] >= 0.4 and len(set(c['views'])) >= 8 and (c['box'][2] * c['box'][3] / area >= 0.02) and (c['features'].get('side', 0) >= 80) and (c['features'].get('eye_span', 1) < 0.2) and (0.35 < abs(c['features'].get('nose_side', 0)) < 1)]
    if any((a['features']['nose_side'] * b['features']['nose_side'] < 0 and evidence.overlap(a['box'], b['box']) == 0 for a in profiles for b in profiles)):
        return 'FACE_NOT_FRONTAL'
    for c in candidates:
        f = c['features']
        ratio = c['box'][2] * c['box'][3] / area
        if 'eyes' not in f:
            continue
        skin = min((p[0] for p in f['eyes'][:2])) > 80 and max((p[1] for p in f['eyes'][:2])) < 60
        detail = min((p[0] for p in f['eyes'][:2])) > 30 and min((p[2] for p in f['eyes'][:2])) > 300 and min((p[1] for p in f['eyes'][:2])) > 80
        if c['score'] >= 0.35 and len(set(c['views'])) >= 3 and (0.02 <= ratio <= 0.5) and (f.get('eye_span', 1) < 0.2) and (abs(f.get('nose_side', 0)) > 0.6) and (skin or detail):
            return 'FACE_NOT_FRONTAL'
    return reason

def same_face(a, b):
    p,q=a['points'],b['points']
    distance = min(math.dist(p[0],q[0])+math.dist(p[1],q[1]),math.dist(p[0],q[1])+math.dist(p[1],q[0]))/2
    overlap = evidence.overlap(a['box'], b['box'])
    return overlap > 0.2 and distance < (0.4 if overlap > 0.75 else 0.25) * min(*a['box'][2:], *b['box'][2:])

def same_output(a,b):
    if same_face(a,b):return True
    overlap = evidence.overlap(a['box'], b['box'])
    side = min(*a['box'][2:], *b['box'][2:])
    low=min((a,b),key=lambda c:c['score'])
    if overlap > 0.7 and (min(low['confirm']) < 0.6 or max(low['confirm']) < 0.7):
        ap,bp=np.array(a['points']),np.array(b['points'])
        ac=np.array(a['box'][:2])+np.array(a['box'][2:])/2
        bc=np.array(b['box'][:2])+np.array(b['box'][2:])/2
        u=ap[3]-ap[:2].mean(axis=0);v=bp[3]-bp[:2].mean(axis=0)
        cosine=np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-6)
        if (np.linalg.norm(ac-bc) < 0.4*side and cosine > 0.25
                and max(np.linalg.norm(ap[2]-bp[2]),np.linalg.norm(ap[3]-bp[3])) < 0.4*side
                and a['features']['nose_side']*b['features']['nose_side'] >= -0.15):
            return True
    large,small=sorted((a,b),key=lambda c:min(c['box'][2:]),reverse=True)
    if (overlap > 0.85 and min(large['box'][2:]) > 2*min(small['box'][2:])
            and max(large['confirm']) == 0 and min(small['confirm']) >= 0.6
            and large['features'].get('eye_span',1) < 0.15
            and math.dist(large['points'][2],small['points'][2]) < 0.5*side
            and math.dist(large['points'][3],small['points'][3]) < 0.7*side):
        return True
    return False

def select(row):

    def unsupported(c):
        f = c['features']
        x,y,w,h=c['box']
        if (x < 0 and y < 0 and f.get('visible',1) < 0.65
                and c['score'] < 0.8 and max(c['confirm']) < 0.5):
            return True
        # Flat monochrome UI marks are not facial detail.
        if c['score'] < 0.8 and f.get('gray_fraction', 0) > 0.98 and min(p[1] for p in f.get('eyes', [[0, 255]])) <= 1 and f.get('white', 0) > 0.03:
            return True
        # Detailed texture with no local human-face confirmation is unreliable.
        detailed_profile = (len(f.get('eyes', [])) >= 4
                            and min(p[1] for p in f['eyes']) >= 50
                            and min(p[2] for p in f['eyes']) >= 100)
        span = f.get('eye_span', 1)
        bad_geometry = span < 0.15 and abs(f.get('nose_side', 0)) > 2 and (span > 0.1 or not detailed_profile)
        if c['score'] < 0.7 and max(c['confirm']) < 0.6 and f.get('lap',0) > 150:
            return True
        return c['score'] < 0.8 and ((max(c['confirm']) == 0 and f.get('lap', 0) > 150)
                                  or (max(c['confirm']) == 0 and bad_geometry)
                                  or (c['score'] < 0.7 and max(c['confirm']) < 0.6 and span < 0.08 and abs(f.get('nose_side',0)) > 4 and not detailed_profile))
    base = [c for c in row['candidates'] if not unsupported(c)]
    accepted, reason = evidence.decide(base)
    for c in base:
        f=c['features']
        if (c not in accepted and c['score'] >= 0.6 and len(set(c['views'])) >= 3
                and 0.5 <= abs(f.get('nose_side',0)) < 1.5 and f.get('side',0) >= 40
                and 0.15 <= f.get('eye_span',0) <= 0.4
                and f.get('p50',0) >= 60 and 2.5 <= f.get('lap',0) < 30
                and min(p[1] for p in f.get('eyes',[[0,0]])) >= 18
                and min(p[1] for p in f['eyes'][:2]) >= 40
                and min(p[2] for p in f['eyes'][:2]) >= 30
                and any(d['score'] >= 0.7 and len(set(d['views'])) >= 12 and same_face(c,d) for d in row.get('diagnostics',[]))
                and not evidence.quality(c) and not any(same_face(c,a) for a in accepted)):
            accepted.append(c)
    canonical = []
    for c in row.get('supplements', []) + row.get('profiles', []) + row.get('edges', []) + row.get('border', []) + row.get('tonal',[]):
        f = c['features']
        # Closely facing people can produce one strong box spanning both
        # profiles. A second, geometrically distinct local face still needs
        # repeated model views and real source-pixel landmark detail.
        border_companion = (c in row.get('border',[])
            and c['score'] >= 0.675 and len(set(c['views'])) >= 20
            and f.get('side',0) >= 180 and 0.18 <= f.get('eye_span',0) <= 0.35
            and abs(f.get('nose_side',0)) < 1.5
            and 0.7 <= f.get('mouth_depth',0) <= 2.3
            and min(p[1] for p in f['eyes']) >= 28
            and max(p[2] for p in f['eyes']) >= 30
            and not evidence.quality(c)
            and any(a['score'] >= 0.75 and max(a['confirm']) >= 0.5
                and 0.3 < evidence.overlap(c['box'],a['box']) < 0.8
                and not same_output(c,a) for a in accepted))
        if c in row.get('tonal',[]) and not (c['score'] >= 0.65 and len(set(c['views'])) >= 3 and min(c['confirm']) >= 0.6 and min(p[1] for p in f.get('eyes',[[0,0]])) >= 40):
            continue
        if c in row.get('border',[]) and not (border_companion or (min(c['confirm']) >= 0.6 and (c['score'] >= 0.8 or (c['score'] >= 0.65 and max(c['confirm']) >= 0.7 and len(set(c['views'])) >= 6)))):
            continue
        if c in row.get('border',[]) and any(same_output(c,b) for b in accepted + canonical):
            continue
        independent_local = (c['score'] >= 0.8 and len(set(c['views'])) >= 3
                              and 0.12 <= f['eye_span'] <= 0.55
                              and abs(f['nose_side']) < 1.8 and 0.4 < f['mouth_depth'] < 3
                              and min(p[1] for p in f['eyes']) >= 20
                              and max(p[2] for p in f['eyes'][:2]) >= 15
                              and (len(set(c['views'])) >= 6 or any(
                                  d['score'] >= 0.7 and len(set(d['views'])) >= 8 and same_face(c,d)
                                  for d in row.get('diagnostics', []))))
        independent_local = independent_local or (
            c['score'] >= 0.7 and len(set(c['views'])) >= 8 and f.get('side',0) >= 80
            and 0.5 <= abs(f['nose_side']) < 2 and 0.15 <= f['eye_span'] <= 0.4
            and min(p[1] for p in f['eyes']) >= 18
            and any(d['score'] >= 0.6 and max(d['confirm']) >= 0.5
                    and len(set(d['views'])) >= 8 and same_face(c,d)
                    for d in row.get('diagnostics',[])))
        independent_local = independent_local or (
            c['score'] >= 0.6 and len(set(c['views'])) >= 6
            and 0.5 <= abs(f.get('nose_side',0)) < 1.5 and 0.15 <= f.get('eye_span',0) <= 0.35
            and 2.5 <= f.get('lap',0) < 30 and f.get('p50',0) >= 60
            and min(p[1] for p in f['eyes']) >= 25
            and min(p[1] for p in f['eyes'][:2]) >= 40
            and any(d['score'] >= 0.6 and len(set(d['views'])) >= 12 and same_face(c,d)
                    for d in row.get('diagnostics',[]))
            and any(b['score'] >= 0.5 and len(set(b['views'])) >= 2 and same_face(c,b)
                    and not evidence.quality(b) for b in base))
        independent_local = independent_local or (
            c['score'] >= 0.6 and len(set(c['views'])) >= 6 and f['side'] >= 120
            and max(c['confirm']) >= 0.5
            and 0.5 <= abs(f['nose_side']) < 1.5 and 0.15 <= f['eye_span'] <= 0.35
            and min(p[1] for p in f['eyes']) >= 30 and max(p[2] for p in f['eyes'][:2]) >= 10
            and any(d['score'] >= 0.5 and len(set(d['views'])) >= 8 and same_face(c,d)
                    for d in row.get('diagnostics',[])))
        independent_local = independent_local or (
            c['score'] >= 0.55 and len(set(c['views'])) >= 3 and 40 <= f['side'] < 120
            and abs(f['nose_side']) < 0.5 and 0.25 <= f['eye_span'] <= 0.5
            and 0.8 <= f['mouth_depth'] < 1.6 and f['p50'] >= 60
            and min(p[1] for p in f['eyes']) >= 20
            and min(p[1] for p in f['eyes'][:2]) >= 50 and min(p[2] for p in f['eyes'][:2]) >= 20
            and any(b['score'] >= 0.6 and min(p[1] for p in b['features']['eyes'][:2]) >= 50 and same_face(c,b) for b in base)
            and any(d['score'] >= 0.7 and len(set(d['views'])) >= 20 and same_face(c,d) for d in row.get('diagnostics',[])))
        if min(c['box'][2:]) < 45 and c['score'] < 0.8 and all((v.startswith('edge:') for v in c['views'])):
            continue
        independent_local = independent_local or (
            c['score'] >= 0.65 and len(set(c['views'])) >= 6 and f['side'] >= 150
            and max(c['confirm']) >= 0.5 and 3 <= f['lap'] < 6 and f['native_lap'] >= 2
            and min(p[1] for p in f['eyes']) >= 20
            and min(p[1] for p in f['eyes'][:2]) >= 40
            and min(p[1] for p in f['eyes'][:2]) >= 0.6*max(p[1] for p in f['eyes'][:2])
            and any(d['score'] >= 0.6 and len(set(d['views'])) >= 8 and same_face(c,d) for d in row.get('diagnostics',[])))
        independent_local = independent_local or border_companion
        readable_alternative = (c['score'] >= 0.8 and min(c['confirm']) >= 0.55
                                and max(c['confirm']) >= 0.65
                                and min(p[1] for p in c['features']['eyes'][:2]) >= 18
                                and max(p[2] for p in c['features']['eyes'][:2]) >= 12)
        readable_alternative = readable_alternative or (c['score'] >= 0.65 and len(set(c['views'])) >= 4
            and min(c['confirm']) >= 0.6 and max(c['confirm']) >= 0.7
            and min(p[1] for p in f['eyes']) >= 25 and min(p[2] for p in f['eyes'][:2]) >= 30)
        dark_alternative = (min(c['confirm']) >= 0.6 and max(c['confirm']) >= 0.7
                            and min(p[1] for p in c['features']['eyes']) >= 8
                            and max(p[1] for p in c['features']['eyes'][:2]) >= 18)
        vetoes = {'FACE_UNRECOGNIZABLE'}
        if not readable_alternative: vetoes.add('FACE_BLURRY')
        if not dark_alternative: vetoes.add('FACE_TOO_DARK')
        if any((b['score'] >= 0.7 and len(set(b['views'])) >= 2 and evidence.quality(b) in vetoes and same_face(c, b) for b in base)):
            continue
        if not (c['score'] >= 0.8 and min(c['confirm']) >= 0.55 and max(c['confirm']) >= 0.65 and f['side'] >= 40) and any(29 <= b['features']['side'] < 40 and b['features']['lap'] < 3 and 0.7 <= b['score'] < 0.8 and max(b['confirm']) >= 0.6 and evidence.quality(b) == 'FACE_TOO_SMALL' and same_face(c, b) for b in base):
            continue
        views = set(c['views'])
        edge_confirmed = 0.4 <= c['features']['visible'] < 0.65 and max(c['confirm']) >= 0.5
        context_confirmed = (f.get('side', 0) >= 80 and abs(f.get('nose_side',0)) >= 0.5 and max(c['confirm']) >= 0.5 and any(
            d['score'] >= 0.65 and len(set(d['views'])) >= 8 and max(d['confirm']) >= 0.5
            and same_face(c,d) and not evidence.quality(d) for d in row.get('diagnostics', [])))
        context_confirmed = context_confirmed or (c['score'] >= 0.65 and f.get('side',0) >= 40
            and abs(f.get('nose_side',0)) >= 1 and min(p[1] for p in f.get('eyes',[[0,0]])) >= 20
            and any(d['score'] >= 0.65 and len(set(d['views'])) >= 20 and max(d['confirm']) >= 0.55
                    and same_face(c,d) and not evidence.quality(d) for d in row.get('diagnostics', [])))
        if unsupported(c) or (len(views) < 2 and max(c['confirm']) < 0.7 and not edge_confirmed and not context_confirmed):
            continue
        if c['features']['visible'] >= 0.65:
            if max(c['confirm']) < 0.5 and not independent_local:
                continue
            supported = any((d['score'] >= 0.65 and len(set(d['views'])) >= 5 and same_face(c, d) for d in row.get('diagnostics', [])))
            supported = supported or (c['score'] >= 0.8 and len(views) >= 6)
            supported = supported or independent_local
            supported = supported or (len(views) >= 12 and c['score'] >= 0.7 and c['features']['side'] < 60 and any(d['score'] >= 0.6 and len(set(d['views'])) >= 8 and same_face(c, d) for d in row.get('diagnostics', [])))
            if max(c['confirm']) < 0.6 and min(c['confirm']) < 0.5 and (not supported):
                continue
        if evidence.quality(c):
            continue
        canonical.append(c)
    for c in row.get('diagnostics', []):
        f = c['features']
        if not f.get('eyes'):
            continue
        if (c['score'] >= 0.5 and len(set(c['views'])) >= 6
                and min(c['confirm']) >= 0.6 and max(c['confirm']) >= 0.7
                and min(p[1] for p in f['eyes']) >= 25 and f['native_lap'] >= 30
                and not any(evidence.overlap(c['box'],b['box']) > 0.3 for b in accepted + canonical)
                and not any(evidence.quality(b) and same_face(c,b) for b in base)
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
        ih, iw = row['shape'][:2]
        # A lower image edge may hide the mouth while preserving the two
        # eyes and the nose bridge. Confirm repeated geometry on source pixels.
        eyes_inside = all(0 <= px < iw and 0 <= py < ih for px,py in c['points'][:2])
        nx,ny = c['points'][2]
        partial_lower = (0.4 <= f['visible'] < 0.7 and f['side'] >= 80
                         and c['score'] >= 0.4 and len(set(c['views'])) >= 3
                         and eyes_inside and 0 <= nx < iw and ih - 0.2*f['side'] <= ny <= ih + 0.3*f['side']
                         and c['points'][3][1] > ih
                         and min(p[1] for p in f['eyes'][:2]) >= 20
                         and min(p[2] for p in f['eyes'][:2]) >= 8
                         and (f['lap'] >= 20 or len(set(c['views'])) >= 8 or any(
                             p['score'] >= 0.5 and same_face(c,p) for p in row.get('profiles', []))))
        if partial_lower and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        # Consistent independently transformed observations can support a face
        # whose aligned local confirmation is unstable.
        native = [b for b in base if same_face(c, b)]
        soft_profile = (f['side'] >= 120 and c['score'] >= 0.7
                        and len(set(c['views'])) >= 12 and max(c['confirm']) >= 0.5
                        and abs(f['nose_side']) >= 0.5
                        and min(p[1] for p in f['eyes'][:2]) >= 25
                        and min(p[1] for p in f['eyes']) >= 18
                        and all(evidence.quality(b) in (None,'FACE_BLURRY') for b in native))
        if soft_profile and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        repeated_profile = (c['score'] >= 0.65 and len(set(c['views'])) >= 8
                            and max(c['confirm']) >= 0.5 and f['side'] >= 80
                            and 0.5 <= abs(f['nose_side']) < 2
                            and 0.15 <= f['eye_span'] <= 0.4
                            and min(p[1] for p in f['eyes']) >= 20
                            and not any(evidence.quality(b) for b in native))
        if repeated_profile and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        visible_profile = (c['score'] >= 0.6 and len(set(c['views'])) >= 20
                           and 0.5 <= abs(f['nose_side']) < 1.5
                           and 0.15 <= f['eye_span'] <= 0.4 and f['visible'] >= 0.85
                           and 2.5 <= f['lap'] < 30 and f['p50'] >= 60
                           and min(p[1] for p in f['eyes']) >= 20
                           and max(p[1] for p in f['eyes'][:2]) >= 50
                           and max(p[2] for p in f['eyes'][:2]) >= 50
                           and min(p[1] for p in f['eyes'][2:]) >= 25
                           and any(b['score'] >= 0.5 and len(set(b['views'])) >= 2 for b in native)
                           and not any(evidence.quality(b) for b in native))
        if visible_profile and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        if (c['score'] >= 0.65 and len(set(c['views'])) >= 20 and min(c['confirm']) >= 0.6
                and f['side'] >= 40 and min(p[1] for p in f['eyes']) >= 25
                and min(p[1] for p in f['eyes'][:2]) >= 35
                and not any(same_face(c,b) for b in accepted)
                and not any(evidence.quality(b) not in (None,'FACE_TOO_SMALL') for b in native)
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
        eye_detail = (min(p[1] for p in f['eyes'][:2]) >= 12
                      or (abs(f['nose_side']) >= 0.5
                          and max(p[1] for p in f['eyes'][:2]) >= 25
                          and max(p[2] for p in f['eyes'][:2]) >= 15
                          and min(p[1] for p in f['eyes'][2:]) >= 25))
        consensus = (c['score'] >= 0.65 and len(set(c['views'])) >= 8
                     and f['visible'] >= 0.85 and f['points_inside'] >= 4
                     and 0.18 <= f['eye_span'] <= (0.5 if abs(f['nose_side']) >= 0.5 or max(c['confirm']) >= 0.6 else 0.35)
                     and abs(f['nose_side']) < 1.5 and 0.4 < f['mouth_depth'] < 3
                     and eye_detail
                     and (max(c['confirm']) >= 0.6 or (f['eyes'][2][1] >= 12 and (f['p50'] >= 60 or abs(f['nose_side']) >= 0.5)))
                     and f['lap'] < 20
                     and not any(evidence.quality(b) for b in native)
                     and (any(b['score'] >= 0.6 and len(set(b['views'])) >= 2 for b in native)
                          or (c['score'] >= 0.7 and len(set(c['views'])) >= 12
                              and (abs(f['nose_side']) >= 0.5 or max(c['confirm']) >= 0.6)
                              and any(b['score'] >= 0.5 for b in native))))
        if consensus and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        confirmed_diagnostic = (c['score'] >= 0.8 and len(set(c['views'])) >= 8
                                and min(c['confirm']) >= 0.6 and max(c['confirm']) >= 0.75
                                and not any(evidence.quality(b) for b in native))
        if confirmed_diagnostic and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        readable_diagnostic = (c['score'] >= 0.8 and len(set(c['views'])) >= 12
                               and min(c['confirm']) >= 0.6 and max(c['confirm']) >= 0.65
                               and min(p[1] for p in f['eyes'][:2]) >= 0.6*max(p[1] for p in f['eyes'][:2])
                               and min(p[1] for p in f['eyes']) >= 25
                               and max(p[2] for p in f['eyes'][:2]) >= 20
                               and all(evidence.quality(b) in (None,'FACE_BLURRY','FACE_TOO_SMALL') for b in native))
        side_diagnostic = (c['score'] >= 0.7 and len(set(c['views'])) >= 20
                           and max(c['confirm']) >= 0.55 and f['side'] >= 40
                           and 0.08 <= f['eye_span'] < 0.18 and 1 <= abs(f['nose_side']) < 2
                           and min(p[1] for p in f['eyes']) >= 25
                           and min(p[1] for p in f['eyes'][:2]) >= 35
                           and all(evidence.quality(b) in (None,'FACE_TOO_SMALL') for b in native))
        if (readable_diagnostic or side_diagnostic) and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        local_profile_consensus = (c['score'] >= 0.6 and len(set(c['views'])) >= 12
            and f['side'] >= 120 and 0.5 <= abs(f['nose_side']) < 1.5
            and 0.15 <= f['eye_span'] <= 0.35
            and min(p[1] for p in f['eyes']) >= 25
            and max(p[2] for p in f['eyes'][:2]) >= 10
            and any(d['score'] >= 0.55 and len(set(d['views'])) >= 3
                    and d['features'].get('nose_side',0)*f['nose_side'] > 0.25
                    and same_face(c,d) and not evidence.quality(d)
                    for name in ('profiles','edges','border') for d in row.get(name,[])))
        if local_profile_consensus and not any(same_face(c,b) for b in accepted) and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        if (c['score'] >= 0.9 and len(set(c['views'])) >= 8
                and min(c['confirm']) >= 0.7 and max(c['confirm']) >= 0.8
                and min(p[1] for p in f['eyes'][:2]) >= 25
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
        if (c['score'] >= 0.8 and len(set(c['views'])) >= 8
                and min(c['confirm']) >= 0.6 and f['side'] < 80
                and all(evidence.quality(b) in (None, 'FACE_TOO_SMALL') for b in native)
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
        if 0.4 <= f['visible'] < 0.65 and c['score'] >= 0.5 and (len(set(c['views'])) >= 5) and (max((p[1] for p in f['eyes'][:2])) >= 30) and (not unsupported(c)) and (not evidence.quality(c)):
            canonical.append(c)
    for c in row.get('chromatic', []):
        if c['score'] < 0.75 and c['features']['side'] < 100 and max(c['confirm']) < 0.5 and not any(d['score'] >= 0.7 and len(set(d['views'])) >= 6 and same_face(c,d) for d in row.get('diagnostics',[])):
            continue
        if c['score'] >= 0.6 and (len(set(c['views'])) >= 6 or (len(set(c['views'])) >= 5 and max(c['confirm']) >= 0.6)) and (len({v.split(':')[1] for v in c['views']}) >= 2) and (not unsupported(c)) and (not evidence.quality(c)):
            canonical.append(c)
    for c in row.get('tonal',[]):
        f=c['features']
        # Repeated, independently confirmed views can preserve a low-light
        # face even when the native view alone is classified as blurry.
        recovered_low_light = (c['score'] >= 0.78 and len(set(c['views'])) >= 20
            and max(c['confirm']) >= 0.65 and f.get('side',0) >= 45
            and f.get('native_lap',0) >= 30 and f.get('p50',0) >= 40
            and 0.18 <= f.get('eye_span',0) <= 0.4
            and 0.4 < f.get('mouth_depth',0) < 3
            and any(b['score'] >= 0.7 and same_face(c,b)
                and evidence.quality(b) == 'FACE_BLURRY' for b in base))
        if recovered_low_light and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        bright_detail = (c['score'] >= 0.55 and len(set(c['views'])) >= 5
            and f.get('p50',0) >= 190 and f.get('side',0) >= 29
            and min(p[1] for p in f.get('eyes',[[0,0,0]])) >= 30
            and min(p[2] for p in f.get('eyes',[[0,0,0]])) >= 40
            and any(b['score'] >= 0.65 and same_face(c,b) for b in base)
            and any(d['score'] >= 0.65 and len(set(d['views'])) >= 8 and same_face(c,d) for d in row.get('diagnostics',[])))
        if bright_detail and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        if (c['score'] >= 0.7 and len(set(c['views'])) >= 8 and max(c['confirm']) >= 0.5
                and f.get('p50',0) >= 190 and f.get('side',0) >= 29
                and min(p[1] for p in f.get('eyes',[[0,0]])) >= 25
                and any(d['score'] >= 0.6 and len(set(d['views'])) >= 2 and same_face(c,d)
                        for d in row.get('diagnostics',[]))
                and not any(evidence.quality(b) not in (None,'FACE_TOO_SMALL') and same_face(c,b) for b in base)
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
    for c in row.get('bands',[]):
        f=c['features']
        if (c['score'] >= 0.8 and len(set(c['views'])) >= 3
                and f.get('side',0) >= 29
                and 0.18 <= f.get('eye_span',0) <= 0.4
                and abs(f.get('nose_side',0)) < 1.5
                and min(p[1] for p in f['eyes'][:2]) >= 35
                and f['eyes'][2][1] >= 25 and f['eyes'][3][1] >= 15
                and not unsupported(c) and not evidence.quality(c)
                and not any(evidence.overlap(c['box'],other['box']) > 0.5
                    for other in accepted+canonical)):
            canonical.append(c)
    for c in row.get('border', []) + row.get('pose',[]):
        f = c['features']
        if c in row.get('pose',[]) and any(evidence.overlap(c['box'],b['box']) > 0.5 for b in accepted):
            continue
        x, y, w, h = c['box']
        ih,iw=row['shape'][:2]
        confirmed_pose = (c in row.get('pose',[])
            and 0.7 <= c['score'] < 0.8 and len(set(c['views'])) >= 8
            and max(c['confirm']) >= 0.7 and 40 <= f.get('side',0) < 120
            and 0.12 <= f.get('eye_span',0) <= 0.4
            and abs(f.get('nose_side',0)) < 1.5
            and 0.5 < f.get('mouth_depth',0) < 3
            and min(p[1] for p in f['eyes'][:2]) >= 30
            and min(p[2] for p in f['eyes'][:2]) >= 30)
        if confirmed_pose and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        if (0.4 <= f['visible'] < 0.7 and f['side'] >= 80 and c['score'] >= 0.4
                and len(set(c['views'])) >= 3
                and all(0 <= px < iw and 0 <= py < ih for px,py in c['points'][:2])
                and 0 <= c['points'][2][0] < iw
                and ih-0.2*f['side'] <= c['points'][2][1] <= ih+0.3*f['side']
                and c['points'][3][1] > ih
                and min(p[1] for p in f.get('eyes',[[0,0]] )[:2]) >= 20
                and min(p[2] for p in f['eyes'][:2]) >= 8
                and (f['lap'] >= 20 or len(set(c['views'])) >= 8)
                and not unsupported(c) and not evidence.quality(c)):
            canonical.append(c)
        detailed_side = (0.03 <= f.get('eye_span',0) < 0.15 and abs(f.get('nose_side',0)) > 1 and (f.get('mouth_depth',0) >= 1 or (f['side'] >= 80 and min(p[1] for p in f['eyes']) >= 80))
                         and f.get('points_inside',0) >= 3 and 0.4 <= f['visible'] <= 1.1
                         and min(w,h) >= 40 and (x < w or x+w > row['shape'][1]-w)
                         and min(p[2] for p in f.get('eyes',[[0,0,0]])) >= 100
                         and ((c['score'] >= 0.6 and len(set(c['views'])) >= 6
                               and min(p[1] for p in f['eyes']) >= 45)
                              or (c['score'] >= 0.55 and len(set(c['views'])) >= 4
                                  and min(p[1] for p in f['eyes']) >= 80)
                              or (c['score'] >= 0.65 and len(set(c['views'])) >= 2
                                  and f['side'] >= 60 and f['eye_span'] < 0.1
                                  and min(p[1] for p in f['eyes']) >= 90)
                              or (c in row.get('pose',[]) and c['score'] >= 0.55
                                  and len(set(c['views'])) >= 3 and f['side'] >= 60
                                  and f['mouth_depth'] >= 2
                                  and min(p[1] for p in f['eyes']) >= 75
                                  and min(p[2] for p in f['eyes']) >= 100)))
        if detailed_side and not unsupported(c) and not evidence.quality(c):
            canonical.append(c)
        if c in row.get('border', []) and c['score'] >= 0.6 and len(set(c['views'])) >= 6 and (0.4 <= f['visible'] <= 1.05) and (x < 0 or x + w > row['shape'][1]) and (f.get('points_inside', 0) >= 3) and (min((p[1] for p in f['eyes'][:2])) >= 50) and (not unsupported(c)) and (not evidence.quality(c)):
            canonical.append(c)
    # At an image boundary, sharply turned faces can have unstable BlazeFace
    # landmarks. Require independent source-pixel observations of the same
    # region before admitting one of these weak boxes as a usable face.
    partial_rescued = []
    for group in ('pose', 'edges', 'diagnostics'):
        for c in row.get(group, []):
            f=c['features'];x,y,w,h=c['box'];side=min(w,h)
            if not f.get('eyes') or any(evidence.overlap(c['box'],a['box']) > 0.3 for a in accepted):
                continue
            near_boundary=(x < w or x+w > row['shape'][1]-w
                           or y < h or y+h > row['shape'][0]-h)
            if not near_boundary or evidence.quality(c):
                continue
            extreme_pose=(group=='pose' and c['score']>=0.5 and side>=55
                and f['eye_span']<0.12 and f['mouth_depth']>=3
                and min(p[1] for p in f['eyes'])>=50
                and min(p[2] for p in f['eyes'])>=80
                and 150<=f['native_lap']<=500 and f['p50']<120
                and (len(set(c['views']))>=3 or any(
                    d['score']>=0.45 and len(set(d['views']))>=4
                    and evidence.overlap(c['box'],d['box'])>0.4
                    for d in row.get('diagnostics',[]))))
            edge_profile=(group=='edges' and c['score']>=0.5 and side>=40
                and f['eye_span']<0.15 and abs(f['nose_side'])>=1.3
                and f['mouth_depth']>=2 and min(p[1] for p in f['eyes'])>=30
                and 350<=f['native_lap']<=475 and 60<=f['p50']<=120
                and not any(d['score']>=0.5
                    and evidence.overlap(c['box'],d['box'])>0.2
                    for d in row.get('diagnostics',[]))
                and any(
                    (d['score']>=0.35 and len(set(d['views']))>=2
                     or d['score']>=0.28 and len(set(d['views']))>=5)
                    and evidence.overlap(c['box'],d['box'])>0.2
                    for d in row.get('diagnostics',[])))
            small_profile=(group=='pose' and c['score']>=0.5 and side>=40
                and f['eye_span']<0.25 and 300<=f['native_lap']<=600
                and (abs(f['nose_side'])>=0.2 or f['mouth_depth']>=2)
                and max(p[1] for p in f['eyes'][:2])>=100
                and max(p[2] for p in f['eyes'][:2])>=100
                and min(p[2] for p in f['eyes'])>=30
                and any(b['score']>=0.65 and len(set(b['views']))>=8
                    and min(b['box'][2:])>=30
                    and evidence.overlap(c['box'],b['box'])>0.5
                    for b in row.get('border',[])))
            lower_border=(group=='diagnostics' and c['score']>=0.4
                and len(set(c['views']))>=3 and side>=150
                and f['visible']>=0.8 and y+h>row['shape'][0]
                and all(0<=px<row['shape'][1] and 0<=py<row['shape'][0]
                        for px,py in c['points'][:2])
                and any(b['score']>=0.6 and len(set(b['views']))>=10
                    and min(b['box'][2:])>=40 and not evidence.quality(b)
                    and evidence.overlap(c['box'],b['box'])>0.45
                    for b in row.get('border',[])))
            clipped_consensus=(group=='diagnostics' and 0.25<=c['score']<0.35
                and len(set(c['views']))==1 and 60<=side<=110
                and (x<0 or x+w>row['shape'][1])
                and 40<=f['p50']<=90 and f['lap']>=15
                and 150<=f['native_lap']<=300
                and max(p[1] for p in f['eyes'])>=100
                and max(p[2] for p in f['eyes'])>=200
                and any(d is not c and 0.4<=d['score']<0.5
                    and len(set(d['views']))==1
                    and min(d['box'][2:])>=60
                    and (d['box'][0]<0 or d['box'][0]+d['box'][2]>row['shape'][1])
                    and evidence.overlap(c['box'],d['box'])>0.5
                    for d in row.get('diagnostics',[]))
                and any(b['score']>=0.5 and 10<=min(b['box'][2:])<30
                    and evidence.overlap(c['box'],b['box'])>0.4
                    for b in row.get('border',[])))
            if extreme_pose or edge_profile or small_profile or lower_border or clipped_consensus:
                partial_rescued.append(c)
                canonical.append(c)
    output = []
    def weak_duplicate(a, b):
        low = min((a, b), key=lambda c: c['score'])
        if low['score'] >= 0.7 or max(low['confirm']) >= 0.6:
            return False
        side = min(*a['box'][2:], *b['box'][2:])
        ac = (a['box'][0] + a['box'][2]/2, a['box'][1] + a['box'][3]/2)
        bc = (b['box'][0] + b['box'][2]/2, b['box'][1] + b['box'][3]/2)
        return (evidence.overlap(a['box'], b['box']) > 0.55
                and math.dist(ac, bc) < 0.4 * side)
    for c in sorted(accepted + canonical, key=lambda c: c['score'], reverse=True):
        if c in row.get('border', []):
            x, y, w, h = c['box']
            ih, iw = row['shape'][:2]
            crosses_image_edge = x < 0 or x + w > iw or y < 0 or y + h > ih
            if (crosses_image_edge and c['score'] < 0.8
                    and c['features'].get('eye_span', 1) < 0.18
                    and max(c['confirm']) < 0.6):
                continue
        if unsupported(c) and c not in partial_rescued:
            continue
        if not any((same_face(c, a) or weak_duplicate(c, a) for a in output)):
            output.append(c)
    if any((c['features'].get('gray_fraction', 1) < 0.3 for c in output)):
        output = [c for c in output if not (c['features'].get('gray_fraction', 0) > 0.8 and c['features']['white'] > 0.15)]
    if any((c['score'] >= 0.8 for c in output)):
        output = [c for c in output if not (c['score'] < 0.8 and (max(c['confirm']) < 0.6 and min(c['features'].get('block', [0])) > 0.25 and c['features']['eye_span'] > 0.3 or (c['features']['p50'] > 190 and c['features']['white'] > 0.15 and (max((p[0] for p in c['features']['eyes'][:2])) < 70) and (min((p[2] for p in c['features']['eyes'][:2])) > 200))))]
    # Weak upside-down proposals on the neck/body can overlap a confirmed face.
    # Conflicting orientation alone does not reject separately supported faces.
    def opposite_weak(c):
        if c['score'] >= 0.8 or max(c['confirm']) >= 0.6:
            return False
        direction = np.array(c['points'][3]) - np.mean(c['points'][:2], axis=0)
        for other in output:
            if other is c or other['score'] < 0.8 or min(other['confirm']) < 0.6:
                continue
            axis = np.array(other['points'][3]) - np.mean(other['points'][:2], axis=0)
            cosine = np.dot(direction, axis) / max(np.linalg.norm(direction)*np.linalg.norm(axis), 1e-6)
            if evidence.overlap(c['box'], other['box']) > 0.2 and cosine < -0.5:
                return True
        return False
    output = [c for c in output if not opposite_weak(c)]
    output = [c for c in output if not (
        c['score'] < 0.8 and max(c['confirm']) == 0
        and any(a is not c and a['score'] >= 0.8 and min(a['confirm']) >= 0.6
            and evidence.overlap(c['box'],a['box']) > 0.5
            and c['box'][2]*c['box'][3] > 1.5*a['box'][2]*a['box'][3]
            for a in output))]
    # Small weak texture proposals must survive a stronger local reobservation.
    # Visible side profiles may instead have consistent asymmetric landmarks.
    def strong_support(c):
        if c in partial_rescued:
            return True
        if c['score'] >= 0.7 and len(set(c['views'])) >= 6 and min(c['confirm']) >= 0.55:
            return True
        if (c in row.get('pose',[]) and c['score'] >= 0.55 and len(set(c['views'])) >= 4
                and c['features']['side'] >= 45 and c['features']['eye_span'] < 0.1
                and min(p[1] for p in c['features']['eyes']) >= 80
                and min(p[2] for p in c['features']['eyes']) >= 100):
            return True
        if (c in row.get('pose',[]) and c['score'] >= 0.65 and len(set(c['views'])) >= 2
                and c['features']['side'] >= 60 and c['features']['eye_span'] < 0.1
                and min(p[1] for p in c['features']['eyes']) >= 90
                and min(p[2] for p in c['features']['eyes']) >= 100):
            return True
        if (c in row.get('pose',[]) and c['score'] >= 0.55
                and len(set(c['views'])) >= 3 and c['features']['side'] >= 60
                and c['features']['eye_span'] < 0.15
                and 1 < abs(c['features']['nose_side']) < 3
                and c['features']['mouth_depth'] >= 2
                and min(p[1] for p in c['features']['eyes']) >= 75
                and min(p[2] for p in c['features']['eyes']) >= 100):
            return True
        if (c in row.get('border',[])+row.get('pose',[]) and len(set(c['views'])) >= 6
                and min(p[1] for p in c['features']['eyes']) >= 45
                and min(p[2] for p in c['features']['eyes']) >= 100):
            return True
        if any(v.startswith('channel:') for v in c['views']):
            return True
        if (max(c['confirm']) >= 0.6 and min(p[1] for p in c['features']['eyes']) >= 40
                and max(p[2] for p in c['features']['eyes'][:2]) >= 50
                and any(d['score'] >= 0.6 and len(set(d['views'])) >= 6 and same_face(c,d)
                        for d in row.get('diagnostics', []))):
            return True
        if (abs(c['features']['nose_side']) >= 0.5
                and min(p[1] for p in c['features']['eyes']) >= 20
                and max(p[2] for p in c['features']['eyes'][:2]) >= 50
                and any(d['score'] >= 0.7 and len(set(d['views'])) >= 6 and same_face(c,d)
                        for d in row.get('diagnostics', []))):
            return True
        if c['score'] >= 0.7 and len(set(c['views'])) >= 4 and max(c['confirm']) >= 0.6:
            return True
        if c in base[:1] and c['score'] >= 0.6 and len({v for v in c['views'] if v.startswith(('whole:', 'roll:'))}) >= 2:
            return True
        if any(d['score'] >= 0.5 and len(set(d['views'])) >= 5 and min(d['confirm']) >= 0.6
               and max(d['confirm']) >= 0.7 and same_face(c,d) for d in row.get('diagnostics', [])):
            return True
        return any(d['score'] >= 0.7 and len(set(d['views'])) >= 12
                   and same_face(c, d) for d in row.get('diagnostics', [])) or (
            len(set(c['views'])) >= 5 and any(d['score'] >= 0.65 and len(set(d['views'])) >= 12
                                            and same_face(c,d) for d in row.get('diagnostics', [])))
    def weak_texture(c):
        if c in partial_rescued:
            return False
        if any(v.startswith('channel:') for v in c['views']):
            return False
        # A narrow landmark layout on a weak, unconfirmed border proposal
        # can reframe part of a nearby already confirmed face as a second one.
        if (c in row.get('border',[]) and c['score'] < 0.7
                and max(c['confirm']) == 0
                and c['features'].get('eye_span',1) < 0.1
                and abs(c['features'].get('nose_side',0)) > 1.5
                and any(other is not c and other['score'] >= 0.75
                    and min(other['confirm']) >= 0.6
                    and evidence.overlap(c['box'],other['box']) > 0.25
                    for other in output)):
            return True
        # A large border-only proposal can repeat across transformed views
        # on upholstery or clothing. Repetition alone does not restore the
        # source pixels' absent landmark detail.
        if (c in row.get('border',[]) and c['score'] < 0.7
                and max(c['confirm']) == 0 and c['features']['side'] >= 100
                and (len(set(c['views'])) < 20 or c['features'].get('eye_span',0) < 0.18)
                and max(p[2] for p in c['features']['eyes']) < 45):
            return True
        if c in row.get('pose',[]) and strong_support(c):
            return False
        if (c['score'] >= 0.55 and len(set(c['views'])) >= 3
                and min(p[1] for p in c['features'].get('eyes',[[0,0]])) >= 20
                and min(p[1] for p in c['features']['eyes'][:2]) >= 50
                and min(p[2] for p in c['features']['eyes'][:2]) >= 20
                and any(b['score'] >= 0.6 and min(p[1] for p in b['features'].get('eyes',[[0,0]])[:2]) >= 50 and same_face(c,b) for b in base)
                and any(d['score'] >= 0.7 and max(d['confirm']) >= 0.5 and len(set(d['views'])) >= 20 and same_face(c,d) for d in row.get('diagnostics',[]))):
            return False
        if (c['score'] < 0.8 and max(c['confirm']) == 0 and len(set(c['views'])) < 3
                and (c not in base[:1] or any(
                    ((a['score'] >= 0.75 and min(a['confirm']) >= 0.6)
                     or (a['score'] >= 0.6 and max(a['confirm']) >= 0.5 and evidence.overlap(c['box'],a['box']) > 0.2))
                    and not same_face(c,a) for a in output))
                and not (c['box'][1]+c['box'][3] > row['shape'][0]
                         and all(0 <= p[1] < row['shape'][0] for p in c['points'][:2]))
                and not any(same_face(c,d) and (max(d['confirm']) >= 0.5
                        or (d['score'] >= 0.75 and len(set(d['views'])) >= 12))
                    for name in ('candidates','diagnostics','supplements','profiles','edges') for d in row.get(name,[]))):
            return True
        if (c['score'] < 0.8 and max(c['confirm']) == 0
                and abs(c['features']['nose_side']) < 0.5 and c['features']['side'] < 100
                and not any(max(d['confirm']) >= 0.6 and same_face(c,d)
                            for name in ('candidates','diagnostics','supplements','profiles','edges')
                            for d in row.get(name,[]))):
            return True
        return not strong_support(c) and (
        c['features']['side'] < 100 and (
            (c['score'] < 0.65 and max(c['confirm']) < 0.75 and min(c['confirm']) < 0.6)
            or (c['features']['eye_span'] < 0.1 and max(c['confirm']) < 0.75)
            or (c['score'] < 0.8 and max(c['confirm']) == 0
                and abs(c['features']['nose_side']) < 0.5)
        ))
    revised = []
    for c in output:
        if weak_texture(c):
            alternatives = [a for a in accepted + canonical if same_face(c,a) and not weak_texture(a)]
            if not alternatives:continue
            c = max(alternatives,key=lambda a:a['score'])
        revised.append(c)
    output = revised
    output = [c for c in output if not any(
        b['score'] >= 0.6 and len(set(b['views'])) >= 2
        and evidence.quality(b) == 'FACE_OCCLUDED' and b['features']['p50'] < 60
        and max(b['confirm']) < 0.8 and same_face(c,b) for b in base)]
    # Rotated square proposals can inflate an otherwise stable face box.
    # Prefer an already qualified native-view observation of the same face.
    # Avoid switching box sources on tiny confidence changes after resizing;
    # this changes localization only, never the accepted count.
    for index, c in enumerate(output):
        anchors = [b for b in accepted if same_face(c, b)]
        if anchors:
            anchor = max(anchors, key=lambda b: b['score'])
            if max(anchor['confirm']) >= 0.5 or min(c['confirm']) < 0.6:
                output[index] = anchor
    # A highly stable local border observation can locate a face more tightly
    # than a large native box. This refines geometry after qualification.
    for index,c in enumerate(output):
        if c not in accepted or c['score'] >= 0.8:
            continue
        alternatives=[b for b in row.get('border',[])
            if b['score'] >= 0.9 and len(set(b['views'])) >= 100
            and max(b['confirm']) >= 0.7 and not unsupported(b)
            and not evidence.quality(b)
            and 0.35 <= min(b['box'][2:])/max(min(c['box'][2:]),1) < 0.75
            and evidence.overlap(b['box'],c['box']) > 0.8]
        if alternatives:
            output[index]=max(alternatives,key=lambda b:b['score'])
    # A repeatedly observed full profile can supersede a smaller nose/mouth
    # proposal. Identity and source-pixel quality were already qualified.
    for index,c in enumerate(output):
        if max(c['confirm']) != 0 or c['features'].get('eye_span',1) >= 0.15:
            continue
        area=c['box'][2]*c['box'][3]
        alternatives=[d for d in canonical if d in row.get('pose',[])
            and same_output(c,d) and 1.5*area < d['box'][2]*d['box'][3] < 4*area
            and len(set(d['views'])) >= 4 and min(p[1] for p in d['features']['eyes']) >= 80
            and min(p[2] for p in d['features']['eyes']) >= 100 and not weak_texture(d)]
        if alternatives:output[index]=max(alternatives,key=lambda d:d['score'])
    # Localization must not turn separate proposals into duplicate references
    # to one accepted face. Count each final face only once.
    unique = []
    for c in output:
        if not any(same_output(c, other) for other in unique):
            unique.append(c)
    output = unique
    for c in output:
        if c not in partial_rescued:
            continue
        if c in row.get('edges', []):
            side=min(c['box'][2:])
            anchors=[d for d in row.get('diagnostics',[])
                if d['score']>=0.28 and len(set(d['views']))>=4
                and 2*side<=min(d['box'][2:])<=4*side
                and evidence.overlap(c['box'],d['box'])>0.8
                and not evidence.quality(d)]
            if anchors:
                c['localization_box']=max(anchors,key=lambda d:d['score'])['box']
        elif (c in row.get('pose',[]) and c['score']<0.55
                and min(c['box'][2:])<50):
            x,y,w,h=c['box']
            c['localization_box']=[x-w/2,y-h/2,2*w,2*h]
    n = len(output)
    if not n:
        reason = diagnosis(row, reason)
    if not n and reason == 'NO_FACE' and any((c['score'] >= 0.6 and len(set(c['views'])) >= 3 and (4 <= min(c['box'][2:]) < 24) for c in row.get('tiny', []))):
        reason = 'FACE_TOO_SMALL'
    return (output, reason)
