"""BlazeFace-only view search, confirmation and source-pixel quality analysis.

All per-photo observations live only for this call. No file identity, expected
answer, external detector, regression report or persistent result cache is read.
"""
import copy
from . import evidence, probe, canonical, profiles, edges, chromatic, tiny
from . import border_rescue, policy, tonal, perspective

def analyze(image, detector, proposal_detector, mp):
    height, width = image.shape[:2]
    row = {'shape': image.shape, 'candidates': evidence.collect(image, detector, mp)}
    row['diagnostics'] = probe.merge(copy.deepcopy(row['candidates']) + probe.augmented(image, proposal_detector, mp))
    for c in row['diagnostics']:
        c['features'] = evidence.features(image, c)
        c['confirm'] = evidence.confirm(image, c, detector, mp)
    row['supplements'] = canonical.collect(image, row['diagnostics'], detector, mp)
    row['profiles'] = profiles.collect(image, row['diagnostics'], detector, mp)

    def enrich(candidates):
        for c in candidates:
            c['features'] = evidence.features(image, c)
            c['confirm'] = evidence.confirm(image, c, detector, mp)
        return candidates
    if any((c['score'] >= 0.25 and min(c['box'][2:]) >= min(height, width) * 0.07 and (c['box'][0] < 0 or c['box'][0] + c['box'][2] > width) for c in row['diagnostics'])):
        row['edges'] = enrich(probe.merge(edges.collect(image, detector, mp)))
    means = image.mean(axis=(0, 1))
    if means.min() < means.max() * 0.5:
        row['chromatic'] = enrich(probe.merge(chromatic.collect(image, detector, mp)))
    accepted, reason = policy.select(row)
    if not accepted and reason == 'NO_FACE':
        row['tiny'] = probe.merge(tiny.collect(image, detector, mp))
    seeds = [c for c in row['diagnostics'] if not any(policy.same_face(c, a) for a in accepted)]
    row['border'] = enrich(border_rescue.collect(image, seeds, detector, mp))
    accepted, _ = policy.select(row)
    seeds = [c for c in row['diagnostics'] if not any(policy.same_face(c, a) for a in accepted)]
    row['tonal'] = enrich(tonal.collect(image, seeds, detector, mp))
    accepted, _ = policy.select(row)
    uncertain = [a for a in accepted if max(a['confirm']) == 0 and a['features'].get('eye_span',1) < 0.15]
    seeds = [c for c in row['diagnostics'] if not any(policy.same_face(c,a) for a in accepted if a not in uncertain)]
    # A weak profile at the image boundary may be observed more clearly from
    # a different local pose even when it was absent from upright diagnostics.
    boundary_seeds = [c for c in row.get('border',[]) + row.get('edges',[])
        if c['score'] >= 0.5 and min(c['box'][2:]) >= 30
        and c['features'].get('eye_span',1) < 0.2
        and c['features'].get('native_lap',0) >= 100
        and not any(policy.same_face(c,a) for a in accepted if a not in uncertain)]
    row['pose'] = enrich(perspective.collect(image, seeds + boundary_seeds + uncertain, detector, mp))
    # Keep the ordinary pipeline's seeds/results stable. Extra windows for
    # extreme aspect ratios contribute only at the final decision stage.
    row['bands'] = enrich(probe.merge(evidence.collect_bands(image, detector, mp)))
    return policy.select(row)
