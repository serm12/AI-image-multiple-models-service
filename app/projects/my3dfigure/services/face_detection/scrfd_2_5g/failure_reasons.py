"""Private zero-usable-face diagnostics; never changes face acceptance."""
from __future__ import annotations

from . import evidence, quality


def aggregate(records, extra_reasons=(), *, unknown=False):
    """Deduplicate independent causes; unknown established faces stay generic."""
    reasons = set(extra_reasons)
    for record in records:
        reasons.update(record['quality_issues'])
        unknown = unknown or record.get('reason_unclear', False)
    reasons = sorted(reasons)
    return reasons, bool(unknown), reasons[0] if len(reasons) == 1 and not unknown else 'NO_FACE'


def independent_extra_reasons(records, extra_reasons):
    discarded = {issue for record in records for issue in record.get('discarded_secondary_issues', [])}
    retained = {issue for record in records for issue in record['quality_issues']}
    return [issue for issue in extra_reasons if issue not in discarded or issue in retained]


def primary_records(observations, context_reasons=(), image_shape=None):
    records = []
    for candidate, reason in observations:
        f = candidate['features']
        support = max(candidate.get('confirm', [0]))
        established = (candidate['score'] >= .75 and support >= .7
                       and f['visible'] >= .85 and f['points_inside'] >= 4
                       and .15 <= f['eye_span'] <= .55
                       and .7 <= f['mouth_depth'] <= 1.8)
        pose_hint = ('FACE_NOT_FRONTAL' in context_reasons
                     and candidate['score'] >= .5 and f['side'] >= 140
                     and f['eye_span'] < .08
                     and (abs(f['nose_side']) > 2 or f['mouth_depth'] > 5)
                     and support < .1)
        if reason is None and not (established or pose_hint):
            continue
        raw_reason = reason
        derived = []
        if ('FACE_OVEREXPOSED' in context_reasons
                and reason in {'FACE_UNRECOGNIZABLE', 'FACE_BLURRY'}
                and f['p10'] >= 240 and f['core_white'] > .5):
            derived.append(reason)
            reason = 'FACE_OVEREXPOSED'
        native_blur = (reason == 'FACE_BLURRY' and f['visible'] >= .85
                       and f['points_inside'] >= 4 and f['p50'] >= 30
                       and f['core_p90'] >= 35 and f['native_lap'] < 25
                       and f['lap'] < 6 and max(p[2] for p in f['eyes']) < 15)
        unreliable_pose = (native_blur
                           or reason in {'FACE_TOO_DARK', 'FACE_OVEREXPOSED', 'FACE_TOO_SMALL', 'FACE_UNRECOGNIZABLE'}
                           or reason == 'FACE_INCOMPLETE' and (f['visible'] < .85 or f['points_inside'] < 4))
        if pose_hint and not unreliable_pose and reason == 'FACE_OCCLUDED':
            derived.append(reason)
            reason = 'FACE_NOT_FRONTAL'
        reasons = {reason} if reason else set()
        if 'FACE_OVEREXPOSED' in context_reasons and f['p10'] >= 240 and f['core_white'] > .5:
            reasons.add('FACE_OVEREXPOSED')
        if pose_hint:
            if unreliable_pose:
                derived.append('FACE_NOT_FRONTAL')
            else:
                reasons.add('FACE_NOT_FRONTAL')
        # An already confirmed cut-off face may also have independently dark
        # visible eyes and nose. Samples outside the image cannot prove darkness.
        points = candidate.get('points')
        visible_features = (image_shape is not None and points is not None
                            and all(0 <= points[i][0] < image_shape[1]
                                    and 0 <= points[i][1] < image_shape[0] for i in (0, 1, 2)))
        if (raw_reason == 'FACE_INCOMPLETE' and visible_features
                and candidate['score'] >= .8 and support >= .95
                and f['visible'] >= .4 and f['points_inside'] >= 3
                and f['core_p90'] < 24
                and all(p[0] + p[1] < 24 for p in f['eyes'][:3])):
            reasons.add('FACE_TOO_DARK')
        if reason and f['visible'] < .4 and f['points_inside'] < 3 and support >= .6:
            reasons.add('FACE_INCOMPLETE')
        records.append({'box': list(candidate['box']), 'score': float(candidate['score']),
                        'quality_issues': sorted(reasons), 'reason_unclear': not reasons,
                        'evidence': 'primary_source_quality',
                        'source_quality_reason': raw_reason,
                        'discarded_secondary_issues': sorted(set(derived))})
    return records


def _soft_source_head(candidate):
    """Severe native face-detail loss with sufficient local exposure.

    Diagnostic only. Unreliable pose from the smear is deliberately not used
    as proof of a second pose/occlusion problem.
    """
    f = candidate['features']
    return (45 <= f['side'] <= 120 and f['visible'] >= .9
            and f['points_inside'] >= 4 and f['p50'] >= 30
            and 8 <= f['p10'] < 240 and f['core_p90'] >= 35
            and f['core_white'] < .5
            and .12 <= f['eye_span'] <= .5
            and .7 <= f['mouth_depth'] <= 3 and abs(f['nose_side']) <= 1
            and f['native_lap'] < 25 and f['lap'] < 6
            and max(p[1] for p in f['eyes']) < 25
            and max(p[2] for p in f['eyes']) < 8)


def repeated_soft_heads(image, detector):
    """Name blur only for source heads reproduced at a second whole scale."""
    def heads(size=None):
        boxes, landmarks = detector.detect(image, det_thresh=.20,
                                           **({'input_size': size} if size else {}))
        result = []
        for (left, top, right, bottom, score), points in zip(boxes, landmarks):
            if not (.20 <= score < .75 and min(right-left, bottom-top) >= 45):
                continue
            candidate = {'box': list(map(float, (left, top, right-left, bottom-top))),
                         'points': points, 'score': float(score)}
            candidate['features'] = quality.features(image, candidate)
            if _soft_source_head(candidate):
                result.append(candidate)
        return result
    source = heads()
    if not source:
        return []
    alternate = heads((800, 800))
    confirmed = []
    for candidate in source:
        matches = [other for other in alternate
                   if evidence.same_location(candidate['box'], other['box'])
                   and max(candidate['score'], other['score']) >= .30]
        if matches:
            confirmed.append({'box': candidate['box'], 'score': candidate['score'],
                              'quality_issues': ['FACE_BLURRY'], 'reason_unclear': False,
                              'evidence': 'repeatable_source_blur',
                              'confirmation_score': max(other['score'] for other in matches)})
    return confirmed
