"""Additional evidence for uncertain faces in any submitted photograph."""
from __future__ import annotations

import copy
import cv2
import numpy as np

from . import evidence,quality,graphics


def _candidate(image,box,points,score,view):
    candidate={'box':list(map(float,box)),'points':np.asarray(points),'score':float(score),'confirm':[float(score)]*2,'views':[view]}
    candidate['features']=quality.features(image,candidate)
    return candidate


def _same_face(a,b):
    overlap=evidence._overlap_smaller(a,b)
    if overlap>.75:return True
    if overlap<=.35:
        return False
    delta=np.abs(np.array(a[:2])+np.array(a[2:])/2-np.array(b[:2])-np.array(b[2:])/2)
    return delta[0]<.65*min(a[2],b[2]) and delta[1]<.65*min(a[3],b[3])


def _represented(candidate,boxes):
    points=candidate['points']
    for box in boxes:
        if _same_face(candidate['box'],box):return True
        x,y,w,h=box
        if (evidence._overlap_smaller(candidate['box'],box)>.2
                and all(x<=px<=x+w and y<=py<=y+h for px,py in points[:3])):
            return True
    return False


def _same_anchor(a,b):
    side=min(*a['box'][2:],*b['box'][2:])
    return (_same_face(a['box'],b['box'])
            and np.linalg.norm(a['points'][:3].mean(axis=0)-b['points'][:3].mean(axis=0))<.4*side)


def _distinct_local_confirmation(image, candidate, detector, established):
    """A weak neighboring proposal must confirm its own facial landmarks.

    Reuse the two confirmation views. Overlap alone can borrow a nearby
    established face's confidence and count hair as a second face.
    """
    x, y, width, height = candidate['box']
    center = np.array([x+width/2, y+height/2])
    eyes = np.asarray(candidate['points'][:2], dtype=float)
    delta = eyes[1]-eyes[0]
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    anchor = np.asarray(candidate['points'])[:3].mean(axis=0)
    for factor in (1.6, 2.3):
        side = max(64, round(max(width, height)*factor))
        matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
        matrix[:, 2] += np.array([side/2, side/2])-center
        view = cv2.warpAffine(image, matrix, (side, side))
        inverse = cv2.invertAffineTransform(matrix)
        boxes, points = detector.detect(view)
        found = False
        for b, p in zip(boxes, points):
            corners = np.array([[b[0], b[1], 1], [b[2], b[1], 1],
                                [b[0], b[3], 1], [b[2], b[3], 1]])@inverse.T
            low, high = corners.min(axis=0), corners.max(axis=0)
            box = [*low, *(high-low)]
            mapped = np.column_stack([p, np.ones(len(p))])@inverse.T
            if evidence._overlap_smaller(box, candidate['box']) < .4:
                continue
            if np.linalg.norm(mapped[:3].mean(axis=0)-anchor) > .3*min(width, height):
                continue
            if np.linalg.norm(mapped[2]-candidate['points'][2]) > .4*min(width, height):
                continue
            if _represented({'box': box, 'points': mapped}, established):
                continue
            found = True
            break
        if not found:
            return False
    return True


def local_observations(image,candidate,detector,angles=(0,),size=320,enhance=False,*,factors=(1.6,2.3,3.2)):
    if not hasattr(detector, 'local_observation_get'):
        return _local_observations(image,candidate,detector,angles,size,enhance,factors=factors)
    f = candidate['features']
    geometry = (tuple(map(float,candidate['box'])),
                tuple(map(float,np.asarray(candidate['points']).ravel())),
                float(f['visible']),float(f['eye_span']),size,enhance)
    source_key = detector.evidence_key('local_observations:exact-view:v2',image,geometry)
    observed = []
    # Preserve original angle/factor sequence, including duplicate views.
    # Each view is a pure independent iteration of the unchanged old body.
    for angle in angles:
        for factor in factors:
            key = (source_key, (angle,str(angle)), (factor,str(factor)))
            result = detector.local_observation_get(key)
            if result is None:
                result = _local_observations(image,candidate,detector,(angle,),size,enhance,factors=(factor,))
                detector.local_observation_put(key,result)
            observed.extend(result)
    return observed


def _local_observations(image,candidate,detector,angles=(0,),size=320,enhance=False,*,factors=(1.6,2.3,3.2)):
    x,y,width,height=candidate['box'];observations=[]
    center=(x+width/2,y+height/2)
    for angle in angles:
        for factor in factors:
            side=max(64,round(max(width,height)*factor))
            matrix=cv2.getRotationMatrix2D(center,angle,1.)
            matrix[:,2]+=np.array([side/2,side/2])-center
            view=cv2.warpAffine(image,matrix,(side,side))
            if enhance:
                lab=cv2.cvtColor(view,cv2.COLOR_BGR2LAB)
                lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(lab[:,:,0])
                view=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            inverse=cv2.invertAffineTransform(matrix)
            boxes,points=detector.detect(view,input_size=None if size==640 else (size,size),det_thresh=.15)
            matches=[]
            for (left,top,right,bottom,score),landmarks in zip(boxes,points):
                corners=np.array([[left,top,1],[right,top,1],[left,bottom,1],[right,bottom,1]])@inverse.T
                low,high=corners.min(axis=0),corners.max(axis=0);box=[*low,*(high-low)]
                if evidence._overlap_smaller(box,candidate['box'])<.4 or not evidence.same_location(box,candidate['box']):continue
                inter=evidence._overlap_smaller(box,candidate['box'])*min(box[2]*box[3],width*height)
                if inter/max(1,box[2]*box[3]+width*height-inter)<.35:continue
                mapped=np.column_stack([landmarks,np.ones(len(landmarks))])@inverse.T
                cf=candidate['features']
                if (cf['visible']>=.9 and cf['eye_span']>=.18 and min(width,height)<120
                        and (np.linalg.norm(mapped[:3].mean(axis=0)-candidate['points'][:3].mean(axis=0))>.3*min(width,height)
                             or np.linalg.norm(mapped[2]-candidate['points'][2])>.4*min(width,height))):continue
                matches.append(_candidate(image,box,mapped,score,f'local:{angle}:{factor}'))
            if matches:observations.append(max(matches,key=lambda c:c['score']))
    return observations


def _source_quality(candidate):
    reason=quality.quality(candidate);f=candidate['features']
    if reason == 'FACE_TOO_SMALL' and compact_visible_face(candidate):
        return None
    if reason == 'FACE_BLURRY' and stable_soft_details(candidate):
        return None
    if reason in ('FACE_BLURRY','FACE_OCCLUDED','FACE_TOO_SMALL') and f['visible']>=.6:
        eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
        readable=(29<=f['side']<120 and (f['lap']>=2.5 or f['native_lap']>=30)
                  and .18<=f['eye_span']<=.55 and .65<=f['mouth_depth']<=2.5
                  and abs(f['nose_side'])<=.8 and min(p[1] for p in eyes)>=15
                  and min(p[2] for p in eyes)>=5 and min(nose[1],mouth[1])>=20
                  and min(nose[2],mouth[2])>=5)
        if readable:return None
        dark_profile=(reason=='FACE_BLURRY' and 35<=f['side']<120 and f['visible']>=.8
                      and f['native_lap']>=40 and f['lap']>=2.5
                      and .15<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2.5
                      and abs(f['nose_side'])<=.8 and min(p[1] for p in eyes)>=12
                      and min(p[2] for p in eyes)>=2.5 and max(p[2] for p in eyes)>=9
                      and nose[1]>=40 and mouth[1]>=4 and min(nose[2],mouth[2])>=3
                      and min(candidate.get('confirm',[0]))>=.65)
        if dark_profile:return None
    return reason


def _agree(observations,minimum=.5,maximum=.6):
    good=[c for c in observations if c['score']>=minimum and c['features'].get('visible')
          and _source_quality(c) is None and not _unrecoverable(c)]
    for a in good:
        cluster=[b for b in good if evidence._overlap_smaller(a['box'],b['box'])>=.7 and _same_anchor(a,b)]
        if len(cluster)>=3 and max(c['score'] for c in cluster)>=maximum:
            return max(cluster,key=lambda c:c['score'])
    return None


def _source_detail(candidate):
    f=candidate['features']
    if f['side']<35:return compact_visible_face(candidate)
    nose,mouth=f['eyes'][2:4]
    if (candidate['score']>=.65 and min(candidate['confirm'])>=.65
            and 35<=f['side']<120 and f['visible']>=.9
            and .18<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2.5
            and abs(f['nose_side'])<=.8 and f['native_lap']>=40 and f['lap']>=2.5
            and min(p[1] for p in f['eyes'][:2])>=12
            and min(p[2] for p in f['eyes'][:2])>=2.5
            and nose[1]>=40 and mouth[1]>=4):return True
    if _soft_profile(candidate) and f['p10']>120:return True
    if (f['p50']>140 and f['native_lap']>40 and f['lap']>3
            and candidate['score']>=.6 and min(candidate['confirm'])>=.65):return True
    if f['side']<100 and f['eye_span']<.13:
        return (nose[1]>=25 and mouth[1]>=25 and nose[2]>=10 and mouth[2]>=10)
    if f['side']<60:
        return nose[1]>=25 and mouth[1]>=15
    return True


def compact_pixels(candidate):
    """Real nose, mouth and bilateral eye detail in a compact source face."""
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    return (24<=f['side']<40 and f['visible']>=.9 and f['points_inside']>=4
            and f['native_lap']>=100 and f['lap']>=2
            and .2<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=1.8
            and abs(f['nose_side'])<=.5 and min(p[1] for p in eyes)>=25
            and min(p[2] for p in eyes)>=8
            and min(nose[1],mouth[1])>=20 and min(nose[2],mouth[2])>=5)


def compact_visible_face(candidate):
    return (candidate['score']>=.5 and min(candidate.get('confirm',[0]))>=.75
            and compact_pixels(candidate))


def stable_soft_details(candidate):
    """Soft source pixels remain usable when all facial parts are readable."""
    f=candidate['features'];eyes=f['eyes'][:2]
    return (candidate['score']>=.8 and min(candidate.get('confirm',[0]))>=.84
            and f['side']>=100 and f['visible']>=.9 and f['points_inside']>=4
            and f['native_lap']>=5 and f['lap']>=3 and f['p90']-f['p10']>=30
            and .2<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=1.8
            and abs(f['nose_side'])<=.5 and min(p[1] for p in eyes)>=20
            and min(p[1] for p in eyes)>=.6*max(p[1] for p in eyes)
            and min(p[2] for p in eyes)>=4
            and min(p[1] for p in f['eyes'][2:4])>=30)


def _half_face_consensus(image,candidate,detector):
    """Do not treat unstable out-of-image landmarks as missing source pixels."""
    if 'half_consensus' in candidate:return candidate['half_consensus']
    f=candidate['features'];h,w=image.shape[:2];x,y,bw,bh=candidate['box']
    readable=(f['side']>=120 and .45<=f['visible']<.75
              and (x<0 or x+bw>w) and f['native_lap']>=20 and f['lap']>=20
              and .18<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2
              and abs(f['nose_side'])<=.8
              and max(p[1] for p in f['eyes'][:2])>=40
              and max(p[2] for p in f['eyes'][:2])>=30)
    candidate['half_consensus']=False
    if readable:
        observed=local_observations(image,candidate,detector,angles=(0,-20,20))
        good=[c for c in observed if c['features'].get('visible',0)>=.4
              and max(p[1] for p in c['features']['eyes'][:2])>=40
              and max(p[2] for p in c['features']['eyes'][:2])>=30
              # Rotation may move a predicted nose slightly outside the
              # boundary. It cannot supply an absent mouth: require one
              # source-space mouth corner with genuine interior clearance.
              and -.08*c['features']['side'] <= c['points'][2][0] <= w+.08*c['features']['side']
              and max(min(float(px),w-float(px)) for px,py in c['points'][3:]) >= .04*c['features']['side']]
        candidate['half_consensus']=_agree(good,minimum=.5,maximum=.65) is not None
    return candidate['half_consensus']


def _unrecoverable(candidate):
    f=candidate['features'];eyes=f['eyes'][:2]
    return (f['core_p90']<24 or (f['core_white']>.5 and min(p[0] for p in eyes)>180)
            or quality.asymmetric_blur(candidate) or quality.blocky_eye_pixels(candidate)
            or quality.severe_pixel_blocks(candidate) or quality.unreliable_detail(candidate)
            or (candidate['box'][0]<0 and candidate['box'][1]<0
                and f['side']>=100 and f['eye_span']<.13 and f['mouth_depth']>3)
            or (f['points_inside']>=4 and f['eye_span']>=.2 and min(p[1] for p in eyes)<8
                and max(p[1] for p in eyes)>=40 and min(p[2] for p in eyes)<5))


def _edge_damage(image,candidate):
    height,width=image.shape[:2];x,y,w,h=candidate['box'];f=candidate['features']
    if f['eye_span']>=.18:return False
    if (x>=0 and x+w<=width) or f['visible']>=.85:return False
    mouths=candidate['points'][3:]
    return max(min(float(px),width-float(px)) for px,py in mouths)<.12*f['side']


def _upper_damage(candidate):
    x,y,w,h=candidate['box']
    return y<-.25*h and min(float(p[1]) for p in candidate['points'][:2])<.02*min(w,h)


def _blocking_fragment(image, detector):
    """Keep established results when a readable edge profile is severely cut.

    A narrow fragment with no confirmation cannot support a broader search or
    replacement of existing faces. The original quality and recovery stages
    still run; this only limits the additional reassessment in this module.
    """
    height, width = image.shape[:2]
    detections, points = detector.detect(image, det_thresh=.1)
    for b, p in zip(detections, points):
        if b[0] >= 0 and b[2] <= width:
            continue
        c = _candidate(image, [b[0], b[1], b[2]-b[0], b[3]-b[1]], p, b[4], 'whole:0')
        f = c['features']
        if not f.get('visible'):
            continue
        if not (f['side'] >= 29 and .45 <= f['visible'] < .8
                and f['eye_span'] < .32 and f['mouth_depth'] > 1.4
                and max(q[1] for q in f['eyes'][:2]) >= 40
                and max(q[2] for q in f['eyes'][:2]) >= 10
                and f['eyes'][2][1] >= 15):
            continue
        clearance = max(min(float(px), width-float(px)) for px, py in p[3:])
        if clearance >= .4 * f['side']:
            continue
        c['confirm'] = evidence.confirm(image, c, detector)
        if max(c['confirm']) < .2:
            return c
    return None


def supports_reassessment(image, detector, present):
    # A damaged edge fragment cannot veto unrelated intact source faces.
    return True


def _supplement_quality_safe(candidate):
    # Confirmation establishes face identity; it cannot restore covered eyes
    # or remove a motion streak shared by every facial landmark patch.
    if quality.quality(candidate) in ('FACE_OCCLUDED','FACE_UNRECOGNIZABLE',
                                      'FACE_OVEREXPOSED','FACE_INCOMPLETE'):
        return False
    gradients=candidate['features'].get('patch_gradients',[])[:4]
    uniform_motion=(len(gradients)==4 and
                    (all(x<.15*y for x,y in gradients) or all(y<.15*x for x,y in gradients)))
    return not uniform_motion


def soft_readable_source(candidate):
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    return (candidate['score']>=.65 and min(candidate.get('confirm',[0]))>=.7
            and max(candidate.get('confirm',[0]))>=.75 and f['side']>=100
            and f['visible']>=.9 and f['points_inside']>=4
            and f['native_lap']>=1.8 and f['lap']>=3.25
            and .18<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2
            and abs(f['nose_side'])<=1.1 and f['p90']-f['p10']>=30
            and min(p[1] for p in eyes)>=25 and min(p[2] for p in eyes)>=4
            and nose[1]>=25 and nose[2]>=2 and mouth[1]>=12 and mouth[2]>=1.5
            and _supplement_quality_safe(candidate)
            and not quality.severe_pixel_blocks(candidate)
            and not quality.blocky_eye_pixels(candidate))


def soft_bilateral_source(candidate):
    f=candidate['features'];eyes=f['eyes'][:2]
    return (soft_readable_source(candidate) and candidate['score']>=.8
            and min(candidate.get('confirm',[0]))>=.83
            and f['native_lap']>=2.5 and min(p[1] for p in eyes)>=30
            and min(p[1] for p in eyes)>=.65*max(p[1] for p in eyes)
            and min(p[2] for p in eyes)>=4
            and f['eyes'][2][1]>=25 and f['eyes'][3][1]>=20)


def dim_readable_core(candidate):
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    return (candidate['score']>=.55 and f['visible']>=.9 and f['points_inside']>=4
            and 35<=f['side']<100 and f['native_lap']>=30 and f['lap']>=2.5
            and .15<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2
            and abs(f['nose_side'])<=.8 and min(p[1] for p in eyes)>=10
            and max(p[1] for p in eyes)>=30 and min(p[2] for p in eyes)>=2.5
            and nose[1]>=25 and nose[2]>=5 and mouth[1]>=15 and mouth[2]>=5)


def dark_source_core(candidate):
    f=candidate['features'];eyes=f['eyes'][:2]
    return (candidate['score']>=.15 and f['side']>=120 and f['visible']>=.9
            and 24<=f['p90']<45 and f['core_p90']>=24
            and f['native_lap']>=2.5 and f['lap']>=2.5
            and .15<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2
            and abs(f['nose_side'])<=.8 and min(p[1] for p in eyes)>=8
            and min(p[2] for p in eyes)>=2
            and min(p[1] for p in f['eyes'][2:4])>=10
            and min(p[2] for p in f['eyes'][2:4])>=2
            and _supplement_quality_safe(candidate))


def _detailed_soft_bilateral(candidate):
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    gradients=f.get('patch_gradients',[])[:4]
    return (candidate['score']>=.75 and min(candidate.get('confirm',[0]))>=.8
            and 200<=f['side']<600 and f['visible']>=.9 and f['points_inside']>=4
            and f['native_lap']>=2.5 and f['lap']>=7
            and .2<=f['eye_span']<=.4 and .8<=f['mouth_depth']<=1.8
            and abs(f['nose_side'])<=.8
            and min(p[1] for p in eyes)>=60 and min(p[2] for p in eyes)>=8
            and nose[1]>=20 and nose[2]>=3 and mouth[1]>=20 and mouth[2]>=3
            and len(gradients)==4
            and not (all(x<.3*y for x,y in gradients) or all(y<.3*x for x,y in gradients))
            and _supplement_quality_safe(candidate))


def _post_recoverable_source(candidate):
    """Permit faint sampled detail only; retain every hard source defect."""
    if not _supplement_quality_safe(candidate):return False
    if not _unrecoverable(candidate):return True
    f=candidate['features'];eyes=f['eyes'][:2]
    hard=(f['core_p90']<24 or (f['core_white']>.5 and min(p[0] for p in eyes)>180)
          or quality.blocky_eye_pixels(candidate) or quality.severe_pixel_blocks(candidate)
          or (candidate['box'][0]<0 and candidate['box'][1]<0
              and f['side']>=100 and f['eye_span']<.13 and f['mouth_depth']>3)
          or (f['points_inside']>=4 and f['eye_span']>=.2 and min(p[1] for p in eyes)<8
              and max(p[1] for p in eyes)>=40 and min(p[2] for p in eyes)<5))
    if hard:return False
    if quality.asymmetric_blur(candidate) and not (soft_bilateral_source(candidate) or _detailed_soft_bilateral(candidate)):return False
    return (soft_readable_source(candidate) or dark_source_core(candidate) or _detailed_soft_bilateral(candidate)
            or _bright_large_lateral_core(candidate))


def _source_veto(image,candidate,source,detector,*,allow_source_soft=False,allow_profile_landmarks=False):
    """Independent views cannot repair a source face's missing visual details."""
    f=candidate['features']
    height,width=image.shape[:2]
    for c in source:
        sf=c['features']
        # Tiny native proposals require readable nose and mouth pixels even
        # when glasses produce a sharp eye patch. A rotated envelope cannot
        # turn an otherwise too-small source face into a usable one.
        minimum = .15 if sf['side'] < 40 else .25
        if not (c['score']>=minimum and _insufficient_compact_pixels(c)):
            continue
        if (evidence._overlap_smaller(c['box'],candidate['box'])>=.8
                and np.linalg.norm(c['points'][:3].mean(axis=0)-candidate['points'][:3].mean(axis=0))<.45*sf['side']):
            return True
    for c in source:
        x,y,w,h=c['box'];sf=c['features']
        if (c['score']>=.35 and evidence.same_location(c['box'],candidate['box'])
                and sf['side']>=120 and sf['visible']<.7 and sf['points_inside']<2
                and (x<0 or x+w>width)):
            if not _half_face_consensus(image,candidate,detector):return True
    for c in source:
        sf=c['features']
        if (c['score']>=.15 and evidence.same_location(c['box'],candidate['box'])
                and sf['side']>=120 and sf['visible']>=.85 and sf['p10']<120
                and max(p[1] for p in sf['eyes'][:3])<12):
            return True
    related=[c for c in source if c['score']>=.5 and evidence.same_location(c['box'],candidate['box'])]
    if not related:return False
    def overlap(c):
        a,b=c['box'],candidate['box'];inter=evidence._overlap_smaller(a,b)*min(a[2]*a[3],b[2]*b[3])
        return inter/max(1,a[2]*a[3]+b[2]*b[3]-inter)
    nearest=max(related,key=overlap)
    nf=nearest['features']
    if nearest['score']<.5:return False
    if 'source_reason' not in nearest:
        nearest['confirm']=evidence.confirm(image,nearest,detector)
        nearest['source_reason']=quality.quality(nearest)
        nearest['source_texture']=evidence.inconsistent_texture(image,nearest,detector)
    if nearest['source_reason']=='FACE_TOO_DARK' and nf['p90']<45:return True
    frontal=(nf['eye_span']>=.25 and .5<nf['mouth_depth']<1.8 and abs(nf['nose_side'])<.5)
    if nearest['source_texture'] and frontal:return True
    readable_native=(allow_source_soft and _same_anchor(candidate,nearest)
                     and _post_recoverable_source(nearest)
                     and (soft_readable_source(nearest) or _detailed_soft_bilateral(nearest)
                          or _bright_large_lateral_core(nearest)))
    if _unrecoverable(nearest) and not readable_native:return True
    if _edge_damage(image,nearest):return True
    if (nf['visible']<.8 and nf['eye_span']<.18 and nf['mouth_depth']>2.5
            and max(nearest['confirm'])<.2):return True
    if _upper_damage(nearest) and max(nearest['confirm'])<.2:return True
    if (nf['eye_span']<.06 and (nf['mouth_depth']>5 or abs(nf['nose_side'])>2)
            and (max(nearest['confirm'])<.2 or (min(p[1] for p in nf['eyes'][2:])<25 and min(p[2] for p in nf['eyes'][2:])<10)) and f['eye_span']<.15
            and (f['mouth_depth']>2 or abs(f['nose_side'])>2) and not allow_profile_landmarks):return True
    if (nf['eye_span']<.08 and nf['mouth_depth']>4
            and max(p[1] for p in nf['eyes'][2:])<25
            and max(p[2] for p in nf['eyes'][2:])<10 and not allow_profile_landmarks):return True
    bilateral_soft=(nf['p10']<120 and max(p[2] for p in nf['eyes'][:2])<10
                    and (nearest['source_reason']=='FACE_BLURRY' or min(p[1] for p in nf['eyes'][:2])<15))
    return nf['side']>=120 and ((bilateral_soft and not readable_native and not (_soft_profile(nearest) and min(nearest['confirm'])>=.65)) or nearest['source_reason'] in ('FACE_TOO_DARK','FACE_UNRECOGNIZABLE','FACE_OVEREXPOSED'))


def _visible_profile_core(image,candidate):
    f=candidate['features'];h,w=image.shape[:2];points=candidate['points']
    inside=lambda point:0<=point[0]<w and 0<=point[1]<h
    return (35<=f['side'] and .5<=f['visible']<.9
            and .06<=f['eye_span']<=.45 and .65<=f['mouth_depth']<=4
            and abs(f['nose_side'])<=1.3
            and any(inside(p) for p in points[:2]) and inside(points[2])
            and any(inside(p) for p in points[3:])
            and f['native_lap']>=40 and f['lap']>=5
            and max(p[1] for p in f['eyes'][:2])>=40
            and min(p[1] for p in f['eyes'][2:4])>=15)


def prune(image,detector,boxes,recovered=()):
    ds,ks=detector.detect(image,det_thresh=.05);kept=[]
    for box in boxes:
        proposals=[]
        for d,p in zip(ds,ks):
            if d[4]<.15:continue
            cb=[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])]
            inter=evidence._overlap_smaller(cb,box)*min(cb[2]*cb[3],box[2]*box[3]);iou=inter/max(1,cb[2]*cb[3]+box[2]*box[3]-inter)
            proposals.append((iou,d,p,cb))
        iou,d,p,cb=max(proposals,key=lambda q:q[0],default=(0,None,None,None))
        if iou<.35 or not .15<=d[4]<.8:
            kept.append(box);continue
        candidate=_candidate(image,cb,p,d[4],'whole:0')
        if _edge_damage(image,candidate):continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        if _upper_damage(candidate) and max(candidate['confirm'])<.2:continue
        f=candidate['features']
        neighbors=[other for other in boxes if other is not box
                   and abs(other[0]+other[2]/2-cb[0]-cb[2]/2)<.7*max(other[2],cb[2])
                   and .25*cb[3]<other[1]+other[3]/2-cb[1]-cb[3]/2<1.5*cb[3]]
        collapsed_texture=(f['side']<100 and f['visible']>=.95 and f['eye_span']<.13
                           and f['mouth_depth']>3 and f['p50']<140 and f['lap']<20
                           and d[4]>=.5 and min(candidate['confirm'])<.55)
        suspicious=collapsed_texture or (neighbors and f['side']<120 and f['native_lap']>=100 and 4<=f['lap']<=20
                    and f['p50']<140 and f['p90']>=130 and f['visible']>=.7
                    and (f['eye_span']>=.15 or (f['eye_span']<.13 and min(candidate['confirm'])<.5)))
        if not suspicious:
            kept.append(box);continue
        delta=np.asarray(p[1])-np.asarray(p[0]);angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
        if (not collapsed_texture and abs(angle)>=45) or min(candidate['confirm'])>=.8:
            kept.append(box);continue
        native=local_observations(image,candidate,detector)
        if _agree(native,minimum=.75,maximum=.8) is not None:
            kept.append(box)
        elif collapsed_texture and f['native_lap']>=40:
            restored=next((c['source_candidate'] for c in recovered if c['box'] is box),None)
            if restored is not None:
                rf=restored['features']
                confirmed_profile=(restored['score']>=.75 and min(restored.get('confirm',[0]))>=.7
                    and .03<=rf['eye_span']<.13 and 3<rf['mouth_depth']<7
                    and 35<=rf['side']<100 and rf['visible']>=.95
                    and rf['native_lap']>=40 and rf['lap']>=3
                    and min(v[1] for v in rf['eyes'])>=25 and min(v[2] for v in rf['eyes'])>=20)
                detailed_profile=(restored['score']>=.65 and min(restored.get('confirm',[0]))>=.65
                    and max(restored.get('confirm',[0]))>=.7
                    and .03<=rf['eye_span']<.13 and 3<rf['mouth_depth']<10
                    and 35<=rf['side']<100 and rf['visible']>=.95
                    and rf['native_lap']>=100 and rf['lap']>=5
                    and min(v[1] for v in rf['eyes'])>=30 and min(v[2] for v in rf['eyes'])>=18)
                if confirmed_profile or detailed_profile:
                    kept.append(box);continue
            if restored is not None and min(restored.get('confirm',[0]))>=.7 and _source_detail(restored):
                delta=restored['points'][1]-restored['points'][0]
                angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
                rotated=local_observations(image,restored,detector,angles=(angle,))
                if _agree(rotated,minimum=.65,maximum=.7) is not None:kept.append(box)
    return kept


def _sharp_collapsed_profile(candidate):
    """Original facial core can remain sharp when profile eye axes collapse."""
    f=candidate['features']
    return (35<=f['side']<120 and f['visible']>=.65
            and f['eye_span']<.04 and f['mouth_depth']>5
            and f['native_lap']>=100 and f['lap']>=10
            and min(p[1] for p in f['eyes'][:4])>=60
            and min(p[2] for p in f['eyes'][:4])>=80)


def _rotation_envelope_supported(candidate, observations, source):
    """A large rotation-only frame must not promote a weak native texture.

    Reuse observations already measured for this recovery. Three crop sizes
    at one rotation are correlated evidence, especially around hair/ears.
    Strong native profiles and native-orientation support retain their path.
    """
    view = candidate['views'][0].split(':')
    if view[0] not in ('local', 'proposal'):
        return True
    angle = float(view[1] if view[0] == 'local' else view[3])
    f = candidate['features']
    if _sharp_collapsed_profile(candidate):return True
    if abs(angle) < 20 or f['side'] >= 120 or f['p50'] >= 100:
        return True
    related = [c for c in source if _same_anchor(candidate, c)]
    if not related or max(c['score'] for c in related) >= .35:
        return True
    nearest = max(related, key=lambda c: c['score'])
    original = nearest['box']
    if candidate['box'][2]*candidate['box'][3] <= 2*original[2]*original[3]:
        return True
    native = [c for c in observations if c['views'][0].startswith('local:0:')
              and _same_anchor(candidate, c)]
    return any(c['score'] >= .5 and _source_quality(c) is None for c in native)


def _linked_recovery_confirmation(image, candidate, detector, established):
    """A recovered overlapping proposal must confirm its own five-point face."""
    if candidate['score'] >= .8 or not established:
        return True
    x,y,width,height=candidate['box']
    if not any(evidence._overlap_smaller(candidate['box'], box) >= .2 for box in established):
        return True
    center=np.array([x+width/2,y+height/2])
    delta=candidate['points'][1]-candidate['points'][0]
    angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
    anchor=np.asarray(candidate['points'])[:3].mean(axis=0)
    for factor in (1.6,2.3):
        side=max(64,round(max(width,height)*factor))
        matrix=cv2.getRotationMatrix2D(tuple(center),angle,1.)
        matrix[:,2]+=np.array([side/2,side/2])-center
        view=cv2.warpAffine(image,matrix,(side,side))
        inverse=cv2.invertAffineTransform(matrix)
        boxes,points=detector.detect(view)
        valid=False
        for b,p in zip(boxes,points):
            mapped=np.column_stack([p,np.ones(len(p))])@inverse.T
            if np.linalg.norm(mapped[:3].mean(axis=0)-anchor)>.3*min(width,height):
                continue
            if np.linalg.norm(mapped[2]-candidate['points'][2])>.4*min(width,height):
                continue
            corners=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[0],b[3],1],[b[2],b[3],1]])@inverse.T
            lo,hi=corners.min(axis=0),corners.max(axis=0);box=[*lo,*(hi-lo)]
            if evidence._overlap_smaller(box,candidate['box'])<.4:
                continue
            if _represented({'box':box,'points':mapped},established):
                continue
            valid=True;break
        if not valid:return False
    return True


def unsupported_native_texture(candidate):
    """A weak fully visible textured proposal needs local face support."""
    f=candidate['features'];confirm=candidate.get('confirm',[1.,1.])
    return (candidate['score'] < .7 and min(confirm) < .2 and max(confirm) < .7
            and f['visible'] >= .9 and f['side'] < 120
            and f['eye_span'] >= .15 and .65 <= f['mouth_depth'] <= 2.5
            and f['p90'] < 130 and f['lap'] < 25)


def _partial_consensus(image,candidate,detector,raw):
    f=candidate['features'];h,w=image.shape[:2]
    if not (f['side']>=100 and .5<=f['visible']<.9 and candidate['score']>=.45):return None
    if f['native_lap']<20 or f['lap']<10 or min(p[1] for p in f['eyes'][:2])<25:return None
    flipped=cv2.flip(image,1)
    ds,ks=detector.detect(flipped,input_size=(640,640),det_thresh=.4)
    observations=[c for c in raw if c['score']>=.4 and evidence.same_location(c['box'],candidate['box']) and not _unrecoverable(c)]
    for b,p in zip(ds,ks):
        mapped=np.column_stack([w-1-p[:,0],p[:,1]])
        # Mirroring also swaps the semantic left/right eyes and mouth corners.
        mapped=mapped[[1,0,2,4,3]]
        c=_candidate(image,[w-1-b[2],b[1],b[2]-b[0],b[3]-b[1]],mapped,b[4],'mirror:640')
        if evidence.same_location(c['box'],candidate['box']) and c['features'].get('visible') and not _unrecoverable(c):observations.append(c)
    if len({c['views'][0] for c in observations})<3 or max(c['score'] for c in observations)<.6:return None
    return max(observations,key=lambda c:c['score'])


def _soft_profile(candidate):
    f=candidate['features'];eyes=f['eyes'][:2]
    if stable_soft_details(candidate):return True
    return (f['visible']>=.8 and f['lap']>=2.5 and f['p90']-f['p10']>=20
            and .18<=f['eye_span']<=.55 and .65<=f['mouth_depth']<=2.5
            and abs(f['nose_side'])<=1.2 and min(p[1] for p in eyes)>=25
            and min(p[2] for p in eyes)>=4 and f['eyes'][2][1]>=20
            and f['eyes'][3][1]>=8)


def _insufficient_compact_pixels(candidate):
    """Skip enlarging small proposals with no source eye or nose detail."""
    f=candidate['features']
    faint_features = (max(p[1] for p in f['eyes'][:3]) < 25
                      and max(p[2] for p in f['eyes'][:3]) < 12)
    tiny_without_core = (f['side'] < 40
                         and max(p[1] for p in f['eyes'][2:4]) < 15
                         and max(p[2] for p in f['eyes'][2:4]) < 5)
    return (f.get('visible',0)>=.9 and f['side']<60 and f['lap']<3
            and f['native_lap']<40 and (faint_features or tiny_without_core))


def proposal_views(image,detector):
    height,width=image.shape[:2];candidates=[]
    layouts=[(image,0,0,angle,pad) for angle,pad in ((0,.3),(0,.15),(-45,0),(45,0),(-20,0),(-45,.15))]
    short,long=min(width,height),max(width,height)
    if long>3*short:
        side=min(long,3*short)
        count=int(np.ceil((long-side)/max(side/2,1)))+1
        for start in np.linspace(0,long-side,count).astype(int):
            x,y=(0,start) if height>width else (start,0)
            view=image[y:y+(side if height>width else height),x:x+(side if width>height else width)]
            layouts.extend((view,x,y,angle,0) for angle in (0,-45))
    for source,ox,oy,angle,padding in layouts:
        h,w=source.shape[:2];matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        corners=np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]])@matrix.T
        low,high=corners.min(axis=0),corners.max(axis=0);pad=round(max(h,w)*padding)
        matrix[:,2]+=-low+pad;vw,vh=np.ceil(high-low+2*pad).astype(int)
        transformed=cv2.warpAffine(source,matrix,(vw,vh));inverse=cv2.invertAffineTransform(matrix)
        ds,ks=detector.detect(transformed,input_size=(320,320),det_thresh=.15)
        for d,points in zip(ds,ks):
            corners=np.array([[d[0],d[1],1],[d[2],d[1],1],[d[0],d[3],1],[d[2],d[3],1]])@inverse.T+np.array([ox,oy])
            lo,hi=corners.min(axis=0),corners.max(axis=0)
            mapped=np.column_stack([points,np.ones(len(points))])@inverse.T+np.array([ox,oy])
            candidates.append(_candidate(image,[*lo,*(hi-lo)],mapped,d[4],f'proposal:{ox}:{oy}:{angle}:{padding}'))
    return candidates


def _native_refine_recovered_box(image, candidate, detector):
    """Tighten a rotation envelope using three native-orientation predictions.

    This runs after face acceptance and pruning and only changes coordinates.
    Native predictions must agree on the same landmarks and a smaller frame.
    """
    original = candidate['box']
    view = candidate['views'][0].split(':')
    if view[0] == 'proposal':
        angle = float(view[3])
    elif view[0] == 'local':
        angle = float(view[1])
    else:
        return original
    if abs(angle) < 15 or candidate['features']['visible'] < .9:
        return original
    x, y, width, height = original
    if min(width, height) < 100:
        return original
    center = np.array([x + width/2, y + height/2])
    anchor = np.asarray(candidate['points'])[:3].mean(axis=0)
    frames = []
    scores = []
    for factor in (1.6, 2.3, 3.2):
        side = max(64, round(max(width, height)*factor))
        matrix = np.array([[1., 0., side/2-center[0]], [0., 1., side/2-center[1]]])
        native = cv2.warpAffine(image, matrix, (side, side))
        boxes, points = detector.detect(native, input_size=(320, 320), det_thresh=.5)
        matches = []
        for b, p in zip(boxes, points):
            mapped = p - matrix[:, 2]
            box = [float(b[0]-matrix[0, 2]), float(b[1]-matrix[1, 2]),
                   float(b[2]-b[0]), float(b[3]-b[1])]
            area = box[2]*box[3]/max(width*height, 1.)
            if not .25 <= area <= .75:
                continue
            if evidence._overlap_smaller(box, original) < .9:
                continue
            if np.linalg.norm(mapped[:3].mean(axis=0)-anchor) > .2*min(width, height):
                continue
            matches.append((float(b[4]), box))
        if not matches:
            return original
        score, box = max(matches, key=lambda r:r[0])
        frames.append(box)
        scores.append(score)
    if max(scores) < .7:
        return original
    for i, a in enumerate(frames):
        for b in frames[i+1:]:
            inter = evidence._overlap_smaller(a, b)*min(a[2]*a[3], b[2]*b[3])
            if inter/max(a[2]*a[3]+b[2]*b[3]-inter, 1.) < .65:
                return original
    corners = np.array([[a[0], a[1], a[0]+a[2], a[1]+a[3]] for a in frames])
    left, top, right, bottom = np.median(corners, axis=0)
    pad_x, pad_y = .03*(right-left), .03*(bottom-top)
    return [float(left-pad_x), float(top-pad_y),
            float(right-left+2*pad_x), float(bottom-top+2*pad_y)]


def refine_recovered_box(image, candidate, detector):
    """Keep established localization; tighten a confirmed inverted envelope."""
    original = candidate['box']
    refined = _native_refine_recovered_box(image, candidate, detector)
    if refined != original:
        return refined
    view = candidate['views'][0].split(':')
    angle = float(view[3]) if view[0] == 'proposal' else float(view[1]) if view[0] == 'local' else 0.
    x, y, width, height = original
    f = candidate['features']
    if not (30 <= abs(angle) <= 60 and f['visible'] >= .9
            and min(width, height) >= 100 and .9 <= width/height <= 1.1
            and .1 <= f['eye_span'] <= .2):
        return original
    center = np.array([x+width/2, y+height/2])
    anchor = np.asarray(candidate['points'])[:3].mean(axis=0)
    frames, landmarks, scores = [], [], []
    for factor in (1.6, 2.3, 3.2):
        side = max(64, round(max(width, height)*factor))
        matrix = cv2.getRotationMatrix2D(tuple(center), 180., 1.)
        matrix[:, 2] += np.array([side/2, side/2])-center
        upright = cv2.warpAffine(image, matrix, (side, side))
        inverse = cv2.invertAffineTransform(matrix)
        boxes, points = detector.detect(upright, input_size=(320, 320), det_thresh=.8)
        matches = []
        for b, p in zip(boxes, points):
            corners = np.array([[b[0],b[1],1],[b[2],b[1],1],
                                [b[0],b[3],1],[b[2],b[3],1]]) @ inverse.T
            low, high = corners.min(axis=0), corners.max(axis=0)
            box = [*low, *(high-low)]
            mapped = np.column_stack([p, np.ones(len(p))]) @ inverse.T
            span = np.linalg.norm(mapped[1]-mapped[0])
            if not (.25 <= box[2]*box[3]/(width*height) <= .55
                    and evidence._overlap_smaller(box, original) >= .95
                    and np.linalg.norm(mapped[:3].mean(axis=0)-anchor) <= .1*min(width,height)
                    and mapped[1,0] < mapped[0,0]
                    and mapped[3:,1].mean() < mapped[:2,1].mean()-.15*span):
                continue
            matches.append((float(b[4]), box, mapped))
        if not matches:
            return original
        score, box, mapped = max(matches, key=lambda item:item[0])
        scores.append(score); frames.append(box); landmarks.append(mapped)
    if max(scores) < .9:
        return original
    for i, a in enumerate(frames):
        for j, b in enumerate(frames[i+1:], i+1):
            inter = evidence._overlap_smaller(a,b)*min(a[2]*a[3],b[2]*b[3])
            if (inter/max(1,a[2]*a[3]+b[2]*b[3]-inter) < .85
                    or np.max(np.linalg.norm(landmarks[i]-landmarks[j],axis=1)) > .08*min(width,height)):
                return original
    corners = np.array([[b[0],b[1],b[0]+b[2],b[1]+b[3]] for b in frames])
    left, top, right, bottom = np.median(corners,axis=0)
    return [float(left),float(top),float(right-left),float(bottom-top)]


def _needs_additional_views(image, detector, candidates, present, anchors):
    """Escalate only when native predictions leave a plausible face unresolved."""
    if not present:
        return True
    # Additional legacy recoveries and locally confirmed primary proposals
    # are still uncertain. Keep the complete reassessment for these images.
    if len(present)!=len(anchors) or any(
            c['source_candidate']['confirm'] != [1.,1.] for c in anchors):
        return True
    for c in candidates:
        f=c['features']
        if _unrecoverable(c) or _edge_damage(image,c) or _insufficient_compact_pixels(c):
            continue
        if c['score']>=.25 and (f['side']>=100 or f['lap']>=2.5 or f['native_lap']>=40):
            return True
        if _visible_profile_core(image,c):
            return True
        if not (f['visible']>=.8 and .18<=f['eye_span']<=.55
                and .65<=f['mouth_depth']<=2.5 and abs(f['nose_side'])<=.8
                and f['native_lap']>=30 and min(p[1] for p in f['eyes'][:3])>=15):
            continue
        native=local_observations(image,c,detector)
        if sum(o['score']>=.65 and _source_quality(o) is None for o in native)>=2:
            return True
    return False


def _dim_compact_native_core(candidate):
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    return (candidate['score']>=.6 and 35<=f['side']<80 and f['visible']>=.9
            and f['points_inside']>=4 and f['native_lap']>=40 and f['lap']>=2.5
            and f['p90']<100 and .18<=f['eye_span']<=.5
            and .7<=f['mouth_depth']<=2 and abs(f['nose_side'])<=.5
            and min(v[1] for v in eyes)>=10 and min(v[2] for v in eyes)>=2.5
            and nose[1]>=40 and nose[2]>=20 and mouth[1]>=20 and mouth[2]>=10)


def _bright_large_lateral_core(candidate):
    f=candidate['features'];eyes=f['eyes'][:2];nose,mouth=f['eyes'][2:4]
    return (candidate['score']>=.75 and f['side']>=300 and f['visible']>=.9
            and f['points_inside']>=4 and f['native_lap']>=1.2 and f['lap']>=3.25
            and f['p10']>=160 and f['p50']>=200 and f['p90']<245
            and .18<=f['eye_span']<=.35 and 1.3<=f['mouth_depth']<=2
            and .6<=abs(f['nose_side'])<=1.2
            and min(v[1] for v in eyes)>=30 and min(v[2] for v in eyes)>=3
            and nose[1]>=25 and nose[2]>=3 and mouth[1]>=12 and mouth[2]>=1.5)


def _confirmed_compact_detail(candidate):
    """Readable original lip pixels plus two confirmations for a small face."""
    f=candidate['features'];patches=f['eyes'];eyes=patches[:2];nose,mouth=patches[2:4]
    return (29<=f['side']<35 and f['visible']>=.95 and f['points_inside']>=4
            and candidate['score']>=.65 and min(candidate.get('confirm',[0]))>=.7
            and max(candidate.get('confirm',[0]))>=.75 and f['native_lap']>=100 and f['lap']>=3
            and .18<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2 and abs(f['nose_side'])<=.8
            and min(p[1] for p in eyes)>=25 and min(p[2] for p in eyes)>=4
            and nose[1]>=25 and nose[2]>=2.5 and mouth[1]>=10 and mouth[2]>=2
            and not quality.blocky_eye_pixels(candidate) and not quality.severe_pixel_blocks(candidate))


def _sharp_profile_source(candidate):
    f=candidate['features']
    return (candidate['score']>=.55 and f['visible']>=.95 and f['points_inside']>=4
            and 35<=f['side']<100 and f['native_lap']>=80 and f['lap']>=5
            and .03<=f['eye_span']<.15 and 2.5<f['mouth_depth']<7 and abs(f['nose_side'])<=1
            and min(p[1] for p in f['eyes'])>=25 and min(p[2] for p in f['eyes'])>=15)


def supplement_pruned_profiles(image,detector,present,recovered,source_pool,anchors=()):
    """Recheck tight native seeds only where prune removed an admitted face."""
    removed=[c for c in recovered if not any(c['box'] is b for b in present)]
    if not removed:return []
    added=[];attempted=[]
    for seed in source_pool:
        if not _sharp_profile_source(seed):continue
        if (_represented(seed,present+[c['box'] for c in added])
                or not any(_same_anchor(seed,c['source_candidate']) for c in removed)
                or any(_same_anchor(seed,c) for c in attempted)):continue
        if (_edge_damage(image,seed) or _upper_damage(seed) or _unrecoverable(seed)
                or _source_quality(seed) is not None):continue
        seed['confirm']=evidence.confirm(image,seed,detector)
        if min(seed['confirm'])<.6 or max(seed['confirm'])<.65:continue
        attempted.append(seed)
        observed=local_observations(image,seed,detector,angles=(0,))
        good=[c for c in observed if c['score']>=.6 and _same_anchor(seed,c)
              and c['features']['visible']>=.95 and c['features']['points_inside']>=4
              and c['features']['native_lap']>=40
              and not _edge_damage(image,c) and not _unrecoverable(c)]
        if not any(_same_anchor(a,b) and evidence._overlap_smaller(a['box'],b['box'])>=.7
                   for i,a in enumerate(good) for b in good[i+1:]):continue
        if (_source_veto(image,seed,source_pool,detector) or unsupported_native_texture(seed)
                or evidence.inconsistent_texture(image,seed,detector)
                or not _linked_recovery_confirmation(image,seed,detector,present)):continue
        added.append({'box':list(seed['box']),'score':seed['score'],'source_candidate':seed})
    photographic=[dict(c,score=max(c['score'],min(c['source_candidate'].get('confirm',[0])))) for c in anchors]
    kept=graphics.filter_candidates(image,photographic+added)
    return [c for c in added if any(c is k for k in kept)]


def _large_profile_source(candidate):
    f=candidate['features']
    return (candidate['score']>=.5 and f['visible']>=.95 and f['points_inside']>=4
            and 120<=f['side']<250 and f['native_lap']>=30 and f['lap']>=20
            and .025<=f['eye_span']<.13 and 3<f['mouth_depth']<10
            and min(p[1] for p in f['eyes'][:2])>=75 and min(p[2] for p in f['eyes'][:2])>=50
            and f['eyes'][3][1]>=40 and f['eyes'][3][2]>=15)


def supplement_missing_profiles(image,detector,present,source_pool,anchors=()):
    """Confirm original native side faces after all ordinary/prune paths finish."""
    added=[];attempted=[]
    for seed in source_pool:
        if not (_sharp_profile_source(seed) or _large_profile_source(seed)):continue
        if (_represented(seed,present+[c['box'] for c in added])
                or any(_same_anchor(seed,c) for c in attempted)):continue
        if (_edge_damage(image,seed) or _upper_damage(seed) or _unrecoverable(seed)
                or _source_quality(seed) is not None or not _supplement_quality_safe(seed)):continue
        seed['confirm']=evidence.confirm(image,seed,detector)
        large=_large_profile_source(seed)
        strong_native=(not large and seed['score']>=.7 and seed['features']['native_lap']>=100)
        if min(seed['confirm'])<(.7 if large else .6) or max(seed['confirm'])<(.7 if large else (.64 if strong_native else .65)):continue
        attempted.append(seed)
        observed=local_observations(image,seed,detector,angles=(0,))
        delta=seed['points'][1]-seed['points'][0]
        axis=float(np.degrees(np.arctan2(delta[1],delta[0])))
        aligned=local_observations(image,seed,detector,angles=(axis,))
        if large:
            good=[c for c in aligned if c['score']>=.6 and _same_anchor(seed,c)
                  and c['features']['visible']>=.95 and c['features']['points_inside']>=4
                  and min(p[1] for p in c['features']['eyes'])>=40
                  and min(p[2] for p in c['features']['eyes'])>=30
                  and not _edge_damage(image,c) and not _unrecoverable(c)]
        else:
            good=[c for c in observed+aligned if c['score']>=.55 and _same_anchor(seed,c)
                  and c['features']['visible']>=.95 and c['features']['points_inside']>=4
                  and min(p[1] for p in c['features']['eyes'])>=25
                  and min(p[2] for p in c['features']['eyes'])>=15
                  and not _edge_damage(image,c) and not _unrecoverable(c)]
        if not (strong_native and any(c['score']>=.55 for c in good)) and not any(len(cluster:=[b for b in good if _same_anchor(a,b)
                    and evidence._overlap_smaller(a['box'],b['box'])>=.7])>=3
                   and max(b['score'] for b in cluster)>=.6 for a in good):continue
        if (_source_veto(image,seed,source_pool,detector,allow_profile_landmarks=large)
                or unsupported_native_texture(seed) or evidence.inconsistent_texture(image,seed,detector)
                or not _linked_recovery_confirmation(image,seed,detector,present)):continue
        added.append({'box':list(seed['box']),'score':seed['score'],'source_candidate':seed})
    photographic=[dict(c,score=max(c['score'],min(c['source_candidate'].get('confirm',[0])))) for c in anchors]
    kept=graphics.filter_candidates(image,photographic+added)
    return [c for c in added if any(c is k for k in kept)]


def _top_boundary_source(image,candidate):
    f=candidate['features'];x,y,w,h=candidate['box'];height,width=image.shape[:2]
    return (candidate['score']>=.25 and x>=0 and x+w<=width and y<0
            and .6<=f['visible']<1 and 35<=f['side']<90 and f['points_inside']>=4
            and f['native_lap']>=30 and f['lap']>=1.8
            and .15<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2.5 and abs(f['nose_side'])<=.8
            and all(0<=px<width and 0<=py<height for px,py in candidate['points'])
            and max(p[1] for p in f['eyes'][:2])>=20 and max(p[2] for p in f['eyes'][:2])>=8
            and f['eyes'][2][1]>=40 and f['eyes'][2][2]>=15
            and f['eyes'][3][1]>=25 and f['eyes'][3][2]>=15)


def supplement_top_boundary_faces(image,detector,present,source_pool,anchors=()):
    """Re-sample retained eye/nose/mouth pixels after native landmark truncation."""
    added=[];attempted=[];height,width=image.shape[:2]
    # This rescue corrects native eye patches truncated by the source top edge.
    # Use exactly the existing quality.features patch radius, not a new margin.
    native_sources=[c for c in source_pool if c['views'] in (['whole:0'],['scale:960'])
                    and _top_boundary_source(image,c)
                    and any(round(float(py))-max(3,round(c['features']['side']*.09))<0
                            for px,py in c['points'][:2])]
    for seed in sorted(source_pool,key=lambda c:c['score'],reverse=True):
        if not any(_same_anchor(seed,c) for c in native_sources):continue
        if not _top_boundary_source(image,seed) or _represented(seed,present+[c['box'] for c in added]):continue
        if any(_same_anchor(seed,c) for c in attempted):continue
        attempted.append(seed)
        observed=local_observations(image,seed,detector,angles=(0,20,45,60))
        good=[c for c in observed if c['score']>=.25 and _same_anchor(seed,c)
              and all(0<=px<width and .02*c['features']['side']<=py<height for px,py in c['points'])
              and min(p[1] for p in c['features']['eyes'])>=20
              and min(p[2] for p in c['features']['eyes'])>=10
              and c['features']['eyes'][2][1]>=40
              and _source_quality(c) is None and not _unrecoverable(c)
              and not _edge_damage(image,c)]
        seed_pixels=(min(p[1] for p in seed['features']['eyes'])>=20
                     and min(p[2] for p in seed['features']['eyes'])>=10
                     and _source_quality(seed) is None
                     and all(0<=px<width and .02*seed['features']['side']<=py<height for px,py in seed['points']))
        cluster=[]
        for c in good:
            nearby=[o for o in good if _same_anchor(c,o) and evidence._overlap_smaller(c['box'],o['box'])>=.7]
            axes={o['views'][0].split(':')[1] for o in nearby}
            if len(nearby)>=3 and len(axes)>=2 and max([o['score'] for o in nearby]+([seed['score']] if seed_pixels else []))>=.5:
                cluster=nearby;break
        if not cluster:continue
        best=seed if seed_pixels else max(cluster,key=lambda c:c['score'])
        candidate=_candidate(image,seed['box'],best['points'],seed['score'],'whole:0')
        # Local landmarks may belong to an already represented face below the seed.
        if _represented(candidate,present+[c['box'] for c in added]):continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        if min(candidate['confirm'])<.5 or max(candidate['confirm'])<.55:continue
        if (_source_quality(candidate) is not None or _unrecoverable(candidate)
                or _upper_damage(candidate) or _edge_damage(image,candidate)
                or _source_veto(image,candidate,source_pool,detector)
                or unsupported_native_texture(candidate) or evidence.inconsistent_texture(image,candidate,detector)
                or not _linked_recovery_confirmation(image,candidate,detector,present)):continue
        x,y,w,h=candidate['box'];box=[x,0.,w,y+h]
        added.append({'box':box,'score':candidate['score'],'source_candidate':candidate})
    photographic=[dict(c,score=max(c['score'],min(c['source_candidate'].get('confirm',[0])))) for c in anchors]
    kept=graphics.filter_candidates(image,photographic+added)
    return [c for c in added if any(c is k for k in kept)]


def recover(image,detector,present,anchors=(),*,collect_candidates=False,source_pool=None):
    fragment=_blocking_fragment(image,detector) if present else None
    height,width=image.shape[:2];detections,points=detector.detect(image,det_thresh=.05)
    larger,larger_points=detector.detect(image,input_size=(960,960),det_thresh=.15)
    pools=[(detections,points,'whole:0'),(larger,larger_points,'scale:960')]
    candidates=[]
    raw=[];source=[]
    for boxes,landmarks,view in pools:
        for b,p in zip(boxes,landmarks):
            if b[4]<.15 and min(b[2]-b[0],b[3]-b[1])<29:
                continue
            candidate=_candidate(image,[b[0],b[1],b[2]-b[0],b[3]-b[1]],p,b[4],view)
            f=candidate['features']
            if source_pool is not None and (_sharp_profile_source(candidate) or _large_profile_source(candidate) or _top_boundary_source(image,candidate)):source_pool.append(candidate)
            if f.get('visible') and view=='whole:0':source.append(candidate)
            if not f.get('visible') or f['visible']<.4 or (f['side']<29 and not compact_pixels(candidate)):continue
            if candidate['score']<.15 and not (_visible_profile_core(image,candidate) or (f['side']>=35 and .18<=f['eye_span']<=.55
                    and .65<=f['mouth_depth']<=2.5 and abs(f['nose_side'])<=.8
                    and f['native_lap']>=30 and min(p[1] for p in f['eyes'][:2])>=25
                    and min(p[2] for p in f['eyes'][:2])>=8
                    and min(p[1] for p in f['eyes'][2:])>=25)):continue
            if _represented(candidate,present):continue
            raw.append(candidate)
    if _needs_additional_views(image,detector,raw,present,anchors):
        pooled=proposal_views(image,detector)
        raw.extend(c for c in pooled if c['features'].get('visible',0)>=.4
                   and c['features']['side']>=29 and not _represented(c,present))
    else:
        # Ordinary recovery has no work. Still inspect already available native
        # seeds in the narrow source-core path; do not launch proposal search.
        raw=[]
    raw=[c for c in raw if not _unrecoverable(c) and not _edge_damage(image,c)]
    for candidate in sorted(raw,key=lambda c:(_source_quality(c) is None,c['features']['visible']>=.9 and c['features']['points_inside']>=4,c['views']==['whole:0'] and c['score']>=.5,c['score']),reverse=True):
            # A third native prediction can refine a real face, but an almost
            # absent prediction must not seed another search around that face.
            limit=3 if candidate['score']>=.15 else 2
            if sum(_same_anchor(candidate,c) for c in candidates)>=limit:continue
            candidates.append(candidate)
    if source_pool is not None:
        source_pool.extend(c for c in raw if _top_boundary_source(image,c) and c['views'] not in (['whole:0'],['scale:960']))
    recovered=[];admitted=[];deferred_profiles=[];deferred_sharp=[];deferred_compact=[]
    def ordinary_then_deferred():
        for item in candidates:yield item,False
        for item in deferred_profiles:yield item,True
    for candidate,deferred_accept in ordinary_then_deferred():
        left,top,box_width,box_height=candidate['box'];right,bottom=left+box_width,top+box_height;score=candidate['score']
        f=candidate['features']
        if not f.get('visible') or f['visible']<.4:continue
        if _unrecoverable(candidate):continue
        if _edge_damage(image,candidate):continue
        if fragment is not None and evidence.same_location(fragment['box'],candidate['box']):continue
        if _source_veto(image,candidate,source,detector):continue
        if _insufficient_compact_pixels(candidate):continue
        if (f['side']<100 and f['eye_span']<.13 and f['visible']>=.9
                and f['p50']<140 and box_height/max(box_width,1)>1.5):continue
        if _represented(candidate,present+recovered):continue
        related=[c for c in source if evidence.same_location(c['box'],candidate['box'])]
        strongest=max((c['score'] for c in related),default=0.)
        boundary_profile=any(_visible_profile_core(image,c) for c in related)
        if (f['visible']>=.9 and f['eye_span']>=.18 and strongest<.25 and not boundary_profile):
            upright=local_observations(image,candidate,detector)
            supported=sum(c['score']>=.65 and _source_quality(c) is None for c in upright)
            if supported<2:
                delta=candidate['points'][1]-candidate['points'][0]
                angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
                if 20<=abs(angle)<=160:
                    upright.extend(local_observations(image,candidate,detector,angles=(angle,)))
                    supported=sum(c['score']>=.65 and _source_quality(c) is None for c in upright)
            native_dark=(f['side']>=120 and 24<=f['p90']<45 and f['core_p90']>=24
                         and f['p90']-f['p10']>=18 and f['p50']>=12
                         and min(p[1] for p in f['eyes'][:2])>=12
                         and min(p[2] for p in f['eyes'][:2])>=2)
            native_edge=(.15<=candidate['score']<.25
                         and (candidate['box'][0]<0 or candidate['box'][0]+candidate['box'][2]>width)
                         and 40<=f['side']<120 and f['visible']>=.75
                         and f['native_lap']>=40 and _source_quality(candidate) is None
                         and _source_detail(candidate))
            if supported<2 and native_edge:
                candidate['confirm']=evidence.confirm(image,candidate,detector)
                native_edge=min(candidate['confirm'])>=.5
            if supported<2 and dark_source_core(candidate):
                enhanced=local_observations(image,candidate,detector,angles=(0,-45,45),enhance=True)
                supported=2 if _agree(enhanced,minimum=.5,maximum=.6) is not None else 0
            if supported<2 and not (native_dark or native_edge):continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        if deferred_accept:
            accepted=True
        else:
            reason=_source_quality(candidate)
            if reason in ('FACE_BLURRY','FACE_OCCLUDED') and not (_soft_profile(candidate) or ((f['lap']>=2.5 or f['native_lap']>=30) and min(p[1] for p in f['eyes'][:2])>=25 and f['eyes'][2][1]>=40 and f['eyes'][3][1]>=18)):
                localized=None
                if (f['native_lap']>=40 and f['lap']>=2.5
                        and .15<=f['eye_span']<=.55 and .65<=f['mouth_depth']<=2.5
                        and abs(f['nose_side'])<=.8
                        and min(p[1] for p in f['eyes'][:2])>=12
                        and min(p[2] for p in f['eyes'][:2])>=2.5
                        and f['eyes'][2][1]>=40 and f['eyes'][3][1]>=12):
                    localized=_agree(local_observations(image,candidate,detector,angles=(0,-45,45)),minimum=.5,maximum=.65)
                if localized is None:continue
                candidate=localized;left,top,box_width,box_height=candidate['box'];right,bottom=left+box_width,top+box_height
                f=candidate['features'];score=candidate['score'];reason=_source_quality(candidate)
            if compact_visible_face(candidate):
                accepted=True
            elif (score>=.65 and reason is None and 35<=f['side']<120
                    and f['visible']>=.9 and .18<=f['eye_span']<=.5
                    and .65<=f['mouth_depth']<=2.5 and abs(f['nose_side'])<=.8
                    and f['native_lap']>=40 and f['lap']>=2.5
                    and min(candidate['confirm'])>=.65):
                accepted=True
            elif score>=.25 and reason in ('FACE_BLURRY','FACE_TOO_SMALL',None) and min(candidate['confirm'])>=.7 and _soft_profile(candidate):
                accepted=True
            elif (score>=.5 and reason is None and f['side']>=100 and f['eye_span']<.12
                  and min(candidate['confirm'])>=.55 and min(p[1] for p in f['eyes'])>=25):
                accepted=True
            elif .25<=score<.5 and reason is None and min(candidate['confirm'])>=.75 and max(candidate['confirm'])>=.8:
                accepted=True
            elif score>=.5 and reason is None and f['eye_span']<.2:
                native=local_observations(image,candidate,detector)
                strong=[c for c in native if c['score']>=.6 and _source_quality(c) is None]
                accepted=len(strong)>=2 and max(c['score'] for c in strong)>=.7 and all(evidence._overlap_smaller(a['box'],b['box'])>=.7 for a in strong for b in strong)
            else:accepted=False
            if not accepted:
                partial=_partial_consensus(image,candidate,detector,raw)
                if partial is not None:
                    candidate=partial;left,top,box_width,box_height=partial['box'];right,bottom=left+box_width,top+box_height;accepted=True
            if not accepted and reason in (None,'FACE_BLURRY','FACE_OCCLUDED','FACE_TOO_SMALL'):
                observations=local_observations(image,candidate,detector,angles=(0,-45,45))
                observations.extend(c for c in raw if c['score']>=.5
                                    and evidence.same_location(c['box'],candidate['box']) and not _unrecoverable(c))
                best=_agree(observations,minimum=.45 if f['visible']<.85 else .5,maximum=.7 if f['visible']<.85 else .6)
                if best is None and (left<0 or left<.15*box_width or right>width) and 60<=f['side']<120 and f['native_lap']>=40 and f['lap']>=10:
                    profiles=[o for o in observations if .5<=o['score'] and _source_quality(o) is None and not _edge_damage(image,o)
                              and max(min(float(px),width-float(px)) for px,py in o['points'][3:])>=.12*o['features']['side']]
                    fallback=_agree(profiles,minimum=.5,maximum=.55)
                    if fallback is not None and _source_detail(fallback) and _rotation_envelope_supported(fallback,observations,source):
                        deferred_profiles.append(fallback)
                if best is not None:
                    bf=best['features']
                    if (bf['side']>=35 and bf['visible']>=.6 and bf['p90']>=45 and _source_detail(best)
                            and _rotation_envelope_supported(best,observations,source)):
                        candidate=best;left,top,box_width,box_height=best['box'];right,bottom=left+box_width,top+box_height
                        accepted=True
                if not accepted and _visible_profile_core(image,candidate):
                    observations.extend(local_observations(image,candidate,detector,angles=(0,),enhance=True))
                    supported=[c for c in observations if c['score']>=.2 and c['features'].get('visible') and _visible_profile_core(image,c)]
                    if len(supported)>=3 and max(c['score'] for c in supported)>=.3:
                        candidate=max(supported,key=lambda c:c['score']);left,top,box_width,box_height=candidate['box'];right,bottom=left+box_width,top+box_height
                        accepted=True
                if not accepted:
                    observations.extend(local_observations(image,candidate,detector,angles=(-20,20)))
                    best=_agree(observations,minimum=.45 if f['visible']<.85 else .5,maximum=.7 if f['visible']<.85 else .6)
                    if best is None and (left<0 or left<.15*box_width or right>width) and 60<=f['side']<120 and f['native_lap']>=40 and f['lap']>=10:
                        profiles=[o for o in observations if .5<=o['score'] and _source_quality(o) is None and not _edge_damage(image,o)
                                  and max(min(float(px),width-float(px)) for px,py in o['points'][3:])>=.12*o['features']['side']]
                        fallback=_agree(profiles,minimum=.5,maximum=.55)
                        if fallback is not None and _source_detail(fallback) and _rotation_envelope_supported(fallback,observations,source):
                            deferred_profiles.append(fallback)
                    if (best is not None and _source_detail(best)
                            and _rotation_envelope_supported(best,observations,source)):
                        candidate=best;left,top,box_width,box_height=best['box'];right,bottom=left+box_width,top+box_height
                        accepted=True
            if not accepted and _half_face_consensus(image,candidate,detector):
                accepted=True
            if (not accepted and f['side']>=120 and 24<=f['p90']<45
                    and f['core_p90']>=24 and f['p90']-f['p10']>=18 and f['p50']>=12
                    and .18<=f['eye_span']<=.5 and .65<=f['mouth_depth']<=2.5
                    and abs(f['nose_side'])<=.8
                    and min(p[1] for p in f['eyes'][:2])>=12
                    and min(p[2] for p in f['eyes'][:2])>=2):
                best=_agree(local_observations(image,candidate,detector,angles=(0,-45,45),enhance=True),minimum=.25,maximum=.45)
                if best is not None and _source_detail(best):
                    candidate=best;left,top,box_width,box_height=best['box'];right,bottom=left+box_width,top+box_height;accepted=True
            if (not accepted and .15<=score<.25 and (left<0 or right>width)
                    and 40<=f['side']<120 and f['visible']>=.75 and f['native_lap']>=40
                    and min(candidate['confirm'])>=.5 and _source_quality(candidate) is None
                    and _source_detail(candidate)):
                best=_agree(local_observations(image,candidate,detector,angles=(0,-45,45)),minimum=.5,maximum=.55)
                if best is not None and _source_detail(best):
                    candidate=best;left,top,box_width,box_height=best['box'];right,bottom=left+box_width,top+box_height;accepted=True
            if not accepted:continue
        if _unrecoverable(candidate):continue
        if candidate['features']['mouth_depth']<.65:continue
        if _edge_damage(image,candidate):continue
        if _source_veto(image,candidate,source,detector):continue
        compact_supplement=not _source_detail(candidate)
        if compact_supplement and not _confirmed_compact_detail(candidate):continue
        if _represented(candidate,present+recovered):continue
        if not _linked_recovery_confirmation(image,candidate,detector,present+recovered):continue
        strong_faces = [c['box'] for c in anchors if c['score'] >= .8]
        if (candidate['score'] < .5 and candidate['features']['side'] < 120
                and candidate['features']['visible'] >= .9
                and candidate['features']['eye_span'] >= .18
                and any(evidence._overlap_smaller(candidate['box'], b) >= .2 for b in strong_faces)
                and not _distinct_local_confirmation(image, candidate, detector, strong_faces)):
            continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        cf=candidate['features']
        # A new frontal proposal clipped above the image lacks a facial core.
        if (candidate['score']<.5 and candidate['box'][1]<0 and cf['visible']<.85
                and cf['eye_span']>=.15 and cf['points_inside']<4
                and max(candidate['confirm'])<.2):continue
        if unsupported_native_texture(candidate):continue
        # Rotation consensus may agree on textured ears, fingers or a phone
        # while both eye-aligned contexts find no face at all. Existing faces
        # never enter this gate; strong native source evidence still wins.
        if (min(candidate['confirm']) < .2 and max(candidate['confirm']) < .7 and cf['side'] < 120
                and not _sharp_collapsed_profile(candidate)
                and (cf['visible'] >= .9
                     or (cf['visible'] >= .6 and .06 <= cf['eye_span'] < .15
                         and cf['mouth_depth'] <= 4))
                and not any(c['score'] >= .5 and _same_anchor(candidate,c) for c in source)):
            continue
        if (candidate['score'] < .65 and cf['p10'] > 150
                and cf['eye_span'] >= .15 and min(candidate['confirm']) < .8
                and cf['eyes'][3][1] < 15 and cf['eyes'][3][2] < 3):
            continue
        frontal=(cf['eye_span']>=.18 and .5<cf['mouth_depth']<1.8 and abs(cf['nose_side'])<.5)
        source_profiles=[c for c in source if c['score']>=.5 and evidence.same_location(c['box'],candidate['box'])
                         and c['features']['eye_span']<.18 and c['features']['mouth_depth']>=1.8
                         and not _unrecoverable(c)]
        if frontal and not source_profiles and evidence.inconsistent_texture(image,candidate,detector):continue
        if (candidate['features']['side']>=120 and candidate['features']['visible']>=.9
                and candidate['features']['lap']>=60 and candidate['features']['eye_span']>=.18
                and min(candidate['confirm'])<.82 and not source_profiles):
            deferred_sharp.append(candidate)
            continue
        if compact_supplement:
            deferred_compact.append(candidate)
            continue
        x1,y1=max(0.,float(left)),max(0.,float(top));x2,y2=min(float(width),float(right)),min(float(height),float(bottom))
        if x2>x1 and y2>y1:
            box=[x1,y1,x2-x1,y2-y1];recovered.append(box)
            admitted.append({'box':box,'score':candidate['score'],'source_candidate':candidate})
    # Sharp consensus is supplemental: ordinary candidates keep their first box.
    for candidate in deferred_sharp:
        if _represented(candidate,present+recovered):continue
        delta=candidate['points'][1]-candidate['points'][0]
        angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
        checked=local_observations(image,candidate,detector,angles=(angle,))
        if _agree(checked,minimum=.65,maximum=.75) is None:continue
        x,y,bw,bh=candidate['box'];x1,y1=max(0.,float(x)),max(0.,float(y));x2,y2=min(float(width),float(x+bw)),min(float(height),float(y+bh))
        if x2>x1 and y2>y1:
            box=[x1,y1,x2-x1,y2-y1];recovered.append(box)
            admitted.append({'box':box,'score':candidate['score'],'source_candidate':candidate})
    # Preserve the complete ordinary path first; readable rescue never replaces it.
    attempted=[]
    for seed in source+candidates:
        descriptor=dict(seed,confirm=[1.,1.])
        if not (soft_readable_source(descriptor) or dark_source_core(descriptor) or dim_readable_core(descriptor)):continue
        if _represented(seed,present+recovered) or any(_same_anchor(seed,c) for c in attempted):continue
        if fragment is not None and evidence.same_location(fragment['box'],seed['box']):continue
        seed['confirm']=evidence.confirm(image,seed,detector)
        readable=(soft_readable_source(seed) or dark_source_core(seed) or dim_readable_core(seed))
        if not readable or _edge_damage(image,seed) or _upper_damage(seed):continue
        if not _post_recoverable_source(seed):continue
        if _source_veto(image,seed,source,detector,allow_source_soft=True):continue
        if quality.blocky_eye_pixels(seed) or quality.severe_pixel_blocks(seed):continue
        # Genuine one-eye damage cannot be repaired by changing the sampling angle.
        if quality.asymmetric_blur(seed) and not (soft_bilateral_source(seed) or _detailed_soft_bilateral(seed)):continue
        attempted.append(seed)
        delta=seed['points'][1]-seed['points'][0]
        angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
        native_identity=(seed['views']==['whole:0'] and seed['score']>=.65
                         and min(seed['confirm'])>=.7 and max(seed['confirm'])>=.75
                         and soft_readable_source(seed))
        observations=local_observations(image,seed,detector,angles=(angle,),size=640,enhance=dark_source_core(seed),
                                        factors=(1.6,2.3) if native_identity else (1.6,2.3,3.2))
        if native_identity:observations.append(seed)
        good=[c for c in observations if c['score']>=.6 and _same_anchor(seed,c)
              and c['features']['visible']>=.6 and c['features']['points_inside']>=3
              and .15<=c['features']['eye_span']<=.55
              and .65<=c['features']['mouth_depth']<=2.5
              and abs(c['features']['nose_side'])<=1.2
              and not quality.blocky_eye_pixels(c) and not quality.severe_pixel_blocks(c)
              and not _edge_damage(image,c)]
        best=None
        for c in good:
            cluster=[o for o in good if _same_anchor(c,o) and evidence._overlap_smaller(c['box'],o['box'])>=.7]
            if len(cluster)>=3 and max(o['score'] for o in cluster)>=(.65 if dim_readable_core(seed) else .75):
                best=seed if native_identity and any(o is seed for o in cluster) else max(cluster,key=lambda o:o['score']);break
        if best is None or _represented(best,present+recovered):continue
        best['confirm']=evidence.confirm(image,best,detector)
        if min(best['confirm'])<.6 or evidence.inconsistent_texture(image,best,detector):continue
        if not _post_recoverable_source(best) or _upper_damage(best):continue
        if _source_veto(image,best,source,detector,allow_source_soft=True):continue
        if quality.asymmetric_blur(best) and not (soft_bilateral_source(best) or _detailed_soft_bilateral(best)):continue
        x,y,bw,bh=best['box'];x1,y1=max(0.,float(x)),max(0.,float(y));x2,y2=min(float(width),float(x+bw)),min(float(height),float(y+bh))
        if x2>x1 and y2>y1:
            box=[x1,y1,x2-x1,y2-y1];recovered.append(box)
            admitted.append({'box':box,'score':best['score'],'source_candidate':best})
    # Additional rescue runs after all ordinary decisions and keeps accepted boxes.
    for seed in source:
        if seed['views']!=['whole:0'] or _represented(seed,present+recovered):continue
        dim=_dim_compact_native_core(seed);bright=_bright_large_lateral_core(seed)
        if not (dim or bright):continue
        if (_edge_damage(image,seed) or _upper_damage(seed) or quality.asymmetric_blur(seed)
                or quality.blocky_eye_pixels(seed) or quality.severe_pixel_blocks(seed)
                or _source_veto(image,seed,source,detector,allow_source_soft=bright)):continue
        seed['confirm']=evidence.confirm(image,seed,detector)
        if dim and (min(seed['confirm'])<.55 or max(seed['confirm'])<.65):continue
        if bright and (min(seed['confirm'])<.7 or max(seed['confirm'])<.75):continue
        selected=seed
        if dim:
            observed=local_observations(image,seed,detector,angles=(0,-45,45))
            good=[c for c in observed if c['score']>=.5 and _same_anchor(c,seed)
                  and c['features']['visible']>=.9]
            cluster=next((v for c in good if len(v:=[o for o in good if _same_anchor(c,o)
                         and evidence._overlap_smaller(c['box'],o['box'])>=.7])>=3),None)
            if cluster is None or max(c['score'] for c in cluster)<.65:continue
            selected=max(cluster,key=lambda c:c['score'])
            if _represented(selected,present+recovered):continue
            selected['confirm']=evidence.confirm(image,selected,detector)
        if (min(selected.get('confirm',[0]))<(.55 if dim else .7)
                or max(selected.get('confirm',[0]))<(.65 if dim else .75)
                or _edge_damage(image,selected) or _upper_damage(selected)
                or quality.asymmetric_blur(selected) or quality.blocky_eye_pixels(selected)
                or quality.severe_pixel_blocks(selected)
                or _source_veto(image,selected,source,detector,allow_source_soft=bright)
                or not _linked_recovery_confirmation(image,selected,detector,present+recovered)):
            continue
        if dim and _unrecoverable(selected):continue
        x,y,w,h=selected['box'];box=[max(0.,x),max(0.,y),min(float(width),x+w)-max(0.,x),min(float(height),y+h)-max(0.,y)]
        if min(box[2:])<=0:continue
        recovered.append(box);admitted.append({'box':box,'score':selected['score'],'source_candidate':selected})
    # New compact evidence is supplemental to every established recovery path.
    for candidate in deferred_compact:
        if _represented(candidate,present+recovered):continue
        if not _linked_recovery_confirmation(image,candidate,detector,present+recovered):continue
        x,y,bw,bh=candidate['box'];x1,y1=max(0.,float(x)),max(0.,float(y));x2,y2=min(float(width),float(x+bw)),min(float(height),float(y+bh))
        if x2>x1 and y2>y1:
            box=[x1,y1,x2-x1,y2-y1];recovered.append(box)
            admitted.append({'box':box,'score':candidate['score'],'source_candidate':candidate})
    photographic=[dict(c,score=max(c['score'],min(c['source_candidate'].get('confirm',[0])))) for c in anchors]
    filtered=graphics.filter_candidates(image,photographic+admitted)
    kept=[c for c in admitted if any(c is kept for kept in filtered)]
    return kept if collect_candidates else [c['box'] for c in kept]


