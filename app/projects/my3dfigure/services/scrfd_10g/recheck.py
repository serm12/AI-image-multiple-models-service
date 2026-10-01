"""Additional evidence for uncertain faces in any submitted photograph."""
from __future__ import annotations

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


def local_observations(image,candidate,detector,angles=(0,),size=320,enhance=False):
    x,y,width,height=candidate['box'];observations=[]
    center=(x+width/2,y+height/2)
    for angle in angles:
        for factor in (1.6,2.3,3.2):
            side=max(64,round(max(width,height)*factor))
            matrix=cv2.getRotationMatrix2D(center,angle,1.)
            matrix[:,2]+=np.array([side/2,side/2])-center
            view=cv2.warpAffine(image,matrix,(side,side))
            if enhance:
                lab=cv2.cvtColor(view,cv2.COLOR_BGR2LAB)
                lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(lab[:,:,0])
                view=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            inverse=cv2.invertAffineTransform(matrix)
            boxes,points=detector.detect(view,input_size=(size,size),det_thresh=.15)
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
    if f['side']<35:return False
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


def _unrecoverable(candidate):
    f=candidate['features'];eyes=f['eyes'][:2]
    return (f['core_p90']<24 or (f['core_white']>.5 and min(p[0] for p in eyes)>180)
            or quality.asymmetric_blur(candidate) or quality.blocky_eye_pixels(candidate)
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


def supports_reassessment(image, detector, present):
    """Keep established results when a readable edge profile is severely cut.

    A narrow fragment with no confirmation cannot support a broader search or
    replacement of existing faces. The original quality and recovery stages
    still run; this only limits the additional reassessment in this module.
    """
    if not present:
        return True
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
            return False
    return True


def _source_veto(image,candidate,source,detector):
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
            return True
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
    if _unrecoverable(nearest):return True
    if _edge_damage(image,nearest):return True
    if (nf['visible']<.8 and nf['eye_span']<.18 and nf['mouth_depth']>2.5
            and max(nearest['confirm'])<.2):return True
    if _upper_damage(nearest) and max(nearest['confirm'])<.2:return True
    if (nf['eye_span']<.06 and (nf['mouth_depth']>5 or abs(nf['nose_side'])>2)
            and (max(nearest['confirm'])<.2 or (min(p[1] for p in nf['eyes'][2:])<25 and min(p[2] for p in nf['eyes'][2:])<10)) and f['eye_span']<.15
            and (f['mouth_depth']>2 or abs(f['nose_side'])>2)):return True
    if (nf['eye_span']<.08 and nf['mouth_depth']>4
            and max(p[1] for p in nf['eyes'][2:])<25
            and max(p[2] for p in nf['eyes'][2:])<10):return True
    bilateral_soft=(nf['p10']<120 and max(p[2] for p in nf['eyes'][:2])<10
                    and (nearest['source_reason']=='FACE_BLURRY' or min(p[1] for p in nf['eyes'][:2])<15))
    return nf['side']>=120 and ((bilateral_soft and not (_soft_profile(nearest) and min(nearest['confirm'])>=.65)) or nearest['source_reason'] in ('FACE_TOO_DARK','FACE_UNRECOGNIZABLE','FACE_OVEREXPOSED'))


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


def prune(image,detector,boxes):
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
    return kept


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
    layouts=[(image,0,0,angle,pad) for angle,pad in ((0,.3),(0,.15),(-45,0),(45,0),(-20,0))]
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
        ds,ks=detector.detect(transformed,input_size=(320,320),det_thresh=.25)
        for d,points in zip(ds,ks):
            corners=np.array([[d[0],d[1],1],[d[2],d[1],1],[d[0],d[3],1],[d[2],d[3],1]])@inverse.T+np.array([ox,oy])
            lo,hi=corners.min(axis=0),corners.max(axis=0)
            mapped=np.column_stack([points,np.ones(len(points))])@inverse.T+np.array([ox,oy])
            candidates.append(_candidate(image,[*lo,*(hi-lo)],mapped,d[4],f'proposal:{ox}:{oy}:{angle}:{padding}'))
    return candidates


def refine_recovered_box(image, candidate, detector):
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


def recover(image,detector,present,anchors=(),*,collect_candidates=False):
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
            if f.get('visible') and view=='whole:0':source.append(candidate)
            if not f.get('visible') or f['visible']<.4 or f['side']<29:continue
            if candidate['score']<.15 and not (_visible_profile_core(image,candidate) or (f['side']>=35 and .18<=f['eye_span']<=.55
                    and .65<=f['mouth_depth']<=2.5 and abs(f['nose_side'])<=.8
                    and f['native_lap']>=30 and min(p[1] for p in f['eyes'][:2])>=25
                    and min(p[2] for p in f['eyes'][:2])>=8
                    and min(p[1] for p in f['eyes'][2:])>=25)):continue
            if _represented(candidate,present):continue
            raw.append(candidate)
    if not _needs_additional_views(image,detector,raw,present,anchors):
        return []
    pooled=proposal_views(image,detector)
    raw.extend(c for c in pooled if c['features'].get('visible',0)>=.4
               and c['features']['side']>=29 and not _represented(c,present))
    raw=[c for c in raw if not _unrecoverable(c) and not _edge_damage(image,c)]
    for candidate in sorted(raw,key=lambda c:(_source_quality(c) is None,c['features']['visible']>=.9 and c['features']['points_inside']>=4,c['views']==['whole:0'] and c['score']>=.5,c['score']),reverse=True):
            if sum(_same_anchor(candidate,c) for c in candidates)>=2:continue
            candidates.append(candidate)
    recovered=[];admitted=[]
    for candidate in candidates:
        left,top,box_width,box_height=candidate['box'];right,bottom=left+box_width,top+box_height;score=candidate['score']
        f=candidate['features']
        if not f.get('visible') or f['visible']<.4:continue
        if _unrecoverable(candidate):continue
        if _edge_damage(image,candidate):continue
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
            if sum(c['score']>=.65 and _source_quality(c) is None for c in upright)<2:continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        reason=_source_quality(candidate)
        if reason in ('FACE_BLURRY','FACE_OCCLUDED') and not (_soft_profile(candidate) or ((f['lap']>=2.5 or f['native_lap']>=30) and min(p[1] for p in f['eyes'][:2])>=25 and f['eyes'][2][1]>=40 and f['eyes'][3][1]>=18)):
            continue
        if (score>=.65 and reason is None and 35<=f['side']<120
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
            if best is not None:
                bf=best['features']
                if bf['side']>=35 and bf['visible']>=.6 and bf['p90']>=45 and _source_detail(best):
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
                if best is not None and _source_detail(best):
                    candidate=best;left,top,box_width,box_height=best['box'];right,bottom=left+box_width,top+box_height
                    accepted=True
        if not accepted:continue
        if _unrecoverable(candidate):continue
        if candidate['features']['mouth_depth']<.65:continue
        if _edge_damage(image,candidate):continue
        if _source_veto(image,candidate,source,detector):continue
        if not _source_detail(candidate):continue
        if _represented(candidate,present+recovered):continue
        strong_faces = [c['box'] for c in anchors if c['score'] >= .8]
        if (candidate['score'] < .5 and candidate['features']['side'] < 120
                and candidate['features']['visible'] >= .9
                and candidate['features']['eye_span'] >= .18
                and any(evidence._overlap_smaller(candidate['box'], b) >= .2 for b in strong_faces)
                and not _distinct_local_confirmation(image, candidate, detector, strong_faces)):
            continue
        candidate['confirm']=evidence.confirm(image,candidate,detector)
        cf=candidate['features']
        frontal=(cf['eye_span']>=.18 and .5<cf['mouth_depth']<1.8 and abs(cf['nose_side'])<.5)
        source_profiles=[c for c in source if c['score']>=.5 and evidence.same_location(c['box'],candidate['box'])
                         and c['features']['eye_span']<.18 and c['features']['mouth_depth']>=1.8
                         and not _unrecoverable(c)]
        if frontal and not source_profiles and evidence.inconsistent_texture(image,candidate,detector):continue
        if (candidate['features']['side']>=120 and candidate['features']['visible']>=.9
                and candidate['features']['lap']>=60 and candidate['features']['eye_span']>=.18
                and min(candidate['confirm'])<.82 and not source_profiles):continue
        x1,y1=max(0.,float(left)),max(0.,float(top));x2,y2=min(float(width),float(right)),min(float(height),float(bottom))
        if x2>x1 and y2>y1:
            box=[x1,y1,x2-x1,y2-y1];recovered.append(box)
            admitted.append({'box':box,'score':candidate['score'],'source_candidate':candidate})
    photographic=[dict(c,score=max(c['score'],min(c['source_candidate'].get('confirm',[0])))) for c in anchors]
    filtered=graphics.filter_candidates(image,photographic+admitted)
    kept=[c for c in admitted if any(c is kept for kept in filtered)]
    return kept if collect_candidates else [c['box'] for c in kept]
