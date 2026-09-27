"""Shared YuNet image-only consistency policy, frozen from the v33 experiment.

This module reads only the image submitted to the current request.  All human
styles use the same result before their expected face count is applied.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import traceback
import weakref

import cv2
import numpy as np

from . import face_detection as _face_detection


class _RequestTrace(list):
    """Keep policy evidence inside the detector's existing request context."""

    @staticmethod
    def _rows():
        work = _face_detection._request_work.get()
        if work is None:
            return []
        return work.setdefault("consistency_policy_trace", [])

    def clear(self):
        self._rows().clear()

    def append(self, item):
        self._rows().append(item)

    def __iter__(self):
        return iter(self._rows())

    def __reversed__(self):
        return reversed(self._rows())

    def __len__(self):
        return len(self._rows())

    def __getitem__(self, index):
        return self._rows()[index]


TRACE = _RequestTrace()


# Frozen source: combined_v1.py=fa6d495a9d9c265f8c34aeb6f5885745a0905be4532955a2ede5d4c50da3bfa2
def _same_face(detector, first, second):
    a=np.asarray(first['box'],dtype=float);b=np.asarray(second['box'],dtype=float)
    def located(c):
        if 'face' in c:return c['face']
        if c.get('native_consistency_confirmed'):return c.get('raw_face')
        return None
    first_face=located(first);second_face=located(second)
    if first_face is not None and second_face is not None and min(*a[2:],*b[2:])>=100:
        first_eyes=np.asarray(first_face[4:8]).reshape(2,2).mean(axis=0)
        second_eyes=np.asarray(second_face[4:8]).reshape(2,2).mean(axis=0)
        if np.linalg.norm(first_eyes-second_eyes)>min(*a[2:],*b[2:])*.40:
            return False
    centers=np.abs(a[:2]+a[2:]/2-b[:2]-b[2:]/2)
    # Overlapping head rectangles are not enough to merge adjacent people.
    aligned=bool(np.all(centers<=np.maximum(a[2:],b[2:])*.35))
    return aligned and (detector._box_iou(a,b)>=.3 or detector._box_overlap_over_smaller(a,b)>=.65)

def _eye_features_resolved(gray,box,face,strict=False):
    if min(box[2:])<100:return False
    radius=max(2,round(min(box[2:])*.10));details=[]
    for px,py in face[4:8].reshape(2,2):
        px,py=round(float(px)),round(float(py))
        patch=gray[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
        if not patch.size:return False
        compact=cv2.resize(patch,(16,16),interpolation=cv2.INTER_AREA)
        sharp=float(cv2.Laplacian(compact,cv2.CV_64F).var())
        contrast=float(np.percentile(patch,90)-np.percentile(patch,10))
        details.append((not strict and sharp>=20 and contrast>=40) or (sharp>=30 and contrast>=32))
    return all(details)

def _small_native_features_resolved(gray,box,face):
    """Require actual source-pixel eye and central-face detail for weak small faces."""
    for fraction in (.10,.15):
        radius=max(2,round(min(box[2:])*fraction))
        for px,py in face[4:10].reshape(3,2):
            px,py=round(float(px)),round(float(py))
            patch=gray[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
            if not patch.size:return False
            contrast=float(np.percentile(patch,90)-np.percentile(patch,10))
            sharpness=float(cv2.Laplacian(patch,cv2.CV_64F).var())
            if contrast<32 or sharpness<30:return False
    return True

def proposals(detector, image):
    h,w=image.shape[:2]
    clusters=[]
    for scale in (.75,1.,1.5,2.,3.):
        if max(h,w)*scale>3200: continue
        view=image if scale==1 else cv2.resize(image,None,fx=scale,fy=scale)
        _,faces=detector._get_yunet_detector((view.shape[1],view.shape[0]),.45).detect(view)
        for raw in (() if faces is None else faces):
            if not np.all(np.isfinite(raw)) or raw[14]<.55: continue
            face=raw.copy();face[:14]/=scale
            box=detector._map_face_to_original(face,1,w,h)
            if min(box[2:])<20: continue
            item=dict(box=box,face=face,score=float(raw[14]))
            for c in clusters:
                if _same_face(detector,item,c):
                    if item['score']>c['score']: c.update(item)
                    break
            else: clusters.append(item)
    if not clusters:
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        for scale in (.5,.75,1.):
            view=cv2.resize(enhanced,None,fx=scale,fy=scale)
            _,faces=detector._get_yunet_detector((view.shape[1],view.shape[0]),.65).detect(view)
            for raw in (() if faces is None else faces):
                if not np.all(np.isfinite(raw)):continue
                face=raw.copy();face[:14]/=scale
                box=detector._map_face_to_original(face,1,w,h)
                if min(box[2:])<100:continue
                item=dict(box=box,face=face,score=float(raw[14]))
                if not any(_same_face(detector,item,c) for c in clusters):clusters.append(item)
    return sorted(clusters,key=lambda c:c['score'],reverse=True)+roll_proposals(detector,image)

def roll_proposals(detector,image):
    h,w=image.shape[:2];clusters=[]
    for angle in (-60,-45,45,60):
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        inverse=cv2.invertAffineTransform(matrix)
        view=cv2.warpAffine(image,matrix,(w,h))
        for scale in (.35,.5,.75):
            if max(h,w)*scale>1600:continue
            scaled=cv2.resize(view,None,fx=scale,fy=scale)
            _,faces=detector._get_yunet_detector((scaled.shape[1],scaled.shape[0]),.8).detect(scaled)
            for raw in (() if faces is None else faces):
                if not np.all(np.isfinite(raw)):continue
                native=raw.copy();native[:14]/=scale
                points=cv2.transform(native[4:14].reshape(1,5,2),inverse)[0]
                if np.any(points[:,0]<1) or np.any(points[:,0]>=w-1) or np.any(points[:,1]<1) or np.any(points[:,1]>=h-1):continue
                x,y,bw,bh=native[:4]
                corners=cv2.transform(np.array([[[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]]],dtype=np.float32),inverse)[0]
                x0,y0=corners.min(axis=0);x1,y1=corners.max(axis=0)
                face=np.array([x0,y0,x1-x0,y1-y0,*points.flatten(),raw[14]],dtype=np.float32)
                box=detector._map_face_to_original(face,1,w,h)
                item=dict(box=box,face=face,score=float(raw[14]),turn=-angle,roll_views={(angle,scale)})
                matched=next((c for c in clusters if _same_face(detector,item,c)),None)
                if matched is None:clusters.append(item)
                else:
                    views=matched['roll_views']|item['roll_views']
                    if item['score']>matched['score']:matched.update(item)
                    matched['roll_views']=views
    return [{k:v for k,v in c.items() if k!='roll_views'} for c in clusters if c['score']>=.88 and len(c['roll_views'])>=3]

def orientation_proposals(detector,image):
    h,w=image.shape[:2];output=[]
    for angle in (90,180,270):
        view,matrix=detector._quarter_turn_with_matrix(image,angle)
        inverse=cv2.invertAffineTransform(matrix)
        for scale in (.75,1.,1.5):
            if max(h,w)*scale>2400:continue
            scaled=view if scale==1 else cv2.resize(view,None,fx=scale,fy=scale)
            _,faces=detector._get_yunet_detector((scaled.shape[1],scaled.shape[0]),.55).detect(scaled)
            for raw in (() if faces is None else faces):
                if not np.all(np.isfinite(raw)):continue
                native=raw.copy();native[:14]/=scale
                x,y,bw,bh=native[:4]
                corners=cv2.transform(np.array([[[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]]],dtype=np.float32),inverse)[0]
                x0,y0=corners.min(axis=0);x1,y1=corners.max(axis=0)
                points=cv2.transform(native[4:14].reshape(1,5,2),inverse)[0]
                face=np.array([x0,y0,x1-x0,y1-y0,*points.flatten(),raw[14]],dtype=np.float32)
                box=detector._map_face_to_original(face,1,w,h)
                if min(box[2:])>=20:output.append(dict(box=box,face=face,score=float(raw[14]),turn=angle))
    return output

def tile_proposals(detector,image):
    h,w=image.shape[:2]
    if max(h,w)<min(h,w)*2: return []
    side=max(256,round(min(h,w)*1.5));step=max(64,side//2)
    positions=list(range(0,max(h,w)-side,step))+[max(0,max(h,w)-side)]
    output=[]
    for pos in positions:
        left,top=(0,pos) if h>w else (pos,0)
        tile=image[top:min(h,top+side),left:min(w,left+side)]
        scale=max(1,360/max(tile.shape[:2]))
        pad=max(32,round(min(tile.shape[:2])*.3))
        tile=cv2.copyMakeBorder(tile,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(127,127,127))
        view=cv2.resize(tile,None,fx=scale,fy=scale)
        _,faces=detector._get_yunet_detector((view.shape[1],view.shape[0]),.55).detect(view)
        for raw in (() if faces is None else faces):
            if not np.all(np.isfinite(raw)) or raw[14]<.55: continue
            face=raw.copy();face[:14]/=scale
            face[0]+=left-pad;face[1]+=top-pad;face[4:14:2]+=left-pad;face[5:14:2]+=top-pad
            box=detector._map_face_to_original(face,1,w,h)
            if min(box[2:])>=20: output.append(dict(box=box,face=face,score=float(raw[14])))
    return output

def filter_monochrome_objects(detector,image,candidates):
    """Existing optional classifier corroborates grayscale object overlays only.

    A separate color portrait must first be accepted. Never reject standalone
    illustrations or monochrome portraits from their color alone. ImageNet
    IDs: ocarina, piggy bank, teddy, comic book (torchvision/models/_meta.py).
    """
    regions={}
    for c in candidates:
        x,y,w,h=map(int,c['box']);roi=image[y:y+h,x:x+w]
        if roi.size:regions[id(c)]=(roi,cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)[:,:,1])
    anchors=[c for c in candidates if c.get('score',0)>=.90 and id(c) in regions
             and float(np.median(regions[id(c)][1]))>=25]
    if not anchors:return candidates
    kept=[]
    for c in candidates:
        region=regions.get(id(c))
        if region is None or any(c is a for a in anchors) or np.percentile(region[1],90)>8:
            kept.append(c);continue
        try:
            blob=cv2.dnn.blobFromImage(region[0],1/255.,detector.PET_CLASSIFIER_INPUT_SIZE,swapRB=True,crop=True)
            logits=detector._pet_classifier_logits(detector._get_pet_classifier(),(blob-detector.PET_CLASSIFIER_MEAN)/detector.PET_CLASSIFIER_STD)
            probabilities=np.exp(logits-logits.max());probabilities/=max(float(probabilities.sum()),1e-9)
            object_score=float(probabilities[[684,719,850,917]].sum())
        except (cv2.error,RuntimeError,ValueError,IndexError):
            object_score=0
        if object_score<.85:kept.append(c)
    return kept

def context_evidence(detector,image,proposal):
    h,w=image.shape[:2]; x,y,bw,bh=proposal['box']
    observations=[]; bounds_seen=set()
    native_roi=image[max(0,y):min(h,y+bh),max(0,x):min(w,x+bw)]
    low,high=np.percentile(cv2.cvtColor(native_roi,cv2.COLOR_BGR2GRAY),(10,90))
    use_enhanced_view=low>=100 and high-low<100 and min(bw,bh)>=100
    source_gray=None
    work=detector._request_work.get()
    view_cache=work.setdefault('consistency_context_view_cache',{}) if work is not None else {}
    # The same request can review both the submitted image and its mirrored
    # copy. A rectangle/scale/angle is equivalent only within one source
    # array, so assign a request-local, non-recycled source identity. Weak
    # references avoid retaining every temporary flipped image.
    source_tag=0
    if work is not None:
        sources=work.setdefault('consistency_context_sources',{})
        source_entry=sources.get(id(image))
        if source_entry is None or source_entry[0]() is not image:
            source_tag=work.get('consistency_context_next_source',0)
            work['consistency_context_next_source']=source_tag+1
            sources[id(image)]=(weakref.ref(image),source_tag)
        else:
            source_tag=source_entry[1]
    def detect_view(view,key):
        # Bounds, exact scale/angle and enhancement inputs determine every
        # source pixel in this view. Reuse only identical request-local views;
        # keep a private copy because callers may combine/modify face arrays.
        cached=view_cache.get(key)
        if cached is not None:
            retval,faces=cached
            return retval,None if faces is None else faces.copy()
        retval,faces=detector._get_yunet_detector((view.shape[1],view.shape[0]),.65).detect(view)
        if len(view_cache)<4096:
            view_cache[key]=(retval,None if faces is None else faces.copy())
        return retval,faces
    for padding in (.4,.8,1.2):
        l=max(0,round(x-bw*padding));t=max(0,round(y-bh*padding))
        r=min(w,round(x+bw*(1+padding)));b=min(h,round(y+bh*(1+padding)))
        bounds=(l,t,r,b)
        if bounds in bounds_seen: continue
        bounds_seen.add(bounds)
        region=image[t:b,l:r]
        face_sizes=(40,60,80,120,200) if min(bw,bh)<=60 else (80,120,200)
        for face_size in face_sizes:
            scale=min(face_size/min(bw,bh),768/max(region.shape[:2]))
            view=cv2.resize(region,None,fx=scale,fy=scale)
            # Native quarter turns use clockwise angles; OpenCV's affine
            # rotation API uses counterclockwise angles.
            turn=-proposal.get('turn',0)
            for angle in (turn,turn-15,turn+15,turn-30,turn+30,turn-45,turn+45,turn-60,turn+60):
                matrix=cv2.getRotationMatrix2D((view.shape[1]/2,view.shape[0]/2),angle,1)
                rotated=view if angle==0 else cv2.warpAffine(view,matrix,(view.shape[1],view.shape[0]))
                inverse=cv2.invertAffineTransform(matrix)
                _,faces=detect_view(rotated,(source_tag,bounds,scale,angle,False,view.shape))
                if use_enhanced_view:
                    enhanced=np.clip((rotated.astype(np.float32)-low)*min(2.0,180/max(1,high-low))+40,0,255).astype(np.uint8)
                    _,extra=detect_view(enhanced,(source_tag,bounds,scale,angle,True,view.shape,float(low),float(high)))
                    if extra is not None:faces=extra if faces is None else np.vstack((faces,extra))
                native_view=native_gray=None
                for raw in (() if faces is None else faces):
                    if not np.all(np.isfinite(raw)) or raw[14]<.65: continue
                    points=cv2.transform(raw[4:14].reshape(1,5,2),inverse)[0]/scale
                    # Never use a keypoint supplied by synthetic rotation padding.
                    if np.any(points[:,0]<1) or np.any(points[:,1]<1) or np.any(points[:,0]>=r-l-1) or np.any(points[:,1]>=b-t-1): continue
                    rx,ry,rw,rh=raw[:4]
                    corners=cv2.transform(np.array([[[rx,ry],[rx+rw,ry],[rx,ry+rh],[rx+rw,ry+rh]]],dtype=np.float32),inverse)[0]/scale+np.array([l,t])
                    x0,y0=corners.min(axis=0);x1,y1=corners.max(axis=0)
                    raw_box=(float(x0),float(y0),float(x1-x0),float(y1-y0))
                    box=(max(0,round(x0)),max(0,round(y0)),round(min(w,x1)-max(0,x0)),round(min(h,y1)-max(0,y0)))
                    # Context views can shift landmarks strongly for the same
                    # low-light or rotated face. Bind observations to the
                    # proposal geometrically here; strict landmark identity
                    # remains in global clustering and duplicate removal.
                    if not _same_face(detector,dict(box=box),proposal): continue
                    points+=np.array([l,t])
                    face=np.array([*raw_box,*points.flatten(),raw[14]],dtype=np.float32)
                    if native_view is None:
                        native_matrix=cv2.getRotationMatrix2D((region.shape[1]/2,region.shape[0]/2),angle,1)
                        native_view=region if angle==0 else cv2.warpAffine(region,native_matrix,(region.shape[1],region.shape[0]))
                        native_gray=cv2.cvtColor(native_view,cv2.COLOR_BGR2GRAY)
                    native_face=raw.copy();native_face[:14]/=scale
                    native_box=detector._map_face_to_original(native_face,1,region.shape[1],region.shape[0])
                    native_occlusion=detector._get_face_occlusion_issue(native_gray,native_box,native_face)
                    native_quality=detector._get_face_quality_issue(native_gray,native_box)
                    native_core=detector._native_face_core_issue(native_view,native_box,float(raw[14]),gray_image=native_gray)
                    # A rotated head enclosure includes black hair/background.
                    # Corroborate dim facial detail at the actual landmarks,
                    # using the existing strict native-pixel exposure gate.
                    feature_points=points
                    feature_low=feature_points.min(axis=0);feature_high=feature_points.max(axis=0)
                    dim_proof=False
                    if min(native_box[2:])>=120:
                        if source_gray is None:
                            source_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        for margin in (.15,.25,.35,.5):
                            feature_margin=(feature_high-feature_low)*margin
                            feature_l=np.maximum(0,feature_low-feature_margin).astype(int)
                            feature_h=np.minimum((w,h),feature_high+feature_margin).astype(int)
                            feature_box=(*feature_l,*(feature_h-feature_l))
                            if detector._has_readable_dark_detail(source_gray,feature_box):
                                dim_proof=True;break
                    if dim_proof and native_quality=='FACE_TOO_DARK':
                        enhanced=np.clip(native_view.astype(np.float32)*4,0,255).astype(np.uint8)
                        enhanced_gray=cv2.cvtColor(enhanced,cv2.COLOR_BGR2GRAY)
                        native_occlusion=detector._get_face_occlusion_issue(enhanced_gray,native_box,native_face)
                        native_quality=detector._get_face_quality_issue(enhanced_gray,native_box)
                        native_core=detector._native_face_core_issue(enhanced,native_box,float(raw[14]),gray_image=enhanced_gray)
                    observations.append(dict(box=box,face=face,score=float(raw[14]),context=padding,size=face_size,angle=angle,fully_inside=bool(x0>=0 and y0>=0 and x1<=w and y1<=h),aligned_occlusion=native_occlusion,aligned_quality=native_quality,aligned_core=native_core))
    return observations

def _complete(detector, image, candidates, *, confident_pet_image=False):
    """Review independent locations only after the ordinary quality analysis."""
    threshold = .82
    settled=list(candidates)
    TRACE.append(dict(stage='complete-input',boxes=[dict(box=c['box'],score=c.get('score'),previously_accepted=c.get('previously_accepted')) for c in settled]))
    if confident_pet_image:
        native_humans=[p for p in proposals(detector,image) if p['score']>=.85 and not detector._is_confident_pet_image(image,p['box'])]
        if not native_humans:return settled
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hints=proposals(detector,image)
    TRACE.append(dict(stage='complete-hints',boxes=[dict(box=c['box'],score=c.get('score'),turn=c.get('turn',0)) for c in hints]))
    # Orientation recovery is only needed when the submitted orientation
    # has no confident localization; do not mine rotated textures beside
    # already clear upright portraits.
    turned=orientation_proposals(detector,image) if not hints or max(c['score'] for c in hints)<.82 else []
    for p in turned:
        existing=next((c for c in hints if c.get('turn',0)==p['turn'] and _same_face(detector,p,c)),None)
        if existing is None:hints.append(p)
        elif p['score']>existing['score']:existing.update(p)
    for p in tile_proposals(detector,image):
        if not any(_same_face(detector,p,c) for c in hints): hints.append(p)
    TRACE.append(dict(stage='complete-all-hints',boxes=[dict(box=c['box'],score=c.get('score'),turn=c.get('turn',0)) for c in hints]))
    def already_usable(c):
        if c.get('previously_accepted'): return True
        core=detector._native_face_core_issue(image,c['box'],c['score'])
        if (core=='FACE_BLURRY' and c['score']>=.88
                and not detector._is_face_box_cut_by_frame(c['box'],image.shape[1],image.shape[0])):
            core=None
        return (c.get('quality_issue') is None
                and core is None
                and not (c['score']<.88 and detector._native_extreme_motion_blur(image,c['box'])))
    def trace_skip(p,reason,**details):
        record=dict(stage='proposal-rejected',box=p.get('box'),score=p.get('score'),reason=reason)
        record.update(details)
        TRACE.append(record)
    for p in hints:
        if any(already_usable(c) and _same_face(detector,p,c) for c in settled): continue
        observations=context_evidence(detector,image,p)
        strong=[o for o in observations if o['score']>=threshold]
        TRACE.append(dict(stage='proposal-evidence',box=p['box'],score=p.get('score'),observations=len(observations),strong=len(strong),contexts=len({o['context'] for o in strong}),sizes=sorted({o['size'] for o in strong}),angles=sorted({o['angle'] for o in strong})))
        strong_whole=[o for o in strong if o['score']>=.90]
        whole_view_proof=(len({o['size'] for o in strong_whole})>=2 and len({o['angle'] for o in strong_whole})>=2)
        if (len({o['context'] for o in strong})<2 and not whole_view_proof) or len(strong)<3 or len({o['size'] for o in observations if o['score']>=.70})<2:
            trace_skip(p,'insufficient_repeatability');continue
        # Match the ordinary final pipeline's strong-face review: a
        # >=.90 face with clean native core can override an ambiguous
        # eye-patch label, but never an actual frame/quality failure.
        def source_core_clear(o):
            core=o['aligned_core']
            return core is None or (core=='FACE_BLURRY' and o['score']>=.88 and o['fully_inside'])
        clean=[o for o in strong if o['aligned_occlusion'] is None or (
            o['score']>=(.88 if o['aligned_occlusion']=='FACE_BLURRY' else .90) and o['aligned_occlusion'] in {'FACE_BLURRY','FACE_OCCLUDED'}
            and source_core_clear(o)) or (
                o['score']>=.85 and o['fully_inside'] and o['aligned_occlusion']=='FACE_BLURRY'
                and o['aligned_quality'] in {None,'FACE_BLURRY'} and o['aligned_core'] in {None,'FACE_BLURRY'}
                and _eye_features_resolved(gray,o['box'],o['face'],strict=True))]
        if len({o['context'] for o in clean})<2 and not (
            whole_view_proof and len(clean)>=3
            and len({o['size'] for o in clean})>=2
            and len({o['angle'] for o in clean})>=2):
            trace_skip(p,'insufficient_clean_contexts',clean=len(clean));continue
        contained=[o for o in clean if o['fully_inside']]
        choices=contained if min(p['box'][2:])>=100 and contained else clean
        feature_choices=[o for o in choices if o['score']>=.88 and o['fully_inside']
                         and _eye_features_resolved(gray,o['box'],o['face'],strict=True)]
        if len({o['context'] for o in feature_choices})>=2:choices=feature_choices
        best=max(choices,key=lambda o:o['score'])
        box=best['box'];face=best['face'];score=best['score']
        TRACE.append(dict(stage='proposal-selected',source_box=p['box'],box=box,score=score,angle=best['angle'],context=best['context'],quality=best['aligned_quality'],occlusion=best['aligned_occlusion'],core=best['aligned_core'],eye_strict=_eye_features_resolved(gray,box,face,strict=True)))
        repeated_native_small=(detector.YUNET_MIN_FACE_SIDE<=min(p['box'][2:])<=60
                and (_small_native_features_resolved(gray,p['box'],p['face']) or
                     len({o['context'] for o in strong if o['fully_inside']
                          and _small_native_features_resolved(gray,o['box'],o['face'])})>=2)
                and detector._get_face_quality_issue(gray,p['box']) is None
                and detector._get_face_occlusion_issue(gray,p['box'],p['face']) is None
                and len({o['context'] for o in strong if o['angle']%360==0 and o['aligned_quality'] is None and o['aligned_occlusion'] is None})>=3
                and len({o['size'] for o in strong if o['angle']%360==0 and o['aligned_quality'] is None and o['aligned_occlusion'] is None})>=2)
        if score<(.83 if repeated_native_small else .84):
            trace_skip(p,'score_floor',chosen=box,score=score);continue
        if min(box[2:])>=100 and score<.88:
            trace_skip(p,'large_face_score_floor',chosen=box,score=score);continue
        evidence_points=np.asarray([o['face'][4:14] for o in strong]).reshape(-1,5,2)
        median_points=np.median(evidence_points,axis=0)
        dispersion=np.median(np.linalg.norm(evidence_points-median_points,axis=2),axis=0)/min(p['box'][2:])
        median_eye=float(np.median(np.linalg.norm(evidence_points[:,0]-evidence_points[:,1],axis=1)))/min(p['box'][2:])
        if score<.90 and median_eye<.25 and max(dispersion[:2])>.15 and float(np.median(dispersion[3:]))>.10:
            trace_skip(p,'profile_landmark_dispersion',chosen=box,score=score,median_eye=median_eye,dispersion=dispersion.tolist());continue
        if score<.90 and median_eye>=.25 and float(np.median(dispersion))>.08:
            trace_skip(p,'landmark_dispersion',chosen=box,score=score,median_eye=median_eye,dispersion=dispersion.tolist());continue
        # Weak scene-level background proposals need stronger evidence
        # than a dominant portrait. This applies to every submitted image.
        area=box[2]*box[3]
        dominant=max((c['box'][2]*c['box'][3] for c in settled),default=area*3)
        if area/float(image.shape[0]*image.shape[1])<.008 and dominant>area*2 and score<.92 and not repeated_native_small:
            trace_skip(p,'weak_small_scene_proposal',chosen=box,score=score);continue
        x,y,bw,bh=box;native_gray=gray[y:y+bh,x:x+bw]
        points=face[4:14].reshape(5,2)
        dark_hair_satellite=False
        if min(bw,bh)<100 and float(native_gray.mean())<60 and score<.92:
            for other in settled:
                ox,oy,ow,oh=other['box']
                if (area<ow*oh*.25 and ox-ow*.1<=x+bw/2<=ox+ow*1.1
                        and oy-oh<=y and y+bh<=oy+oh*.15):
                    dark_hair_satellite=True;break
        if dark_hair_satellite:
            trace_skip(p,'dark_hair_satellite',chosen=box,score=score);continue
        if not best['fully_inside'] and min(bw,bh)<100 and score<.90:
            trace_skip(p,'edge_small_face',chosen=box,score=score);continue
        # A predicted mouth in another person's foreground hair region
        # does not establish visibility of the rear person's lower face.
        mouth=points[3:5].mean(axis=0)
        if any(
            not _same_face(detector,best,other)
            and .5<=area/(other['box'][2]*other['box'][3])<=2
            and other['box'][0]<=mouth[0]<=other['box'][0]+other['box'][2]
            and other['box'][1]-other['box'][3]*.35<=mouth[1]<=other['box'][1]+other['box'][3]*.10
            for other in settled):
            continue
        source_occlusion=detector._get_face_occlusion_issue(gray,p['box'],p['face'])
        roll_clear=[o for o in strong if o['score']>=.87 and 30<=abs(o['angle'])%180<=60
                    and o['fully_inside'] and not (o['aligned_occlusion'] or o['aligned_quality'] or o['aligned_core'])]
        roll_proof=(score>=.88 and len({o['context'] for o in roll_clear})>=2
                    and len({o['size'] for o in roll_clear})>=2)
        if source_occlusion=='FACE_OCCLUDED' and min(p['box'][2:])>=80 and score<.92 and not roll_proof:
            trace_skip(p,'source_occlusion',chosen=box,score=score,source=source_occlusion,roll_proof=roll_proof);continue
        work=detector._request_work.get()
        if (work is not None and work.get('consistency_initial_code')=='FACE_OCCLUDED'
                and min(bw,bh)>=100 and score<.90 and not roll_proof):
            trace_skip(p,'request_initial_occlusion',chosen=box,score=score,roll_proof=roll_proof);continue
        source_quality=detector._get_face_quality_issue(gray,p['box'])
        eye_detail=[]
        if score>=.88 and best['fully_inside'] and min(box[2:])>=100:
            radius=max(2,round(min(box[2:])*.10))
            for px,py in face[4:8].reshape(2,2):
                px,py=round(float(px)),round(float(py))
                patch=gray[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
                compact=cv2.resize(patch,(16,16),interpolation=cv2.INTER_AREA)
                eye_detail.append((float(cv2.Laplacian(compact,cv2.CV_64F).var()),float(np.percentile(patch,90)-np.percentile(patch,10))))
        resolved_eye_detail=_eye_features_resolved(gray,box,face) if score>=.88 and best['fully_inside'] else False
        repeatable_native_proof=(score>=.90 and best['fully_inside']
            and len({o['context'] for o in clean if o['score']>=.87})>=3
            and len({o['size'] for o in clean if o['score']>=.87})>=2
            and len({o['angle'] for o in clean if o['score']>=.87})>=2
            and source_quality in {None,'FACE_TOO_DARK'}
            and best['aligned_quality'] is None and best['aligned_core'] is None
            and best['aligned_occlusion'] is None)
        if not resolved_eye_detail and (detector._native_extreme_motion_blur(image,p['box']) or detector._native_extreme_motion_blur(image,box)) and not repeatable_native_proof:
            trace_skip(p,'native_extreme_blur',chosen=box,score=score,resolved_eye_detail=resolved_eye_detail,repeatable_native_proof=repeatable_native_proof,source_quality=source_quality,clean_contexts=len({o['context'] for o in clean if o['score']>=.87}),clean_sizes=len({o['size'] for o in clean if o['score']>=.87}),clean_angles=len({o['angle'] for o in clean if o['score']>=.87}));continue
        if source_quality=='FACE_BLURRY' and not resolved_eye_detail and not repeatable_native_proof:
            trace_skip(p,'source_blurry',chosen=box,score=score,source=source_quality,resolved_eye_detail=resolved_eye_detail);continue
        quality=best['aligned_quality']
        if quality=='FACE_BLURRY' and resolved_eye_detail:quality=None
        core=best['aligned_core']
        if core=='FACE_BLURRY' and score>=.88 and best['fully_inside']:core=None
        occlusion=best['aligned_occlusion'] if score<(.88 if best['aligned_occlusion']=='FACE_BLURRY' else .90) else None
        edge=(not best['fully_inside']) and detector._is_unusable_edge_cropped_face(face,box,image.shape[1],image.shape[0],score)
        if quality or core or occlusion or edge:
            trace_skip(p,'final_quality',chosen=box,score=score,quality=quality,core=core,occlusion=occlusion,edge=edge);continue
        if score<.90:
            eye_vector=points[1]-points[0]
            eye_length=float(np.linalg.norm(eye_vector))
            nose_offset=abs(float(np.dot(points[2]-points[:2].mean(axis=0),eye_vector)))/max(eye_length**2,1)
            px,py,pw,ph=p['box']
            mirrored_proposal=dict(box=(image.shape[1]-px-pw,py,pw,ph),turn=(-p.get('turn',0))%360)
            mirrored=context_evidence(detector,cv2.flip(image,1),mirrored_proposal)
            mirror_score=max((o['score'] for o in mirrored),default=0)
            profile_eye=median_eye
            if min(bw,bh)>=100 and eye_length/min(bw,bh)<.15:
                profile_eye=min(profile_eye,.10)
            mirror_floor=(.84 if .12<=profile_eye<.25 else .90) if profile_eye<.25 else (.85 if nose_offset>=.30 else (.86 if min(bw,bh)>=100 else .88))
            if min(bw,bh)<100 and .08<=median_eye<.25 and nose_offset>=.5:
                mirror_floor=.84
            if mirror_score<mirror_floor:
                trace_skip(p,'mirror_floor',chosen=box,score=score,mirror_score=mirror_score,mirror_floor=mirror_floor);continue
        if any(already_usable(c) and _same_face(detector,best,c) for c in settled):
            trace_skip(p,'already_usable_same_face',chosen=box,score=score);continue
        candidate=dict(box=box,raw_face=face,score=score,ratio=box[2]*box[3]/float(image.shape[0]*image.shape[1]),quality_issue=None,native_context_confirmed_primary=True,native_edge_readable=best['fully_inside'],native_consistency_confirmed=True)
        # A rotated enclosing rectangle can dilute a small ink-on-paper
        # overlay with surrounding photograph pixels. Check the original
        # proposal footprint as well as the final enclosing rectangle.
        source_candidate=dict(candidate,box=p['box'],raw_face=p['face'])
        source_filtered=detector._filter_pet_face_candidates(image,[*settled,source_candidate])
        if not any(c is source_candidate for c in source_filtered):
            trace_skip(p,'source_pet_filter',chosen=box,score=score);continue
        filtered=detector._filter_pet_face_candidates(image,[*settled,candidate])
        if not any(c is candidate for c in filtered):
            trace_skip(p,'final_pet_filter',chosen=box,score=score);continue
        # Replace only a candidate proven to be the same face. A large
        # overlapping proposal can span two adjacent people (for example a
        # close kissing pair); overlap alone must never delete one accepted
        # person from the baseline.
        replacements=[c for c in settled if _same_face(detector,c,candidate)]
        if len(replacements)>1:
            trace_skip(p,'multiple_replacements',chosen=box,score=score,count=len(replacements));continue
        settled=[c for c in settled if not any(c is previous for previous in replacements)]
        settled.append(candidate)
        TRACE.append(dict(stage='candidate-added',box=box,score=score,replaced=[c['box'] for c in replacements]))
    # The shared neck-satellite heuristic may discard a previously accepted
    # face when a newly recovered, overlapping proposal is larger. Keep the
    # completed baseline immutable here; only newly added candidates may be
    # screened by this post-addition rule.
    filtered=[c for c in settled if c.get('previously_accepted')]
    for c in (item for item in settled if not item.get('previously_accepted')):
        comparison=[c,*[other for other in settled if other is not c and (
            c.get('previously_accepted') or other.get('previously_accepted'))]]
        if any(item is c for item in detector._filter_neck_satellite_candidates(comparison)):
            filtered.append(c)
    verified=[]
    for c in filtered:
        if (.60<=c.get('score',0)<.85 and c.get('previously_accepted')
                and not c.get('native_roll_confirmed')
                and any(other is not c and other.get('score',0)>=.90 for other in filtered)):
            own=context_evidence(detector,image,c)
            if max((o['score'] for o in own),default=0)<.82:
                x,y,bw,bh=c['box']
                mirrored=context_evidence(detector,cv2.flip(image,1),dict(box=(image.shape[1]-x-bw,y,bw,bh)))
                if max((o['score'] for o in mirrored),default=0)<.82:
                    continue
        verified.append(c)
    result=filter_monochrome_objects(detector,image,verified)
    TRACE.append(dict(stage='complete-output',boxes=[dict(box=c['box'],score=c.get('score'),previously_accepted=c.get('previously_accepted')) for c in result]))
    return result

def _review_existing_evidence(detector,image,candidates):
    if len(candidates)<2:return candidates
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    evidence={id(c):context_evidence(detector,image,c) for c in candidates if c.get('score',0)<.90}
    bests={key:max(obs,key=lambda o:o['score'],default=None) for key,obs in evidence.items()}
    consumed=set();coalesced=[]
    for first in candidates:
        if id(first) in consumed:continue
        b1=bests.get(id(first))
        partner=None
        if first.get('score',0)<.85 and b1 is not None and b1['score']>=.90:
            for second in candidates:
                if second is first or id(second) in consumed or second.get('score',0)>=.85:continue
                b2=bests.get(id(second))
                if b2 is None or b2['score']<.90:continue
                p1=b1['face'][4:14].reshape(5,2);p2=b2['face'][4:14].reshape(5,2)
                agreement=float(np.median(np.linalg.norm(p1-p2,axis=1)))/min(*b1['box'][2:],*b2['box'][2:])
                if detector._box_iou(b1['box'],b2['box'])>=.55 and agreement<=.12:
                    partner=second;break
        if partner is not None:
            b2=bests[id(partner)];best=b1 if b1['score']>=b2['score'] else b2
            consumed.update((id(first),id(partner)))
            item=dict(first,box=best['box'],raw_face=best['face'],score=best['score'],native_consistency_confirmed=True)
            TRACE.append(dict(reason='converging_landmarks_merge',boxes=[first['box'],partner['box']],best_box=best['box'],agreement=agreement))
            coalesced.append(item)
        else:coalesced.append(first)
    candidates=coalesced
    verified=[]
    for item in candidates:
        score=item.get('score',0)
        if score>=.90 or not any(other is not item and (other.get('score',0)>=.90 or (bests.get(id(other)) or {}).get('score',0)>=.90) for other in candidates):
            verified.append(item);continue
        own=evidence.get(id(item))
        if own is None:own=context_evidence(detector,image,item)
        best=max(own,key=lambda o:o['score'],default=None)
        x,y,w,h=item['box']
        mirror=context_evidence(detector,cv2.flip(image,1),dict(box=(image.shape[1]-x-w,y,w,h)))
        peak=best['score'] if best else 0;mirror_peak=max((o['score'] for o in mirror),default=0)
        mirror_best=max(mirror,key=lambda o:o['score'],default=None)
        mirror_displacement=None
        if best is not None and mirror_best is not None:
            reflected=mirror_best['face'][4:14].reshape(5,2).copy()
            reflected[:,0]=image.shape[1]-reflected[:,0]
            reflected=reflected[[1,0,2,4,3]]
            own_points=best['face'][4:14].reshape(5,2)
            mirror_displacement=float(np.median(np.linalg.norm(reflected-own_points,axis=1)))/min(w,h)
        boundary=any(detector._frame_edges_touched(item['box'],image.shape[1],image.shape[0]))
        reason=None
        raw=item.get('raw_face')
        established_profile=False
        if raw is not None and len(raw)>=15 and raw[2]>0 and not item.get('native_roll_confirmed') and min(peak,mirror_peak)>=.81:
            nose=(float(raw[8])-float(raw[0]))/float(raw[2])
            eye_span=float(np.linalg.norm(raw[4:6]-raw[6:8]))/float(raw[2])
            established_profile=(detector._is_yunet_profile(raw) or (eye_span<.22 and (nose<.35 or nose>.65)))
        native_quality=detector._get_face_quality_issue(gray,item['box'])
        prior_dark_face=item.get('previously_accepted') and native_quality=='FACE_TOO_DARK'
        if (item.get('previously_accepted') and native_quality is not None and not prior_dark_face
                and not established_profile and not boundary and peak<.88 and mirror_peak<.88):
            reason='weak_symmetric_repeatability'
        if (item.get('previously_accepted') and not established_profile and not boundary
                and score<.80 and max(peak,mirror_peak)<.90
                and mirror_displacement is not None and mirror_displacement>.15):
            reason='inconsistent_mirrored_landmarks'
        points=best['face'][4:14].reshape(5,2) if best is not None else None
        mouth=points[3:5].mean(axis=0) if points is not None else None
        area=w*h
        if mouth is not None and any(not _same_face(detector,item,other)
                and .5<=area/(other['box'][2]*other['box'][3])<=2
                and other['box'][0]-other['box'][2]*.30<=mouth[0]<=other['box'][0]+other['box'][2]*1.30
                and other['box'][1]-other['box'][3]*.35<=mouth[1]<=other['box'][1]+other['box'][3]*.10
                for other in candidates):
            reason='mouth_hidden_by_foreground_head'
        TRACE.append(dict(box=item['box'],score=score,peak=peak,mirror_peak=mirror_peak,mirror_displacement=mirror_displacement,established_profile=established_profile,native_quality=native_quality,prior_dark_face=prior_dark_face,reason=reason))
        if reason is None:verified.append(item)
    return verified

def _observe_region(detector, image, region_box, pad_fraction, angles, scales, floor):
    height, width = image.shape[:2]
    left, top, region_width, region_height = region_box
    region = image[top:top + region_height, left:left + region_width]
    observations = []
    for reflected in (False, True):
        source = cv2.flip(region, 1) if reflected else region
        pad = round(min(source.shape[:2]) * pad_fraction)
        padded = cv2.copyMakeBorder(source, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=(127, 127, 127))
        padded_height, padded_width = padded.shape[:2]
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((padded_width / 2, padded_height / 2), angle, 1)
            inverse = cv2.invertAffineTransform(matrix)
            view = cv2.warpAffine(padded, matrix, (padded_width, padded_height),
                                 borderValue=(127, 127, 127))
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            for scale in scales:
                if max(view.shape[:2]) * scale > 1600:
                    continue
                scaled = cv2.resize(view, None, fx=scale, fy=scale)
                _, faces = detector._get_yunet_detector((scaled.shape[1], scaled.shape[0]), floor).detect(scaled)
                for raw in (() if faces is None else faces):
                    if len(raw) < 15 or not np.all(np.isfinite(raw)):
                        continue
                    native = raw.copy()
                    native[:14] /= scale
                    x, y, box_width, box_height = native[:4]
                    corners = cv2.transform(np.array([[[x, y], [x + box_width, y],
                                                        [x, y + box_height], [x + box_width, y + box_height]]],
                                                     dtype=np.float32), inverse)[0] - pad
                    points = cv2.transform(native[4:14].reshape(1, 5, 2), inverse)[0] - pad
                    if reflected:
                        corners[:, 0] = region_width - corners[:, 0]
                        points[:, 0] = region_width - points[:, 0]
                    corners += np.array([left, top])
                    points += np.array([left, top])
                    x0, y0 = corners.min(0)
                    x1, y1 = corners.max(0)
                    box = (max(0, round(float(x0))), max(0, round(float(y0))),
                           round(min(width, float(x1)) - max(0, float(x0))),
                           round(min(height, float(y1)) - max(0, float(y0))))
                    if min(box[2:]) < 32 or min(box[2:]) > 250 or not (x0 < 3 or x1 > width - 3):
                        continue
                    if (np.any(points[:, 0] < 2) or np.any(points[:, 0] > width - 2)
                            or np.any(points[:, 1] < 2) or np.any(points[:, 1] > height - 2)):
                        continue
                    if (np.any(points[:, 0] < left) or np.any(points[:, 0] > left + region_width)
                            or np.any(points[:, 1] < top) or np.any(points[:, 1] > top + region_height)):
                        continue
                    eye_span = float(np.linalg.norm(points[0] - points[1])) / min(box[2:])
                    if eye_span > .15:
                        continue
                    native_box = detector._map_face_to_original(native, 1, padded_width, padded_height)
                    face = np.array([x0, y0, x1 - x0, y1 - y0, *points.flatten(), raw[14]], dtype=np.float32)
                    observations.append(dict(
                        box=box, face=face, score=float(raw[14]), points=points,
                        flip=reflected, angle=angle, context=region_box, scale=scale,
                        quality=detector._get_face_quality_issue(gray, native_box),
                        core=detector._native_face_core_issue(view, native_box, float(raw[14])),
                        occlusion=detector._get_face_occlusion_issue(gray, native_box, native),
                        eye_span=eye_span,
                    ))
    return observations

def _profile_observations(detector, image, same_face):
    height, width = image.shape[:2]
    tile_width, tile_height = min(width, 192), min(height, round(192 * 1.6))
    seed = []
    seen = set()
    positions = list(range(0, max(1, height - tile_height), max(48, tile_height // 2))) + [max(0, height - tile_height)]
    for top in positions:
        for left in (0, max(0, width - tile_width)):
            region_box = (left, top, tile_width, tile_height)
            if region_box in seen:
                continue
            seen.add(region_box)
            seed.extend(_observe_region(detector, image, region_box, .25, (-45, 45), (3.,), .78))
    locations = []
    for observation in sorted(seed, key=lambda item: item['score'], reverse=True):
        if not any(same_face(detector, observation, previous) for previous in locations):
            locations.append(observation)
    observations = list(seed)
    seen = set()
    for proposal in locations:
        x, y, box_width, box_height = proposal['box']
        for context in (.4, .8, 1.2):
            left = max(0, round(x - box_width * context))
            top = max(0, round(y - box_height * context))
            right = min(width, round(x + box_width * (1 + context)))
            bottom = min(height, round(y + box_height * (1 + context)))
            region_box = (left, top, right - left, bottom - top)
            if region_box in seen:
                continue
            seen.add(region_box)
            for pad in (0, .25):
                observations.extend(_observe_region(
                    detector, image, region_box, pad,
                    (-52.5, -45, -37.5, -30, 30, 37.5, 45, 52.5), (2., 3.), .70,
                ))
    return observations

def review_edge_profiles(detector, image, candidates, same_face):
    """Add a profile only with real, repeatable visible facial landmarks."""
    settled = list(candidates)
    groups = []
    for observation in _profile_observations(detector, image, same_face):
        for group in groups:
            if same_face(detector, observation, group[0]):
                group.append(observation)
                break
        else:
            groups.append([observation])
    for group in groups:
        best = max(group, key=lambda item: item['score'])
        if any(same_face(detector, best, candidate) for candidate in settled):
            continue
        strong = [item for item in group if item['score'] >= .80
                  and item['quality'] is None and item['core'] is None and item['occlusion'] is None]
        if best['score'] < .82 or len(strong) < 3:
            continue
        # Reflection reverses rotation handedness. Do not count a reflected
        # +45 view and a direct -45 view as two different source angles.
        if (len({-item['angle'] if item['flip'] else item['angle'] for item in strong}) < 2
                or len({item['flip'] for item in group}) < 2):
            continue
        if min(max(item['score'] for item in group if item['flip'] == reflected) for reflected in (False, True)) < .72:
            continue
        stable_points = np.array([np.stack((item['points'][:2].mean(0), item['points'][2],
                                           item['points'][3:].mean(0))) for item in strong])
        dispersion = np.median(np.linalg.norm(stable_points - np.median(stable_points, axis=0), axis=2)) / min(best['box'][2:])
        if dispersion > .07:
            continue
        best = max(strong, key=lambda item: item['score'])
        box, face, score = best['box'], best['face'], best['score']
        if not any(detector._frame_edges_touched(box,image.shape[1],image.shape[0])):
            continue
        if detector._native_extreme_motion_blur(image, box):
            continue
        candidate = dict(box=box, raw_face=face, score=score,
                         ratio=box[2] * box[3] / float(image.shape[0] * image.shape[1]),
                         quality_issue=None, native_context_confirmed_primary=True,
                         native_edge_readable=True, native_consistency_confirmed=True)
        if any(item is candidate for item in detector._filter_pet_face_candidates(image, [*settled, candidate])):
            settled.append(candidate)
            work = detector._request_work.get()
            if work is not None:
                work.setdefault("refinement_edge_profiles", []).append(candidate)
    return settled

def _review_v1(detector, image_path, baseline, locale=None):
    """Corroborate usable faces using only this submitted image.

    Preserve the completed baseline when no count change is justified. The
    product headcount is applied later by the existing shared evaluator.
    """
    TRACE.clear()
    if baseline.get('code') in {'IMAGE_READ_ERROR','FACE_DETECTION_FAILED'}: return baseline
    image=detector._read_image(image_path)
    if image is None: return baseline
    work=detector._request_work.get()
    known=work.get('consistency_accepted_candidates',{}) if work is not None else {}
    count=baseline.get('usable_face_count',0)
    candidates=[]
    if count:
        for box in baseline.get('usable_face_boxes',[]):
            item=deepcopy(known.get(tuple(box),{}))
            item.update(box=box,quality_issue=None,previously_accepted=True,
                        ratio=box[2]*box[3]/float(image.shape[0]*image.shape[1]))
            item.setdefault('score',baseline.get('face_score',0) if list(box)==list(baseline.get('face',[])) else 0)
            candidates.append(item)
    if len(candidates)!=count: return baseline
    try:
        completed=_complete(detector,image,candidates,confident_pet_image=detector._is_confident_pet_image(image))
    except Exception:
        TRACE.append(dict(stage='complete-exception',traceback=traceback.format_exc()))
        return baseline
    completed=_review_existing_evidence(detector,image,completed)
    TRACE.append(dict(stage='after-existing-review',boxes=[c['box'] for c in completed]))
    if not detector._is_confident_pet_image(image):
        completed=review_edge_profiles(detector,image,completed,_same_face)
    TRACE.append(dict(stage='after-edge-review',boxes=[c['box'] for c in completed]))
    if not completed: return baseline
    if len(completed)==count and sorted(tuple(c['box']) for c in completed)==sorted(tuple(b) for b in baseline.get('usable_face_boxes',[])): return baseline
    result=detector._yunet_candidate_result(completed,image.shape[1],image.shape[0],locale)
    if not result: return baseline
    TRACE.append(dict(stage='result-build',face_count=result.get('face_count'),boxes=result.get('usable_face_boxes')))
    if result['face_count']==1 and result.get('face_small') and not detector._face_has_sufficient_native_detail(image_path,result):
        return baseline
    return detector._with_usable_face_count(result)



# Frozen source: combined_v22_small_group_guard.py=5f85acec95f31c7d0c6b457fad862ec7647b6614e85599e9a7ae278edb3497c0
def _native_detail(image, box):
    import cv2
    import numpy as np
    x, y, width, height = map(int, box)
    roi = image[max(0, y):min(image.shape[0], y + height),
                max(0, x):min(image.shape[1], x + width)]
    if roi.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    return sharpness, contrast

def _review_v22(detector, image_path, baseline, locale=None):
    result = _review_v1(detector, image_path, baseline, locale)
    if result is baseline:
        return baseline
    before = [tuple(map(int, box)) for box in baseline.get("usable_face_boxes", [])]
    after = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    added = [box for box in after if box not in before]
    if not added:
        return result
    image = detector._read_image(image_path)
    if image is None:
        return baseline

    selected = [item for item in TRACE if item.get("stage") == "proposal-selected"]
    inputs = [item for item in TRACE if item.get("stage") == "complete-input"]
    initial_scores = {
        tuple(map(int, item["box"])): float(item.get("score", 0))
        for row in inputs for item in row.get("boxes", [])
    }
    baseline_count = int(baseline.get("usable_face_count", 0) or 0)
    # A very small face is accepted only when it is high-confidence and is
    # supported by a multi-face context: either the baseline already has
    # multiple usable faces or at least two independently selected additions
    # are present. This avoids promoting an isolated tiny background proposal.
    small_group_context = baseline_count >= 2 or len(added) >= 2
    rejected = []
    for box in added:
        evidence = next((item for item in reversed(selected)
                         if tuple(item.get("box", ())) == box), None)
        score = float(evidence.get("score", 0)) if evidence else 0.0
        side = min(box[2:])
        reason = None
        reliable_landmarks = bool(
            evidence and score >= .90 and evidence.get("eye_strict", False)
            and (float(evidence.get("context", 0)) >= .75
                 or evidence.get("occlusion") == "FACE_BLURRY")
        )
        if (detector._native_extreme_motion_blur(image, box)
                and not reliable_landmarks):
            reason = "native_extreme_motion_blur"
        elif (evidence and evidence.get("occlusion") == "FACE_BLURRY"
              and not reliable_landmarks):
            reason = "aligned_face_blurry"
        elif baseline_count >= 12 and evidence and (
                score < .92 or not evidence.get("eye_strict", False)):
            reason = "dense_scene_conservative_addition"
        elif side < 50 and not (
                side >= 20 and evidence and score >= .90 and small_group_context):
            reason = "undersized_candidate"
        elif not evidence and side < 90:
            reason = "small_candidate_without_repeatability_record"
        elif evidence and score < .90 and not evidence.get("eye_strict", False):
            source = evidence.get("source_box") or box
            if min(map(int, source[2:])) > 60:
                reason = "weak_landmarks_on_non_small_face"
            else:
                sharpness, contrast = _native_detail(image, box)
                if sharpness < 35 or contrast < 60:
                    reason = "insufficient_native_feature_detail"

        if (reason is None and evidence and baseline_count and evidence.get("source_box")):
            source = evidence["source_box"]
            source_area = max(1.0, float(source[2]) * float(source[3]))
            final_area = float(box[2]) * float(box[3])
            if final_area / source_area > 2.0:
                for previous in before:
                    if detector._box_overlap_over_smaller(previous, box) >= .55:
                        reason = "expanded_box_overlaps_existing_face"
                        break

        if (reason is None and baseline_count == 1 and evidence
                and evidence.get("source_box") and score >= .94
                and evidence.get("eye_strict", False)):
            original = next((item for item in before
                             if initial_scores.get(item, 0) < .85), None)
            source_area = float(evidence["source_box"][2]) * float(evidence["source_box"][3])
            if (original is not None and original[2] * original[3] > 0
                    and source_area / (original[2] * original[3]) > 20):
                reason = "disproportionate_candidate_over_weak_seed"

        if reason:
            rejected.append({"box": list(box), "reason": reason, "score": score})

    if rejected:
        TRACE.append({"stage": "feature-guard-rejected-additions", "items": rejected})
        if len(rejected) == len(added):
            return baseline
        removed = {tuple(item["box"]) for item in rejected}
        kept = [box for box in after if box not in removed]
        revised = dict(result)
        revised["usable_face_boxes"] = kept
        revised["face_count"] = len(kept)
        revised["usable_face_count"] = len(kept)
        if kept:
            revised["face"] = max(kept, key=lambda box: box[2] * box[3])
        return detector._with_usable_face_count(revised)
    return result



# Frozen source: combined_v24_evidence_tuned.py=e53037d11789f94fea9b222ba2ce6ada93f470391ca2323890323f4b1f440a65
def _review_v24(detector, image_path, baseline, locale=None):
    result = _review_v22(detector, image_path, baseline, locale)
    rejected_rows = [row for row in TRACE
                     if row.get("stage") == "feature-guard-rejected-additions"]
    rejected_items = [item for row in rejected_rows for item in row.get("items", [])]
    if not rejected_items:
        return result

    selected = [item for item in TRACE if item.get("stage") == "proposal-selected"]
    evidence_rows = [item for item in TRACE if item.get("stage") == "proposal-evidence"]
    built = [item for item in TRACE if item.get("stage") == "result-build"]
    if not built:
        return result

    baseline_boxes = [tuple(map(int, box)) for box in baseline.get("usable_face_boxes", [])]
    baseline_count = int(baseline.get("usable_face_count", len(baseline_boxes)) or 0)
    pre_guard_boxes = [tuple(map(int, box)) for box in built[-1].get("boxes", [])]
    current_boxes = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    restored = []
    for item in rejected_items:
        box = tuple(map(int, item.get("box", ())))
        selection = next((row for row in reversed(selected)
                          if tuple(map(int, row.get("box", ()))) == box), None)
        if selection is None:
            continue
        source = tuple(map(int, selection.get("source_box", box)))
        evidence = next((row for row in reversed(evidence_rows)
                         if tuple(map(int, row.get("box", ()))) == source), {})
        score = float(selection.get("score", 0.0))
        side = min(box[2:])
        observations = int(evidence.get("observations", 0))
        strong = int(evidence.get("strong", 0))
        contexts = int(evidence.get("contexts", 0))
        reason = item.get("reason")

        tiny_group_face = (
            reason == "undersized_candidate" and side >= 20
            and baseline_count >= 1 and score >= .90
            and observations >= 60 and strong >= 20 and contexts >= 2
            and selection.get("quality") != "FACE_BLURRY"
        )
        distant_group_face = (
            reason == "undersized_candidate" and side >= 30
            and baseline_count >= 2 and score >= .83
            and observations >= 30 and strong >= 4 and contexts >= 2
            and selection.get("quality") != "FACE_BLURRY"
        )
        repeated_blurred_face = (
            reason == "native_extreme_motion_blur" and score >= .89
            and selection.get("eye_strict") is True
            and observations >= 60 and strong >= 30 and contexts >= 2
            and selection.get("occlusion") == "FACE_BLURRY"
        )
        if ((tiny_group_face or distant_group_face or repeated_blurred_face)
                and box in pre_guard_boxes):
            restored.append(box)

    if not restored:
        return result
    merged = list(current_boxes)
    for box in restored:
        if box not in merged:
            merged.append(box)
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in merged]
    revised["face_count"] = len(merged)
    revised["usable_face_count"] = len(merged)
    if merged:
        revised["face"] = list(max(merged, key=lambda box: box[2] * box[3]))
    TRACE.append({"stage": "v24-evidence-restored", "boxes": [list(box) for box in restored]})
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v25_dark_face_candidate.py=310d26f64e54b26a00a08b91b55166b51b3068ad4573fe7fe05f64639ece6dff
def _review_v25(detector, image_path, baseline, locale=None):
    result = _review_v24(detector, image_path, baseline, locale)
    rejects = [row for row in TRACE if row.get("stage") == "proposal-rejected"]
    selected = [row for row in TRACE if row.get("stage") == "proposal-selected"]
    evidence_rows = [row for row in TRACE if row.get("stage") == "proposal-evidence"]
    existing = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    if not rejects or not existing:
        return result

    added = []
    for reject in rejects:
        if reject.get("reason") != "native_extreme_blur":
            continue
        if reject.get("source_quality") != "FACE_TOO_DARK":
            continue
        source = tuple(map(int, reject.get("box", ())))
        selection = next((row for row in reversed(selected)
                          if tuple(map(int, row.get("source_box", ())) or ()) == source), None)
        evidence = next((row for row in reversed(evidence_rows)
                         if tuple(map(int, row.get("box", ()))) == source), None)
        if selection is None or evidence is None:
            continue
        box = tuple(map(int, reject.get("chosen", ())))
        if len(box) != 4 or box[2] < 120 or box[3] < 120:
            continue
        if (float(reject.get("score", 0.0)) < .87
                or int(evidence.get("observations", 0)) < 40
                or int(evidence.get("strong", 0)) < 15
                or int(evidence.get("contexts", 0)) < 3
                or len(evidence.get("sizes", [])) < 3
                or len(evidence.get("angles", [])) < 3
                or selection.get("quality") is not None
                or selection.get("occlusion") is not None
                or selection.get("core") is not None):
            continue
        if any(detector._box_overlap_over_smaller(old, box) >= .35 for old in existing + added):
            continue
        added.append(box)

    if not added:
        return result
    merged = existing + added
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in merged]
    revised["face_count"] = len(merged)
    revised["usable_face_count"] = len(merged)
    revised["face"] = list(max(merged, key=lambda box: box[2] * box[3]))
    TRACE.append({"stage": "v25-dark-face-restored", "boxes": [list(box) for box in added]})
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v27_repeatable_blur_guard.py=7ca874977a2593ce867b582b1eb4a578c71c870f60fdc6928d916e3711d678fd
def _review_v27(detector, image_path, baseline, locale=None):
    result = _review_v25(detector, image_path, baseline, locale)
    rejected_rows = [row for row in TRACE
                     if row.get("stage") == "feature-guard-rejected-additions"]
    rejected = [item for row in rejected_rows for item in row.get("items", [])
                if item.get("reason") == "aligned_face_blurry"]
    if not rejected:
        return result
    selected = [row for row in TRACE if row.get("stage") == "proposal-selected"]
    evidence_rows = [row for row in TRACE if row.get("stage") == "proposal-evidence"]
    built = [row for row in TRACE if row.get("stage") == "result-build"]
    if not built:
        return result
    possible = [tuple(map(int, box)) for box in built[-1].get("boxes", [])]
    boxes = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    restored = []
    for item in rejected:
        box = tuple(map(int, item.get("box", ())))
        if box not in possible:
            continue
        selection = next((row for row in reversed(selected)
                          if tuple(map(int, row.get("box", ()))) == box), None)
        if selection is None or not selection.get("eye_strict"):
            continue
        source = tuple(map(int, selection.get("source_box", ())))
        evidence = next((row for row in reversed(evidence_rows)
                         if tuple(map(int, row.get("box", ()))) == source), {})
        if (float(selection.get("score", 0.0)) < .88
                or float(selection.get("context", 0.0)) < .4
                or int(evidence.get("observations", 0)) < 60
                or int(evidence.get("strong", 0)) < 30
                or int(evidence.get("contexts", 0)) < 3
                or len(evidence.get("sizes", [])) < 3
                or len(evidence.get("angles", [])) < 5):
            continue
        if any(detector._box_overlap_over_smaller(old, box) >= .45
               for old in boxes + restored):
            continue
        restored.append(box)
    if not restored:
        return result
    merged = boxes + restored
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in merged]
    revised["face_count"] = len(merged)
    revised["usable_face_count"] = len(merged)
    revised["face"] = list(max(merged, key=lambda box: box[2] * box[3]))
    TRACE.append({"stage": "v27-repeatable-blur-restored",
                  "boxes": [list(box) for box in restored]})
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v28_single_face_repeatability.py=8d07329e955ce2f57bb49c9393294bd2cc6644466e6d859c924b2297f2fb3e51
def _review_v28(detector, image_path, baseline, locale=None):
    result = _review_v27(detector, image_path, baseline, locale)
    if result.get("usable_face_boxes"):
        return result
    rejected_rows = [row for row in TRACE
                     if row.get("stage") == "feature-guard-rejected-additions"]
    rejected = [item for row in rejected_rows for item in row.get("items", [])
                if item.get("reason") == "weak_landmarks_on_non_small_face"]
    if not rejected:
        return result

    selected = [row for row in TRACE if row.get("stage") == "proposal-selected"]
    evidence_rows = [row for row in TRACE if row.get("stage") == "proposal-evidence"]
    built = [row for row in TRACE if row.get("stage") == "result-build"]
    if not built:
        return result
    result_boxes = [tuple(map(int, box)) for box in built[-1].get("boxes", [])]
    if len(result_boxes) != 1:
        return result
    chosen = result_boxes[0]
    item = next((row for row in rejected
                 if tuple(map(int, row.get("box", ()))) == chosen), None)
    if item is None:
        return result
    selection = next((row for row in reversed(selected)
                      if tuple(map(int, row.get("box", ()))) == chosen), None)
    if selection is None:
        return result
    source = tuple(map(int, selection.get("source_box", ())))
    evidence = next((row for row in reversed(evidence_rows)
                     if tuple(map(int, row.get("box", ()))) == source), {})
    if (float(selection.get("score", 0.0)) < .84
            or min(chosen[2:]) < 60
            or int(evidence.get("observations", 0)) < 50
            or int(evidence.get("strong", 0)) < 20
            or int(evidence.get("contexts", 0)) < 3
            or len(evidence.get("sizes", [])) < 3
            or len(evidence.get("angles", [])) < 5
            or selection.get("quality") is not None
            or selection.get("occlusion") is not None
            or selection.get("core") is not None):
        return result
    revised = dict(result)
    revised["usable_face_boxes"] = [list(chosen)]
    revised["face"] = list(chosen)
    revised["face_count"] = revised["usable_face_count"] = 1
    TRACE.append({"stage": "v28-single-face-repeatability-restored",
                  "boxes": [list(chosen)]})
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v30_repeatable_tiny.py=bc8aa0108e1e33195fed350362ac07289b72ff6592b22c6896d2c167edf0b088
def _review_v30(detector, image_path, baseline, locale=None):
    result = _review_v28(detector, image_path, baseline, locale)
    if result.get("usable_face_boxes"):
        return result
    rejected_rows = [row for row in TRACE
                     if row.get("stage") == "feature-guard-rejected-additions"]
    rejected = [item for row in rejected_rows for item in row.get("items", [])
                if item.get("reason") == "undersized_candidate"]
    built = [row for row in TRACE if row.get("stage") == "result-build"]
    if not rejected or not built:
        return result
    result_boxes = [tuple(map(int, box)) for box in built[-1].get("boxes", [])]
    if len(result_boxes) != 1:
        return result
    chosen = result_boxes[0]
    if not any(tuple(map(int, item.get("box", ()))) == chosen for item in rejected):
        return result
    selected = [row for row in TRACE if row.get("stage") == "proposal-selected"]
    evidence_rows = [row for row in TRACE if row.get("stage") == "proposal-evidence"]
    selection = next((row for row in reversed(selected)
                      if tuple(map(int, row.get("box", ()))) == chosen), None)
    if selection is None:
        return result
    source = tuple(map(int, selection.get("source_box", ())))
    evidence = next((row for row in reversed(evidence_rows)
                     if tuple(map(int, row.get("box", ()))) == source), {})
    # Strong repeatability/scale/angle evidence is required; only relax the
    # minimum pixel side from v29's 30px to 20px after that evidence is met.
    if (min(chosen[2:]) < 20 or min(chosen[2:]) > 52
            or float(selection.get("score", 0.0)) < .90
            or int(evidence.get("observations", 0)) < 60
            or int(evidence.get("strong", 0)) < 40
            or int(evidence.get("contexts", 0)) < 2
            or len(evidence.get("sizes", [])) < 4
            or len(evidence.get("angles", [])) < 4
            or selection.get("quality") is not None
            or selection.get("occlusion") is not None
            or selection.get("core") is not None):
        return result
    revised = dict(result)
    revised["usable_face_boxes"] = [list(chosen)]
    revised["face"] = list(chosen)
    revised["face_count"] = revised["usable_face_count"] = 1
    TRACE.append({"stage": "v30-repeatable-tiny-restored", "boxes": [list(chosen)]})
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v31_weak_face_corrob.py=82fb1eab1d6b16789bb188c0987bb2c62de7a6e572ad35143c53ec30d4ad151a
def _context_helper(detector):
    source = Path(detector.__file__).with_name("face_consistency.py")
    spec = importlib.util.spec_from_file_location("crop_under10_installed_context_v31", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.context_evidence

def _review_v31(detector, image_path, baseline, locale=None):
    result = _review_v30(detector, image_path, baseline, locale)
    boxes = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    if len(boxes) < 2:
        return result
    work = detector._request_work.get()
    known = work.get("consistency_accepted_candidates", {}) if work is not None else {}
    old_boxes = {tuple(map(int, box)) for box in baseline.get("usable_face_boxes", [])}
    strong = [box for box in boxes if float(known.get(box, {}).get("score", 0)) >= .90]
    if not strong:
        return result
    weak = [box for box in boxes if box in old_boxes and
            .60 <= float(known.get(box, {}).get("score", 0)) < .86 and
            not known.get(box, {}).get("native_roll_confirmed")]
    if not weak:
        return result
    image = detector._read_image(image_path)
    if image is None:
        return result
    context_evidence = _context_helper(detector)
    removed = []
    for box in weak:
        candidate = known[box]
        observations = context_evidence(detector, image, candidate)
        best = max((float(item["score"]) for item in observations), default=0.0)
        if best < .82:
            removed.append(box)
            TRACE.append({"stage": "v31-weak-face-unconfirmed", "box": list(box),
                          "score": float(candidate["score"]), "context_peak": best})
    if not removed or len(removed) >= len(boxes):
        return result
    kept = [box for box in boxes if box not in removed]
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in kept]
    revised["face_count"] = revised["usable_face_count"] = len(kept)
    revised["face"] = list(max(kept, key=lambda box: box[2] * box[3]))
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v32_dark_soft_candidate.py=9fb51c3ed1918fb9ce98ecd93fde6dc85f5ee1d08e5a2dd633fe4b7ff794b970
def _review_v32(detector, image_path, baseline, locale=None):
    result = _review_v31(detector, image_path, baseline, locale)
    boxes = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    if len(boxes) < 2:
        return result
    original = {tuple(map(int, box)) for box in baseline.get("usable_face_boxes", [])}
    selected = {tuple(map(int, row["box"])): row for row in TRACE
                if row.get("stage") == "proposal-selected"}
    possible = [box for box in boxes if box not in original
                and box in selected and min(box[2:]) >= 70
                and float(selected[box].get("context", 1)) <= .4]
    if not possible:
        return result
    image = detector._read_image(image_path)
    if image is None:
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    removed = []
    for box in possible:
        x, y, width, height = box
        roi = gray[y:y + height, x:x + width]
        if roi.size == 0:
            continue
        compact = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_LINEAR)
        brightness = float(compact.mean())
        sharpness = float(cv2.Laplacian(compact, cv2.CV_64F).var())
        if 30 <= brightness < 40 and sharpness < 85:
            removed.append(box)
            TRACE.append({"stage": "v32-dark-soft-candidate-rejected",
                          "box": list(box), "brightness": brightness,
                          "sharpness": sharpness})
    if not removed or len(removed) >= len(boxes):
        return result
    kept = [box for box in boxes if box not in removed]
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in kept]
    revised["face_count"] = revised["usable_face_count"] = len(kept)
    revised["face"] = list(max(kept, key=lambda box: box[2] * box[3]))
    return detector._with_usable_face_count(revised)



# Frozen source: combined_v33_dense_scene_evidence.py=f57cd1511524b066c6cb44f9ef049e0c1991226348d4cfa06e951392cff58945
def _patch_detail(gray, box):
    x, y, width, height = box
    roi = gray[y:y + height, x:x + width]
    if roi.size == 0:
        return None
    compact = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_LINEAR)
    return float(compact.mean()), float(cv2.Laplacian(compact, cv2.CV_64F).var())

def _review_v33(detector, image_path, baseline, locale=None):
    result = _review_v32(detector, image_path, baseline, locale)
    boxes = [tuple(map(int, box)) for box in result.get("usable_face_boxes", [])]
    if len(boxes) < 5:
        return result
    image = detector._read_image(image_path)
    if image is None:
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    work = detector._request_work.get()
    known = work.get("consistency_accepted_candidates", {}) if work is not None else {}
    selections = {tuple(map(int, row["box"])): row for row in TRACE
                  if row.get("stage") == "proposal-selected"}
    evidence = {tuple(map(int, row["box"])): row for row in TRACE
                if row.get("stage") == "proposal-evidence"}
    initial = next((row for row in TRACE if row.get("stage") == "complete-input"), {})
    initial_scores = {tuple(map(int, item["box"])): float(item.get("score", 0))
                      for item in initial.get("boxes", [])}

    removed = []
    for box in boxes:
        if min(box[2:]) < 70:
            continue
        candidate = known.get(box, {})
        score = float(candidate.get("score") or initial_scores.get(box)
                      or selections.get(box, {}).get("score", 0))
        if not (.60 <= score < .89):
            continue
        detail = _patch_detail(gray, box)
        if detail and 30 <= detail[0] < 45 and detail[1] < 85:
            removed.append(box)
            TRACE.append({"stage": "v33-dense-weak-face-rejected", "box": list(box),
                          "score": score, "brightness": detail[0],
                          "sharpness": detail[1]})
    kept = [box for box in boxes if box not in removed]

    restored = []
    if len(kept) >= 10:
        for row in TRACE:
            if row.get("stage") != "feature-guard-rejected-additions":
                continue
            for item in row.get("items", []):
                if item.get("reason") != "dense_scene_conservative_addition":
                    continue
                box = tuple(map(int, item.get("box", ())))
                selection = selections.get(box)
                if selection is None:
                    continue
                source = tuple(map(int, selection.get("source_box", ())))
                proof = evidence.get(source, {})
                if (float(selection.get("score", 0)) < .905
                        or int(proof.get("observations", 0)) < 90
                        or int(proof.get("strong", 0)) < 55
                        or int(proof.get("contexts", 0)) < 3
                        or len(proof.get("sizes", [])) < 5
                        or len(proof.get("angles", [])) < 5
                        or any(selection.get(key) is not None
                               for key in ("quality", "core", "occlusion"))):
                    continue
                detail = _patch_detail(gray, box)
                if detail is None or detail[0] < 55:
                    continue
                if any(detector._box_overlap_over_smaller(box, old) >= .45
                       for old in kept + restored):
                    continue
                restored.append(box)
                TRACE.append({"stage": "v33-repeatable-dense-face-restored",
                              "box": list(box), "score": float(selection["score"]),
                              "observations": int(proof["observations"])})

    if not removed and not restored:
        return result
    merged = kept + restored
    if not merged:
        return result
    revised = dict(result)
    revised["usable_face_boxes"] = [list(box) for box in merged]
    revised["face_count"] = revised["usable_face_count"] = len(merged)
    revised["face"] = list(max(merged, key=lambda box: box[2] * box[3]))
    return detector._with_usable_face_count(revised)


def review_human_faces(detector, image_path, baseline, locale=None):
    from . import face_consistency_refinement

    reviewed = _review_v33(detector, image_path, baseline, locale)
    return face_consistency_refinement.refine(
        detector, sys.modules[__name__], image_path, baseline, reviewed, locale,
    )
