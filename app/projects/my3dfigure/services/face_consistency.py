"""Shared image-only YuNet localisation and quality corroboration.

The caller supplies the ordinary detector helpers and request-local evidence.
No crop provenance, filenames, original-image anchors or expected counts are
used to accept a face. Both manual and automated requests use this path.
"""
from copy import deepcopy

import cv2
import numpy as np
def _same_face(detector, first, second):
    a=np.asarray(first['box'],dtype=float);b=np.asarray(second['box'],dtype=float)
    centers=np.abs(a[:2]+a[2:]/2-b[:2]-b[2:]/2)
    # Overlapping head rectangles are not enough to merge adjacent people.
    aligned=bool(np.all(centers<=np.maximum(a[2:],b[2:])*.35))
    return aligned and (detector._box_iou(a,b)>=.3 or detector._box_overlap_over_smaller(a,b)>=.65)


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
    return sorted(clusters,key=lambda c:c['score'],reverse=True)


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
    for padding in (.4,.8,1.2):
        l=max(0,round(x-bw*padding));t=max(0,round(y-bh*padding))
        r=min(w,round(x+bw*(1+padding)));b=min(h,round(y+bh*(1+padding)))
        bounds=(l,t,r,b)
        if bounds in bounds_seen: continue
        bounds_seen.add(bounds)
        region=image[t:b,l:r]
        for face_size in (80,120,200):
            scale=min(face_size/min(bw,bh),768/max(region.shape[:2]))
            view=cv2.resize(region,None,fx=scale,fy=scale)
            # Native quarter turns use clockwise angles; OpenCV's affine
            # rotation API uses counterclockwise angles.
            turn=-proposal.get('turn',0)
            for angle in (turn,turn-15,turn+15,turn-30,turn+30):
                matrix=cv2.getRotationMatrix2D((view.shape[1]/2,view.shape[0]/2),angle,1)
                rotated=view if angle==0 else cv2.warpAffine(view,matrix,(view.shape[1],view.shape[0]))
                inverse=cv2.invertAffineTransform(matrix)
                _,faces=detector._get_yunet_detector((view.shape[1],view.shape[0]),.65).detect(rotated)
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
                    if not _same_face(detector,dict(box=box),proposal): continue
                    points+=np.array([l,t])
                    face=np.array([*raw_box,*points.flatten(),raw[14]],dtype=np.float32)
                    native_matrix=cv2.getRotationMatrix2D((region.shape[1]/2,region.shape[0]/2),angle,1)
                    native_view=region if angle==0 else cv2.warpAffine(region,native_matrix,(region.shape[1],region.shape[0]))
                    native_gray=cv2.cvtColor(native_view,cv2.COLOR_BGR2GRAY)
                    native_face=raw.copy();native_face[:14]/=scale
                    native_box=detector._map_face_to_original(native_face,1,region.shape[1],region.shape[0])
                    native_occlusion=detector._get_face_occlusion_issue(native_gray,native_box,native_face)
                    native_quality=detector._get_face_quality_issue(native_gray,native_box)
                    native_core=detector._native_face_core_issue(native_view,native_box,float(raw[14]))
                    observations.append(dict(box=box,face=face,score=float(raw[14]),context=padding,size=face_size,angle=angle,fully_inside=bool(x0>=0 and y0>=0 and x1<=w and y1<=h),aligned_occlusion=native_occlusion,aligned_quality=native_quality,aligned_core=native_core))
    return observations



def _complete(detector, image, candidates, *, confident_pet_image=False):
    """Review independent locations only after the ordinary quality analysis."""
    threshold = .82
    settled=list(candidates)
    if confident_pet_image: return settled
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hints=proposals(detector,image)
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
    def already_usable(c):
        if c.get('previously_accepted'): return True
        core=detector._native_face_core_issue(image,c['box'],c['score'])
        if (core=='FACE_BLURRY' and c['score']>=.88
                and not detector._is_face_box_cut_by_frame(c['box'],image.shape[1],image.shape[0])):
            core=None
        return (c.get('quality_issue') is None
                and core is None
                and not (c['score']<.88 and detector._native_extreme_motion_blur(image,c['box'])))
    for p in hints:
        if any(already_usable(c) and _same_face(detector,p,c) for c in settled): continue
        observations=context_evidence(detector,image,p)
        strong=[o for o in observations if o['score']>=threshold]
        strong_whole=[o for o in strong if o['score']>=.90]
        whole_view_proof=(len({o['size'] for o in strong_whole})>=2 and len({o['angle'] for o in strong_whole})>=2)
        if (len({o['context'] for o in strong})<2 and not whole_view_proof) or len(strong)<3 or len({o['size'] for o in observations if o['score']>=.70})<2: continue
        # Match the ordinary final pipeline's strong-face review: a
        # >=.90 face with clean native core can override an ambiguous
        # eye-patch label, but never an actual frame/quality failure.
        def source_core_clear(o):
            core=o['aligned_core']
            return core is None or (core=='FACE_BLURRY' and o['score']>=.88 and o['fully_inside'])
        clean=[o for o in strong if o['aligned_occlusion'] is None or (
            o['score']>=(.88 if o['aligned_occlusion']=='FACE_BLURRY' else .90) and o['aligned_occlusion'] in {'FACE_BLURRY','FACE_OCCLUDED'}
            and source_core_clear(o))]
        if len({o['context'] for o in clean})<2 and not (
            whole_view_proof and len(clean)>=3
            and len({o['size'] for o in clean})>=2
            and len({o['angle'] for o in clean})>=2): continue
        contained=[o for o in clean if o['fully_inside']]
        best=max(contained if min(p['box'][2:])>=100 and contained else clean,key=lambda o:o['score'])
        box=best['box'];face=best['face'];score=best['score']
        if score<.84:
            continue
        if min(box[2:])>=100 and score<.88:
            continue
        evidence_points=np.asarray([o['face'][4:14] for o in strong]).reshape(-1,5,2)
        median_points=np.median(evidence_points,axis=0)
        dispersion=np.median(np.linalg.norm(evidence_points-median_points,axis=2),axis=0)/min(p['box'][2:])
        median_eye=float(np.median(np.linalg.norm(evidence_points[:,0]-evidence_points[:,1],axis=1)))/min(p['box'][2:])
        if score<.90 and median_eye>=.25 and float(np.median(dispersion))>.08:
            continue
        # Weak scene-level background proposals need stronger evidence
        # than a dominant portrait. This applies to every submitted image.
        area=box[2]*box[3]
        dominant=max((c['box'][2]*c['box'][3] for c in settled),default=area*3)
        if area/float(image.shape[0]*image.shape[1])<.008 and dominant>area*2 and score<.92:
            continue
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
            continue
        if not best['fully_inside'] and min(bw,bh)<100 and score<.90:
            continue
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
        if source_occlusion=='FACE_OCCLUDED' and min(p['box'][2:])>=80 and score<.92:
            continue
        work=detector._request_work.get()
        if (work is not None and work.get('consistency_initial_code')=='FACE_OCCLUDED'
                and min(bw,bh)>=100 and score<.90):
            continue
        source_quality=detector._get_face_quality_issue(gray,p['box'])
        eye_detail=[]
        if score>=.88 and best['fully_inside'] and min(box[2:])>=100:
            radius=max(2,round(min(box[2:])*.10))
            for px,py in face[4:8].reshape(2,2):
                px,py=round(float(px)),round(float(py))
                patch=gray[max(0,py-radius):py+radius+1,max(0,px-radius):px+radius+1]
                compact=cv2.resize(patch,(16,16),interpolation=cv2.INTER_AREA)
                eye_detail.append((float(cv2.Laplacian(compact,cv2.CV_64F).var()),float(np.percentile(patch,90)-np.percentile(patch,10))))
        resolved_eye_detail=len(eye_detail)==2 and all(sharp>=20 and contrast>=40 for sharp,contrast in eye_detail)
        if not resolved_eye_detail and (detector._native_extreme_motion_blur(image,p['box']) or detector._native_extreme_motion_blur(image,box)):
            continue
        if source_quality=='FACE_BLURRY' and not resolved_eye_detail:
            continue
        quality=best['aligned_quality']
        if quality=='FACE_BLURRY' and resolved_eye_detail:quality=None
        core=best['aligned_core']
        if core=='FACE_BLURRY' and score>=.88 and best['fully_inside']:core=None
        occlusion=best['aligned_occlusion'] if score<(.88 if best['aligned_occlusion']=='FACE_BLURRY' else .90) else None
        edge=(not best['fully_inside']) and detector._is_unusable_edge_cropped_face(face,box,image.shape[1],image.shape[0],score)
        if quality or core or occlusion or edge: continue
        if score<.90:
            eye_vector=points[1]-points[0]
            eye_length=float(np.linalg.norm(eye_vector))
            nose_offset=abs(float(np.dot(points[2]-points[:2].mean(axis=0),eye_vector)))/max(eye_length**2,1)
            px,py,pw,ph=p['box']
            mirrored_proposal=dict(box=(image.shape[1]-px-pw,py,pw,ph),turn=(-p.get('turn',0))%360)
            mirrored=context_evidence(detector,cv2.flip(image,1),mirrored_proposal)
            mirror_score=max((o['score'] for o in mirrored),default=0)
            mirror_floor=(.84 if .12<=median_eye<.25 else .88) if median_eye<.25 else (.85 if nose_offset>=.30 else (.86 if min(bw,bh)>=100 else .88))
            if mirror_score<mirror_floor:
                continue
        if any(already_usable(c) and _same_face(detector,best,c) for c in settled):
            continue
        candidate=dict(box=box,raw_face=face,score=score,ratio=box[2]*box[3]/float(image.shape[0]*image.shape[1]),quality_issue=None,native_context_confirmed_primary=True,native_edge_readable=best['fully_inside'],native_consistency_confirmed=True)
        # A rotated enclosing rectangle can dilute a small ink-on-paper
        # overlay with surrounding photograph pixels. Check the original
        # proposal footprint as well as the final enclosing rectangle.
        source_candidate=dict(candidate,box=p['box'],raw_face=p['face'])
        source_filtered=detector._filter_pet_face_candidates(image,[*settled,source_candidate])
        if not any(c is source_candidate for c in source_filtered):
            continue
        filtered=detector._filter_pet_face_candidates(image,[*settled,candidate])
        if not any(c is candidate for c in filtered):
            continue
        replacements=[c for c in settled if _same_face(detector,c,candidate) or (
            area>c['box'][2]*c['box'][3]*2
            and detector._box_overlap_over_smaller(box,c['box'])>=.55)]
        if len(replacements)>1:
            continue
        settled=[c for c in settled if not any(c is previous for previous in replacements)]
        settled.append(candidate)
    filtered=[]
    for c in settled:
        comparison=[c,*[other for other in settled if other is not c and (
            not c.get('previously_accepted') or not other.get('previously_accepted'))]]
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
    return filter_monochrome_objects(detector,image,verified)


def review_human_faces(detector, image_path, baseline, locale=None):
    """Corroborate usable faces using only this submitted image.

    Preserve the completed baseline when no count change is justified. The
    product headcount is applied later by the existing shared evaluator.
    """
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
    completed=_complete(detector,image,candidates,confident_pet_image=detector._is_confident_pet_image(image))
    if len(completed)==count or not completed: return baseline
    result=detector._yunet_candidate_result(completed,image.shape[1],image.shape[0],locale)
    if not result: return baseline
    if result['face_count']==1 and result.get('face_small') and not detector._face_has_sufficient_native_detail(image_path,result):
        return baseline
    return detector._with_usable_face_count(result)
