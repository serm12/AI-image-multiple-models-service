"""Local head evidence, independent from other detector implementations."""
import numpy as np

def analyze(head, size, context=None):
    x,y,width,height=head['box'];ih,iw=size
    evidence=head['evidence'];species=max(evidence,key=lambda s:evidence[s]['visible']);ev=evidence[species]
    if ev['nonpet']>=ev['visible']-.015:return None
    p=head['pixels'];scores=np.asarray(head['landmark_scores']);points=np.asarray(head['landmarks'])
    eye=int(np.argmax(scores[:2]));selected=points[[eye,2]];span=float(np.linalg.norm(selected[0]-selected[1]))
    reliable=bool(scores[eye]>=2 and scores[2]>=2 and span>=3)
    inside=bool(np.all(selected>=[x,y]) and np.all(selected<[x+width,y+height]))
    issues=set();dark_fur=bool(min(scores)>=4 and p['p90']>=50 and p['std']>=15)
    if min(width,height)<16:issues.add('PET_TOO_SMALL')
    if p['mean']<35 and not dark_fur:issues.add('PET_TOO_DARK')
    elif p['mean']>235 or p['bright']>.65:issues.add('PET_OVEREXPOSED')
    elif not dark_fur and 'PET_TOO_SMALL' not in issues and p['lap']<25:issues.add('PET_BLURRY')
    margin=max(1.,span*.2)
    if reliable and (np.any(selected<margin) or np.any(selected>=[iw-margin,ih-margin])):issues.add('PET_INCOMPLETE')
    if not issues:
        if ev['occluded']>=.28 and ev['occluded']>max(ev['visible']+.015,ev['back']+.01,ev['blur']+.01):issues.add('PET_OCCLUDED')
        elif max(ev['back'],ev['fur'])>max(ev['visible']+.015,ev['blur']+.015):issues.add('PET_FACE_UNRECOGNIZABLE')
    usable=bool(not issues and reliable and inside)
    # Contradictory scene and face species evidence cannot certify a domestic pet.
    if context and usable:
        scene=max(context,key=lambda s:context[s]['visible'])
        if scene!=species and context[scene]['visible']>context[species]['visible']+.02 and context[scene]['visible']<context[scene]['nonpet']+.015:
            usable=False
    face=dict(usable=usable,issues=sorted(issues),landmarks=head['landmarks'],landmark_scores=head['landmark_scores'],visual_similarities=ev)
    if usable:
        # Retain chin/muzzle around the regressed head, including rotated faces.
        left=max(0,int(np.floor(x-width*.08)));right=min(iw,int(np.ceil(x+width*1.08)))
        top=max(0,int(np.floor(y-height*.18)));bottom=min(ih,int(np.ceil(y+height*1.18)))
        face['identity_anchor_box']=[left,top,right-left,bottom-top]
    return {k:v for k,v in {**head,'species':species,'quality_issues':sorted(issues),'usable':usable,'face_analysis':face}.items() if k not in ('evidence','pixels')}

def diagnostic_issues(heads, size, context):
    """Weak proposals only explain rejection; they never create usable faces."""
    issues=set()
    for h in heads:
        if h['score']<.05:continue
        sp=max(h['evidence'],key=lambda s:h['evidence'][s]['visible']);ev=h['evidence'][sp];p=h['pixels']
        if ev['visible']<ev['nonpet']+.02 and not (h['score']>.5 and p['mean']<15 and context and max(e['visible']-e['nonpet'] for e in context.values())>.01):continue
        if p['mean']<35:issues.add('PET_TOO_DARK')
        if p['bright']>.65:issues.add('PET_OVEREXPOSED')
        if p['lap']<25 and ev['motion']>max(ev['visible']+.03,ev['blur']+.025) and ev['motion']>ev['back']+.015:issues.add('PET_BLURRY')
        x,y,w,hgt=h['box'];ih,iw=size
        if h['score']>=.1 and p['mean']>=35 and (x<iw*.08 or y<ih*.015 or x+w>iw*.985 or y+hgt>ih*.985) and ev['cut']>max(ev['visible']+.008,ev['motion']+.005):issues.add('PET_INCOMPLETE')
    if context:
        sp=max(context,key=lambda s:context[s]['visible']);ev=context[sp]
        if ev['visible']>ev['nonpet']+.02:
            if ev['motion']>max(ev['visible']+.05,ev['blur']+.015,ev['back']+.025):issues.add('PET_BLURRY')
            if ev['fur']>max(ev['visible']+.02,ev['back']+.02,ev['motion']+.02):issues.add('PET_OCCLUDED')
            if ev['occluded']>max(ev['visible']+.03,ev['back']+.02):issues.add('PET_OCCLUDED')
    return sorted(issues)
