"""BlazeFace pose probes; all boxes and quality remain in submitted pixel space."""
import cv2
import numpy as np
from . import evidence,probe

def collect(image,candidates,det,mp):
    found=[]
    for j,c in enumerate(candidates):
        f=c['features']
        boundary_detail = (c['score'] >= 0.5 and f.get('native_lap',0) >= 100
                           and f.get('eye_span',1) < 0.2
                           and (c['box'][0] < 0.5*c['box'][2]
                                or c['box'][0]+c['box'][2] > image.shape[1]-0.5*c['box'][2]))
        clipped_profile = (c['score'] >= 0.4 and f.get('native_lap',0) >= 150
                           and min(c['box'][2:]) >= 60 and c['box'][0] < 0
                           and f.get('visible',0) >= 0.8
                           and len(f.get('eyes',[])) >= 2
                           and max(p[1] for p in f['eyes'][:2]) >= 40)
        if (c['score'] < 0.3 or max(c['confirm']) >= 0.6
                or (len(set(c['views'])) < 2 and not (boundary_detail or clipped_profile))
                or min(c['box'][2:]) < 32 or not f.get('eyes')
                or min(p[1] for p in f['eyes']) < 15
                or max(p[2] for p in f['eyes']) < 30
                or not (f.get('eye_span',1) < 0.2 or abs(f.get('nose_side',0)) > 0.8 or clipped_profile)):
            continue
        x,y,w,h=c['box'];center=np.array([x+w/2,y+h/2])
        for factor in (1.5,2.5,4.0):
            side=max(64,round(max(w,h)*factor));scale=min(1,384/side);outsize=round(side*scale)
            for angle in (-60,-30,0,30,60,90):
                for stretch in (0.65,1.5,2.0):
                    m=cv2.getRotationMatrix2D(tuple(center),angle,scale)
                    m[:,2]+=np.array([outsize/2]*2)-center
                    stretch_m=np.array([[stretch,0,(1-stretch)*outsize/2],[0,1,0],[0,0,1.]])
                    m=(stretch_m@np.vstack([m,[0,0,1]]))[:2]
                    view=cv2.warpAffine(image,m,(outsize,outsize),borderMode=cv2.BORDER_REPLICATE)
                    inv=cv2.invertAffineTransform(m)
                    for flip in (False,True):
                        matrix=inv if not flip else (np.vstack([inv,[0,0,1]])@np.array([[-1.,0,outsize-1],[0,1.,0],[0,0,1.]]))[:2]
                        for c2 in evidence.detect_view(cv2.flip(view,1) if flip else view,det,mp,matrix,f'pose:{j}:{factor}:{angle}:{stretch}:{flip}'):
                            if flip:c2['points']=[c2['points'][k] for k in [1,0,2,3,5,4]]
                            if evidence.overlap(c['box'],c2['box'])>.4:found.append(c2)
    return probe.merge(found)
