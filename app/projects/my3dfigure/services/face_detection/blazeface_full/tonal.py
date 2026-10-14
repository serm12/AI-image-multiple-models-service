"""Reobserve uncertain faces after local tonal normalization; retain source quality."""
import cv2
import numpy as np
from . import evidence,probe

def collect(image,candidates,det,mp):
    found=[]
    for j,c in enumerate(candidates):
        if c['score'] < 0.5 or max(c['confirm']) >= 0.7 or min(c['box'][2:]) < 24:
            continue
        x,y,w,h=c['box'];center=(x+w/2,y+h/2)
        delta=np.array(c['points'][1])-np.array(c['points'][0])
        angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
        for factor in (1.3,2.0,3.0):
            side=max(64,round(max(w,h)*factor));scale=min(1,512/side)
            for offset in (-30,0,30):
                m=cv2.getRotationMatrix2D(center,angle+offset,scale)
                m[:,2]+=np.array([side*scale/2]*2)-np.array(center)
                v=cv2.warpAffine(image,m,(round(side*scale),round(side*scale)),borderMode=cv2.BORDER_REPLICATE)
                gray=cv2.cvtColor(v,cv2.COLOR_BGR2GRAY)
                lo,hi=np.percentile(gray,(5,95))
                stretch=np.clip((gray.astype(np.float32)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)
                clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4)).apply(gray)
                for kind,g in [('stretch',stretch),('clahe',clahe)]:
                    view=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR);inv=cv2.invertAffineTransform(m)
                    for flip in (False,True):
                        matrix=inv if not flip else (np.vstack([inv,[0,0,1]])@np.array([[-1.,0,view.shape[1]-1],[0,1.,0],[0,0,1.]]))[:2]
                        for f in evidence.detect_view(cv2.flip(view,1) if flip else view,det,mp,matrix,f'tonal:{j}:{factor}:{offset}:{kind}:{flip}'):
                            if flip:f['points']=[f['points'][k] for k in [1,0,2,3,5,4]]
                            if evidence.overlap(c['box'],f['box'])>.4:found.append(f)
    return probe.merge(found)
