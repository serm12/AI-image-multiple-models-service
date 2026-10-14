"""Bounded local search for pets too small to survive whole-image resizing.

Only returns diagnostic evidence, never an accepted face or a substitute box.
"""
import numpy as np
from .visual import species_evidence

def find_tiny_pet(image, detect):
    height,width=image.shape[:2]
    if min(width,height)<600:return []
    cw,ch=width//6,height//6
    windows=[]
    for y in np.linspace(0,height-ch,9).astype(int):
        for x in np.linspace(0,width-cw,9).astype(int):
            evidence=species_evidence(image[y:y+ch,x:x+cw])
            best=max(evidence.values(),key=lambda e:e['visible'])
            windows.append((best['visible']-best['nonpet'],int(x),int(y)))
    margin,x,y=max(windows)
    if margin<.04:return []
    fine=[];fw,fh=cw//3,ch//3
    for yy in np.linspace(y,y+ch-fh,4).astype(int):
        for xx in np.linspace(x,x+cw-fw,4).astype(int):
            evidence=species_evidence(image[yy:yy+fh,xx:xx+fw])
            best=max(evidence.values(),key=lambda e:e['visible'])
            fine.append((best['visible']-best['nonpet'],int(xx),int(yy)))
    found=[]
    for margin,xx,yy in sorted(fine,reverse=True)[:3]:
        if margin<.04:continue
        crop=image[yy:yy+fh,xx:xx+fw]
        for head in detect(crop):
            bx,by,bw,bh=head['box']
            if max(bw,bh)>48 or max(bw/width,bh/height)>.04:continue
            evidence=species_evidence(crop[by:by+bh,bx:bx+bw])
            species=max(evidence,key=lambda s:evidence[s]['visible']);e=evidence[species]
            if e['visible']<=e['nonpet']+.03:continue
            found.append(dict(box=[xx+bx,yy+by,bw,bh],species=species,score=head['score'],issue='PET_TOO_SMALL'))
    return found
