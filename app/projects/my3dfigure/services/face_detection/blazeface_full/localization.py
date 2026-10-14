"""Final box coverage for readable collapsed-eye side profiles.

This never changes qualification or count and is applied only after the search
finishes. Detailed unconfirmed border observations can localize the nose/mouth
more tightly than the face; reserve forehead space along the facial axis.
"""
import math

def face_box(c):
    if 'localization_box' in c:
        return list(c['localization_box'])
    box=list(c['box']);f=c.get('features',{});patches=f.get('eyes',[])
    if not (any(v.startswith(('border:','border-refine:')) for v in c.get('views',[]))
            and c['score'] < 0.7 and max(c['confirm']) == 0
            and len(set(c['views'])) >= 6 and 40 <= min(box[2:]) < 100
            and .08 <= f.get('eye_span',1) < .15 and 1 <= abs(f.get('nose_side',0)) < 2
            and len(patches)==4 and min(p[1] for p in patches) >= 40
            and min(p[2] for p in patches) >= 100):
        return box
    x,y,w,h=box;points=c['points']
    dx=(points[0][0]+points[1][0])/2-points[3][0]
    dy=(points[0][1]+points[1][1])/2-points[3][1]
    length=max(math.hypot(dx,dy),1);side=min(w,h)
    cx=x+w/2+.2*side*dx/length;cy=y+h/2+.2*side*dy/length
    return [cx-.875*w,cy-.875*h,1.75*w,1.75*h]
