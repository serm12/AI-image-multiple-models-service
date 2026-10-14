"""Independent SCRFD-10G post-validation face-angle evidence."""
import math
import statistics
import numpy as np
import cv2

TEMPLATE=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=float)
Q=TEMPLATE-TEMPLATE.mean(axis=0)

def angle(points):
    p=np.asarray(points,dtype=float);p=p-p.mean(axis=0)
    return math.degrees(math.atan2(float((Q[:,0]*p[:,1]-Q[:,1]*p[:,0]).sum()),float((Q*p).sum())))

def diff(a,b):
    return (a-b+180)%360-180

def consensus(angles):
    seed=min(angles,key=lambda a:sum(abs(diff(x,a)) for x in angles))
    val=(seed+statistics.median([diff(x,seed) for x in angles])+180)%360-180
    spread=max(diff(x,val) for x in angles)-min(diff(x,val) for x in angles)
    return val,spread

def overlap(a,b):
    a,b=np.asarray(a),np.asarray(b)
    low=np.maximum(a[:2],b[:2]);high=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    inter=float(np.maximum(high-low,0).prod())
    return inter/max(1.,float(a[2:].prod()+b[2:].prod())-inter)


def shape_error(points):
    p=np.asarray(points,dtype=float);p=p-p.mean(axis=0)
    a=math.radians(angle(points));c,s=math.cos(a),math.sin(a)
    pred=Q@np.array([[c,s],[-s,c]])
    pred*=float((pred*p).sum()/(pred*pred).sum())
    return float(np.sqrt(np.mean((pred-p)**2))/max(np.linalg.norm(p[1]-p[0]),1))

def hypotheses(image,box,detector,coarse_angles=(0,90,180,270),factor=2.5,input_side=320):
    """Re-detect only an already-usable face in four directions. Does not alter face counts."""
    center=np.asarray(box[:2])+np.asarray(box[2:])/2
    side=max(box[2:])*factor
    observations=[]
    for coarse in coarse_angles:
        m=cv2.getRotationMatrix2D(tuple(center),coarse,input_side/side)
        m[:,2]+=np.array([(input_side-1)/2,(input_side-1)/2])-center
        view=cv2.warpAffine(image,m,(input_side,input_side),flags=cv2.INTER_LINEAR,borderValue=(127,127,127))
        detections,keypoints=detector.detect(view,input_size=(input_side,input_side))
        inv=cv2.invertAffineTransform(m)
        for rect,pts in zip(detections,keypoints):
            # Associate with the original, accepted person; neighbouring faces cannot vote.
            corners=np.array([[rect[0],rect[1]],[rect[2],rect[1]],[rect[0],rect[3]],[rect[2],rect[3]]])
            mapped=corners@inv[:,:2].T+inv[:,2]
            low,high=mapped.min(axis=0),mapped.max(axis=0)
            mapped_box=np.r_[low,high-low]
            iou=overlap(mapped_box,box)
            distance=np.linalg.norm((low+high)/2-center)/max(min(box[2:]),1)
            if iou<.35 or distance>.35:continue
            pts=np.asarray(pts,dtype=float)
            mapped_pts=pts@inv[:,:2].T+inv[:,2]
            view_angle=angle(pts)
            error=shape_error(pts)
            eyes=pts[1]-pts[0];span=np.linalg.norm(eyes)
            down=np.array([-eyes[1],eyes[0]])/max(span,1e-6)
            depth=float(np.dot((pts[3]+pts[4])/2-(pts[0]+pts[1])/2,down)/max(span,1e-6))
            observations.append(dict(coarse_cv_deg=coarse,score=float(rect[4]),angle_deg=angle(mapped_pts),
                                     view_angle_deg=view_angle,shape_error=error,mouth_depth=depth,
                                     association_iou=iou,center_distance=distance,points=mapped_pts.tolist(),box=mapped_box.tolist(),
                                     plausible=bool(abs(view_angle)<=65 and .3<=depth<=2.2 and error<=.4)))
    return observations

def choose(observations):
    candidates=[x for x in observations if x['plausible']]
    # Detector score dominates; geometry only filters impossible layouts.
    # No angle, file name or expected orientation is supplied per image.
    if not candidates:return None
    return max(candidates,key=lambda x:x['score'])

def estimate_face(image,box,detector):
    observations=hypotheses(image,box,detector)
    best=choose(observations)
    if best is None:
        return dict(reliable=False,reason='no_plausible_direction',observations=observations,network_calls=4)
    competitors=[o for o in observations if o['plausible'] and abs(diff(o['angle_deg'],best['angle_deg']))>100]
    opposite=max(competitors,key=lambda o:o['score']) if competitors else None
    direction_margin=best['score']-opposite['score'] if opposite else None
    refinements=[]
    for factor in (2.1,2.8):
        obs=hypotheses(image,box,detector,coarse_angles=(best['angle_deg'],),factor=factor)
        matched=[o for o in obs if o['plausible'] and abs(diff(o['angle_deg'],best['angle_deg']))<=20]
        pick=choose(matched)
        if pick is not None:
            refinements.append(pick)
    if not refinements:
        return dict(reliable=False,reason='upright_recheck_failed',observations=observations,refinements=refinements,network_calls=6)
    stable_angles=[o['angle_deg'] for o in refinements]
    refined,spread=consensus(stable_angles)
    reliable=bool((direction_margin is None or direction_margin>=.05) and spread<=12)
    # Representative points are from the recheck nearest the consensus angle.
    selected=min(refinements,key=lambda o:abs(diff(o['angle_deg'],refined)))
    return dict(reliable=reliable,reason='verified' if reliable else 'ambiguous_direction_or_recheck',
                angle_deg=refined,points=selected['points'],score=selected['score'],association_iou=selected['association_iou'],
                coarse_choice=best['coarse_cv_deg'],initial_angle_deg=best['angle_deg'],direction_margin=direction_margin,
                recheck_spread_deg=spread,observations=observations,refinements=refinements,network_calls=6)

def estimate_image(image,boxes,detector):
    """Return this model's independent face-angle evidence, without rotation policy."""
    faces=[]
    for index,box in enumerate(boxes,1):
        f=estimate_face(image,box,detector)
        faces.append(dict(index=index,box=box,**f))
    credible=[f['angle_deg'] for f in faces if f['reliable']]
    spread=consensus(credible)[1] if credible else None
    return dict(faces=faces,face_angle_spread_deg=spread,
                landmark_faces=len(faces),orientation_faces=len(credible),
                network_calls=sum(f['network_calls'] for f in faces))
