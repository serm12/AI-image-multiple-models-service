"""Prepare a separate human generation source only after successful validation."""
from __future__ import annotations
import os
import importlib
import hashlib
import logging
import time
import numpy as np
import cv2
from app.projects.my3dfigure.services.face_detection import active_pipeline

VERSION = "checked-source-shared-rotation-policy-v3"

# Change the selection policy HERE for every supported orientation model.
SOURCE_ROTATION_SELECTION = "maximum_absolute"  # or "minimum_absolute"
SOURCE_ROTATION_MIN_DEGREES = 3.0


def rotate(bgr,degrees):
    h,w=bgr.shape[:2]
    m=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),degrees,1.)
    corners=np.array([[-.5,-.5],[w-.5,-.5],[-.5,h-.5],[w-.5,h-.5]])
    mapped=corners@m[:,:2].T+m[:,2]
    low,high=mapped.min(axis=0),mapped.max(axis=0)
    nw,nh=np.ceil(high-low).astype(int)
    m[:,2]-=low+.5
    out=cv2.warpAffine(cv2.cvtColor(bgr,cv2.COLOR_BGR2BGRA),m,(int(nw),int(nh)),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
    return out,m


def select_source_rotation(estimate: dict) -> dict:
    """Apply the shared generation rotation policy to independent model evidence."""
    if SOURCE_ROTATION_SELECTION not in ("maximum_absolute", "minimum_absolute"):
        raise ValueError("Unknown source rotation selection policy")
    result = dict(estimate)
    reliable = [face for face in estimate["faces"] if face["reliable"]]
    result["selection_policy"] = SOURCE_ROTATION_SELECTION
    if not reliable:
        result.update(correction_cv_deg=0., orientation_status="no_verified_direction")
        return result
    def signed_angle(face):
        return (float(face["angle_deg"]) + 180.) % 360. - 180.
    direction = -1 if SOURCE_ROTATION_SELECTION == "maximum_absolute" else 1
    selected = min(reliable, key=lambda face: (
        direction * abs(signed_angle(face)), -face["score"], face["index"]))
    correction = signed_angle(selected)
    if abs(correction) < SOURCE_ROTATION_MIN_DEGREES:
        correction = 0.
    result.update(correction_cv_deg=correction,
                  orientation_status=SOURCE_ROTATION_SELECTION + "_verified_angle",
                  selected_face_index=selected["index"],
                  selected_raw_angle_deg=selected["angle_deg"])
    return result


def prepare_checked_source_rotation(source_path: str, validation: dict, output_dir: str):
    boxes=validation.get("usable_face_boxes") or []
    metadata={"version":VERSION,"applied":False,"reason":"validation_not_passed"}
    if validation.get("valid") is not True:
        return source_path, boxes, metadata
    if validation.get("usable_face_count") not in (1,2) or len(boxes)!=validation.get("usable_face_count"):
        metadata["reason"]="inconsistent_validated_faces"
        return source_path, boxes, metadata
    # Never import another detector when the active pipeline is switched.
    pipeline_name = active_pipeline.__name__.split('.')[-2]
    if pipeline_name not in ("scrfd_2_5g", "scrfd_10g"):
        metadata["reason"]="active_pipeline_has_no_source_rotation"
        return source_path, boxes, metadata
    started=time.perf_counter()
    try:
        orientation = importlib.import_module(f"{active_pipeline.__package__}.source_orientation")
        estimate_image = orientation.estimate_image
        metadata["orientation_pipeline"] = pipeline_name
        image=active_pipeline._read_image(source_path)
        if image is None:raise ValueError("Could not decode validated source")
        estimate=select_source_rotation(estimate_image(image,boxes,active_pipeline._native_detector(4)))
        metadata.update(estimate)
        metadata["source_filename"]=os.path.basename(source_path)
        metadata["original_face_boxes"]=[list(b) for b in boxes]
        with open(source_path,"rb") as file:
            metadata["source_sha256"]=hashlib.sha256(file.read()).hexdigest()
        degrees=estimate["correction_cv_deg"]
        if degrees==0:
            metadata["reason"]="below_rotation_threshold" if estimate["orientation_faces"] else "no_reliable_direction"
            metadata["seconds"]=time.perf_counter()-started
            return source_path, boxes, metadata
        corrected,matrix=rotate(image,degrees)
        # Preserve existing PNG alpha in the same decoded coordinate frame.
        raw=cv2.imdecode(np.fromfile(source_path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.ndim==3 and raw.shape[2]==4:
            if raw.shape[:2] != image.shape[:2] or not np.array_equal(raw[:,:,:3],image):
                raise ValueError("Alpha/source coordinate frames disagree")
            corrected=cv2.warpAffine(raw,matrix,(corrected.shape[1],corrected.shape[0]),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
        transformed=[]
        for x,y,w,h in boxes:
            corners=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])@matrix[:,:2].T+matrix[:,2]
            low,high=corners.min(axis=0),corners.max(axis=0)
            low=np.maximum(low,0);high=np.minimum(high,[corrected.shape[1],corrected.shape[0]])
            if (high<=low).any():raise ValueError("Rotated face box outside output")
            transformed.append([*low.tolist(),*(high-low).tolist()])
        filename="validated_source_rotated.png"
        destination=os.path.join(output_dir,filename)
        ok,encoded=cv2.imencode('.png',corrected,[cv2.IMWRITE_PNG_COMPRESSION,3])
        if not ok:raise ValueError("Could not encode rotated source")
        # Never overwrite a browser upload or a file hard-linked into another task.
        if os.path.abspath(source_path)==os.path.abspath(destination):raise ValueError("Output collides with original")
        temporary=destination+".tmp"
        with open(temporary,"xb") as file:file.write(encoded.tobytes())
        os.replace(temporary,destination)
        metadata.update(applied=True,reason=estimate["orientation_status"],output_filename=filename,
                        output_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
                        matrix=matrix.tolist(),rotated_face_boxes=transformed,
                        output_size=[corrected.shape[1],corrected.shape[0]],seconds=time.perf_counter()-started)
        return destination, transformed, metadata
    except Exception:
        logging.getLogger(__name__).exception("Validated source rotation unavailable; retaining original")
        metadata.update(applied=False,reason="rotation_unavailable",seconds=time.perf_counter()-started)
        return source_path, boxes, metadata
