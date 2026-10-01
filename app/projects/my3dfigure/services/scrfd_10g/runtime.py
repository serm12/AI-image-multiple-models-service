"""Request-local SCRFD evidence reuse; cached arrays never cross requests."""
from __future__ import annotations

import hashlib

import numpy as np


class RequestDetector:
    def __init__(self, detector):
        self.detector = detector
        self.cache = {}

    def detect(self, image, input_size=None, max_num=0, metric='default', det_thresh=None):
        threshold = self.detector.det_thresh if det_thresh is None else float(det_thresh)
        if max_num:
            return self.detector.detect(image,input_size=input_size,max_num=max_num,metric=metric,det_thresh=det_thresh)
        key=(image.shape,hashlib.blake2b(np.ascontiguousarray(image).data,digest_size=16).digest(),str(input_size))
        if key not in self.cache:
            self.cache[key]=self.detector.detect(image,input_size=input_size,det_thresh=min(.05,threshold))
        boxes,points=self.cache[key]
        keep=boxes[:,4]>=threshold
        return boxes[keep].copy(),points[keep].copy()
