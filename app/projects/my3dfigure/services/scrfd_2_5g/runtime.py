"""Request-local SCRFD evidence reuse; cached arrays never cross requests."""
from __future__ import annotations

import hashlib
import copy
import sys
from collections import OrderedDict

import numpy as np


class RequestDetector:
    def __init__(self, detector):
        self.detector = detector
        self.source_frame = None
        self.source_frame_signature = None
        self.source_frame_digest = None
        self.cache = {}
        self.evidence_cache = {}
        self.local_observation_cache = OrderedDict()
        self.local_observation_cache_bytes = 0
        self.local_observation_cache_limit = 2 * 1024 * 1024
        self.local_observation_cache_entries = 128

    @staticmethod
    def _frame_signature(image):
        return image.dtype.str, image.shape, image.strides

    def register_source(self, image):
        # Immutable bytes own the pixels: writable caller aliases cannot mutate
        # this frame, nor can setflags(write=True) re-enable writes to its base.
        contiguous = np.ascontiguousarray(image)
        frozen = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
        self.source_frame = frozen
        self.source_frame_signature = self._frame_signature(frozen)
        self.source_frame_digest = hashlib.blake2b(frozen.data, digest_size=16).digest()
        return frozen

    def _pixels_digest(self, image):
        if (image is self.source_frame and not image.flags.writeable
                and self._frame_signature(image) == self.source_frame_signature):
            return self.source_frame_digest
        return hashlib.blake2b(np.ascontiguousarray(image).data, digest_size=16).digest()

    @staticmethod
    def _memo_size(value):
        # Conservative payload estimate; includes nested Python containers and
        # ndarray object/storage. Fixed entry cap also bounds dictionary overhead.
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            return total + sum(RequestDetector._memo_size(k) + RequestDetector._memo_size(v) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return total + sum(RequestDetector._memo_size(v) for v in value)
        if isinstance(value, np.ndarray):
            return total + value.nbytes
        return total

    def local_observation_get(self, key):
        cached = self.local_observation_cache.get(key)
        if cached is None:
            return None
        self.local_observation_cache.move_to_end(key)
        return copy.deepcopy(cached[0])

    def local_observation_put(self, key, value):
        size = self._memo_size(key) + self._memo_size(value)
        if size > self.local_observation_cache_limit:
            return
        previous = self.local_observation_cache.pop(key, None)
        if previous is not None:
            self.local_observation_cache_bytes -= previous[1]
        while self.local_observation_cache and (
                self.local_observation_cache_bytes + size > self.local_observation_cache_limit
                or len(self.local_observation_cache) >= self.local_observation_cache_entries):
            _, evicted = self.local_observation_cache.popitem(last=False)
            self.local_observation_cache_bytes -= evicted[1]
        self.local_observation_cache[key] = (copy.deepcopy(value), size)
        self.local_observation_cache_bytes += size

    def evidence_key(self, namespace, image, geometry):
        """Identify repeated evidence only within this image analysis request."""
        pixels = self._pixels_digest(image)
        return namespace, image.shape, pixels, geometry

    def detect(self, image, input_size=None, max_num=0, metric='default', det_thresh=None):
        threshold = self.detector.det_thresh if det_thresh is None else float(det_thresh)
        if max_num:
            return self.detector.detect(image,input_size=input_size,max_num=max_num,metric=metric,det_thresh=det_thresh)
        key=(image.shape,self._pixels_digest(image),str(input_size))
        if key not in self.cache:
            self.cache[key]=self.detector.detect(image,input_size=input_size,det_thresh=min(.05,threshold))
        boxes,points=self.cache[key]
        keep=boxes[:,4]>=threshold
        return boxes[keep].copy(),points[keep].copy()
