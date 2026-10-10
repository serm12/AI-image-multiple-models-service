"""Request-local SCRFD evidence reuse; cached arrays never cross requests."""
from __future__ import annotations

import hashlib
import copy
import sys
import zlib
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
        self.native_forward_cache = OrderedDict()
        self.native_forward_cache_bytes = 0
        self.native_forward_cache_limit = 32 * 1024 * 1024
        self.native_forward_cache_entries = 64
        self.native_forward_cache_hits = 0
        self.native_forward_cache_misses = 0

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

    @staticmethod
    def _native_forward_key(args, kwargs):
        # The installed SCRFD contract is forward(img, threshold). Unknown
        # parameters bypass reuse and retain the captured original behavior.
        if len(args) > 2 or set(kwargs) - {'img', 'threshold'}:
            return None
        values = []
        for index, name in enumerate(('img', 'threshold')):
            if index < len(args):
                if name in kwargs:
                    return None
                values.append(args[index])
            elif name in kwargs:
                values.append(kwargs[name])
            else:
                return None
        image, threshold = values
        if (not isinstance(image, np.ndarray) or image.dtype.hasobject
                or image.dtype.fields is not None):
            return None
        if not (type(threshold) in (bool, int, float) or isinstance(threshold, np.generic)):
            return None
        scalar = np.asarray(threshold)
        if scalar.shape or scalar.dtype.kind not in 'biuf' or scalar.dtype.hasobject:
            return None
        pixels = np.ascontiguousarray(image)
        pixel_bytes = pixels.tobytes()
        checksum = zlib.crc32(pixel_bytes)
        threshold_key = (type(threshold).__module__, type(threshold).__qualname__,
                         scalar.dtype.str, scalar.tobytes())
        # CRC only selects a bucket; reuse also requires byte-for-byte pixels.
        # Immutable bytes are counted inside the existing bounded memo budget.
        key = image.dtype.str, tuple(image.shape), checksum, threshold_key
        return key, pixel_bytes

    @staticmethod
    def _copy_native_forward_outputs(value):
        if type(value) is np.ndarray and not value.dtype.hasobject:
            return value.copy()
        if isinstance(value, list):
            return [RequestDetector._copy_native_forward_outputs(item) for item in value]
        if isinstance(value, tuple):
            return tuple(RequestDetector._copy_native_forward_outputs(item) for item in value)
        if value is None or type(value) in (bool, int, float, str, bytes):
            return value
        raise TypeError('Unknown forward result is not eligible for optional memo')

    def _native_forward_put(self, key, outputs, pixel_bytes):
        size = self._memo_size(key) + self._memo_size(outputs) + self._memo_size(pixel_bytes)
        if size > self.native_forward_cache_limit:
            return
        stored = self._copy_native_forward_outputs(outputs)
        previous = self.native_forward_cache.pop(key, None)
        if previous is not None:
            self.native_forward_cache_bytes -= previous[2]
        while self.native_forward_cache and (
                self.native_forward_cache_bytes + size > self.native_forward_cache_limit
                or len(self.native_forward_cache) >= self.native_forward_cache_entries):
            _, evicted = self.native_forward_cache.popitem(last=False)
            self.native_forward_cache_bytes -= evicted[2]
        self.native_forward_cache[key] = (pixel_bytes, stored, size)
        self.native_forward_cache_bytes += size

    def _backend_detect(self, *args, **kwargs):
        # The 10G detector is thread-local. Do not mutate session.run, its
        # resolution pool, providers, model, thresholds or postprocessing.
        detector = self.detector
        namespace = getattr(detector, '__dict__', None)
        original = getattr(detector, 'forward', None)
        if not isinstance(namespace, dict) or not callable(original):
            return detector.detect(*args, **kwargs)
        had_instance_forward = 'forward' in namespace
        previous_instance_forward = namespace.get('forward')
        def exact_forward(*forward_args, **forward_kwargs):
            try:
                entry = self._native_forward_key(forward_args, forward_kwargs)
            except (TypeError, ValueError, BufferError, MemoryError):
                entry = None
            if entry is None:
                return original(*forward_args, **forward_kwargs)
            key, pixel_bytes = entry
            cached = self.native_forward_cache.get(key)
            if cached is not None and cached[0] == pixel_bytes:
                try:
                    result = self._copy_native_forward_outputs(cached[1])
                except (TypeError, MemoryError):
                    result = None
                if result is not None:
                    self.native_forward_cache.move_to_end(key)
                    self.native_forward_cache_hits += 1
                    return result
            self.native_forward_cache_misses += 1
            result = original(*forward_args, **forward_kwargs)
            try:
                self._native_forward_put(key, result, pixel_bytes)
            except (TypeError, MemoryError):
                pass  # Optional memo allocation must not fail detection.
            return result
        def restore():
            if had_instance_forward:
                detector.forward = previous_instance_forward
            else:
                del detector.forward
        try:
            detector.forward = exact_forward
        except (AttributeError, TypeError):
            if namespace.get('forward') is exact_forward:
                restore()
            return detector.detect(*args, **kwargs)
        try:
            return detector.detect(*args, **kwargs)
        finally:
            restore()

    def detect_generated_view(self, image, input_size=None, det_thresh=None):
        """Avoid hashing large one-use local views; retain identical inference.

        local_observations already memoizes each source/geometry/angle/factor.
        This removes only the optional full-image memo for large generated
        views. Smaller views retain its cheap reuse; no detection is skipped.
        """
        if image.nbytes <= 4 * 1024 * 1024:
            return self.detect(image, input_size=input_size, det_thresh=det_thresh)
        threshold = self.detector.det_thresh if det_thresh is None else float(det_thresh)
        boxes, points = self._backend_detect(
            image, input_size=input_size, det_thresh=min(.05, threshold))
        keep = boxes[:, 4] >= threshold
        return boxes[keep].copy(), points[keep].copy()

    def detect(self, image, input_size=None, max_num=0, metric='default', det_thresh=None):
        threshold = self.detector.det_thresh if det_thresh is None else float(det_thresh)
        if max_num:
            return self._backend_detect(image,input_size=input_size,max_num=max_num,metric=metric,det_thresh=det_thresh)
        key=(image.shape,self._pixels_digest(image),str(input_size))
        if key not in self.cache:
            self.cache[key]=self._backend_detect(image,input_size=input_size,det_thresh=min(.05,threshold))
        boxes,points=self.cache[key]
        keep=boxes[:,4]>=threshold
        return boxes[keep].copy(),points[keep].copy()
