"""Exact FP32 decoding for selected anchors; network and NMS stay unchanged."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType

import numpy as np

from insightface.model_zoo.scrfd import SCRFD, distance2bbox, distance2kps

_SUPPORTED_SOURCE_SHA256 = 'c06275fe207bd02b08b462cfb318054787058cd0005b2eac9004afb0e1943c64'


def install(detector):
    # Unknown future dependency implementations retain their own forward.
    if (type(detector) is not SCRFD or not detector.use_kps
            or detector.fmc != 3 or detector._num_anchors != 2
            or detector._feat_stride_fpn != [8, 16, 32]):
        return False
    source = inspect.getsourcefile(SCRFD)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != _SUPPORTED_SOURCE_SHA256:
        return False
    detector.forward = MethodType(_forward, detector)
    return True



def _prepare_blob(self, img, input_size):
    if (type(img) is np.ndarray and img.dtype == np.uint8 and img.size
            and img.shape == (input_size[1], input_size[0], 3)
            and type(self.input_mean) in (int, float) and self.input_mean == 127.5
            and type(self.input_std) in (int, float) and self.input_std == 128.0
            and np.dtype(getattr(self, 'input_dtype', np.float32)) == np.dtype(np.float32)):
        # All byte values, subtraction by 127.5 and multiplication by 1/128
        # are exactly representable in float32; preserve native RGB/NCHW.
        value = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None]
        value -= np.float32(127.5)
        value *= np.float32(1 / 128)
        return value
    return self._prepare_input_blob(img, input_size)


def _forward(self, img, threshold):
    input_size = tuple(img.shape[0:2][::-1])
    blob = _prepare_blob(self, img, input_size)
    session_for_size = getattr(self, "_my3dfigure_session_for_size", self._session_for_input_size)
    session = session_for_size(input_size)
    net_outs = session.run(self.output_names, {self.input_name: blob})
    scores_list, bboxes_list, kpss_list = [], [], []
    input_height, input_width = blob.shape[2:4]
    fmc = self.fmc
    for idx, stride in enumerate(self._feat_stride_fpn):
        scores = net_outs[idx][0] if self.batched else net_outs[idx]
        bbox_preds = net_outs[idx + fmc][0] if self.batched else net_outs[idx + fmc]
        height, width = input_height // stride, input_width // stride
        key = (height, width, stride)
        if key in self.center_cache:
            anchor_centers = self.center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
            if len(self.center_cache) < 100:
                self.center_cache[key] = anchor_centers
        pos_inds = np.where(scores >= threshold)[0]
        # Same selected indices, arithmetic, dtype and order as native forward.
        # Background anchors never reach NMS, quality or the public result.
        selected_centers = anchor_centers[pos_inds]
        bboxes = distance2bbox(selected_centers, bbox_preds[pos_inds] * stride)
        scores_list.append(scores[pos_inds])
        bboxes_list.append(bboxes)
        if self.use_kps:
            kps_preds = net_outs[idx + fmc * 2][0] if self.batched else net_outs[idx + fmc * 2]
            kpss = distance2kps(selected_centers, kps_preds[pos_inds] * stride)
            kpss_list.append(kpss.reshape((kpss.shape[0], 5, 2)))
    return scores_list, bboxes_list, kpss_list
