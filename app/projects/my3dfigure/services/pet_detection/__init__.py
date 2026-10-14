"""My3dFigure pet pipeline selector: comment one import, uncomment the other.

Each pipeline is independent; an error never falls back to another detector.
"""
from .yolox_rtmpose import pet_detection as active_pipeline  # Existing YOLOX + RTMPose
# from .keypoint_rcnn import pet_detection as active_pipeline  # Keypoint R-CNN

contains_single_pet = active_pipeline.contains_single_pet
__all__ = ["contains_single_pet"]
