"""My3dFigure human-face pipeline selector.

Switch models by commenting the active import and uncommenting the other.
Neither pipeline imports its counterpart or uses it as a fallback.
"""

# from .yunet import face_detection as active_pipeline
# from .blazeface_full import face_detection as active_pipeline
from .scrfd_10g import face_detection as active_pipeline

analyze_human_faces = active_pipeline.analyze_human_faces
evaluate_human_face_analysis = active_pipeline.evaluate_human_face_analysis
contains_human = active_pipeline.contains_human
get_usable_face_reference_boxes = active_pipeline.get_usable_face_reference_boxes

__all__ = [
    "analyze_human_faces",
    "evaluate_human_face_analysis",
    "contains_human",
    "get_usable_face_reference_boxes",
]
