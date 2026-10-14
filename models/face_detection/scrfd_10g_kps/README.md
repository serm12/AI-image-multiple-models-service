# SCRFD-10G-KPS model

- Detector: InsightFace official `buffalo_l` model pack, `det_10g.onnx`.
- Source: https://github.com/deepinsight/insightface/releases/tag/model-zoo
- Model listing: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- SHA-256: `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`.
- Runtime: InsightFace 2.0, CPU ONNX Runtime, official defaults: auto 128×128
  and 640×640 detection, confidence 0.5, NMS 0.4, no face count limit.
- The model has five facial keypoints. The newer Raccoon 10G detector is the
  `wo` variant and is not the requested KPS model.

InsightFace library code is MIT licensed. Its public pretrained weights are
licensed for non-commercial research only; commercial use requires a separate
model license from InsightFace. This local integration has not been deployed.

For a local backend virtual environment, install `requirements.txt`, then
run `python -m pip install --no-deps insightface==2.0` to retain headless OpenCV.
