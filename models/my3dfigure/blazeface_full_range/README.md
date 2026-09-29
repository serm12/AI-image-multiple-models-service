# BlazeFace Full Range model

- Source: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite
- Official model listing: https://developers.google.com/edge/mediapipe/solutions/vision/face_detector#blazeface_full-range
- Checked against Google's `latest` URL on 2026-09-28.
- SHA-256: `3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b`.
- Variant: regular Full Range, float16; not Full Range Sparse.

The local file is pinned by hash so a future change to Google's `latest` URL cannot silently change detection results.

For a local backend virtual environment, install `requirements.txt` first, then
run `python -m pip install --no-deps mediapipe==1.0.1`. Docker performs the same
two steps. The separate installation keeps MediaPipe's declared
`opencv-contrib-python` dependency from replacing the YuNet environment's
`opencv-python-headless` package.
