import os
from functools import lru_cache

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency guard
    cv2 = None


MIN_FACE_SIZE = (60, 60)
MIN_FACE_RATIO = 0.02
MAX_EYE_Y_DIFF_RATIO = 0.08


@lru_cache(maxsize=4)
def _load_cascade(filename: str):
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python-headless，无法执行人脸检测。")

    cascade_path = os.path.join(cv2.data.haarcascades, filename)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"无法加载 OpenCV 分类器: {filename}")
    return cascade


def contains_human(image_path: str) -> dict:
    """
    使用 OpenCV 检查图片中是否包含清晰且足够大的正面人脸。
    返回 dict，包含 valid / message，检测通过时附带 face 和 img_size。
    """
    try:
        face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
        eye_cascade = _load_cascade("haarcascade_eye.xml")

        image = cv2.imread(image_path)
        if image is None:
            return {"valid": False, "message": "无法读取图片，请重新上传。"}

        img_height, img_width = image.shape[:2]
        img_area = img_width * img_height
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
        )

        if len(faces) == 0:
            return {"valid": False, "message": "未能检测到人脸，请上传包含清晰人脸的正面照片。"}

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_ratio = (w * h) / float(max(img_area, 1))

        if face_ratio < MIN_FACE_RATIO:
            return {
                "valid": False,
                "message": f"检测到的人脸太小（仅占图片 {face_ratio * 100:.0f}%），请上传正面半身照或更近的照片，避免拍摄全身远景。",
            }

        face_roi_gray = gray_image[y:y + h, x:x + w]
        eye_search_h = max(int(h * 0.65), 1)
        eye_roi_gray = face_roi_gray[:eye_search_h, :]

        eyes = eye_cascade.detectMultiScale(
            eye_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(int(w * 0.12), 18), max(int(h * 0.08), 14)),
        )
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        if len(eyes) < 2:
            return {
                "valid": False,
                "message": "检测到人脸关键区域不完整（可能存在头发遮挡、侧脸或闭眼）。请露出双眼并保持正面拍摄。",
            }

        eye_centers = sorted(
            [(ex + ew / 2.0, ey + eh / 2.0) for ex, ey, ew, eh in eyes],
            key=lambda p: p[0],
        )
        eyes = sorted(eyes, key=lambda e: e[0])

        eye_y_diff_ratio = abs(eye_centers[0][1] - eye_centers[1][1]) / float(max(h, 1))
        if eye_y_diff_ratio > MAX_EYE_Y_DIFF_RATIO:
            return {
                "valid": False,
                "message": "检测到头部倾斜较明显，请保持头部端正并平视镜头后重拍。",
            }

        left_eye, right_eye = eyes
        eye_distance_ratio = abs(eye_centers[1][0] - eye_centers[0][0]) / float(max(w, 1))
        eye_mid_x_ratio = ((eye_centers[0][0] + eye_centers[1][0]) / 2.0) / float(max(w, 1))
        eye_size_ratio = (
            max(left_eye[2] * left_eye[3], right_eye[2] * right_eye[3]) /
            max(min(left_eye[2] * left_eye[3], right_eye[2] * right_eye[3]), 1)
        )

        if (
            eye_distance_ratio < 0.28
            or eye_distance_ratio > 0.60
            or not (0.38 <= eye_mid_x_ratio <= 0.62)
            or eye_size_ratio > 1.9
        ):
            return {
                "valid": False,
                "message": "检测到非正面人像（侧脸或转头幅度较大），请正对镜头平视拍摄。",
            }

        left_eye_count = sum(1 for cx, _ in eye_centers if cx < w * 0.45)
        right_eye_count = sum(1 for cx, _ in eye_centers if cx > w * 0.55)
        if left_eye_count == 0 or right_eye_count == 0:
            return {
                "valid": False,
                "message": "检测到头发遮挡面部过多，请确保双眼和眉眼区域清晰可见后重拍。",
            }

        return {
            "valid": True,
            "message": "人脸检测通过",
            "face": (int(x), int(y), int(w), int(h)),
            "img_size": (int(img_width), int(img_height)),
        }
    except Exception as exc:
        return {"valid": False, "message": f"人脸检测失败: {exc}"}
