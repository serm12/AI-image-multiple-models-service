"""Small, shared message catalog for storefront-facing API responses."""

from __future__ import annotations





MESSAGES = {

    "en": {
        "PET_OCCLUDED": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Important facial features are covered. Please use a photo with the pet’s face unobstructed by hands, objects or other animals.",
        "PET_FACE_UNRECOGNIZABLE": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Pet facial features cannot be clearly identified. Please use a photo showing the pet’s face more clearly. A clear side view is also acceptable.",
        "PET_DETECTION_FAILED": "We couldn’t complete pet detection. Please try again in a moment.",

        "FACE_DETECTION_PASSED": "Face detection passed.",

        "PET_DETECTION_PASSED": "Pet detection passed.",

        "PET_NOT_DETECTED": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. No usable pets were detected. Please upload a photo with the pet’s face clearly visible and unobstructed.",

        "PET_COUNT_MISMATCH": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. {pet_count} usable pets were detected. Please choose a photo showing only 1 pet.",

        "PET_WITH_PERSON": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. A person was also detected. Please upload a photo showing only 1 cat or dog.",

        "PET_TOO_SMALL": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Pet details are too small to see clearly. Please upload a closer photo or crop around the pet.",

        "PET_INCOMPLETE": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Important facial features are cut off by the edge of the image. Please use a photo with the pet’s face fully within the frame.",

        "PET_BLURRY": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Pet details are too blurry to see clearly. Please upload a sharper, in-focus photo.",

        "PET_TOO_DARK": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Pet details are too dark to see clearly. Please upload a photo with better lighting.",

        "PET_OVEREXPOSED": "This style requires a photo of exactly 1 cat or dog with a clear face and recognizable features. Pet details are washed out by excessive brightness. Please upload a photo with softer lighting and no strong glare.",

        "IMAGE_READ_ERROR": "Unable to read the image. Please upload it again.",

        "NO_FACE": "No face was detected in the selected image.",

        "FACE_STYLE_REQUIREMENT": "This style requires a clear photo of exactly {expected_people}. {detail}",

        "FACE_STYLE_REQUIREMENT_NO_FACE": "This style requires a clear photo of exactly {expected_people}. No clear human face was detected. Please use a closer photo with exactly {expected_faces} large and visible in the frame.",

        "FACE_STYLE_REQUIREMENT_NO_USABLE_FACE_SINGLE": "This style requires a clear photo of exactly {expected_people}. No usable human face is visible. Please upload a clear, unobstructed photo containing exactly {expected_people}, with the face fully visible.",

        "FACE_STYLE_REQUIREMENT_NO_USABLE_FACE": "This style requires a clear photo of exactly {expected_people}. No usable human face is visible. Please upload a clear, unobstructed photo containing exactly {expected_people}, with both faces fully visible.",

        "FACE_STYLE_REQUIREMENT_COUNT": "This style requires a clear photo of exactly {expected_people}. {face_count} usable human faces were detected. Please upload a photo containing exactly {expected_faces}, with both faces clearly visible and unobstructed.",

        "MULTIPLE_FACES": "Please select a photo with exactly 1 recognizable face. {face_count} recognizable faces were detected.",

        "FACE_COUNT_MISMATCH": "Please select a photo with exactly {expected_face_count} recognizable faces. {face_count} recognizable faces were detected.",

        "FACE_TOO_SMALL": "A face was detected, but it is too small to identify reliably ({face_percent}% of the image). Please use a closer photo or crop around the person.",

        "FACE_INCOMPLETE": "A face was detected, but an important part of it is outside the image. Please use a photo showing the complete face.",

        "FACE_BLURRY": "A face was detected, but it is too blurry to identify reliably. Please use a sharper photo.",

        "FACE_TOO_DARK": "A face was detected, but it is too dark to identify reliably. Please use a better-lit photo.",

        "FACE_OVEREXPOSED": "A face was detected, but it is too bright or washed out to identify reliably. Please use a photo with more facial detail.",

        "FACE_UNRECOGNIZABLE": "A face was detected, but the facial appearance cannot be identified clearly. Please use a photo with visible facial features.",

        "FACE_UNRECOGNIZABLE_PLURAL": "This style requires a clear photo of exactly 2 people. Faces were detected, but their facial appearances cannot be identified clearly. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_TOO_SMALL_DOUBLE": "This style requires a clear photo of exactly 2 people. Faces were detected, but at least 1 face is too small to identify reliably. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_INCOMPLETE_DOUBLE": "This style requires a clear photo of exactly 2 people. Faces were detected, but at least 1 face is incomplete. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_TOO_DARK_DOUBLE": "This style requires a clear photo of exactly 2 people. Faces were detected, but at least 1 face is too dark to identify reliably. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_OVEREXPOSED_DOUBLE": "This style requires a clear photo of exactly 2 people. Faces were detected, but at least 1 face is too bright or washed out to identify reliably. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_NOT_FRONTAL_DOUBLE": "This style requires a clear photo of exactly 2 people. Faces were detected, but at least 1 face is turned too far away to identify the person's appearance. Please upload a clear, unobstructed photo containing exactly 2 people, with both faces fully visible.",

        "FACE_TILTED": "The face is noticeably tilted. Please keep the head upright and look at the camera.",

        "FACE_NOT_FRONTAL": "A face was detected, but it is turned too far away to identify the person's appearance. A slight side angle is acceptable if the facial features remain clear.",

        "FACE_OCCLUDED": "The face is too heavily covered. Please make sure the eyes and eyebrow area are clearly visible.",

        "FACE_DETECTION_FAILED": "We could not complete face detection. Please try again.",

        "GENERATION_REQUEST_INVALID": "The image generation request is invalid. Please try again.",

        "GENERATION_INPUT_REQUIRED": "Please upload a photo before generating the image.",

        "GENERATION_CREATE_FAILED": "Unable to start image generation. Please try again.",

        "GENERATION_FAILED": "Image generation failed. Please try again later.",

        "TASK_NOT_FOUND": "The generation task was not found.",

        "GENERATION_SUBMITTED": "Your request has been submitted and is being processed.",

    },

    "zh-CN": {
        "PET_OCCLUDED": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。重要面部特征被遮挡，请使用宠物面部未被手、物品或其他动物遮挡的照片。",
        "PET_FACE_UNRECOGNIZABLE": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。宠物面部特征无法清楚辨认，请使用面部更清晰的照片；清晰的侧脸也可以。",
        "PET_DETECTION_FAILED": "宠物检测暂未能完成，请稍后重试。",

        "FACE_DETECTION_PASSED": "人脸检测通过。",

        "PET_DETECTION_PASSED": "宠物检测通过。",

        "PET_NOT_DETECTED": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。当前未检测到可用宠物，请上传宠物面部清晰可见、无遮挡的照片。",

        "PET_COUNT_MISMATCH": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。当前检测到 {pet_count} 只可用宠物，请选择仅展示 1 只宠物的照片。",

        "PET_WITH_PERSON": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。当前还检测到人物。请上传只展示单只猫或狗的照片。",

        "PET_TOO_SMALL": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。检测到宠物，但宠物过小，无法可靠辨认。请使用距离更近的照片，或裁剪到宠物周围。",

        "PET_INCOMPLETE": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。重要面部特征被图片边缘截断，请使用宠物面部完整位于画面内的照片。",

        "PET_BLURRY": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。检测到宠物，但画面过于模糊，无法可靠辨认。请使用更清晰的照片。",

        "PET_TOO_DARK": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。检测到宠物，但画面过暗，无法可靠辨认。请使用光线更好的照片。",

        "PET_OVEREXPOSED": "此款式需要恰好包含 1 只面部清晰、关键外观可辨认的猫或狗的照片。检测到宠物，但画面过亮或过曝，无法可靠辨认。请使用细节更清楚的照片。",

        "IMAGE_READ_ERROR": "无法读取图片，请重新上传。",

        "NO_FACE": "所选图片中未检测到人脸。",

        "FACE_STYLE_REQUIREMENT": "此款式需要恰好 {expected_people} 的清晰照片。{detail}",

        "FACE_STYLE_REQUIREMENT_NO_FACE": "此款式需要恰好 {expected_people} 的清晰照片。未检测到清晰人脸，请使用距离更近、让 {expected_faces} 在画面中足够大且清晰可见的照片。",

        "FACE_STYLE_REQUIREMENT_NO_USABLE_FACE_SINGLE": "此款式需要恰好 {expected_people} 的清晰照片。画面中没有可用于生成的真人面部。请上传恰好包含 {expected_people}、人脸完整清晰无遮挡的照片。",

        "FACE_STYLE_REQUIREMENT_NO_USABLE_FACE": "此款式需要恰好 {expected_people} 的清晰照片。画面中没有可用于生成的真人面部。请上传恰好包含 {expected_people}、两张人脸均清晰无遮挡的照片。",

        "FACE_STYLE_REQUIREMENT_COUNT": "此款式需要恰好 {expected_people} 的清晰照片，当前检测到 {face_count} 张可用人脸。请上传恰好包含 {expected_faces}、两张人脸均清晰无遮挡的图片。",

        "MULTIPLE_FACES": "请上传仅包含 1 张可辨认人脸的照片，当前检测到 {face_count} 张可辨认人脸。",

        "FACE_COUNT_MISMATCH": "请上传恰好包含 {expected_face_count} 张可辨认人脸的照片，当前检测到 {face_count} 张可辨认人脸。",

        "FACE_TOO_SMALL": "检测到人脸，但人脸过小，无法可靠辨认（仅占图片 {face_percent}%）。请使用距离更近的照片，或裁剪到人物周围。",

        "FACE_INCOMPLETE": "检测到人脸，但重要面部区域超出图片范围。请使用面部完整的照片。",

        "FACE_BLURRY": "检测到人脸，但画面过于模糊，无法可靠辨认。请使用更清晰的照片。",

        "FACE_TOO_DARK": "检测到人脸，但面部过暗，无法可靠辨认。请使用光线更好的照片。",

        "FACE_OVEREXPOSED": "检测到人脸，但面部过亮或过曝，无法可靠辨认。请使用面部细节更清楚的照片。",

        "FACE_UNRECOGNIZABLE": "检测到人脸，但面部特征无法清楚辨认。请使用五官可见的照片。",

        "FACE_UNRECOGNIZABLE_PLURAL": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但人物面部特征均无法清楚辨认。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_TOO_SMALL_DOUBLE": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但至少一张人脸过小，无法可靠辨认。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_INCOMPLETE_DOUBLE": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但至少一张人脸不完整。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_TOO_DARK_DOUBLE": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但至少一张人脸过暗，无法可靠辨认。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_OVEREXPOSED_DOUBLE": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但至少一张人脸过亮或过曝，无法可靠辨认。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_NOT_FRONTAL_DOUBLE": "此款式需要恰好 2 位真人的清晰照片。检测到人脸，但至少一张人脸侧转角度过大，无法辨认人物样貌。请上传恰好包含 2 位真人、两张人脸均清晰无遮挡的照片。",

        "FACE_TILTED": "检测到头部倾斜较明显，请保持头部端正并平视镜头后重拍。",

        "FACE_NOT_FRONTAL": "检测到人脸，但侧转角度过大，无法辨认人物样貌。只要五官仍然清楚，轻微侧脸可以通过。",

        "FACE_OCCLUDED": "检测到面部遮挡过多，请确保双眼和眉眼区域清晰可见后重拍。",

        "FACE_DETECTION_FAILED": "暂时无法完成人脸检测，请重试。",

        "GENERATION_REQUEST_INVALID": "图片生成请求无效，请重试。",

        "GENERATION_INPUT_REQUIRED": "请先上传照片，再生成图片。",

        "GENERATION_CREATE_FAILED": "无法启动图片生成，请重试。",

        "GENERATION_FAILED": "图片生成失败，请稍后重试。",

        "TASK_NOT_FOUND": "未找到图片生成任务。",

        "GENERATION_SUBMITTED": "请求已提交，正在处理中。",

    },

}





def normalize_locale(locale: str | None) -> str:

    value = (locale or "").strip().replace("_", "-").lower()

    return "zh-CN" if value.startswith("zh") else "en"





def get_message(code: str, locale: str | None, **params) -> str:

    language = normalize_locale(locale)

    template = MESSAGES[language].get(code) or MESSAGES["en"].get(code) or code

    return template.format(**params)





# Presentation-only catalog: do not change detector evidence or shared/double

# templates when refining single-face storefront copy.

SINGLE_FACE_REQUIREMENT = {

    "en": "This style requires a photo with exactly 1 clear, fully visible human face.",

    "zh-CN": "这种风格需要一张仅有一个清晰、完整可见的人脸的照片。",

}

SINGLE_FACE_DETAILS = {

    "en": {

        "NO_FACE": "No clear human faces meeting the requirements were detected. Please use a clearer, unobstructed photo.",

        "MULTIPLE_FACES": "{face_count} usable human faces were detected. Please choose a photo with only 1 clear face.",

        "FACE_TOO_SMALL": "Facial details are too small to see clearly. Please crop closer around the person or upload a closer photo.",

        "FACE_INCOMPLETE": "Facial features are cut off by the edge of the image. Please keep the entire face within the frame.",

        "FACE_TOO_DARK": "Facial details are too dark to see clearly. Please upload a photo with better facial lighting.",

        "FACE_UNRECOGNIZABLE": "Facial features are not clear enough to distinguish. Please upload a photo with clearly visible facial details.",

        "FACE_OCCLUDED": "Facial features are too heavily covered. Please use a photo with key facial features clearly visible and unobstructed.",

        "FACE_BLURRY": "Facial details are too blurry to see clearly. Please upload a sharper, in-focus photo.",

        "FACE_OVEREXPOSED": "Facial details are washed out by excessive brightness. Please use a photo with softer lighting and no strong glare.",

        "FACE_NOT_FRONTAL": "Facial features are not clear enough due to the viewing angle. Please use a view with clearer facial features. A clear side view is acceptable.",

        "FACE_TILTED": "Head tilt makes facial features difficult to see clearly. Please use a photo with the head upright.",

    },

    "zh-CN": {

        "NO_FACE": "未能检测到符合要求的清晰人脸。请使用面部更清晰、无遮挡的照片。",

        "MULTIPLE_FACES": "检测到 {face_count} 张可用人脸。请选择仅有 1 张清晰人脸的照片。",

        "FACE_TOO_SMALL": "面部细节过小，无法看清。请裁剪到人物周围，或上传距离更近的照片。",

        "FACE_INCOMPLETE": "部分面部被图片边缘截断。请确保完整面部位于照片范围内。",

        "FACE_TOO_DARK": "人脸过暗，无法看清面部细节。请上传面部光线更充足的照片。",

        "FACE_UNRECOGNIZABLE": "检测到人脸，但五官不够清晰。请上传面部细节清楚可见的照片。",

        "FACE_OCCLUDED": "面部特征遮挡过多。请使用关键面部特征清晰、无遮挡的照片。",

        "FACE_BLURRY": "人脸过于模糊，无法看清面部细节。请上传对焦准确、更清晰的照片。",

        "FACE_OVEREXPOSED": "人脸过亮或过曝，面部细节不清楚。请使用光线更柔和、面部没有强光反射的照片。",

        "FACE_NOT_FRONTAL": "由于拍摄角度，五官无法清楚辨认。请使用面部更清晰可见的照片；五官清楚的侧脸也可以。",

        "FACE_TILTED": "头部向一侧倾斜过大。请使用头部端正的照片。",

    },

}





def get_single_face_error_message(code: str, locale: str | None, *, face_count=0) -> str:

    """Format a validation error only; technical failures keep their own copy."""

    if code in {"IMAGE_READ_ERROR", "FACE_DETECTION_FAILED"}:

        return get_message(code, locale)

    language = normalize_locale(locale)

    detail_code = {"NON_HUMAN_FACE": "NO_FACE", "FACE_COUNT_MISMATCH": "MULTIPLE_FACES"}.get(code, code)

    details = SINGLE_FACE_DETAILS[language]

    detail = details.get(detail_code, details["NO_FACE"]).format(face_count=face_count)

    if language == "en" and detail_code == "MULTIPLE_FACES" and face_count == 1:

        detail = detail.replace("1 usable human faces were", "1 usable human face was")

    separator = " " if language == "en" else ""

    return SINGLE_FACE_REQUIREMENT[language] + separator + detail





# Double-style presentation is isolated from the detector and single-style copy.

DOUBLE_FACE_REQUIREMENT = {

    "en": "This style requires a photo with exactly 2 clear, fully visible human faces.",

    "zh-CN": "这种风格需要一张恰好包含两张清晰、完整可见的人脸的照片。",

}

DOUBLE_FACE_DETAILS = {

    "en": {

        "NO_FACE": "No clear human faces meeting the requirements were detected. Please use a clearer, unobstructed photo.",

        "ONE_USABLE_FACE": "Only 1 usable human face was detected. Please use a photo with 2 clear faces.",

        "FACE_COUNT_MISMATCH": "{face_count} usable human faces were detected. Please choose a photo with only 2 clear faces.",

        "FACE_TOO_SMALL": "Facial details are too small to see clearly. Please upload a closer photo with both faces clearly visible.",

        "FACE_INCOMPLETE": "Facial features are cut off by the edge of the image. Please keep both faces entirely within the frame.",

        "FACE_TOO_DARK": "Facial details are too dark to see clearly. Please upload a photo with better lighting on both faces.",

        "FACE_UNRECOGNIZABLE": "Facial features are not clear enough to distinguish. Please upload a photo with clearly visible facial details on both people.",

        "FACE_OCCLUDED": "Facial features are too heavily covered. Please use a photo with key features on both faces clearly visible and unobstructed.",

        "FACE_BLURRY": "Facial details are too blurry to see clearly. Please upload a sharper photo with both faces in focus.",

        "FACE_OVEREXPOSED": "Facial details are washed out by excessive brightness. Please upload a photo with softer lighting and no strong glare on either face.",

        "FACE_NOT_FRONTAL": "Facial features are not clear enough due to the viewing angle. Please use a clearer view of both faces; side views are acceptable if the features are clearly visible.",

        "FACE_TILTED": "Head tilt makes facial features difficult to see clearly. Please upload a photo with both heads upright and both faces clearly visible.",

    },

    "zh-CN": {

        "NO_FACE": "未能检测到符合要求的清晰人脸。请使用面部更清晰、无遮挡的照片。",

        "ONE_USABLE_FACE": "仅检测到 1 张可用人脸。请使用包含 2 张清晰人脸的照片。",

        "FACE_COUNT_MISMATCH": "检测到 {face_count} 张可用人脸。请选择仅有 2 张清晰人脸的照片。",

        "FACE_TOO_SMALL": "面部细节过小，无法看清。请上传距离更近、两张人脸均清晰可见的照片。",

        "FACE_INCOMPLETE": "部分面部被图片边缘截断。请确保两张完整面部均位于照片范围内。",

        "FACE_TOO_DARK": "可见的面部细节过暗，无法看清。请上传两张人脸光线都更充足的照片。",

        "FACE_UNRECOGNIZABLE": "检测到面部特征，但不够清晰。请上传两人的面部细节均清楚可见的照片。",

        "FACE_OCCLUDED": "面部特征遮挡过多。请使用两张脸的关键面部特征均清晰、无遮挡的照片。",

        "FACE_BLURRY": "面部细节过于模糊，无法看清。请上传两张人脸均对焦准确、更清晰的照片。",

        "FACE_OVEREXPOSED": "过强的亮光使面部细节变白、难以看清。请上传光线更柔和、两张人脸均无强光反射的照片。",

        "FACE_NOT_FRONTAL": "由于拍摄角度，五官无法清楚辨认。请使用两张脸均更清晰可见的照片；五官清楚的侧脸也可以。",

        "FACE_TILTED": "头部倾斜使面部特征难以看清。请上传两人头部端正、两张人脸均清晰可见的照片。",

    },

}





def get_double_face_error_message(code: str, locale: str | None, *, face_count=0) -> str:

    """Format the already-decided outcome; never infer or alter usable counts."""

    if code in {"IMAGE_READ_ERROR", "FACE_DETECTION_FAILED"}:

        return get_message(code, locale)

    language = normalize_locale(locale)

    detail_code = {"NON_HUMAN_FACE": "NO_FACE", "MULTIPLE_FACES": "FACE_COUNT_MISMATCH"}.get(code, code)

    if detail_code == "FACE_COUNT_MISMATCH":

        if face_count == 1:

            detail_code = "ONE_USABLE_FACE"

        elif face_count == 0:

            detail_code = "NO_FACE"

    details = DOUBLE_FACE_DETAILS[language]

    detail = details.get(detail_code, details["NO_FACE"]).format(face_count=face_count)

    separator = " " if language == "en" else ""

    return DOUBLE_FACE_REQUIREMENT[language] + separator + detail
