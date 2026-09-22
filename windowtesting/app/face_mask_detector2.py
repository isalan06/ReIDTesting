import cv2
import numpy as np


class FaceMaskHeuristicDetector:
    """
    無專用AI模型的PoC口罩／下半臉遮擋判斷器。

    它比較上半臉與下半臉的膚色比例、紋理及色彩變化；
    結果只能用於測試流程，不可視為正式口罩分類結果。
    """

    def __init__(self, mask_threshold=0.55, min_face_size=80):
        self.mask_threshold = float(mask_threshold)
        self.min_face_size = int(min_face_size)

    @staticmethod
    def _clip01(value):
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _skin_ratio(bgr_roi):
        ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        skin = (
            (cr >= 133)
            & (cr <= 173)
            & (cb >= 77)
            & (cb <= 127)
        )
        return float(np.mean(skin))

    @staticmethod
    def _texture_score(gray_roi):
        return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

    def detect(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return self._unknown("EMPTY")

        height, width = face_crop.shape[:2]
        if width < self.min_face_size or height < self.min_face_size:
            return self._unknown("TOO_SMALL")

        normalized = cv2.resize(face_crop, (112, 112))

        # 避開頭髮、耳朵與背景，只分析中央臉部。
        upper = normalized[20:55, 18:94]
        lower = normalized[58:106, 18:94]

        upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)

        upper_skin = self._skin_ratio(upper)
        lower_skin = self._skin_ratio(lower)

        upper_texture = self._texture_score(upper_gray)
        lower_texture = self._texture_score(lower_gray)

        upper_color_std = float(np.mean(np.std(upper.astype(np.float32), axis=(0, 1))))
        lower_color_std = float(np.mean(np.std(lower.astype(np.float32), axis=(0, 1))))

        skin_drop = self._clip01(
            (upper_skin - lower_skin) / max(upper_skin, 0.10)
        )
        texture_drop = self._clip01(
            (upper_texture - lower_texture) / max(upper_texture, 1.0)
        )
        lower_uniformity = self._clip01(
            1.0 - lower_color_std / max(upper_color_std, 1.0)
        )

        mask_score = (
            skin_drop * 0.55
            + texture_drop * 0.25
            + lower_uniformity * 0.20
        )

        is_masked = mask_score >= self.mask_threshold

        return {
            "status": "MASK" if is_masked else "NO_MASK",
            "is_masked": is_masked,
            "score": float(mask_score),
            "skin_drop": skin_drop,
            "texture_drop": texture_drop,
            "lower_uniformity": lower_uniformity,
            "reason": None,
        }

    @staticmethod
    def _unknown(reason):
        return {
            "status": "UNKNOWN",
            "is_masked": None,
            "score": 0.0,
            "skin_drop": 0.0,
            "texture_drop": 0.0,
            "lower_uniformity": 0.0,
            "reason": reason,
        }
