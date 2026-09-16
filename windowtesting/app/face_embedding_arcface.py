from pathlib import Path

import numpy as np

from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop


class ArcFaceEmbedder:
    def __init__(self, model_path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "找不到ArcFace模型："
                f"{self.model_path.resolve()}"
            )

        print(
            "載入ArcFace模型："
            f"{self.model_path.resolve()}"
        )

        self.model = get_model(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.model.prepare(ctx_id=-1)

        print("ArcFace模型載入完成")

    def extract(self, frame, landmarks):
        if landmarks is None or len(landmarks) != 5:
            raise ValueError(
                "ArcFace需要SCRFD五點Landmark"
            )

        landmarks_array = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        # 使用五點Landmark對齊成人臉112×112
        aligned_face = norm_crop(
            frame,
            landmark=landmarks_array,
            image_size=112,
        )

        embedding = self.model.get_feat(
            aligned_face
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        # L2正規化，方便後續計算Cosine Similarity
        embedding_norm = np.linalg.norm(
            embedding
        )

        if embedding_norm > 0:
            embedding = (
                embedding / embedding_norm
            )

        return aligned_face, embedding