from insightface.app import FaceAnalysis


class InsightFaceDetector:
    def __init__(
        self,
        model_name="buffalo_l",
        detection_size=(640, 640),
        detection_threshold=0.5,
    ):
        self.model_name = model_name
        self.detection_size = detection_size
        self.detection_threshold = detection_threshold

        print(
            f"載入InsightFace偵測模型："
            f"{self.model_name}"
        )

        self.app = FaceAnalysis(
            name=self.model_name,

            # 關鍵：只載入detection
            allowed_modules=["detection"],

            providers=[
                "CPUExecutionProvider"
            ],
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=self.detection_size,
            det_thresh=self.detection_threshold,
        )

        print("InsightFace偵測模型載入完成")

    def detect(self, frame):
        faces = self.app.get(frame)

        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()

            landmarks = None

            if face.kps is not None:
                landmarks = (
                    face.kps
                    .astype(float)
                    .tolist()
                )

            results.append(
                {
                    "bbox": bbox,
                    "score": float(
                        face.det_score
                    ),
                    "landmarks": landmarks,
                }
            )

        return results