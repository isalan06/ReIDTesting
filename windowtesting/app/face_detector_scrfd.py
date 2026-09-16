from pathlib import Path

from insightface.model_zoo import get_model


class ScrfdOnnxDetector:
    def __init__(
        self,
        model_path,
        input_size=(640, 640),
        detection_threshold=0.5,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.detection_threshold = detection_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                "找不到 SCRFD ONNX 模型："
                f"{self.model_path.resolve()}"
            )

        print(
            f"載入 SCRFD ONNX："
            f"{self.model_path.resolve()}"
        )

        self.detector = get_model(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.detector.prepare(
            ctx_id=-1,
            input_size=self.input_size,
            det_thresh=self.detection_threshold,
        )

        print("SCRFD ONNX 模型載入完成")

    def detect(self, frame):
        bounding_boxes, landmarks_array = (
            self.detector.detect(
                frame,
                max_num=0,
                metric="default",
            )
        )

        results = []

        if bounding_boxes is None:
            return results

        for index, bounding_box in enumerate(
            bounding_boxes
        ):
            x1, y1, x2, y2, score = bounding_box

            landmarks = None

            if landmarks_array is not None:
                landmarks = (
                    landmarks_array[index]
                    .astype(float)
                    .tolist()
                )

            results.append(
                {
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "score": float(score),
                    "landmarks": landmarks,
                }
            )

        return results