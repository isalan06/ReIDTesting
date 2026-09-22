import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_detector_scrfd import ScrfdOnnxDetector
from face_embedding_arcface import ArcFaceEmbedder
from face_mask_detector2 import FaceMaskHeuristicDetector
from face_quality import evaluate_blur
from head_pose import estimate_head_pose, is_pose_accepted


# =========================================================
# 攝影機、模型與品質設定
# =========================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

DETECTION_SIZE = (480, 480)
DETECTION_THRESHOLD = 0.5

MIN_FACE_WIDTH = 112
MIN_FACE_HEIGHT = 112
LAPLACIAN_THRESHOLD = 80.0
TENENGRAD_THRESHOLD = 1000.0

# 即時影像允許進行比對的姿態範圍。
MATCH_MAX_YAW = 70.0
MATCH_MAX_PITCH = 80.0
MATCH_MAX_ROLL = 30.0

# 只有此範圍內的正面影像可以寫入參考暫存區。
REFERENCE_MAX_YAW = 15.0
REFERENCE_MAX_PITCH = 15.0
REFERENCE_MAX_ROLL = 15.0

ARCFACE_INTERVAL = 5
MASK_HEURISTIC_THRESHOLD = 0.55

# 完整512D與可信維度的初始門檻。
FULL_COSINE_THRESHOLD = 0.40
TRUSTED_COSINE_THRESHOLD = 0.35
REFERENCE_ENROLL_THRESHOLD = 0.40

# 正面可信特徵學習設定。
MAX_REFERENCE_SAMPLES = 20
MIN_TRUSTED_SAMPLES = 5
TRUSTED_DIM_RATIO = 0.50
MIN_TRUSTED_DIMS = 128
REFERENCE_SAMPLE_INTERVAL_SECONDS = 0.25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRFD_MODEL_PATH = PROJECT_ROOT / "models" / "det_10g.onnx"
ARCFACE_MODEL_PATH = PROJECT_ROOT / "models" / "w600k_r50.onnx"


# =========================================================
# 向量與正面可信特徵暫存區
# =========================================================


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 512 or norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("ArcFace特徵必須是有效的512維向量")
    return vector / norm


def cosine_similarity(vector_a, vector_b, indices=None):
    a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"向量維度不同：{a.shape} != {b.shape}")

    if indices is not None:
        a = a[indices]
        b = b[indices]

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denominator = norm_a * norm_b
    if denominator <= 0.0 or not np.isfinite(denominator):
        return None

    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def similarity_percent(similarity):
    if similarity is None:
        return None
    return max(0.0, min(100.0, float(similarity) * 100.0))


class TrustedFrontReferenceBuffer:
    """
    只收集合格正面ArcFace特徵。

    多張正面特徵用來估算每個維度的訊號穩定度：
        reliability = abs(mean) / (std + epsilon)

    選出可靠度最高的一部分維度，用於姿態改變時的實驗性比對。
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.samples = []
        self.prototype = None
        self.dimension_std = None
        self.trusted_indices = None
        self.last_sample_time = 0.0
        self.last_full_similarity = None
        self.last_trusted_similarity = None
        self.last_final_similarity = None
        self.last_result = "WAITING"

    @property
    def ready(self):
        return self.prototype is not None

    @property
    def trusted_ready(self):
        return (
            self.trusted_indices is not None
            and len(self.trusted_indices) >= MIN_TRUSTED_DIMS
        )

    @property
    def sample_count(self):
        return len(self.samples)

    @property
    def trusted_dimension_count(self):
        return 0 if self.trusted_indices is None else len(self.trusted_indices)

    def rebuild(self):
        matrix = np.stack(self.samples, axis=0)
        mean_vector = np.mean(matrix, axis=0)
        self.prototype = normalize_embedding(mean_vector)

        if len(self.samples) < MIN_TRUSTED_SAMPLES:
            self.dimension_std = None
            self.trusted_indices = None
            return

        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        self.dimension_std = std

        # 同時考量維度平均訊號與跨正面樣本穩定度。
        reliability = np.abs(mean) / (std + 1e-4)
        dimension_count = max(
            MIN_TRUSTED_DIMS,
            int(512 * TRUSTED_DIM_RATIO),
        )
        dimension_count = min(512, dimension_count)

        ranked = np.argsort(reliability)[::-1]
        self.trusted_indices = np.sort(ranked[:dimension_count])

    def can_collect_now(self):
        return (
            self.sample_count < MAX_REFERENCE_SAMPLES
            and time.perf_counter() - self.last_sample_time
            >= REFERENCE_SAMPLE_INTERVAL_SECONDS
        )

    def add_sample(self, embedding):
        if not self.can_collect_now():
            return False

        self.samples.append(normalize_embedding(embedding).copy())
        self.last_sample_time = time.perf_counter()
        self.rebuild()
        return True

    def compare(self, embedding):
        if not self.ready:
            return None, None, None, None

        query = normalize_embedding(embedding)
        full_similarity = cosine_similarity(query, self.prototype)

        trusted_similarity = None
        if self.trusted_ready:
            trusted_similarity = cosine_similarity(
                query,
                self.prototype,
                indices=self.trusted_indices,
            )

        if trusted_similarity is not None:
            final_similarity = trusted_similarity
            threshold = TRUSTED_COSINE_THRESHOLD
        else:
            final_similarity = full_similarity
            threshold = FULL_COSINE_THRESHOLD

        recognized = (
            final_similarity is not None
            and final_similarity >= threshold
        )

        self.last_full_similarity = full_similarity
        self.last_trusted_similarity = trusted_similarity
        self.last_final_similarity = final_similarity
        self.last_result = "MATCH" if recognized else "NOT MATCH"

        return (
            full_similarity,
            trusted_similarity,
            final_similarity,
            recognized,
        )


# =========================================================
# 影像與畫面顯示工具
# =========================================================


def clamp_bbox(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(int(x1), frame_width - 1)),
        max(0, min(int(y1), frame_height - 1)),
        max(0, min(int(x2), frame_width)),
        max(0, min(int(y2), frame_height)),
    )


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_strict_front_pose(pose_result):
    if not pose_result.get("success", False):
        return False
    return (
        abs(float(pose_result["yaw"])) <= REFERENCE_MAX_YAW
        and abs(float(pose_result["pitch"])) <= REFERENCE_MAX_PITCH
        and abs(float(pose_result["roll"])) <= REFERENCE_MAX_ROLL
    )


def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    for index, point in enumerate(landmarks[:5]):
        cv2.circle(
            frame,
            (int(point[0]), int(point[1])),
            3,
            colors[index],
            -1,
        )


def draw_text_lines(frame, x, y, lines, color, scale=0.50):
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
        )


def format_similarity(value):
    return "--" if value is None else f"{value:.4f}"


def draw_reference_panel(frame, reference):
    x1, y1, x2, y2 = 10, 180, 540, 390
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if reference.ready:
        state = "READY"
        color = (0, 255, 0)
    else:
        state = "COLLECT FRONT FACE"
        color = (0, 255, 255)

    lines = [
        f"Front-only reference: {state}",
        f"Front samples: {reference.sample_count}/{MAX_REFERENCE_SAMPLES}",
        (
            f"Trusted dimensions: {reference.trusted_dimension_count}/512 "
            f"({'READY' if reference.trusted_ready else 'LEARNING'})"
        ),
        f"Last full cosine: {format_similarity(reference.last_full_similarity)}",
        f"Last trusted cosine: {format_similarity(reference.last_trusted_similarity)}",
        f"Last final: {format_similarity(reference.last_final_similarity)}",
        f"Last result: {reference.last_result}",
        "R: Clear reference   S: Screenshot   ESC: Exit",
    ]
    draw_text_lines(frame, 20, 205, lines, color, scale=0.54)


def save_snapshot_async(frame):
    snapshot = frame.copy()
    output_directory = PROJECT_ROOT / "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_directory / f"main4_{timestamp}.jpg"

    def writer():
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(
                str(output_path),
                snapshot,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if success:
                print(f"畫面已儲存：{output_path}")
            else:
                print(f"畫面儲存失敗：{output_path}")
        except Exception as error:
            print(f"畫面儲存發生錯誤：{error}")

    threading.Thread(
        target=writer,
        name="main4-snapshot-writer",
        daemon=True,
    ).start()


# =========================================================
# 主程式
# =========================================================


def main():
    detector = ScrfdOnnxDetector(
        model_path=SCRFD_MODEL_PATH,
        input_size=DETECTION_SIZE,
        detection_threshold=DETECTION_THRESHOLD,
    )
    arcface = ArcFaceEmbedder(model_path=ARCFACE_MODEL_PATH)
    mask_detector = FaceMaskHeuristicDetector(
        mask_threshold=MASK_HEURISTIC_THRESHOLD,
    )
    reference = TrustedFrontReferenceBuffer()

    capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if not capture.isOpened():
        raise RuntimeError("無法開啟攝影機")

    frame_counter = 0
    previous_time = time.perf_counter()
    display_fps = 0.0

    while True:
        success, frame = capture.read()
        if not success:
            print("無法取得攝影機影像")
            break

        frame_counter += 1
        frame_height, frame_width = frame.shape[:2]

        start = time.perf_counter()
        faces = detector.detect(frame)
        detection_ms = (time.perf_counter() - start) * 1000.0

        primary_index = None
        if faces:
            primary_index = max(
                range(len(faces)),
                key=lambda index: bbox_area(faces[index]["bbox"]),
            )

        run_arcface = frame_counter % ARCFACE_INTERVAL == 0

        for face_index, face in enumerate(faces):
            is_primary = face_index == primary_index
            x1, y1, x2, y2 = clamp_bbox(
                face["bbox"],
                frame_width,
                frame_height,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            face_width = x2 - x1
            face_height = y2 - y1
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            blur_result = evaluate_blur(
                face_crop,
                laplacian_threshold=LAPLACIAN_THRESHOLD,
                tenengrad_threshold=TENENGRAD_THRESHOLD,
            )
            mask_result = mask_detector.detect(face_crop)

            size_accepted = (
                face_width >= MIN_FACE_WIDTH
                and face_height >= MIN_FACE_HEIGHT
            )
            blur_accepted = not blur_result["is_blurry"]

            pose_result = estimate_head_pose(
                landmarks=face["landmarks"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            match_pose_accepted = is_pose_accepted(
                pose_result,
                max_yaw=MATCH_MAX_YAW,
                max_pitch=MATCH_MAX_PITCH,
                max_roll=MATCH_MAX_ROLL,
            )
            strict_front = is_strict_front_pose(pose_result)

            quality_accepted = (
                size_accepted
                and blur_accepted
                and match_pose_accepted
                and face["landmarks"] is not None
            )

            reasons = []
            if not size_accepted:
                reasons.append("SMALL")
            if not blur_accepted:
                reasons.append("BLUR")
            if not match_pose_accepted:
                reasons.append("POSE")
            if face["landmarks"] is None:
                reasons.append("NO-LANDMARK")

            full_similarity = None
            trusted_similarity = None
            final_similarity = None
            recognized = None
            action = ""
            arcface_ms = 0.0

            if quality_accepted and run_arcface:
                arcface_start = time.perf_counter()
                _, embedding = arcface.extract(
                    frame=frame,
                    landmarks=face["landmarks"],
                )
                arcface_ms = (time.perf_counter() - arcface_start) * 1000.0

                if reference.ready:
                    (
                        full_similarity,
                        trusted_similarity,
                        final_similarity,
                        recognized,
                    ) = reference.compare(embedding)

                reference_candidate = (
                    is_primary
                    and strict_front
                    and mask_result["is_masked"] is not True
                )

                if reference_candidate:
                    identity_verified = (
                        not reference.ready
                        or (
                            full_similarity is not None
                            and full_similarity >= REFERENCE_ENROLL_THRESHOLD
                        )
                    )
                    if identity_verified:
                        if reference.add_sample(embedding):
                            action = "FRONT SAMPLE ADDED"
                            if recognized is None:
                                recognized = True
                    else:
                        action = "FRONT IDENTITY FAIL"
                elif strict_front and mask_result["is_masked"] is True:
                    action = "MASK: NO ENROLL"

            if not quality_accepted:
                box_color = (0, 0, 255)
                result_text = "REJECTED"
            elif recognized is True:
                box_color = (0, 255, 0)
                result_text = "MATCH"
            elif recognized is False:
                box_color = (0, 165, 255)
                result_text = "NOT MATCH"
            else:
                box_color = (255, 255, 0)
                result_text = "READY"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3 if is_primary else 2,
            )
            draw_landmarks(frame, face["landmarks"])

            raw_pitch = float(pose_result.get("raw_pitch", 0.0))
            reason_text = ",".join(reasons) if reasons else "NONE"
            lines = [
                f"Face {face_index + 1}{' PRIMARY' if is_primary else ''}",
                f"Size:{face_width}x{face_height} Lap:{blur_result['laplacian']:.1f}",
                (
                    f"Y:{pose_result['yaw']:.1f} "
                    f"P:{pose_result['pitch']:.1f} "
                    f"R:{pose_result['roll']:.1f}"
                ),
                f"RawPitch:{raw_pitch:.1f} StrictFront:{strict_front}",
                f"Mask:{mask_result['status']} Score:{mask_result['score']:.2f}",
                f"Full512:{format_similarity(full_similarity)}",
                f"Trusted:{format_similarity(trusted_similarity)}",
                f"Final:{format_similarity(final_similarity)} {result_text}",
                f"Reason:{reason_text}",
            ]
            if arcface_ms > 0.0:
                lines.append(f"ArcFace:{arcface_ms:.1f}ms")
            if action:
                lines.append(action)

            text_y = max(25, y1 - (len(lines) - 1) * 21 - 8)
            draw_text_lines(frame, x1, text_y, lines, box_color)

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            display_fps = display_fps * 0.9 + instant_fps * 0.1
        previous_time = current_time

        status_lines = [
            "SCRFD + ArcFace trusted frontal dimensions",
            f"Faces:{len(faces)} FPS:{display_fps:.1f}",
            f"Detection:{detection_ms:.1f}ms",
            (
                f"Full threshold:{FULL_COSINE_THRESHOLD:.2f} "
                f"Trusted:{TRUSTED_COSINE_THRESHOLD:.2f}"
            ),
            f"ArcFace interval:{ARCFACE_INTERVAL}",
        ]
        draw_text_lines(frame, 20, 30, status_lines, (0, 255, 255), scale=0.55)
        draw_reference_panel(frame, reference)

        cv2.imshow("Kiosk Trusted Front Face Test - main4", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            reference.clear()
            print("正面可信特徵暫存區已清除")
        if key in (ord("s"), ord("S")):
            save_snapshot_async(frame)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
