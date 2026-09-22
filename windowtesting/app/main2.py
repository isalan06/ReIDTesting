import math
import time
from pathlib import Path

import cv2
import numpy as np

from face_detector_scrfd import ScrfdOnnxDetector
from face_embedding_arcface import ArcFaceEmbedder
from face_quality import evaluate_blur
from head_pose import estimate_head_pose, is_pose_accepted


# =========================================================
# 基本設定
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

MAX_YAW = 70.0
MAX_PITCH = 80.0
MAX_ROLL = 30.0

# 30 fps下每5幀最多執行一次ArcFace，約6次/秒。
ARCFACE_INTERVAL = 5

# ArcFace已正規化向量的Cosine Similarity初始門檻。
# 這只是PoC起始值，正式門檻需以實際資料重新校正。
COSINE_THRESHOLD = 0.40

# 新姿態分數至少改善多少才更新基準，避免浮點抖動反覆覆蓋。
REFERENCE_IMPROVEMENT_MARGIN = 0.01

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRFD_MODEL_PATH = PROJECT_ROOT / "models" / "det_10g.onnx"
ARCFACE_MODEL_PATH = PROJECT_ROOT / "models" / "w600k_r50.onnx"


class FaceReferenceBuffer:
    """單人PoC的記憶體暫存區；程式結束後不保留。"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.embedding = None
        self.frontal_score = float("inf")
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.updated_count = 0
        self.updated_at = None
        # 左側面板常駐顯示最近一次綠色MATCH的分數。
        self.last_match_similarity = None
        self.last_match_percent = None

    @property
    def has_reference(self):
        return self.embedding is not None

    def set_reference(self, embedding, pose_result, frontal_score):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 512 or norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("ArcFace特徵必須是有效的512維向量")

        self.embedding = (vector / norm).copy()
        self.frontal_score = float(frontal_score)
        self.yaw = float(pose_result["yaw"])
        self.pitch = float(pose_result["pitch"])
        self.roll = float(pose_result["roll"])
        self.updated_count += 1
        self.updated_at = time.time()

    def record_match(self, similarity):
        """記錄最近一次成功MATCH的Cosine分數。"""
        self.last_match_similarity = float(similarity)
        self.last_match_percent = similarity_percent(similarity)


def cosine_similarity(vector_a, vector_b):
    """計算Cosine Similarity，回傳範圍約為-1至1。"""
    a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"向量維度不同：{a.shape} != {b.shape}")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denominator = norm_a * norm_b

    if denominator <= 0.0 or not np.isfinite(denominator):
        return None

    similarity = float(np.dot(a, b) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))


def similarity_percent(similarity):
    """僅供UI顯示；辨識判斷仍使用原始Cosine值。"""
    if similarity is None:
        return None
    return max(0.0, min(100.0, similarity * 100.0))


def calculate_frontal_score(pose_result):
    """
    將Yaw/Pitch/Roll依允許範圍正規化後計算距離。
    分數越接近0，表示越接近正面。
    """
    if not pose_result.get("success", False):
        return float("inf")

    yaw = float(pose_result["yaw"]) / MAX_YAW
    pitch = float(pose_result["pitch"]) / MAX_PITCH
    roll = float(pose_result["roll"]) / MAX_ROLL
    return math.sqrt(yaw * yaw + pitch * pitch + roll * roll)


def clamp_bbox(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))
    return x1, y1, x2, y2


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


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


def draw_reference_panel(frame, reference_buffer):
    x1, y1, x2, y2 = 10, 185, 490, 345
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    if reference_buffer.has_reference:
        if reference_buffer.last_match_similarity is None:
            last_match_text = "Last MATCH: waiting..."
        else:
            last_match_text = (
                f"Last MATCH: {reference_buffer.last_match_similarity:.4f} "
                f"({reference_buffer.last_match_percent:.1f}%)"
            )

        lines = [
            "Reference: READY (512D)",
            (
                f"Best Y:{reference_buffer.yaw:.1f} "
                f"P:{reference_buffer.pitch:.1f} "
                f"R:{reference_buffer.roll:.1f}"
            ),
            f"Frontal score: {reference_buffer.frontal_score:.3f}",
            last_match_text,
            f"Updates: {reference_buffer.updated_count}",
            "R: Clear reference   ESC: Exit",
        ]
        color = (0, 255, 0)
    else:
        lines = [
            "Reference: EMPTY",
            "Look straight at the camera",
            "First qualified primary face will be stored",
            "R: Clear reference   ESC: Exit",
        ]
        color = (0, 255, 255)

    draw_text_lines(frame, 20, 210, lines, color, scale=0.55)


def main():
    detector = ScrfdOnnxDetector(
        model_path=SCRFD_MODEL_PATH,
        input_size=DETECTION_SIZE,
        detection_threshold=DETECTION_THRESHOLD,
    )

    arcface = ArcFaceEmbedder(model_path=ARCFACE_MODEL_PATH)
    reference_buffer = FaceReferenceBuffer()

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

        detection_start = time.perf_counter()
        faces = detector.detect(frame)
        detection_ms = (time.perf_counter() - detection_start) * 1000.0

        # 暫存區只由畫面中最大的人臉建立或更新，減少多人污染。
        primary_index = None
        if faces:
            primary_index = max(
                range(len(faces)),
                key=lambda index: bbox_area(faces[index]["bbox"]),
            )

        run_arcface_this_frame = frame_counter % ARCFACE_INTERVAL == 0

        for face_index, face in enumerate(faces):
            display_index = face_index + 1
            is_primary = face_index == primary_index

            bbox = clamp_bbox(face["bbox"], frame_width, frame_height)
            x1, y1, x2, y2 = bbox
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
            pose_accepted = is_pose_accepted(
                pose_result,
                max_yaw=MAX_YAW,
                max_pitch=MAX_PITCH,
                max_roll=MAX_ROLL,
            )

            quality_accepted = (
                size_accepted
                and blur_accepted
                and pose_accepted
                and face["landmarks"] is not None
            )

            rejection_reasons = []
            if not size_accepted:
                rejection_reasons.append("SMALL")
            if not blur_accepted:
                rejection_reasons.append("BLUR")
            if not pose_accepted:
                rejection_reasons.append("POSE")
            if face["landmarks"] is None:
                rejection_reasons.append("NO-LANDMARK")

            frontal_score = calculate_frontal_score(pose_result)
            similarity = None
            recognized = None
            reference_action = ""
            arcface_ms = 0.0

            if quality_accepted and run_arcface_this_frame:
                arcface_start = time.perf_counter()
                _, embedding = arcface.extract(
                    frame=frame,
                    landmarks=face["landmarks"],
                )
                arcface_ms = (time.perf_counter() - arcface_start) * 1000.0

                if reference_buffer.has_reference:
                    similarity = cosine_similarity(
                        embedding,
                        reference_buffer.embedding,
                    )
                    recognized = (
                        similarity is not None
                        and similarity >= COSINE_THRESHOLD
                    )

                    # 僅將綠色MATCH的分數複製到左側常駐面板。
                    if recognized:
                        reference_buffer.record_match(similarity)

                    # 只有主要人臉、已辨識為同一人，而且姿態更正面才覆蓋。
                    is_more_frontal = (
                        frontal_score + REFERENCE_IMPROVEMENT_MARGIN
                        < reference_buffer.frontal_score
                    )
                    if is_primary and recognized and is_more_frontal:
                        reference_buffer.set_reference(
                            embedding,
                            pose_result,
                            frontal_score,
                        )
                        reference_action = "REF UPDATED"
                elif is_primary:
                    reference_buffer.set_reference(
                        embedding,
                        pose_result,
                        frontal_score,
                    )
                    recognized = True
                    similarity = 1.0
                    reference_action = "REF CREATED"

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3 if is_primary else 2)
            draw_landmarks(frame, face["landmarks"])

            primary_text = " PRIMARY" if is_primary else ""
            reason_text = ",".join(rejection_reasons) if rejection_reasons else "NONE"

            if similarity is None:
                similarity_text = "Similarity: --"
            else:
                percent = similarity_percent(similarity)
                similarity_text = (
                    f"Cosine:{similarity:.4f} Display:{percent:.1f}%"
                )

            text_lines = [
                f"Face {display_index}{primary_text} Score:{face['score']:.2f}",
                f"Size:{face_width}x{face_height}",
                f"Lap:{blur_result['laplacian']:.1f} Ten:{blur_result['tenengrad']:.1f}",
                (
                    f"Y:{pose_result['yaw']:.1f} "
                    f"P:{pose_result['pitch']:.1f} "
                    f"R:{pose_result['roll']:.1f}"
                ),
                f"Front:{frontal_score:.3f} Result:{result_text}",
                similarity_text,
                f"Reason:{reason_text}",
            ]
            if arcface_ms > 0.0:
                text_lines.append(f"ArcFace:{arcface_ms:.1f}ms")
            if reference_action:
                text_lines.append(reference_action)

            text_y = max(25, y1 - (len(text_lines) - 1) * 21 - 8)
            draw_text_lines(frame, x1, text_y, text_lines, box_color)

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            display_fps = display_fps * 0.9 + instant_fps * 0.1
        previous_time = current_time

        status_lines = [
            "Detector: SCRFD ONNX + ArcFace",
            f"Faces: {len(faces)}",
            f"FPS: {display_fps:.1f}",
            f"Detection: {detection_ms:.1f}ms",
            f"Cosine threshold: {COSINE_THRESHOLD:.2f}",
            f"ArcFace interval: {ARCFACE_INTERVAL}",
        ]
        draw_text_lines(frame, 20, 30, status_lines, (0, 255, 255), scale=0.56)
        draw_reference_panel(frame, reference_buffer)

        cv2.imshow("Kiosk Face Reference Test - main2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key in (ord("r"), ord("R")):
            reference_buffer.clear()
            print("人臉特徵暫存區已清除")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
