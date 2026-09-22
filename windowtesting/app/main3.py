import math
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

# 允許進入ArcFace比對的最大姿態範圍。
MAX_YAW = 70.0
MAX_PITCH = 80.0
MAX_ROLL = 30.0

ARCFACE_INTERVAL = 5

# 最終辨識門檻與側臉暫存槽寫入門檻。
# 多模板取最大值會提高誤判機會，正式值必須用實測資料校正。
COSINE_THRESHOLD = 0.40
ENROLL_IDENTITY_THRESHOLD = 0.35
REFERENCE_IMPROVEMENT_MARGIN = 0.01

# 口罩影像仍可進行比對，但預設不寫入姿態基準。
ALLOW_MASKED_REFERENCE_UPDATE = False
MASK_HEURISTIC_THRESHOLD = 0.55

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRFD_MODEL_PATH = PROJECT_ROOT / "models" / "det_10g.onnx"
ARCFACE_MODEL_PATH = PROJECT_ROOT / "models" / "w600k_r50.onnx"


# =========================================================
# 姿態暫存槽
# =========================================================
#
# 注意：不同攝影機鏡像設定可能造成Yaw正負方向相反。
# 若實測發現向左轉顯示RIGHT，僅交換LEFT/RIGHT顯示名稱即可，
# 不影響向量比對邏輯。
# =========================================================

SLOT_CONFIGS = {
    "FRONT": {
        "label": "FRONT",
        "target_yaw": 0.0,
        "yaw_scale": 15.0,
    },
    "LEFT_MID": {
        "label": "LEFT 15-40",
        "target_yaw": 27.5,
        "yaw_scale": 12.5,
    },
    "LEFT_LARGE": {
        "label": "LEFT 40-70",
        "target_yaw": 55.0,
        "yaw_scale": 15.0,
    },
    "RIGHT_MID": {
        "label": "RIGHT 15-40",
        "target_yaw": -27.5,
        "yaw_scale": 12.5,
    },
    "RIGHT_LARGE": {
        "label": "RIGHT 40-70",
        "target_yaw": -55.0,
        "yaw_scale": 15.0,
    },
    "PITCH_UP_MID": {
        "label": "PITCH UP 15-40",
        "target_yaw": 0.0,
        "target_pitch": 27.5,
        "yaw_scale": 15.0,
        "pitch_scale": 12.5,
    },
    "PITCH_UP_LARGE": {
        "label": "PITCH UP 40-80",
        "target_yaw": 0.0,
        "target_pitch": 60.0,
        "yaw_scale": 15.0,
        "pitch_scale": 20.0,
    },
    "PITCH_DOWN_MID": {
        "label": "PITCH DOWN 15-40",
        "target_yaw": 0.0,
        "target_pitch": -27.5,
        "yaw_scale": 15.0,
        "pitch_scale": 12.5,
    },
    "PITCH_DOWN_LARGE": {
        "label": "PITCH DOWN 40-80",
        "target_yaw": 0.0,
        "target_pitch": -60.0,
        "yaw_scale": 15.0,
        "pitch_scale": 20.0,
    },
}

# 補齊原本Yaw槽的Pitch目標參數。
for _config in SLOT_CONFIGS.values():
    _config.setdefault("target_pitch", 0.0)
    _config.setdefault("pitch_scale", 25.0)

SLOT_ORDER = [
    "FRONT",
    "LEFT_MID",
    "LEFT_LARGE",
    "RIGHT_MID",
    "RIGHT_LARGE",
    "PITCH_UP_MID",
    "PITCH_UP_LARGE",
    "PITCH_DOWN_MID",
    "PITCH_DOWN_LARGE",
]


def pose_to_slot_key(yaw, pitch):
    """
    Yaw與Pitch同時偏轉時，以相對允許範圍偏離較大的軸分類。

    Pitch正負方向可能受相機安裝與head_pose模型影響；若實測
    抬頭／低頭標籤相反，只需交換UP/DOWN顯示名稱。
    """
    yaw_ratio = abs(float(yaw)) / 70.0
    pitch_ratio = abs(float(pitch)) / 80.0

    if abs(yaw) <= 15.0 and abs(pitch) <= 15.0:
        return "FRONT"

    if pitch_ratio > yaw_ratio:
        if 15.0 < pitch <= 40.0:
            return "PITCH_UP_MID"
        if 40.0 < pitch <= 80.0:
            return "PITCH_UP_LARGE"
        if -40.0 <= pitch < -15.0:
            return "PITCH_DOWN_MID"
        if -80.0 <= pitch < -40.0:
            return "PITCH_DOWN_LARGE"

    if 15.0 < yaw <= 40.0:
        return "LEFT_MID"
    if 40.0 < yaw <= 70.0:
        return "LEFT_LARGE"
    if -40.0 <= yaw < -15.0:
        return "RIGHT_MID"
    if -70.0 <= yaw < -40.0:
        return "RIGHT_LARGE"
    return None


def calculate_slot_quality(pose_result, slot_key):
    """分數越小，代表越接近該姿態槽的理想角度。"""
    if slot_key is None or not pose_result.get("success", False):
        return float("inf")

    config = SLOT_CONFIGS[slot_key]
    yaw_error = (
        float(pose_result["yaw"]) - config["target_yaw"]
    ) / config["yaw_scale"]
    pitch_error = (
        float(pose_result["pitch"]) - config["target_pitch"]
    ) / config["pitch_scale"]
    roll_error = float(pose_result["roll"]) / 20.0

    return math.sqrt(
        yaw_error * yaw_error
        + pitch_error * pitch_error
        + roll_error * roll_error
    )


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 512 or norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("ArcFace特徵必須是有效的512維向量")
    return vector / norm


def cosine_similarity(vector_a, vector_b):
    a = normalize_embedding(vector_a)
    b = normalize_embedding(vector_b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def similarity_percent(similarity):
    if similarity is None:
        return None
    return max(0.0, min(100.0, float(similarity) * 100.0))


class PoseReferenceSlot:
    def __init__(self, slot_key):
        self.slot_key = slot_key
        self.label = SLOT_CONFIGS[slot_key]["label"]
        self.clear()

    def clear(self):
        self.embedding = None
        self.quality_score = float("inf")
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.updated_count = 0

    @property
    def ready(self):
        return self.embedding is not None

    def update(self, embedding, pose_result, quality_score):
        self.embedding = normalize_embedding(embedding).copy()
        self.quality_score = float(quality_score)
        self.yaw = float(pose_result["yaw"])
        self.pitch = float(pose_result["pitch"])
        self.roll = float(pose_result["roll"])
        self.updated_count += 1


class PoseReferenceBank:
    """單一身份的多角度ArcFace暫存區。"""

    def __init__(self):
        self.slots = {
            key: PoseReferenceSlot(key)
            for key in SLOT_ORDER
        }
        self.clear_status()

    def clear_status(self):
        self.last_similarity = None
        self.last_percent = None
        self.last_selected_slot = None
        self.last_result = "WAITING"

    def clear(self):
        for slot in self.slots.values():
            slot.clear()
        self.clear_status()

    @property
    def has_front(self):
        return self.slots["FRONT"].ready

    def compare_all(self, embedding):
        similarities = {}
        for key in SLOT_ORDER:
            slot = self.slots[key]
            if slot.ready:
                similarities[key] = cosine_similarity(
                    embedding,
                    slot.embedding,
                )
        return similarities

    @staticmethod
    def select_best(similarities):
        if not similarities:
            return None, None
        key = max(similarities, key=similarities.get)
        return key, similarities[key]

    def record_result(self, selected_slot, similarity, recognized):
        self.last_selected_slot = selected_slot
        self.last_similarity = similarity
        self.last_percent = similarity_percent(similarity)
        self.last_result = "MATCH" if recognized else "NOT MATCH"

    def try_update(
        self,
        slot_key,
        embedding,
        pose_result,
        quality_score,
        identity_verified,
    ):
        """
        FRONT可建立第一個身份基準；其他槽必須先通過身份驗證。
        已存在的FRONT也必須驗證為同一人才可覆蓋。
        """
        if slot_key is None:
            return None

        slot = self.slots[slot_key]

        if not self.has_front:
            if slot_key != "FRONT":
                return "NEED FRONT"
        elif not identity_verified:
            return "IDENTITY FAIL"

        is_better = (
            not slot.ready
            or quality_score + REFERENCE_IMPROVEMENT_MARGIN
            < slot.quality_score
        )
        if not is_better:
            return None

        slot.update(embedding, pose_result, quality_score)
        return f"{slot.label} UPDATED"


# =========================================================
# 顯示與影像工具
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


def save_snapshot_async(frame):
    """
    在背景執行緒儲存目前已標註的畫面，避免JPEG寫檔阻塞辨識迴圈。
    """
    snapshot = frame.copy()
    output_directory = PROJECT_ROOT / "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_directory / f"main3_{timestamp}.jpg"

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
        name="main3-snapshot-writer",
        daemon=True,
    ).start()


def draw_reference_panel(frame, bank):
    x1, y1, x2, y2 = 10, 185, 540, 525
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if bank.last_similarity is None:
        result_line = "Last result: waiting..."
    else:
        result_line = (
            f"Last {bank.last_result}: {bank.last_similarity:.4f} "
            f"({bank.last_percent:.1f}%) via {bank.last_selected_slot}"
        )

    lines = [
        "Multi-pose reference bank",
        result_line,
    ]

    for key in SLOT_ORDER:
        slot = bank.slots[key]
        if slot.ready:
            lines.append(
                f"{slot.label}: READY "
                f"Y:{slot.yaw:.1f} Q:{slot.quality_score:.2f}"
            )
        else:
            lines.append(f"{slot.label}: EMPTY")

    lines.extend(
        [
            f"Match threshold: {COSINE_THRESHOLD:.2f}",
            "R: Clear references   S: Screenshot   ESC: Exit",
        ]
    )
    draw_text_lines(frame, 20, 210, lines, (0, 255, 255), scale=0.52)


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
    bank = PoseReferenceBank()

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

        primary_index = None
        if faces:
            primary_index = max(
                range(len(faces)),
                key=lambda index: bbox_area(faces[index]["bbox"]),
            )

        run_arcface = frame_counter % ARCFACE_INTERVAL == 0

        for face_index, face in enumerate(faces):
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

            mask_result = mask_detector.detect(face_crop)

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

            yaw = float(pose_result["yaw"])
            pitch = float(pose_result["pitch"])
            slot_key = (
                pose_to_slot_key(yaw, pitch)
                if pose_result["success"]
                else None
            )
            slot_quality = calculate_slot_quality(pose_result, slot_key)

            quality_accepted = (
                size_accepted
                and blur_accepted
                and pose_accepted
                and face["landmarks"] is not None
                and slot_key is not None
            )

            rejection_reasons = []
            if not size_accepted:
                rejection_reasons.append("SMALL")
            if not blur_accepted:
                rejection_reasons.append("BLUR")
            if not pose_accepted or slot_key is None:
                rejection_reasons.append("POSE")
            if face["landmarks"] is None:
                rejection_reasons.append("NO-LANDMARK")

            similarities = {}
            selected_key = None
            selected_similarity = None
            recognized = None
            update_action = ""
            arcface_ms = 0.0

            if quality_accepted and run_arcface:
                start = time.perf_counter()
                _, embedding = arcface.extract(
                    frame=frame,
                    landmarks=face["landmarks"],
                )
                arcface_ms = (time.perf_counter() - start) * 1000.0

                similarities = bank.compare_all(embedding)
                selected_key, selected_similarity = bank.select_best(similarities)

                if selected_similarity is not None:
                    recognized = selected_similarity >= COSINE_THRESHOLD
                    bank.record_result(
                        selected_key,
                        selected_similarity,
                        recognized,
                    )

                identity_verified = (
                    not bank.has_front
                    or (
                        selected_similarity is not None
                        and selected_similarity >= ENROLL_IDENTITY_THRESHOLD
                    )
                )

                # 只有畫面最大的主要人臉可以建立或更新模板。
                if is_primary:
                    mask_blocks_update = (
                        mask_result["is_masked"] is True
                        and not ALLOW_MASKED_REFERENCE_UPDATE
                    )

                    if mask_blocks_update:
                        action = "MASK: NO REFERENCE UPDATE"
                    else:
                        action = bank.try_update(
                            slot_key=slot_key,
                            embedding=embedding,
                            pose_result=pose_result,
                            quality_score=slot_quality,
                            identity_verified=identity_verified,
                        )
                    if action:
                        update_action = action
                        if action.endswith("UPDATED") and selected_similarity is None:
                            recognized = True

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

            front_similarity = similarities.get("FRONT")
            same_pose_similarity = similarities.get(slot_key)
            reason_text = ",".join(rejection_reasons) if rejection_reasons else "NONE"
            slot_label = SLOT_CONFIGS[slot_key]["label"] if slot_key else "OUT"

            lines = [
                f"Face {face_index + 1}{' PRIMARY' if is_primary else ''}",
                f"Size:{face_width}x{face_height} Lap:{blur_result['laplacian']:.1f}",
                (
                    f"Y:{pose_result['yaw']:.1f} "
                    f"P:{pose_result['pitch']:.1f} "
                    f"R:{pose_result['roll']:.1f}"
                ),
                f"Pose slot:{slot_label} Q:{slot_quality:.2f}",
                (
                    f"Mask:{mask_result['status']} "
                    f"Score:{mask_result['score']:.2f}"
                ),
                f"Front:{format_similarity(front_similarity)}",
                f"Same pose:{format_similarity(same_pose_similarity)}",
                (
                    f"Best:{format_similarity(selected_similarity)} "
                    f"via {selected_key or '--'}"
                ),
                f"Result:{result_text} Reason:{reason_text}",
            ]
            if arcface_ms > 0.0:
                lines.append(f"ArcFace:{arcface_ms:.1f}ms")
            if update_action:
                lines.append(update_action)

            text_y = max(25, y1 - (len(lines) - 1) * 21 - 8)
            draw_text_lines(frame, x1, text_y, lines, box_color)

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            display_fps = display_fps * 0.9 + instant_fps * 0.1
        previous_time = current_time

        status_lines = [
            "SCRFD + ArcFace multi-pose references",
            f"Faces:{len(faces)} FPS:{display_fps:.1f}",
            f"Detection:{detection_ms:.1f}ms",
            (
                f"Match:{COSINE_THRESHOLD:.2f} "
                f"Enroll:{ENROLL_IDENTITY_THRESHOLD:.2f}"
            ),
            f"ArcFace interval:{ARCFACE_INTERVAL}",
        ]
        draw_text_lines(frame, 20, 30, status_lines, (0, 255, 255), scale=0.55)
        draw_reference_panel(frame, bank)

        cv2.imshow("Kiosk Multi-Pose Face Test - main3", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            bank.clear()
            print("多角度人臉特徵暫存區已全部清除")
        if key in (ord("s"), ord("S")):
            save_snapshot_async(frame)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
