import time
from pathlib import Path

import cv2

from face_detector_scrfd import ScrfdOnnxDetector
from face_embedding_arcface import ArcFaceEmbedder
from face_quality import evaluate_blur
from head_pose import (
    estimate_head_pose,
    is_pose_accepted,
)


# ==========================================
# 基本設定
# ==========================================

CAMERA_INDEX = 0

DETECTION_SIZE = (480, 480)
DETECTION_THRESHOLD = 0.5

# 階段九：尺寸門檻
MIN_FACE_WIDTH = 112
MIN_FACE_HEIGHT = 112

# 階段九：模糊門檻
LAPLACIAN_THRESHOLD = 80.0
TENENGRAD_THRESHOLD = 1000.0

# 階段十：角度門檻
MAX_YAW = 30.0
MAX_PITCH = 25.0
MAX_ROLL = 20.0

# 階段十一：30fps下，每5幀最多執行一次
ARCFACE_INTERVAL = 5


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

SCRFD_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "det_10g.onnx"
)

ARCFACE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "w600k_r50.onnx"
)


def clamp_bbox(
    bbox,
    frame_width,
    frame_height,
):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))

    return x1, y1, x2, y2


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

    for index, point in enumerate(landmarks):
        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            frame,
            (x, y),
            3,
            colors[index],
            -1,
        )


def draw_text_lines(
    frame,
    x,
    y,
    lines,
    color,
):
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )


def main():
    detector = ScrfdOnnxDetector(
        model_path=SCRFD_MODEL_PATH,
        input_size=DETECTION_SIZE,
        detection_threshold=DETECTION_THRESHOLD,
    )

    arcface = ArcFaceEmbedder(
        model_path=ARCFACE_MODEL_PATH
    )

    capture = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )
    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )
    capture.set(
        cv2.CAP_PROP_FPS,
        30,
    )

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

        inference_start = time.perf_counter()
        faces = detector.detect(frame)
        detection_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        for face_index, face in enumerate(
            faces,
            start=1,
        ):
            bbox = clamp_bbox(
                face["bbox"],
                frame_width,
                frame_height,
            )

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            face_width = x2 - x1
            face_height = y2 - y1

            face_crop = frame[
                y1:y2,
                x1:x2,
            ]

            if face_crop.size == 0:
                continue

            # ==================================
            # 階段九：尺寸與模糊判斷
            # ==================================

            blur_result = evaluate_blur(
                face_crop,
                laplacian_threshold=(
                    LAPLACIAN_THRESHOLD
                ),
                tenengrad_threshold=(
                    TENENGRAD_THRESHOLD
                ),
            )

            size_accepted = (
                face_width >= MIN_FACE_WIDTH
                and face_height >= MIN_FACE_HEIGHT
            )

            blur_accepted = not blur_result[
                "is_blurry"
            ]

            # ==================================
            # 階段十：頭部姿態判斷
            # ==================================

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

            # ==================================
            # 綜合品質判斷
            # ==================================

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

            # ==================================
            # 階段十一：ArcFace
            # ==================================

            embedding = None
            arcface_ms = 0.0

            run_arcface = (
                quality_accepted
                and frame_counter
                % ARCFACE_INTERVAL
                == 0
            )

            if run_arcface:
                arcface_start = time.perf_counter()

                aligned_face, embedding = (
                    arcface.extract(
                        frame=frame,
                        landmarks=face["landmarks"],
                    )
                )

                arcface_ms = (
                    time.perf_counter()
                    - arcface_start
                ) * 1000.0

            if quality_accepted:
                box_color = (0, 255, 0)
                quality_text = "ACCEPTED"
            else:
                box_color = (0, 0, 255)
                quality_text = "REJECTED"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            draw_landmarks(
                frame,
                face["landmarks"],
            )

            pose_text = (
                f"Yaw:{pose_result['yaw']:.1f} "
                f"Pitch:{pose_result['pitch']:.1f} "
                f"Roll:{pose_result['roll']:.1f}"
            )

            if embedding is not None:
                embedding_text = (
                    f"Embedding:{embedding.shape[0]}D "
                    f"{arcface_ms:.1f}ms"
                )
            elif quality_accepted:
                embedding_text = "Embedding:READY"
            else:
                embedding_text = "Embedding:SKIPPED"

            if rejection_reasons:
                reason_text = (
                    "Reason:"
                    + ",".join(rejection_reasons)
                )
            else:
                reason_text = "Reason:NONE"

            text_lines = [
                (
                    f"Face {face_index} "
                    f"Score:{face['score']:.2f}"
                ),
                (
                    f"Size:{face_width}x{face_height}"
                ),
                (
                    f"Lap:{blur_result['laplacian']:.1f} "
                    f"Ten:{blur_result['tenengrad']:.1f}"
                ),
                pose_text,
                f"Quality:{quality_text}",
                reason_text,
                embedding_text,
            ]

            text_x = x1
            text_y = max(25, y1 - 145)

            draw_text_lines(
                frame,
                text_x,
                text_y,
                text_lines,
                box_color,
            )

        # ======================================
        # FPS計算
        # ======================================

        current_time = time.perf_counter()
        elapsed = current_time - previous_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            display_fps = (
                display_fps * 0.9
                + instant_fps * 0.1
            )

        previous_time = current_time

        status_lines = [
            "Detector: SCRFD ONNX",
            f"Faces: {len(faces)}",
            f"FPS: {display_fps:.1f}",
            f"Detection: {detection_ms:.1f} ms",
            (
                f"ArcFace interval: "
                f"{ARCFACE_INTERVAL}"
            ),
            "ESC: Exit",
        ]

        draw_text_lines(
            frame,
            20,
            30,
            status_lines,
            (0, 255, 255),
        )

        cv2.imshow(
            "Kiosk Stage 9-11 Test",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()