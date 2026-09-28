import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import main4 as base
from face_detector_scrfd import ScrfdOnnxDetector
from face_embedding_arcface import ArcFaceEmbedder
from face_mask_detector2 import FaceMaskHeuristicDetector
from face_quality import evaluate_blur
from head_pose import estimate_head_pose, is_pose_accepted


# =========================================================
# 照片比較輸出設定
# =========================================================

COMPARISON_WINDOW_NAME = "Template vs Loaded Photo"
COMPARISON_IMAGE_WIDTH = 360
COMPARISON_IMAGE_HEIGHT = 360
COMPARISON_CANVAS_WIDTH = 960
COMPARISON_CANVAS_HEIGHT = 650
COMPARISON_JPEG_QUALITY = 95


class PhotoReferenceBuffer(base.TrustedFrontReferenceBuffer):
    """在main4可信特徵暫存區上，另外保留最正面的代表照片。"""

    def __init__(self):
        self.template_image = None
        self.template_pose = None
        self.template_quality_score = float("inf")
        super().__init__()

    def clear(self):
        super().clear()
        self.template_image = None
        self.template_pose = None
        self.template_quality_score = float("inf")

    @staticmethod
    def pose_quality_score(pose_result):
        yaw = float(pose_result["yaw"]) / base.REFERENCE_MAX_YAW
        pitch = float(pose_result["pitch"]) / base.REFERENCE_MAX_PITCH
        roll = float(pose_result["roll"]) / base.REFERENCE_MAX_ROLL
        return float(np.sqrt(yaw * yaw + pitch * pitch + roll * roll))

    def add_photo_sample(self, embedding, aligned_face, pose_result):
        added = super().add_sample(embedding)
        if not added:
            return False

        pose_score = self.pose_quality_score(pose_result)
        if self.template_image is None or pose_score < self.template_quality_score:
            self.template_image = aligned_face.copy()
            self.template_pose = {
                "yaw": float(pose_result["yaw"]),
                "pitch": float(pose_result["pitch"]),
                "raw_pitch": float(pose_result.get("raw_pitch", 0.0)),
                "roll": float(pose_result["roll"]),
            }
            self.template_quality_score = pose_score

        return True


def select_image_file():
    """開啟系統檔案選擇視窗；取消時回傳None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="選擇要與人臉模板比較的照片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception as error:
        print(f"無法開啟照片選擇視窗：{error}")
        return None


def read_image_unicode(image_path):
    """支援Windows中文或特殊字元路徑。"""
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception as error:
        print(f"照片讀取失敗：{error}")
        return None


def write_image_unicode(image_path, image, jpeg_quality=95):
    """支援Windows中文或特殊字元輸出路徑。"""
    suffix = image_path.suffix.lower()
    extension = suffix if suffix in (".jpg", ".jpeg", ".png", ".bmp") else ".jpg"
    parameters = []
    if extension in (".jpg", ".jpeg"):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        return False
    try:
        encoded.tofile(str(image_path))
        return True
    except Exception:
        return False


def resize_with_letterbox(image, target_width, target_height):
    """保持原始比例縮放，空白區域以深灰色補齊。"""
    if image is None or image.size == 0:
        return np.full((target_height, target_width, 3), 35, dtype=np.uint8)

    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.full((target_height, target_width, 3), 35, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized
    return canvas


def put_centered_text(image, text, center_x, y, scale, color, thickness=2):
    (text_width, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(10, int(center_x - text_width / 2))
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def build_comparison_image(
    template_image,
    template_pose,
    loaded_image,
    loaded_pose,
    full_similarity,
    trusted_similarity,
    final_similarity,
    recognized,
    loaded_path,
    mask_result,
):
    canvas = np.full(
        (COMPARISON_CANVAS_HEIGHT, COMPARISON_CANVAS_WIDTH, 3),
        245,
        dtype=np.uint8,
    )

    left_x = 70
    right_x = COMPARISON_CANVAS_WIDTH - 70 - COMPARISON_IMAGE_WIDTH
    image_y = 95

    left = resize_with_letterbox(
        template_image,
        COMPARISON_IMAGE_WIDTH,
        COMPARISON_IMAGE_HEIGHT,
    )
    right = resize_with_letterbox(
        loaded_image,
        COMPARISON_IMAGE_WIDTH,
        COMPARISON_IMAGE_HEIGHT,
    )

    canvas[
        image_y:image_y + COMPARISON_IMAGE_HEIGHT,
        left_x:left_x + COMPARISON_IMAGE_WIDTH,
    ] = left
    canvas[
        image_y:image_y + COMPARISON_IMAGE_HEIGHT,
        right_x:right_x + COMPARISON_IMAGE_WIDTH,
    ] = right

    result_color = (30, 150, 30) if recognized else (20, 20, 220)
    border_color = result_color

    cv2.rectangle(
        canvas,
        (left_x - 2, image_y - 2),
        (left_x + COMPARISON_IMAGE_WIDTH + 2, image_y + COMPARISON_IMAGE_HEIGHT + 2),
        border_color,
        3,
    )
    cv2.rectangle(
        canvas,
        (right_x - 2, image_y - 2),
        (right_x + COMPARISON_IMAGE_WIDTH + 2, image_y + COMPARISON_IMAGE_HEIGHT + 2),
        border_color,
        3,
    )

    put_centered_text(
        canvas,
        "PROGRAM TEMPLATE",
        left_x + COMPARISON_IMAGE_WIDTH // 2,
        55,
        0.75,
        (40, 40, 40),
    )
    put_centered_text(
        canvas,
        "LOADED PHOTO",
        right_x + COMPARISON_IMAGE_WIDTH // 2,
        55,
        0.75,
        (40, 40, 40),
    )

    template_angle = (
        f"Yaw {template_pose['yaw']:.1f}  "
        f"Pitch {template_pose['pitch']:.1f}  "
        f"Roll {template_pose['roll']:.1f}"
    )
    loaded_angle = (
        f"Yaw {loaded_pose['yaw']:.1f}  "
        f"Pitch {loaded_pose['pitch']:.1f}  "
        f"Roll {loaded_pose['roll']:.1f}"
    )
    put_centered_text(
        canvas,
        template_angle,
        left_x + COMPARISON_IMAGE_WIDTH // 2,
        485,
        0.50,
        (50, 50, 50),
    )
    put_centered_text(
        canvas,
        loaded_angle,
        right_x + COMPARISON_IMAGE_WIDTH // 2,
        485,
        0.50,
        (50, 50, 50),
    )

    final_text = base.format_similarity(final_similarity)
    full_text = base.format_similarity(full_similarity)
    trusted_text = base.format_similarity(trusted_similarity)
    result_text = "MATCH" if recognized else "NOT MATCH"

    put_centered_text(
        canvas,
        f"RESULT: {result_text}    FINAL: {final_text}",
        COMPARISON_CANVAS_WIDTH // 2,
        535,
        0.82,
        result_color,
        2,
    )
    put_centered_text(
        canvas,
        f"Full512: {full_text}    Trusted: {trusted_text}    Mask: {mask_result['status']}",
        COMPARISON_CANVAS_WIDTH // 2,
        575,
        0.62,
        (40, 40, 40),
        2,
    )
    put_centered_text(
        canvas,
        f"Source: {loaded_path.name}",
        COMPARISON_CANVAS_WIDTH // 2,
        615,
        0.50,
        (80, 80, 80),
        1,
    )
    return canvas


def save_comparison_image(comparison_image, loaded_path):
    output_directory = base.PROJECT_ROOT / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in loaded_path.stem
    )[:50]
    output_path = output_directory / f"main5_compare_{safe_stem}_{timestamp}.jpg"
    success = write_image_unicode(
        output_path,
        comparison_image,
        jpeg_quality=COMPARISON_JPEG_QUALITY,
    )
    if not success:
        raise RuntimeError(f"比較圖片儲存失敗：{output_path}")
    return output_path


def compare_loaded_photo(
    image_path,
    detector,
    arcface,
    mask_detector,
    reference,
):
    if not reference.ready or reference.template_image is None:
        print("尚未建立正面模板，請先面向攝影機收集模板")
        return None

    image = read_image_unicode(image_path)
    if image is None:
        print(f"無法讀取照片：{image_path}")
        return None

    faces = detector.detect(image)
    if not faces:
        print("載入照片中未偵測到人臉")
        return None

    selected_face = max(faces, key=lambda item: base.bbox_area(item["bbox"]))
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = base.clamp_bbox(
        selected_face["bbox"],
        image_width,
        image_height,
    )
    if x2 <= x1 or y2 <= y1:
        print("載入照片的人臉框無效")
        return None

    face_crop = image[y1:y2, x1:x2]
    blur_result = evaluate_blur(
        face_crop,
        laplacian_threshold=base.LAPLACIAN_THRESHOLD,
        tenengrad_threshold=base.TENENGRAD_THRESHOLD,
    )
    mask_result = mask_detector.detect(face_crop)
    pose_result = estimate_head_pose(
        landmarks=selected_face["landmarks"],
        frame_width=image_width,
        frame_height=image_height,
    )

    if selected_face["landmarks"] is None:
        print("載入照片沒有可用的五點Landmark")
        return None
    if not pose_result.get("success", False):
        print(f"載入照片角度估算失敗：{pose_result.get('error')}")
        return None

    aligned_face, embedding = arcface.extract(
        frame=image,
        landmarks=selected_face["landmarks"],
    )
    (
        full_similarity,
        trusted_similarity,
        final_similarity,
        recognized,
    ) = reference.compare(embedding)

    comparison_image = build_comparison_image(
        template_image=reference.template_image,
        template_pose=reference.template_pose,
        loaded_image=aligned_face,
        loaded_pose=pose_result,
        full_similarity=full_similarity,
        trusted_similarity=trusted_similarity,
        final_similarity=final_similarity,
        recognized=recognized,
        loaded_path=image_path,
        mask_result=mask_result,
    )
    output_path = save_comparison_image(comparison_image, image_path)
    cv2.imshow(COMPARISON_WINDOW_NAME, comparison_image)

    print(f"比較圖片已儲存：{output_path}")
    print(
        "照片品質："
        f"Laplacian={blur_result['laplacian']:.1f}, "
        f"Tenengrad={blur_result['tenengrad']:.1f}, "
        f"Mask={mask_result['status']}"
    )
    print(
        "比較分數："
        f"Full512={base.format_similarity(full_similarity)}, "
        f"Trusted={base.format_similarity(trusted_similarity)}, "
        f"Final={base.format_similarity(final_similarity)}, "
        f"Result={'MATCH' if recognized else 'NOT MATCH'}"
    )
    return output_path


def save_live_snapshot_async(frame):
    snapshot = frame.copy()
    output_directory = base.PROJECT_ROOT / "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_directory / f"main5_{timestamp}.jpg"

    def writer():
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            if write_image_unicode(output_path, snapshot):
                print(f"即時畫面已儲存：{output_path}")
            else:
                print(f"即時畫面儲存失敗：{output_path}")
        except Exception as error:
            print(f"即時畫面儲存發生錯誤：{error}")

    threading.Thread(
        target=writer,
        name="main5-snapshot-writer",
        daemon=True,
    ).start()


def draw_main5_reference_panel(frame, reference):
    x1, y1, x2, y2 = 10, 180, 560, 410
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    state = "READY" if reference.ready else "COLLECT FRONT FACE"
    template_state = "READY" if reference.template_image is not None else "EMPTY"
    color = (0, 255, 0) if reference.ready else (0, 255, 255)
    lines = [
        f"Front reference: {state}",
        f"Template photo: {template_state}",
        f"Front samples: {reference.sample_count}/{base.MAX_REFERENCE_SAMPLES}",
        (
            f"Trusted dimensions: {reference.trusted_dimension_count}/512 "
            f"({'READY' if reference.trusted_ready else 'LEARNING'})"
        ),
        f"Last full: {base.format_similarity(reference.last_full_similarity)}",
        f"Last trusted: {base.format_similarity(reference.last_trusted_similarity)}",
        f"Last final: {base.format_similarity(reference.last_final_similarity)}",
        f"Last result: {reference.last_result}",
        "L: Load photo   R: Clear   S: Screenshot   ESC: Exit",
    ]
    base.draw_text_lines(frame, 20, 205, lines, color, scale=0.53)


def main():
    detector = ScrfdOnnxDetector(
        model_path=base.SCRFD_MODEL_PATH,
        input_size=base.DETECTION_SIZE,
        detection_threshold=base.DETECTION_THRESHOLD,
    )
    arcface = ArcFaceEmbedder(model_path=base.ARCFACE_MODEL_PATH)
    mask_detector = FaceMaskHeuristicDetector(
        mask_threshold=base.MASK_HEURISTIC_THRESHOLD,
    )
    reference = PhotoReferenceBuffer()

    capture = cv2.VideoCapture(base.CAMERA_INDEX, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, base.CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, base.CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, base.CAMERA_FPS)
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
                key=lambda index: base.bbox_area(faces[index]["bbox"]),
            )

        run_arcface = frame_counter % base.ARCFACE_INTERVAL == 0

        for face_index, face in enumerate(faces):
            is_primary = face_index == primary_index
            x1, y1, x2, y2 = base.clamp_bbox(
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
                laplacian_threshold=base.LAPLACIAN_THRESHOLD,
                tenengrad_threshold=base.TENENGRAD_THRESHOLD,
            )
            mask_result = mask_detector.detect(face_crop)

            size_accepted = (
                face_width >= base.MIN_FACE_WIDTH
                and face_height >= base.MIN_FACE_HEIGHT
            )
            blur_accepted = not blur_result["is_blurry"]
            pose_result = estimate_head_pose(
                landmarks=face["landmarks"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            pose_accepted = is_pose_accepted(
                pose_result,
                max_yaw=base.MATCH_MAX_YAW,
                max_pitch=base.MATCH_MAX_PITCH,
                max_roll=base.MATCH_MAX_ROLL,
            )
            strict_front = base.is_strict_front_pose(pose_result)
            quality_accepted = (
                size_accepted
                and blur_accepted
                and pose_accepted
                and face["landmarks"] is not None
            )

            reasons = []
            if not size_accepted:
                reasons.append("SMALL")
            if not blur_accepted:
                reasons.append("BLUR")
            if not pose_accepted:
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
                aligned_face, embedding = arcface.extract(
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
                            and full_similarity >= base.REFERENCE_ENROLL_THRESHOLD
                        )
                    )
                    if identity_verified:
                        if reference.add_photo_sample(
                            embedding,
                            aligned_face,
                            pose_result,
                        ):
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
            base.draw_landmarks(frame, face["landmarks"])

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
                f"Full512:{base.format_similarity(full_similarity)}",
                f"Trusted:{base.format_similarity(trusted_similarity)}",
                f"Final:{base.format_similarity(final_similarity)} {result_text}",
                f"Reason:{reason_text}",
            ]
            if arcface_ms > 0.0:
                lines.append(f"ArcFace:{arcface_ms:.1f}ms")
            if action:
                lines.append(action)

            text_y = max(25, y1 - (len(lines) - 1) * 21 - 8)
            base.draw_text_lines(frame, x1, text_y, lines, box_color)

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            display_fps = display_fps * 0.9 + instant_fps * 0.1
        previous_time = current_time

        status_lines = [
            "SCRFD + ArcFace photo comparison",
            f"Faces:{len(faces)} FPS:{display_fps:.1f}",
            f"Detection:{detection_ms:.1f}ms",
            (
                f"Full threshold:{base.FULL_COSINE_THRESHOLD:.2f} "
                f"Trusted:{base.TRUSTED_COSINE_THRESHOLD:.2f}"
            ),
            "Press L to load and compare a photo",
        ]
        base.draw_text_lines(frame, 20, 30, status_lines, (0, 255, 255), scale=0.55)
        draw_main5_reference_panel(frame, reference)

        cv2.imshow("Kiosk Photo Comparison - main5", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            reference.clear()
            print("正面模板與可信特徵暫存區已清除")
        if key in (ord("s"), ord("S")):
            save_live_snapshot_async(frame)
        if key in (ord("l"), ord("L")):
            selected_path = select_image_file()
            if selected_path is not None:
                compare_loaded_photo(
                    image_path=selected_path,
                    detector=detector,
                    arcface=arcface,
                    mask_detector=mask_detector,
                    reference=reference,
                )

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
