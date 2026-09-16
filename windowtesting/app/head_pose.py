import cv2
import numpy as np


# =========================================================
# SCRFD五點Landmark所對應的簡化3D人臉模型
# =========================================================
#
# SCRFD Landmark順序：
# 0：左眼
# 1：右眼
# 2：鼻尖
# 3：左嘴角
# 4：右嘴角
#
# 此模型只適合PoC階段進行正面／側臉篩選，
# 並不是精密的3D頭部角度量測模型。
# =========================================================

MODEL_POINTS = np.array(
    [
        [-30.0, 35.0, -30.0],   # 左眼
        [30.0, 35.0, -30.0],    # 右眼
        [0.0, 0.0, 0.0],        # 鼻尖
        [-25.0, -30.0, -20.0],  # 左嘴角
        [25.0, -30.0, -20.0],   # 右嘴角
    ],
    dtype=np.float64,
)


def create_failed_result(error_message=None):
    """
    建立頭部姿態估算失敗結果。

    即使單一影格的Landmark或solvePnP發生錯誤，
    主程式仍可繼續執行。
    """
    return {
        "success": False,
        "pitch": 0.0,
        "raw_pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "rotation_vector": None,
        "translation_vector": None,
        "error": error_message,
    }


def normalize_front_pitch(raw_pitch):
    """
    將原始Pitch的±180度正面方向轉換為0度。

    OpenCV solvePnP搭配目前的簡化3D模型時，
    正面可能位於Pitch的±180度附近。

    原始數值可能如下：

        160 -> 170 -> 180 -> -180 -> -170 -> -160

    轉換後變成：

        -20 -> -10 -> 0 -> 0 -> 10 -> 20

    轉換範例：

        原始  160度 -> 修正後 -20度
        原始  170度 -> 修正後 -10度
        原始  180度 -> 修正後   0度
        原始 -180度 -> 修正後   0度
        原始 -170度 -> 修正後  10度
        原始 -160度 -> 修正後  20度
    """
    normalized = (
        float(raw_pitch) % 360.0
    ) - 180.0

    # 避免畫面顯示-0.0
    if abs(normalized) < 0.0001:
        normalized = 0.0

    return normalized


def normalize_standard_angle(angle):
    """
    將一般角度限制於-180至180度之間。

    目前主要用於Yaw與Roll的基本正規化。
    """
    normalized = (
        float(angle) + 180.0
    ) % 360.0 - 180.0

    if abs(normalized) < 0.0001:
        normalized = 0.0

    return normalized


def estimate_head_pose(
    landmarks,
    frame_width,
    frame_height,
):
    """
    使用SCRFD五點Landmark近似估算頭部姿態。

    Parameters
    ----------
    landmarks:
        SCRFD回傳的五點Landmark，格式應為：

        [
            [left_eye_x, left_eye_y],
            [right_eye_x, right_eye_y],
            [nose_x, nose_y],
            [left_mouth_x, left_mouth_y],
            [right_mouth_x, right_mouth_y],
        ]

    frame_width:
        原始影像寬度。

    frame_height:
        原始影像高度。

    Returns
    -------
    dict:
        {
            "success": bool,
            "pitch": 修正後Pitch,
            "raw_pitch": OpenCV原始Pitch,
            "yaw": Yaw,
            "roll": Roll,
            "rotation_vector": 旋轉向量,
            "translation_vector": 平移向量,
            "error": 錯誤訊息
        }
    """

    # -----------------------------------------------------
    # 1. 檢查Landmark
    # -----------------------------------------------------

    if landmarks is None:
        return create_failed_result(
            "Landmark不存在"
        )

    try:
        image_points = np.asarray(
            landmarks,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        return create_failed_result(
            f"Landmark無法轉換：{error}"
        )

    if image_points.shape != (5, 2):
        return create_failed_result(
            "Landmark格式錯誤，"
            f"預期(5, 2)，實際{image_points.shape}"
        )

    if not np.all(np.isfinite(image_points)):
        return create_failed_result(
            "Landmark包含NaN或Infinity"
        )

    # -----------------------------------------------------
    # 2. 檢查影像尺寸
    # -----------------------------------------------------

    if frame_width is None or frame_height is None:
        return create_failed_result(
            "影像尺寸不存在"
        )

    if frame_width <= 0 or frame_height <= 0:
        return create_failed_result(
            "影像尺寸必須大於0"
        )

    # -----------------------------------------------------
    # 3. 建立相機內部參數
    # -----------------------------------------------------
    #
    # PoC階段尚未進行實際相機校正，
    # 暫時以影像寬度作為焦距近似值。
    #
    # 正式版本建議使用相機校正後的：
    # - fx
    # - fy
    # - cx
    # - cy
    # - distortion coefficients
    # -----------------------------------------------------

    focal_length = float(frame_width)

    camera_matrix = np.array(
        [
            [
                focal_length,
                0.0,
                frame_width / 2.0,
            ],
            [
                0.0,
                focal_length,
                frame_height / 2.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    # PoC階段假設沒有鏡頭畸變
    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # 4. 使用SQPnP計算頭部姿態
    # -----------------------------------------------------
    #
    # 不使用SOLVEPNP_ITERATIVE。
    #
    # OpenCV 5的ITERATIVE方法可能先用DLT初始化，
    # DLT至少需要6個3D-2D對應點。
    #
    # SCRFD只有5個Landmark，因此使用SQPnP。
    # SQPnP支援3點以上。
    # -----------------------------------------------------

    try:
        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                objectPoints=MODEL_POINTS,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        )

    except cv2.error as error:
        return create_failed_result(
            f"solvePnP發生OpenCV錯誤：{error}"
        )

    if not success:
        return create_failed_result(
            "solvePnP回傳失敗"
        )

    if rotation_vector is None:
        return create_failed_result(
            "solvePnP未回傳rotation vector"
        )

    if translation_vector is None:
        return create_failed_result(
            "solvePnP未回傳translation vector"
        )

    # -----------------------------------------------------
    # 5. 將旋轉向量轉換成Euler Angles
    # -----------------------------------------------------

    try:
        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        decomposition_result = (
            cv2.RQDecomp3x3(
                rotation_matrix
            )
        )

        angles = decomposition_result[0]

        raw_pitch = float(angles[0])
        raw_yaw = float(angles[1])
        raw_roll = float(angles[2])

    except (cv2.error, TypeError, ValueError) as error:
        return create_failed_result(
            f"角度轉換失敗：{error}"
        )

    # -----------------------------------------------------
    # 6. 修正角度範圍
    # -----------------------------------------------------

    # 目前3D模型的正面落在Pitch約±180度，
    # 將±180度重新映射成正面0度。
    pitch = normalize_front_pitch(
        raw_pitch
    )

    yaw = normalize_standard_angle(
        raw_yaw
    )

    roll = normalize_standard_angle(
        raw_roll
    )

    # -----------------------------------------------------
    # 7. 檢查結果
    # -----------------------------------------------------

    angle_values = np.array(
        [
            raw_pitch,
            pitch,
            yaw,
            roll,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(angle_values)):
        return create_failed_result(
            "姿態角度包含無效數值"
        )

    return {
        "success": True,
        "pitch": pitch,
        "raw_pitch": raw_pitch,
        "yaw": yaw,
        "roll": roll,
        "rotation_vector": rotation_vector,
        "translation_vector": translation_vector,
        "error": None,
    }


def is_pose_accepted(
    pose_result,
    max_yaw=30.0,
    max_pitch=25.0,
    max_roll=20.0,
):
    """
    判斷頭部姿態是否適合執行ArcFace。

    預設門檻：
    - Yaw：±30度
    - Pitch：±25度
    - Roll：±20度
    """

    if pose_result is None:
        return False

    if not pose_result.get(
        "success",
        False,
    ):
        return False

    try:
        yaw = float(pose_result["yaw"])
        pitch = float(pose_result["pitch"])
        roll = float(pose_result["roll"])

    except (KeyError, TypeError, ValueError):
        return False

    if not np.all(
        np.isfinite(
            [
                yaw,
                pitch,
                roll,
            ]
        )
    ):
        return False

    yaw_accepted = (
        abs(yaw) <= max_yaw
    )

    pitch_accepted = (
        abs(pitch) <= max_pitch
    )

    roll_accepted = (
        abs(roll) <= max_roll
    )

    return (
        yaw_accepted
        and pitch_accepted
        and roll_accepted
    )