<h1> 參考 </h1>

# 階段一：安裝開發工具
1. 安裝 Python

建議使用：

Python 3.11 64-bit

安裝時勾選：

Add python.exe to PATH

安裝完成後，在 PowerShell 確認：

python --version
pip --version

預期顯示：

Python 3.11.x

2. 安裝 VS Code 擴充套件

在 VS Code Extensions 安裝：

Python，Microsoft
Python Environments，Microsoft
Pylance，Microsoft
Jupyter，Microsoft，可選
GitLens，可選

VS Code 官方目前可以直接建立、切換和管理 .venv 虛擬環境。VS Code Python environments

# 階段二：建立專案

建立資料夾，例如：

D:\Projects\KioskDynamicReID

使用 VS Code：

File → Open Folder → D:\Projects\KioskDynamicReID

建議專案結構：

KioskDynamicReID/
├─ app/
│  ├─ main.py
│  ├─ video_source.py
│  ├─ face_detector.py
│  ├─ face_quality.py
│  ├─ face_pose.py
│  ├─ face_embedding.py
│  ├─ face_tracker.py
│  ├─ feature_fusion.py
│  └─ mask_detector.py
├─ config/
│  └─ settings.yaml
├─ models/
├─ videos/
├─ output/
├─ tests/
├─ requirements.txt
└─ README.md

第一階段不用把所有檔案都實作，但先保留此結構。

# 階段三：建立虛擬環境

在 VS Code 開啟 Terminal：

Terminal → New Terminal

執行：

python -m venv .venv

啟用環境：

.\.venv\Scripts\Activate.ps1

如果 PowerShell 阻擋啟用，可以只對目前終端放寬：

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

成功後終端前面會顯示：

(.venv)

接著在 VS Code 按：

Ctrl + Shift + P
Python: Select Interpreter

選擇：

.\.venv\Scripts\python.exe

# 階段四：先安裝基礎套件

第一批只安裝影像與數值運算：

python -m pip install --upgrade pip
pip install numpy scipy opencv-python pyyaml

建立 requirements.txt：

numpy
scipy
opencv-python
pyyaml

確認套件：

python -c "import cv2; import numpy; print('OpenCV:', cv2.__version__)"

# 階段五：完成影片／Webcam讀取

先建立 app/main.py：

```python
import cv2


def main():
    # 0：筆電內建攝影機
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("無法開啟攝影機")

    while True:
        success, frame = capture.read()

        if not success:
            break

        cv2.putText(
            frame,
            f"Size: {frame.shape[1]}x{frame.shape[0]}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Kiosk Dynamic Re-ID Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```

執行：

```PowerShell

python app/main.py

```

按 Esc 離開。

如果要測試 MP4：

```python

capture = cv2.VideoCapture("videos/test.mp4")

```

本階段驗收
可以開啟筆電攝影機。
可以讀取 MP4。
畫面沒有嚴重延遲。
可以正常離開程式。

# 階段六：開發模糊判斷

建立 app/face_quality.py：

```python

import cv2
import numpy as np


def resize_gray(image, size=(112, 112)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size)


def laplacian_score(image):
    gray = resize_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_score(image):
    gray = resize_gray(image)

    gradient_x = cv2.Sobel(
        gray, cv2.CV_64F, 1, 0, ksize=3
    )
    gradient_y = cv2.Sobel(
        gray, cv2.CV_64F, 0, 1, ksize=3
    )

    score = np.mean(
        gradient_x * gradient_x +
        gradient_y * gradient_y
    )

    return float(score)

```

第一版可以先對整張畫面計算；加入 SCRFD 後，再改成只計算人臉 Crop。

本階段測試影片

各錄製約10～20秒：

靜止正面
正常行走
快速移動
左右搖頭
故意失焦
光線不足
本階段驗收

畫面顯示：

Laplacian: 185.3
Tenengrad: 2345.7
Quality: PASS

先收集數據，不要急著決定正式門檻。

## 擴充修改

ioskDynamicReID/
└─ app/
   ├─ main.py
   └─ face_quality.py

1. face_quality.py

```python

import cv2
import numpy as np


def resize_gray(image, size=(112, 112)):
    """轉換成固定大小的灰階影像。"""
    if image is None or image.size == 0:
        raise ValueError("輸入影像不可為空")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size)


def laplacian_score(image):
    """計算 Laplacian 清晰度分數，通常數值越高越清楚。"""
    gray = resize_gray(image)

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return float(score)


def tenengrad_score(image):
    """計算 Tenengrad 清晰度分數，通常數值越高越清楚。"""
    gray = resize_gray(image)

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    score = np.mean(
        gradient_x * gradient_x
        + gradient_y * gradient_y
    )

    return float(score)


def evaluate_blur(
    image,
    laplacian_threshold=80.0,
    tenengrad_threshold=1000.0,
):
    """
    影像只要同時低於兩個門檻，就判定為模糊。

    目前門檻僅供程式測試，之後需要使用實際攝影機校正。
    """
    laplacian = laplacian_score(image)
    tenengrad = tenengrad_score(image)

    is_blurry = (
        laplacian < laplacian_threshold
        and tenengrad < tenengrad_threshold
    )

    return {
        "laplacian": laplacian,
        "tenengrad": tenengrad,
        "is_blurry": is_blurry,
    }
```

2. 修改 main.py

```python

import cv2

from face_quality import evaluate_blur


def draw_blur_result(frame, result, fps):
    is_blurry = result["is_blurry"]

    if is_blurry:
        state_text = "BLUR"
        state_color = (0, 0, 255)
    else:
        state_text = "CLEAR"
        state_color = (0, 255, 0)

    cv2.putText(
        frame,
        f"Laplacian: {result['laplacian']:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Tenengrad: {result['tenengrad']:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Quality: {state_text}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        state_color,
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def main():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("無法開啟攝影機")

    tick_frequency = cv2.getTickFrequency()
    previous_tick = cv2.getTickCount()

    while True:
        success, frame = capture.read()

        if not success:
            print("無法取得攝影機影像")
            break

        # 現階段先對整張畫面計算。
        blur_result = evaluate_blur(frame)

        current_tick = cv2.getTickCount()
        elapsed = (
            current_tick - previous_tick
        ) / tick_frequency

        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        previous_tick = current_tick

        draw_blur_result(
            frame,
            blur_result,
            fps,
        )

        cv2.imshow(
            "Kiosk Dynamic Re-ID Test",
            frame,
        )

        # Esc 結束
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```

# 階段七：加入ONNX與OpenVINO

安裝：

```PowerShell

pip install onnx onnxruntime openvino

```

確認 ONNX Runtime：

```PowerShell

python -c "import onnxruntime as ort; print(ort.get_available_providers())"

```

一般會先看到：

['CPUExecutionProvider']

確認 OpenVINO：

```PowerShell

python -c "from openvino import Core; print(Core().available_devices)"

```

可能顯示：

['CPU', 'GPU', 'NPU']

實際裝置依驅動程式與模型支援狀態為準。先從 CPU 開始，整個功能穩定後再測 GPU／NPU。OpenVINO 官方支援透過 PyPI 安裝。OpenVINO 安裝說明

# 階段八：加入SCRFD人臉偵測

有兩種方法。

## 方法A：InsightFace快速測試

```PowerShell

pip install insightface

```

第一版設定：

```python

from insightface.app import FaceAnalysis


face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
)

```

取得每張臉的：

Bounding box
五點 Landmark
Detection score
ArcFace embedding

這是最快看到效果的方式。

## 方法B：SCRFD ONNX獨立執行

後續正式架構建議改成：

SCRFD ONNX
   ↓
ONNX Runtime／OpenVINO
   ↓
Bounding Box＋Landmark

這樣未來移植到 TensorRT 比較容易。

本階段驗收

畫面至少顯示：

Face ID: Temporary
Detection: 0.97
Face size: 164 px
Blur: 182

## 補充方法
把兩種方法各自封裝成偵測器，輸出相同格式，然後在 main.py 只切換一個參數：

```python

DETECTOR_TYPE = "insightface"

```

或：

```python

DETECTOR_TYPE = "scrfd_onnx"

```

這兩種方法的差別：

方法	實作方式	適合用途
A	FaceAnalysis 整合介面	快速完成 SCRFD＋後續 ArcFace
B	直接載入 det_10g.onnx	單獨測試 SCRFD、掌握模型與效能

以下先使用 CPU，方便在 ASUS VivoBook 上穩定驗證。

一、專案結構

調整為：

KioskDynamicReID/
├─ app/
│  ├─ main.py
│  ├─ face_quality.py
│  ├─ face_detector_insightface.py
│  └─ face_detector_scrfd.py
├─ models/
├─ videos/
├─ output/
└─ requirements.txt
二、安裝套件

在 VS Code 終端確認已啟用 .venv：

```PowerShell

.\.venv\Scripts\Activate.ps1

```

安裝：

```PowerShell

python -m pip install --upgrade pip
pip install numpy scipy opencv-python onnx onnxruntime
pip install insightface

```

確認：

```PowerShell

python -c "import insightface; import onnxruntime; print('InsightFace OK')"

```

如果 insightface 安裝時出現 C++ 編譯錯誤，通常需要安裝 Microsoft C++ Build Tools；若沒有錯誤就不必安裝。

InsightFace 官方的 SCRFD 實作會使用 ONNX Runtime 執行模型。InsightFace SCRFD；ONNX Runtime 也提供 Windows Python CPU 套件。ONNX Runtime Python

三、共用的輸出格式

兩種偵測器都統一回傳：

```json

[
    {
        "bbox": [x1, y1, x2, y2],
        "score": 0.98,
        "landmarks": [
            [left_eye_x, left_eye_y],
            [right_eye_x, right_eye_y],
            [nose_x, nose_y],
            [left_mouth_x, left_mouth_y],
            [right_mouth_x, right_mouth_y],
        ],
    }
]

```

如此 main.py 不需要知道底層是哪一種方法。

四、方法A：FaceAnalysis整合方式

新增：

```PowerShell

app/face_detector_insightface.py

內容如下：

```python

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

```

第一次執行的行為

第一次使用：



FaceAnalysis(name="buffalo_l")

通常會自動下載模型到：

C:\Users\你的帳號\.insightface\models\buffalo_l\

裡面可能包含：

det_10g.onnx
w600k_r50.onnx
1k3d68.onnx
2d106det.onnx
genderage.onnx

其中：

det_10g.onnx

就是 SCRFD 人臉偵測模型。

第一種方法目前只取：

人臉框
Detection Score
五點 Landmark

雖然 buffalo_l 同時會載入其他模型，但階段八先不使用 ArcFace 特徵值。

五、方法B：直接載入SCRFD ONNX

方法B不使用 FaceAnalysis，而是直接指定 det_10g.onnx。

新增：

app/face_detector_scrfd.py

內容：

```python

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

```

雖然這裡仍利用 InsightFace 官方的 get_model() 解碼 SCRFD 輸出，但和方法A不同：

不使用 FaceAnalysis
不載入 ArcFace等其他模型
明確指定 SCRFD ONNX檔
只執行人臉偵測
更接近未來獨立轉換 TensorRT 的架構
六、準備SCRFD模型

先執行一次方法A，讓 InsightFace 下載 buffalo_l。

模型通常位於：

C:\Users\你的帳號\.insightface\models\buffalo_l\det_10g.onnx

可以在 PowerShell 查看：

```PowerShell

Get-ChildItem "$env:USERPROFILE\.insightface\models\buffalo_l"

```

把：

det_10g.onnx

複製到專案：

KioskDynamicReID\models\det_10g.onnx

完成後：

KioskDynamicReID/
└─ models/
   └─ det_10g.onnx

如果尚未執行方法A，方法B就會因為沒有 ONNX 模型而無法啟動。

七、沿用階段六的 face_quality.py

app/face_quality.py 保持如下：

```python

import cv2
import numpy as np


def resize_gray(image, size=(112, 112)):
    if image is None or image.size == 0:
        raise ValueError("輸入影像不可為空")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.resize(
        gray,
        size,
    )


def laplacian_score(image):
    gray = resize_gray(image)

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def tenengrad_score(image):
    gray = resize_gray(image)

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    score = np.mean(
        gradient_x * gradient_x
        + gradient_y * gradient_y
    )

    return float(score)


def evaluate_blur(
    image,
    laplacian_threshold=80.0,
    tenengrad_threshold=1000.0,
):
    laplacian = laplacian_score(image)
    tenengrad = tenengrad_score(image)

    is_blurry = (
        laplacian < laplacian_threshold
        and tenengrad < tenengrad_threshold
    )

    return {
        "laplacian": laplacian,
        "tenengrad": tenengrad,
        "is_blurry": is_blurry,
    }

```

八、修改 main.py

以下 main.py 可以切換兩種方法。

```python

import time
from pathlib import Path

import cv2

from face_quality import evaluate_blur
from face_detector_insightface import (
    InsightFaceDetector,
)
from face_detector_scrfd import ScrfdOnnxDetector


# ==========================================
# 切換偵測方法
# ==========================================
# 方法A：
DETECTOR_TYPE = "insightface"

# 方法B：
# DETECTOR_TYPE = "scrfd_onnx"


CAMERA_INDEX = 0
DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRFD_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "det_10g.onnx"
)


def create_detector():
    if DETECTOR_TYPE == "insightface":
        return InsightFaceDetector(
            model_name="buffalo_l",
            detection_size=DETECTION_SIZE,
            detection_threshold=DETECTION_THRESHOLD,
        )

    if DETECTOR_TYPE == "scrfd_onnx":
        return ScrfdOnnxDetector(
            model_path=SCRFD_MODEL_PATH,
            input_size=DETECTION_SIZE,
            detection_threshold=DETECTION_THRESHOLD,
        )

    raise ValueError(
        f"未知的偵測器類型：{DETECTOR_TYPE}"
    )


def clamp_bbox(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    return x1, y1, x2, y2


def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return

    colors = [
        (255, 0, 0),      # 左眼
        (0, 255, 0),      # 右眼
        (0, 0, 255),      # 鼻子
        (255, 255, 0),    # 左嘴角
        (255, 0, 255),    # 右嘴角
    ]

    for index, point in enumerate(landmarks):
        x = int(point[0])
        y = int(point[1])

        color = colors[
            min(index, len(colors) - 1)
        ]

        cv2.circle(
            frame,
            (x, y),
            3,
            color,
            -1,
        )


def draw_face_result(
    frame,
    bbox,
    detection_score,
    blur_result,
    face_index,
):
    x1, y1, x2, y2 = bbox

    if blur_result["is_blurry"]:
        state = "BLUR"
        color = (0, 0, 255)
    else:
        state = "CLEAR"
        color = (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    face_width = x2 - x1
    face_height = y2 - y1

    text_lines = [
        (
            f"Face {face_index} "
            f"Score:{detection_score:.2f}"
        ),
        (
            f"Size:{face_width}x{face_height}"
        ),
        (
            f"Lap:{blur_result['laplacian']:.1f}"
        ),
        (
            f"Ten:{blur_result['tenengrad']:.1f}"
        ),
        (
            f"Quality:{state}"
        ),
    ]

    text_y = max(25, y1 - 10)

    for line_index, text in enumerate(text_lines):
        y = text_y + line_index * 22

        cv2.putText(
            frame,
            text,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def main():
    detector = create_detector()

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

    previous_time = time.perf_counter()
    display_fps = 0.0

    while True:
        success, frame = capture.read()

        if not success:
            print("無法取得攝影機影像")
            break

        frame_height, frame_width = frame.shape[:2]

        inference_start = time.perf_counter()

        faces = detector.detect(frame)

        inference_elapsed = (
            time.perf_counter()
            - inference_start
        )

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

            face_crop = frame[
                y1:y2,
                x1:x2,
            ]

            if face_crop.size == 0:
                continue

            blur_result = evaluate_blur(
                face_crop
            )

            draw_face_result(
                frame=frame,
                bbox=bbox,
                detection_score=face["score"],
                blur_result=blur_result,
                face_index=face_index,
            )

            draw_landmarks(
                frame,
                face["landmarks"],
            )

        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            # FPS平滑，避免畫面數值跳動太大
            display_fps = (
                display_fps * 0.9
                + instant_fps * 0.1
            )

        previous_time = current_time

        status_lines = [
            f"Detector: {DETECTOR_TYPE}",
            f"Faces: {len(faces)}",
            f"FPS: {display_fps:.1f}",
            (
                f"Inference: "
                f"{inference_elapsed * 1000:.1f} ms"
            ),
            "ESC: Exit",
        ]

        for index, text in enumerate(
            status_lines
        ):
            cv2.putText(
                frame,
                text,
                (20, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow(
            "Kiosk Face Quality Test",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```

九、測試方法A

確認 main.py：

```python

DETECTOR_TYPE = "insightface"

```

執行：

```PowerShell

python app/main.py

```

第一次可能需要等待模型下載。

應該看到：

Detector: insightface
Faces: 1
FPS: ...
Inference: ... ms

人臉上應顯示：

Face 1 Score:0.98
Size:180x195
Lap:125.3
Ten:1832.5
Quality:CLEAR

以及五個 Landmark 點。

十、測試方法B

先確認模型存在：

```PowerShell

Test-Path .\models\det_10g.onnx

```

預期：

True

修改：

```python

# DETECTOR_TYPE = "insightface"
DETECTOR_TYPE = "scrfd_onnx"

```

執行：

python app/main.py

畫面應顯示：

Detector: scrfd_onnx
十一、建議比較項目

同一個位置、同一段時間，分別測試兩種方法：

項目	方法A	方法B
平均FPS	記錄	記錄
平均Inference ms	記錄	記錄
正面偵測	成功／失敗	成功／失敗
左轉30°	成功／失敗	成功／失敗
右轉30°	成功／失敗	成功／失敗
戴口罩	成功／失敗	成功／失敗
快速移動	成功／失敗	成功／失敗
人臉框穩定度	高／中／低	高／中／低
模糊分數	數值	數值

因為兩種方法本質上可能使用同一個 det_10g.onnx，偵測結果應該非常接近。主要差異會是：

方法A同時管理多個人臉模型，方便後續加入ArcFace。
方法B只載入SCRFD，架構較單純，較適合做效能拆分及未來TensorRT移植。
注意事項

目前 Quality:CLEAR/BLUR 的門檻：

laplacian_threshold=80.0
tenengrad_threshold=1000.0

只是暫時測試值。而且每張臉都被縮放到 112×112 後計算，因此下一步應收集清楚、模糊、快速移動、側臉和口罩影像的分數，再決定正式門檻。

另外，InsightFace程式碼與官方提供模型的授權範圍並不完全相同；目前可用於技術PoC，未來商業交付前需要再確認模型授權

# 階段九-十一

## 階段九：模糊與尺寸剔除

SCRFD 找到臉後：

```python

x1, y1, x2, y2 = map(int, face.bbox)

face_crop = frame[y1:y2, x1:x2]
face_width = x2 - x1
blur = laplacian_score(face_crop)

```

初步判斷：

```python

if face_width < 112:
    quality_state = "REJECT_SIZE"
elif blur < blur_reject_threshold:
    quality_state = "REJECT_BLUR"
else:
    quality_state = "ACCEPT"

```

建議先記錄 CSV：

timestamp
frame_number
face_width
face_height
laplacian_score
tenengrad_score
quality_state

錄製至少100～300張不同狀態的人臉後，再設定門檻。

## 階段十：人臉角度判斷

使用 SCRFD 的五個 Landmark：

左眼
右眼
鼻尖
左嘴角
右嘴角

分兩步開發。

第一版：幾何比例

先判斷：

左右眼高度差 → Roll
鼻子偏向左眼或右眼 → Yaw
鼻子與眼睛、嘴巴距離比例 → Pitch

這一版容易完成，適合先驗證。

第二版：solvePnP

使用：

```python

cv2.solvePnP()

```

輸出：

Yaw
Pitch
Roll

初始品質規則：

```python

if abs(yaw) > 35:
    reject_reason = "YAW"

elif abs(pitch) > 25:
    reject_reason = "PITCH"

elif abs(roll) > 25:
    reject_reason = "ROLL"

```

畫面顯示：

Yaw:   12.5
Pitch: -6.8
Roll:   2.3
Pose: ACCEPT

## 階段十一：ArcFace辨識測試

準備兩類照片：

IDV基準照片
動態影片擷取照片

執行：

對齊人臉。
產生512維特徵。
L2 Normalize。
計算 Cosine Similarity。
記錄正確人員與錯誤人員分數。
similarity = np.dot(
    reference_embedding,
    dynamic_embedding,
)

不要一開始把相似度門檻固定為網路建議值。應使用自己的攝影機、距離、補光、角度及口罩資料建立門檻。

本階段驗收
Reference ID: A001
Cosine similarity: 0.63
Quality: ACCEPT
Result: CANDIDATE

## 補充範例
將「階段九～十一」整合成一個可直接測試的版本，沿用目前的方法 B：

SCRFD ONNX
→ 階段九：人臉尺寸＋模糊篩選
→ 階段十：Yaw／Pitch／Roll 角度篩選
→ 階段十一：合格才執行 ArcFace
→ 顯示 512D Embedding

這一版暫時不加入 Tracker、Top-K、多幀融合，先確認單張影格流程正確。

一、階段九～十一測試目標
階段	測試內容	通過條件
九	人臉尺寸、Laplacian、Tenengrad	臉部夠大且不模糊
十	Yaw、Pitch、Roll	頭部角度在允許範圍
十一	ArcFace特徵提取	顯示Embedding: 512D
剔除	過小、模糊或角度過大	顯示拒絕原因

整體判斷流程：

二、專案結構

確認專案結構如下：

KioskDynamicReID/
├─ app/
│  ├─ main.py
│  ├─ face_quality.py
│  ├─ face_detector_scrfd.py
│  ├─ head_pose.py
│  └─ face_embedding_arcface.py
├─ models/
│  ├─ det_10g.onnx
│  └─ w600k_r50.onnx
├─ videos/
└─ output/

前面已經建立的檔案可以繼續使用：

face_quality.py
face_detector_scrfd.py
models/det_10g.onnx

這次新增：

head_pose.py
face_embedding_arcface.py

最後以新的main.py整合測試。

三、準備ArcFace模型

從InsightFace模型目錄：

C:\Users\你的帳號\.insightface\models\buffalo_l\

複製：

w600k_r50.onnx

到：

KioskDynamicReID\models\w600k_r50.onnx

PowerShell可以確認：

Test-Path .\models\det_10g.onnx
Test-Path .\models\w600k_r50.onnx

兩個結果都應為：

True
四、階段十：建立頭部角度模組

新增：

app/head_pose.py

內容如下：

```python

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

```

初始角度門檻
Yaw：±30°
Pitch：±25°
Roll：±20°

這些先作為PoC門檻。由於目前只用五點Landmark，角度是近似值，之後要依實際鏡頭位置重新校正。

五、階段十一：建立ArcFace模組

新增：

app/face_embedding_arcface.py

內容：

```python

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

```

六、整合版main.py

將app/main.py修改如下：

```python

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

```

七、執行測試

在VS Code終端切換到專案根目錄：

cd KioskDynamicReID

啟用虛擬環境：

.\.venv\Scripts\Activate.ps1

執行：

python app/main.py

正常啟動時，終端應看到：

載入 SCRFD ONNX：...
SCRFD ONNX模型載入完成
載入ArcFace模型：...
ArcFace模型載入完成
八、畫面結果說明
1. 完全合格
Size:180x195
Lap:125.3 Ten:1832.5
Yaw:2.5 Pitch:-1.8 Roll:0.9
Quality:ACCEPTED
Reason:NONE
Embedding:512D 18.5ms

代表：

人臉尺寸足夠
清晰度合格
角度合格
已成功執行ArcFace
輸出512維特徵

因為設定：

ARCFACE_INTERVAL = 5

所以不是每一幀都顯示512D。其他合格影格會顯示：

Embedding:READY

這是正常現象。

2. 人臉過小
Quality:REJECTED
Reason:SMALL
Embedding:SKIPPED
3. 人臉模糊
Quality:REJECTED
Reason:BLUR
Embedding:SKIPPED
4. 側臉或角度過大
Quality:REJECTED
Reason:POSE
Embedding:SKIPPED
5. 同時發生多個問題
Reason:SMALL,BLUR,POSE
九、建議測試步驟

建議按以下順序測試，避免一次改太多條件。

測試1：正面靜止

條件：

距離約50～100公分
正面看攝影機
保持靜止
光線充足

預期：

Quality:ACCEPTED
Embedding:512D
測試2：人臉尺寸

慢慢遠離攝影機，觀察：

Size:寬度x高度

當寬或高小於112時，預期：

Reason:SMALL
Embedding:SKIPPED
測試3：移動模糊

快速左右移動頭部，觀察：

Lap
Ten

預期數值下降，並可能顯示：

Reason:BLUR
測試4：左右轉頭

依序測試：

正面
左轉約15°
左轉約30°
左轉超過45°
右側重複測試

觀察Yaw變化。超過約±30°時，預期：

Reason:POSE
測試5：抬頭與低頭

觀察Pitch變化。超過約±25°時應剔除。

測試6：頭部傾斜

頭部向左右肩膀傾斜，觀察Roll。超過約±20°時應剔除。

測試7：ArcFace效能

正面保持不動，記錄：

Embedding:512D xx.xms

建議記錄約20次，再觀察ArcFace平均推論時間。

十、測試紀錄表
測試條件	Size	Laplacian	Tenengrad	Yaw	Pitch	Roll	結果	ArcFace ms
正面50cm								
正面100cm								
正面150cm								
左轉30°								
右轉30°								
抬頭25°								
低頭25°								
快速移動								
光線不足								
戴口罩								
十一、目前版本的限制
五點Landmark的Yaw／Pitch／Roll只是近似值，不是精密3D量測。
尚未加入Tracker，所以每一幀的人臉仍是獨立結果。
ARCFACE_INTERVAL目前是整體影格計數，多人狀態下還不是每個人分別計時。
尚未保存Embedding，也還沒有與基準照片比較。
Laplacian與Tenengrad門檻需要用實際PoE／GMSL攝影機、補光及安裝距離重新校正。

完成這一輪測試後，下一階段才適合加入：

Track ID
→ 每個Track獨立限制ArcFace頻率
→ 保留Top 3～5張Embedding
→ 基準照片Cosine Similarity
→ 多幀加權融合