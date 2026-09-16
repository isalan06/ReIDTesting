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