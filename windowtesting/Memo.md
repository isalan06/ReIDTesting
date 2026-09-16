<h1> 開發隨記 </h1>

# 安裝環境

## folder path

```PowerShell

cd "C:\Users\alanh\OneDrive\Alan\git program\ReIDTesting"

```

## python

系統已經安裝環境，若尚未安裝環境請先安裝

```PowerShell

python --version
=> Python 3.11.9
pip --version
=> pip 24.0 from C:\Program Files\Python311\Lib\site-packages\pip (python 3.11)

```

# 測試程式操作步驟

## 建立虛擬環境

1. 開啟 Visual Studio Code中的Terminal
2. 進入測試資料夾
```PowerShell

cd windowtesting

```
3. 建立虛擬環境
```PowerShell

python -m venv .venv
=> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "c:\Users\alanh\OneDrive\Alan\git program\ReIDTesting\windowtesting\.venv\Scripts\Activate.ps1")

```

## 測試第一個元件

4. 安裝套件，參考 [階段四：先安裝基礎套件](Reference.md#階段四先安裝基礎套件)

```PowerShell

python -m pip install --upgrade pip
pip install numpy scipy opencv-python pyyaml

```

5. 執行參考 [階段五：完成影片／Webcam讀取](Reference.md#階段五完成影片webcam讀取)

6. 執行參考 [階段六：開發模糊判斷](Reference.md#階段六開發模糊判斷)

7. 執行參考 [階段七：加入ONNX與OpenVINO](Reference.md#階段七加入onnx與openvino)

8. 執行參考 [階段八：加入SCRFD人臉偵測](Reference.md#階段八加入scrfd人臉偵測)

    - 執行方法A
    ```PowerShell

    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')" 

    ```
