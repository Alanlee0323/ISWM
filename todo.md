# 部署 ISWM 專案至 Jetson Xavier NX To-Do List

本文件旨在引導您完成將 ISWM 專案從開發環境成功部署到 Jetson Xavier NX 邊緣裝置的整個流程。

**核心策略：** 先在 Jetson 的原生環境中調試並成功運行，再將其封裝到 Docker 容器中，最後進行模型優化以達到最佳性能。

---

## 階段一：原生環境設置與驗證 (Debug on Bare Metal)
*目標：確保程式碼可以在 Jetson 上直接執行，並驗證所有依賴項都已正確安裝。*

- [ ] **1.1. 準備 Jetson 系統環境**
    - [ ] 確認 Jetson Xavier NX 已安裝 JetPack 5.1.2。
    - [ ] 開啟終端機，執行 `sudo apt update && sudo apt upgrade` 更新系統。
    - [ ] 安裝必要的開發工具：`sudo apt install -y git python3.8-venv libopenmpi-dev`

- [ ] **1.2. 建立 Python 虛擬環境**
    - [ ] 在您的家目錄 (home directory) 建立一個專案資料夾。
    - [ ] 建立虛擬環境：`python3 -m venv jetson_env`
    - [ ] 啟用虛擬環境：`source jetson_env/bin/activate`

- [ ] **1.3. 安裝 Python 核心依賴**
    - [ ] 將 `torch-2.1.0a0+...linux_aarch64.whl` 檔案複製到 Jetson 中。
    - [ ] 安裝 PyTorch：`pip install /path/to/your/torch-2.1.0a0+...whl`
    - [ ] 複製 `requirements.txt` 到 Jetson，並嘗試安裝：`pip install -r requirements.txt`

- [ ] **1.4. 複製並測試您的專案**
    - [ ] 將您的 ISWM 專案程式碼 `git clone` 或直接複製到 Jetson。
    - [ ] 將 `best_deeplabv3plus_...pth` 模型檔案放到 `checkpoints` 資料夾。
    - [ ] 執行您的預測腳本 (`predict.py`) 進行一次完整的推論測試。
    - [ ] **目標：** 成功運行並得到預期結果。在此階段解決所有 Python 錯誤。

---

## 階段二：模型量化與優化 (On-Device)
*目標：成功將 DeeplabV3+ PyTorch 模型轉換成 INT8 量化模型，並匯出為 TensorRT `.plan` 檔案。*

### Day 1: PyTorch 量化概念學習
- [ ] **上午：** 重新審視現有模型程式碼 (`network/modeling.py`, `predict.py`)，確保模型可以順利載入和推論。
- [ ] **下午：** 閱讀 PyTorch 官方關於 **Post-Training Quantization (PTQ)** 的文件，理解 FP32 vs. INT8 的差異。

### Day 2: 模型量化程式碼實作
- [ ] **上午：** 修改 PyTorch 程式碼，加入量化 API (例如 `torch.quantization.prepare_qat` 和 `torch.quantization.convert`)。
- [ ] **下午：** 準備一個小型的**校準數據集 (calibration dataset)**（約 100-200 張具代表性的圖片）。
- [ ] **下午：** 執行模型校準，讓量化器收集權重分佈資訊以生成量化參數。

### Day 3: 量化模型性能驗證
- [ ] **上午：** 使用測試集，比較**原始 FP32 模型**和**量化後 INT8 模型**的推論準確度 (例如 mIoU)。
- [ ] **下午：** 編寫腳本以精確測量並記錄兩者的平均推論時間和模型檔案大小。
- [ ] **目標：** 確認 INT8 模型在可接受的準確度下降範圍內，速度顯著提升。

### Day 4: TensorRT 轉換學習與 ONNX 匯出
- [ ] **上午：** 閱讀 NVIDIA TensorRT 官方文件，了解其優化原理。
- [ ] **下午：** 執行 `export_onnx.py` 腳本，將**原始的 FP32 PyTorch 模型**轉換為 ONNX 格式。
- [ ] **驗證：** 確保生成的 ONNX 模型可以被 ONNX Runtime 等工具正確載入和推論。

### Day 5: TensorRT `.plan` 檔案生成與驗證
- [ ] **上午：** 使用 Jetson 上的 `trtexec` 工具或撰寫 Python 腳本，將 `.onnx` 檔案轉換為最終的 TensorRT 引擎。
    - **FP16 引擎:** `trtexec --onnx=model.onnx --saveEngine=model_fp16.plan --fp16`
    - **INT8 引擎:** `trtexec --onnx=model.onnx --saveEngine=model_int8.plan --int8 --calib=<calibration_data_cache>`
- [ ] **下午：** 撰寫一個新的預測腳本，專門用於載入 `.plan` 檔案並進行推論。
- [ ] **最終驗證：** 確認載入 `.plan` 檔案的推論結果正確，並記錄其最終的推論速度。

---

## 階段三：Docker 化與最終部署
*目標：將優化後的模型與應用程式封裝，實現可移植、自動化的部署。*

- [ ] **3.1. 建立 `requirements.jetson.txt`**
    - [ ] 基於階段一成功安裝的套件，建立一個 Jetson 專用的需求文件。
    - [ ] 在虛擬環境中執行 `pip freeze > requirements.jetson.txt`。

- [ ] **3.2. 撰寫 `Dockerfile.edge`**
    - [ ] 基於 `nvcr.io/nvidia/l4t-tensorrt` 或 `l4t-pytorch` 的基礎映像檔。
    - [ ] 文件內容應包含：安裝 Python 依賴、複製專案程式碼、複製優化後的 `.plan` 模型檔案。
    - [ ] (可選) 加入在容器啟動時，自動從 ONNX 生成 `.plan` 的腳本。

- [ ] **3.3. 建置並測試 Docker 映像檔**
    - [ ] 建置映像檔：`docker build -t iswm-jetson:latest -f Dockerfile.edge .`
    - [ ] 執行容器進行測試：`docker run --runtime nvidia -it iswm-jetson:latest`

- [ ] **3.4. 撰寫 `docker-compose.jetson.yml`**
    - [ ] 建立一個 docker-compose 檔案來簡化啟動流程。
    - [ ] 設定資料卷 (volumes) 掛載，方便管理模型和資料。
    - [ ] 加入 `restart: unless-stopped` 策略，讓容器在開機時自動啟動。