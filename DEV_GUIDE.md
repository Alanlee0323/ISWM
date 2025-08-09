# 開發環境部署指南 (Development Guide)

> 本專案原生設計為 Linux 系統執行，透過 Docker 實現跨平台部署
> 
> **設計理念**: Linux First - Windows/macOS Compatible

## 🎯 核心理念

### Environment as Code (環境即程式碼)
透過 `Dockerfile` 和 `docker-compose.yml`，將整個專案的運行環境程式碼化，確保在任何機器上都能一鍵完美重現。

### Separation of Concerns (關注點分離)
將不同功能的服務（如訓練程式、MLflow 伺服器）拆分成獨立的容器，讓架構更清晰、更穩定、更易於維護。

---

## 📋 系統需求

> **重要說明**: 本專案原生設計為 Linux 系統執行。以下提供不同平台的部署方式：

### Linux 系統 (推薦)
- **作業系統**: Ubuntu 20.04+ / CentOS 8+ / 其他主流 Linux 發行版
- **容器平台**: Docker + Docker Compose
- **開發環境**: 任何支援的編輯器 (VS Code, Vim, etc.)

### Windows 系統 (透過 WSL2)
- **作業系統**: Windows 11/10
- **虛擬化**: WSL2 + Ubuntu
- **容器平台**: Docker Desktop
- **開發環境**: VS Code + Remote-WSL 擴充功能

### macOS 系統
- **作業系統**: macOS 10.15+
- **容器平台**: Docker Desktop for Mac
- **開發環境**: VS Code 或其他編輯器

---

## 🏗️ 核心架構文件

### 1. `Dockerfile` - 服務建置藍圖
定義單一服務（如 training 環境）的映像檔建置過程。

**關鍵實踐**：
- 選擇包含 CUDA 和 PyTorch 的官方基礎映像檔
- 區分系統層和應用層的套件安裝
- 使用多階段建置優化映像檔大小

```dockerfile
FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y git
COPY requirements.docker.txt .
RUN pip install -r requirements.docker.txt
```

### 2. `requirements.docker.txt` - Python 依賴清單
專門為 Docker 環境定義的 Python 函式庫列表。

**重要提醒**：
- 不包含基礎映像檔已有的核心套件（如 `torch`, `torchvision`）
- 從 `pip freeze` 結果中提煉出的最小化依賴列表

### 3. `docker-compose.yml` - 服務編排配置
定義多個容器服務的協同工作方式。

**核心配置**：
- `services`: 定義各獨立服務
- `volumes`: 實現資料持久化和程式碼同步
- `environment`: 配置容器間通訊
- `deploy.resources`: GPU 資源分配
- `shm_size`: 避免 DataLoader 記憶體問題

---

## 🚀 快速開始

### 平台特定設定

#### Linux 系統 (原生環境，推薦)

1. **安裝 Docker 和 Docker Compose**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   # 登出後重新登入以使群組變更生效
   
   # CentOS/RHEL
   sudo yum install docker docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```

2. **複製專案**
   ```bash
   git clone <your-repo-url> ~/ISWM
   cd ~/ISWM
   ```

#### Windows 系統 (透過 WSL2)

1. **安裝必要工具**
   ```bash
   # 1. 啟用 WSL2 並安裝 Ubuntu
   # 2. 安裝 Docker Desktop for Windows
   # 3. 在 Docker Desktop 設定中啟用 WSL2 整合
   # 4. VS Code 安裝 Remote-WSL 擴充功能
   ```

2. **複製專案到 WSL**
   ```bash
   # 在 WSL Ubuntu 終端機中執行
   git clone <your-repo-url> ~/ISWM
   cd ~/ISWM
   ```

3. **開啟開發環境**
   ```bash
   # 在專案目錄中啟動 VS Code
   code .
   ```

#### macOS 系統

1. **安裝 Docker Desktop**
   ```bash
   # 下載並安裝 Docker Desktop for Mac
   # 確保 Docker Desktop 正在運行
   ```

2. **複製專案**
   ```bash
   git clone <your-repo-url> ~/ISWM
   cd ~/ISWM
   ```

### 日常開發流程 (適用所有平台)

> **注意**: 以下指令在 Linux 原生環境、WSL2 和 macOS 中均可執行

#### 步驟 1: 啟動服務
```bash
# 基本啟動
docker-compose up -d

# 如果修改了 Dockerfile 或 requirements.docker.txt
docker-compose up --build -d
```

#### 步驟 2: 檢查服務狀態
```bash
docker ps
# 確認 training 和 mlflow 服務都處於 Up 狀態
```

#### 步驟 3: 進入開發容器
```bash
docker-compose exec training bash
# 您會得到容器內部的 shell 環境
```

#### 步驟 4: 執行開發任務
```bash
# 在容器內執行您的腳本
./Original_CE.sh
```

#### 步驟 5: 監控日誌（可選）
```bash
# 在新的終端機視窗中
docker-compose logs -f training
```

#### 步驟 6: 查看結果
開啟瀏覽器訪問：`http://localhost:5000` 查看 MLflow UI

#### 步驟 7: 結束工作
```bash
docker-compose down
```

---

## 🔧 常見問題排除

| 錯誤現象 | 根本原因 | 解決方案 |
|---------|---------|---------|
| `Permission denied` | 檔案所有權問題 (常見於 WSL) | `sudo chown -R $USER:$USER .` |
| `python: command not found` | 腳本使用 `python`，環境只有 `python3` | 將腳本中的 `python` 改為 `python3` |
| `Connection refused` (MLflow) | 容器連接 `localhost` 而非其他容器 | 使用容器名稱，如 `http://mlflow:5000` |
| 容器啟動後立刻 `Exited (0)` | 非互動模式下 bash 執行完就退出 | 在 `docker-compose.yml` 中加入 `stdin_open: true` 和 `tty: true` |
| 容器啟動後立刻 `Exited (137)` | 記憶體不足 (OOM) | 調高 Docker 記憶體限制 (Linux: 修改 daemon.json / Windows: Docker Desktop 設定) |
| `Bus error`, `out of shared memory` | DataLoader 共享記憶體不足 | 在 `docker-compose.yml` 中加入 `shm_size: '8g'` |
| `docker: command not found` (Linux) | Docker 未安裝或未啟動 | `sudo systemctl start docker` 或重新安裝 Docker |
| `docker-compose: command not found` | Docker Compose 未安裝 | Linux: `sudo apt install docker-compose` / 其他平台: 檢查 Docker Desktop |

---

## 📚 進階使用

### 重新建置映像檔
```bash
docker-compose build --no-cache
```

### 查看服務日誌
```bash
# 查看所有服務日誌
docker-compose logs

# 查看特定服務日誌
docker-compose logs training

# 即時跟蹤日誌
docker-compose logs -f training
```

### 停止特定服務
```bash
docker-compose stop training
docker-compose start training
```

### 清理環境
```bash
# 停止並移除容器
docker-compose down

# 同時移除相關映像檔
docker-compose down --rmi all

# 清理所有未使用的 Docker 資源
docker system prune -a
```

---

## 💡 最佳實踐

### 平台特定建議

#### Linux 系統 (原生環境)
- ✅ 直接在本機檔案系統中開發，效能最佳
- ✅ 使用 `systemctl` 管理 Docker 服務
- ✅ 考慮使用 `nvidia-docker2` 支援 GPU (如需要)

#### Windows + WSL2
- ✅ 程式碼永遠存放在 WSL 檔案系統中 (`~/` 而非 `/mnt/c/`)
- ✅ 使用 `code .` 在 WSL 中啟動 VS Code
- ✅ 避免跨檔案系統操作以獲得最佳效能

#### macOS
- ✅ 使用 Docker Desktop 的檔案共享功能
- ✅ 注意 M1/M2 晶片的 ARM 架構相容性

### 通用建議
- ✅ 定期執行 `docker system prune` 清理空間
- ✅ 使用 `-d` 參數在背景運行服務
- ✅ 修改 Dockerfile 後記得 `--build`
- ✅ 開發完成後執行 `docker-compose down`

---

## 🤝 貢獻指南

1. Fork 本專案
2. 在 WSL 環境中進行開發
3. 確保所有測試通過
4. 提交 Pull Request

---

## 📞 支援

### 平台特定問題
- **Linux**: 檢查 Docker daemon 狀態 (`sudo systemctl status docker`)
- **Windows/WSL2**: 確保 Docker Desktop 和 WSL2 整合正確設定
- **macOS**: 檢查 Docker Desktop 是否正在運行

### 通用支援資源
1. 本文件的「常見問題排除」章節
2. 專案的 Issues 頁面
3. Docker 和相關平台官方文件

**適用於所有平台的開發體驗 🚀**