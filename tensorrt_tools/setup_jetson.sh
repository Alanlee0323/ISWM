#!/bin/bash

# =================================================================
# ISWM 專案 Jetson NX 環境自動化安裝腳本
# =================================================================
# 說明：
# 這個腳本會自動化完成所有在 Jetson NX 上設定 Python 虛擬環境
# 並安裝所有必要相依套件的步驟。
#
# 使用方法：
# 1. 將 PyTorch 的 .whl 檔案 (例如 torch-2.1.0a0...) 放在此腳本同目錄下。
# 2. 給予執行權限： chmod +x setup_jetson.sh
# 3. 執行：         ./setup_jetson.sh
# =================================================================

# 當任何指令失敗時，立即中止腳本
set -e

echo "🚀 (1/5) 更新 apt 並安裝系統級相依套件..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libjpeg-dev zlib1g-dev libopenblas-dev libopenmpi-dev libomp-dev libpng-dev

echo "🚀 (2/5) 建立並啟用 Python 虛擬環境 (edgeAI_ISWM_env)..."
# 在專案外部的家目錄建立 venv
python3 -m venv --system-site-packages ~/edgeAI_ISWM_env
source ~/edgeAI_ISWM_env/bin/activate

echo "✅ 虛擬環境已啟用，目前 Python 版本："
python3 --version

echo "🚀 (3/5) 升級 pip 並安裝 requirements.txt 中的 Python 套件..."
pip3 install --upgrade pip
# 回到專案目錄來尋找 requirements_nx.txt
cd $(dirname "$0")
pip3 install -r requirements_nx.txt

echo "🚀 (4/5) 安裝 Jetson 專用的 PyTorch..."
# 尋找當前目錄下的 torch .whl 檔案
TORCH_WHL=$(find . -maxdepth 1 -name "torch*.whl")
if [ -z "$TORCH_WHL" ]; then
    echo "❌ 錯誤：在專案目錄中找不到 PyTorch 的 .whl 檔案！"
    exit 1
fi
echo "找到 PyTorch .whl 檔案: $TORCH_WHL"
pip3 install "$TORCH_WHL"

echo "🚀 (5/5) 從原始碼編譯並安裝 Torchvision (這一步會需要很長時間)..."
# 確保舊的下載被清除
rm -rf torchvision
git clone --branch v0.16.2 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.2
python3 setup.py install --user
cd ..
rm -rf torchvision # 安裝完畢後清理原始碼

echo "🎉🎉🎉 環境設定完成！🎉🎉🎉"
echo "您現在可以透過 'source ~/edgeAI_ISWM_env/bin/activate' 來啟用環境。"