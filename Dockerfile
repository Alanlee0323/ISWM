# 根據 torch==2.5.0+cu124，我們選擇對應的官方基礎映像檔
# pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime 是一個很好的選擇
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 設定工作目錄
WORKDIR /app

# 複製您剛剛整理好的、乾淨的 requirements.txt
COPY requirements.docker.txt .

# 安裝核心依賴
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.docker.txt

# 複製您專案的所有程式碼
COPY . .

# 設定預設指令
CMD ["/bin/bash"]