# 根據 torch==2.5.0+cu124，我們選擇對應的官方基礎映像檔
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.docker.txt .

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

CMD ["/bin/bash"]