# 使用 CUDA 12.2 基礎鏡像
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# 設置非交互式環境和時區
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "tzdata tzdata/Areas select Etc" > /tmp/tzdata.conf && \
    echo "tzdata tzdata/Zones/Etc select UTC" >> /tmp/tzdata.conf && \
    debconf-set-selections /tmp/tzdata.conf && \
    rm /tmp/tzdata.conf

# 切換到更快的鏡像源（可選）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list

# 安裝 Python 3.9 和必要的工具，包括 distutils
RUN apt-get update && apt-get install -y --no-install-recommends python3.9 python3.9-dev python3-distutils wget && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

WORKDIR /app
COPY requirements.txt .

# 使用 python3.9 -m pip 安裝 PyTorch 和其他依賴
RUN python3.9 -m pip install --no-cache-dir torch==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    python3.9 -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
# COPY models/bge-m3 /app/models/bge-m3 

EXPOSE 8000
CMD ["/usr/bin/python3.9", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
