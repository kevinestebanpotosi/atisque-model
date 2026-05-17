# Usamos una imagen oficial y ligera de Python
FROM --platform=linux/amd64 python:3.10-slim

# Instalamos dependencias básicas
RUN apt-get update && apt-get install -y \
    wget \
    libicu-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalamos .NET 7.0
RUN wget -q https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh && \
    chmod +x ./dotnet-install.sh && \
    ./dotnet-install.sh --channel 7.0 --install-dir /usr/share/dotnet

ENV DOTNET_ROOT=/usr/share/dotnet
ENV PATH=$PATH:$DOTNET_ROOT

# CAMBIO CRÍTICO: Instalamos nncase 2.9.0 descargando directamente desde GitHub (No está en PyPI normal)
RUN pip install uv && uv pip install --system --no-cache-dir \
    https://github.com/kendryte/nncase/releases/download/v2.9.0/nncase-2.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    https://github.com/kendryte/nncase/releases/download/v2.9.0/nncase_kpu-2.9.0-py2.py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    numpy \
    onnx \
    onnxsim \
    onnxruntime \
    wheel \
    setuptools \
    # Pillow y flatbuffers útiles para manejo de modelos/imagenes
    pillow \
    flatbuffers

# Configuramos el path de los plugins para la v2.9.0
ENV NNCASE_PLUGIN_PATH=/usr/local/lib/python3.10/site-packages/nncase_kpu

WORKDIR /app