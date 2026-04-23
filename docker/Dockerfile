# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
#  ct2-whisper-gfx906 : CTranslate2 + faster-whisper + pyannote on AMD MI50
#  Base : ROCm 6.3 PyTorch 2.4 (Ubuntu 22.04, Python 3.10)
#  Target GPU arch : gfx906 (Vega 20 / CDNA0, AMD MI50 / Radeon VII)
# ---------------------------------------------------------------------------
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config \
        libopenblas-dev libopenblas0 libsndfile1 libomp-dev \
        wget curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- oneDNN v3.1.1 (static) ------------------------------------------------
WORKDIR /build
RUN git clone --depth 1 -b v3.1.1 https://github.com/oneapi-src/oneDNN.git && \
    cd oneDNN && mkdir build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DDNNL_LIBRARY_TYPE=STATIC \
        -DDNNL_BUILD_TESTS=OFF \
        -DDNNL_BUILD_EXAMPLES=OFF && \
    ninja -j"$(nproc)" && ninja install && \
    cd /build && rm -rf oneDNN

# --- CTranslate2-rocm ------------------------------------------------------
RUN git clone --recursive https://github.com/arlo-phoenix/CTranslate2-rocm.git && \
    cd CTranslate2-rocm && git checkout rocm && \
    mkdir build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
        -DCMAKE_HIP_ARCHITECTURES="gfx906" \
        -DWITH_HIP=ON -DWITH_CUDA=OFF \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
        -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP \
        -DBUILD_CLI=OFF -DBUILD_TESTS=OFF && \
    ninja -j"$(nproc)" && ninja install && \
    cd ../python && \
    CTRANSLATE2_ROOT=/opt/ctranslate2 pip install --no-cache-dir -e . && \
    cd /build && rm -rf CTranslate2-rocm

# ---------------------------------------------------------------------------
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libopenblas0 libomp-dev curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Bring in the compiled CTranslate2 runtime
COPY --from=builder /opt/ctranslate2 /opt/ctranslate2
# Python bindings are installed in the system interpreter, copy them over
COPY --from=builder /opt/conda/lib/python3.10/site-packages/ctranslate2* /opt/conda/lib/python3.10/site-packages/

ENV CTRANSLATE2_ROOT=/opt/ctranslate2 \
    LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH} \
    CT2_CUDA_ALLOCATOR=cub_caching \
    CT2_CUDA_CACHING_ALLOCATOR_CONFIG=4,3,12,419430400 \
    HSA_OVERRIDE_GFX_VERSION=9.0.6 \
    PYTHONUNBUFFERED=1

# --- Python deps -----------------------------------------------------------
RUN pip install --no-cache-dir \
        "huggingface-hub<1.0" \
        faster-whisper==1.0.3 \
        transformers tokenizers \
        librosa soundfile scipy numpy \
        fastapi "uvicorn[standard]" python-multipart aiofiles \
        noisereduce && \
    pip install --no-cache-dir \
        torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1 && \
    pip install --no-cache-dir pyannote.audio==3.3.2

# --- app -------------------------------------------------------------------
WORKDIR /app
COPY server/whisper_server.py /app/whisper_server.py
COPY server/static/index.html /app/static/index.html

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT:-8000}/v1/models || exit 1

# HF_TOKEN is expected to be provided at runtime (e.g. -e HF_TOKEN=hf_xxx)
# If absent, diarization is silently disabled and transcription still works.
ENTRYPOINT ["python3", "/app/whisper_server.py"]
CMD ["--model-path", "large-v3-turbo", "--port", "8000", "--host", "0.0.0.0"]
