# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
#  ct2-whisper-gfx906 : CTranslate2 + faster-whisper + pyannote on AMD MI50
#  Base : ROCm 6.3 PyTorch 2.4 (Ubuntu 22.04, Python 3.10)
#  Target GPU arch : gfx906 (Vega 20 / CDNA0, AMD MI50 / Radeon VII)
#
#  Size note: the upstream base image is already ~30 GB. We keep a single
#  stage (no multi-stage) because the ROCm toolchain (HIP/hipcc) present in
#  that image is what we use to build CTranslate2. We do, however, clean up
#  build artefacts and avoid duplicating PyTorch ROCm, which is what made
#  previous iterations of this image balloon to >100 GB.
# ---------------------------------------------------------------------------
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    CTRANSLATE2_ROOT=/opt/ctranslate2 \
    LD_LIBRARY_PATH=/opt/ctranslate2/lib:/opt/rocm/lib:/usr/local/lib \
    CT2_CUDA_ALLOCATOR=cub_caching \
    CT2_CUDA_CACHING_ALLOCATOR_CONFIG=4,3,12,419430400 \
    HSA_OVERRIDE_GFX_VERSION=9.0.6 \
    PYTHONUNBUFFERED=1 \
    PYANNOTE_METRICS_ENABLED=0

# --- system deps (build + runtime in one layer, cleaned at the end) --------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config \
        libopenblas-dev libopenblas0 libsndfile1 libomp-dev \
        wget curl ca-certificates ffmpeg && \
    ln -sf /usr/lib/x86_64-linux-gnu/libomp.so.5 \
        /opt/conda/envs/py_3.10/lib/libomp.so && \
    rm -rf /var/lib/apt/lists/*

# --- oneDNN v3.1.1 (static) ------------------------------------------------
ARG ONEDNN_VERSION=3.1.1
RUN cd /tmp && \
    wget -q "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz" && \
    tar xf "v${ONEDNN_VERSION}.tar.gz" && \
    cd "oneDNN-${ONEDNN_VERSION}" && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DONEDNN_LIBRARY_TYPE=STATIC \
        -DONEDNN_BUILD_EXAMPLES=OFF \
        -DONEDNN_BUILD_TESTS=OFF \
        -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
        -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" \
        -DONEDNN_BUILD_GRAPH=OFF . && \
    ninja -j"$(nproc)" install && \
    cd / && rm -rf /tmp/oneDNN* "/tmp/v${ONEDNN_VERSION}.tar.gz"

# --- CTranslate2-rocm (gfx906) ---------------------------------------------
RUN cd /tmp && \
    git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/arlo-phoenix/CTranslate2-rocm.git ct2 && \
    cd ct2 && git checkout rocm && \
    CLANG_CMAKE_CXX_COMPILER=clang++ CXX=clang++ \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -G Ninja -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
        -DCMAKE_HIP_ARCHITECTURES="gfx906" \
        -DWITH_HIP=ON -DWITH_CUDA=OFF \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
        -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP \
        -DBUILD_CLI=OFF -DBUILD_TESTS=OFF \
        -DENABLE_CPU_DISPATCH=OFF && \
    cmake --build build --parallel "$(nproc)" && \
    cmake --install build && \
    cd python && pip install --no-cache-dir -r install_requirements.txt && \
    pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/ct2 ~/.cache/pip

# --- torchaudio built against existing ROCm-6.3 torch (no CUDA/ROCm ext) ---
# This avoids pulling a second copy of libtorch_hip.so which is the main
# reason previous images were 70+ GB heavier than necessary.
RUN pip install --no-cache-dir "numpy<2" && \
    cd /tmp && \
    git clone --depth 1 -b v2.4.0 --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/audio.git ta && \
    cd ta && USE_CUDA=0 USE_ROCM=0 BUILD_SOX=0 \
        pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/ta ~/.cache/pip

# --- Python deps for the server -------------------------------------------
RUN pip install --no-cache-dir \
        "huggingface-hub<1.0" \
        faster-whisper==1.0.3 \
        transformers tokenizers \
        librosa soundfile scipy \
        fastapi "uvicorn[standard]" python-multipart aiofiles \
        noisereduce \
        "pyannote.audio==3.3.2" && \
    rm -rf ~/.cache/pip

# --- strip build toolchain (keeps runtime libs) ---------------------------
RUN apt-get purge -y --auto-remove \
        git cmake ninja-build build-essential pkg-config \
        libopenblas-dev libomp-dev wget && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/.cache/*

# --- app -------------------------------------------------------------------
WORKDIR /app
COPY server/whisper_server.py /app/whisper_server.py
COPY server/static/index.html /app/static/index.html

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT:-8000}/v1/models || exit 1

# HF_TOKEN is expected to be provided at runtime (e.g. -e HF_TOKEN=hf_xxx).
# If absent, diarization is silently disabled and transcription still works.
ENTRYPOINT ["python3", "/app/whisper_server.py"]
CMD ["--model-path", "large-v3-turbo", "--port", "8000", "--host", "0.0.0.0"]
