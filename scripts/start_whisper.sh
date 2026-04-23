#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  ct2-whisper-gfx906 :: start helper
#
#  Usage:
#    HF_TOKEN=hf_xxxx MODELS_DIR=/srv/models ./start_whisper.sh
#
#  Env vars:
#    IMAGE        docker image        (default: nickoptimal/ct2-whisper-gfx906:latest)
#    CONTAINER    container name      (default: ct2-whisper)
#    PORT         host port           (default: 8000)
#    MODELS_DIR   host dir w/ models  (default: $HOME/models)
#    MODEL_PATH   faster-whisper id   (default: large-v3-turbo)
#    MODEL_NAME   served-model-name   (default: whisper)
#    COMPUTE_TYPE int8 | float16 | ...(default: int8)
#    BEAM_SIZE    beam size           (default: 2)
#    HF_TOKEN     HuggingFace token   (optional; required only for diarization)
# -----------------------------------------------------------------------------
set -euo pipefail

IMAGE="${IMAGE:-nickoptimal/ct2-whisper-gfx906:latest}"
CONTAINER="${CONTAINER:-ct2-whisper}"
PORT="${PORT:-8000}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_PATH="${MODEL_PATH:-large-v3-turbo}"
MODEL_NAME="${MODEL_NAME:-whisper}"
COMPUTE_TYPE="${COMPUTE_TYPE:-int8}"
BEAM_SIZE="${BEAM_SIZE:-2}"

mkdir -p "${MODELS_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "[*] Removing existing container ${CONTAINER}"
    docker rm -f "${CONTAINER}" >/dev/null
fi

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_ARG+=(-e "HF_TOKEN=${HF_TOKEN}")
    echo "[*] HF_TOKEN is set  -> diarization enabled"
else
    echo "[!] HF_TOKEN is empty -> diarization DISABLED (transcription still works)"
fi

echo "[*] Launching ${IMAGE} (model=${MODEL_PATH} ct=${COMPUTE_TYPE})"

docker run -d \
    --name "${CONTAINER}" \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add 44 --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --restart unless-stopped \
    -v "${MODELS_DIR}:/models" \
    "${TOKEN_ARG[@]}" \
    "${IMAGE}" \
        --model-path  "${MODEL_PATH}" \
        --port        "${PORT}" \
        --host        0.0.0.0 \
        --model-name  "${MODEL_NAME}" \
        --compute-type "${COMPUTE_TYPE}" \
        --beam-size   "${BEAM_SIZE}"

echo "[*] Started. UI: http://localhost:${PORT}/   API: http://localhost:${PORT}/v1/models"
