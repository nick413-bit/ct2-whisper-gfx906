# ct2-whisper-gfx906

**CTranslate2 + faster-whisper + pyannote.audio speaker diarization** —
OpenAI-compatible speech-to-text server with a built-in web UI, tuned for
**AMD Instinct MI50 / Radeon VII (gfx906)** on ROCm 6.3.

| | |
|---|---|
| Base image         | `rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0` |
| CTranslate2 branch | [`arlo-phoenix/CTranslate2-rocm` / `rocm`](https://github.com/arlo-phoenix/CTranslate2-rocm) |
| Target GPU arch    | `gfx906` (Vega 20 / CDNA0) |
| Docker image       | `nickoptimal/ct2-whisper-gfx906:latest` |

## Features

- **Multi-GPU parallel transcription** — input audio is split at VAD silence
  boundaries and each chunk is decoded on its own GPU via `ThreadPoolExecutor`,
  then merged back in timeline order. Near-linear speed-up up to 8 × MI50.
- **Speaker diarization** via `pyannote/speaker-diarization-3.1`
  (requires a HuggingFace token — passed at runtime, never baked into the image).
- **Audio preprocessing** — `ffmpeg` to 16 kHz mono → `noisereduce`
  (spectral gating 0.6) → Butterworth bandpass 80 – 7600 Hz (order 5)
  → RMS normalization.
- **OpenAI-compatible HTTP API** — drop-in replacement for
  `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`.
- **Web UI** (`/`) — drag & drop, 4-stage progress bar, live log streaming,
  speaker-coloured segments, Copy-Text / Copy-JSON / Download-SRT.
- **Healthcheck** on `/health` reporting loaded models, GPUs, diarization state.

## Quick start

### 1. Pull the image

```bash
docker pull nickoptimal/ct2-whisper-gfx906:latest
```

### 2. Run

```bash
docker run -d \
    --name ct2-whisper \
    --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$HOME/models:/models" \
    -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
    nickoptimal/ct2-whisper-gfx906:latest \
        --model-path large-v3-turbo \
        --port 8000 --host 0.0.0.0 \
        --model-name whisper \
        --compute-type int8 \
        --beam-size 2
```

Or simply:

```bash
HF_TOKEN=hf_xxx MODELS_DIR=/srv/models ./scripts/start_whisper.sh
```

Open <http://localhost:8000/> for the UI or hit the API directly:

```bash
curl -F file=@recording.mp3 -F response_format=verbose_json \
     http://localhost:8000/v1/audio/transcriptions
```

### 3. HF_TOKEN & diarization

- If `HF_TOKEN` is **unset** the server starts normally, diarization is disabled,
  transcription still works.
- To enable diarization, accept the model license at
  <https://huggingface.co/pyannote/speaker-diarization-3.1> first, then pass
  the token with `-e HF_TOKEN=...`.

## CLI flags

| flag | default | description |
|---|---|---|
| `--model-path`      | *(required)* | faster-whisper id (`large-v3-turbo`, `medium`, …) or path to a CT2-converted model directory |
| `--model-name`      | `whisper`    | value returned from `/v1/models` (`served-model-name`) |
| `--port`            | `8000`       | HTTP port |
| `--host`            | `0.0.0.0`   | bind address |
| `--compute-type`    | `int8`       | `int8`, `int8_float16`, `float16`, `float32` |
| `--beam-size`       | `2`          | beam search width |
| `--num-gpus`        | auto         | override detected GPU count |

## API

### `POST /v1/audio/transcriptions`  (OpenAI-compatible)

Form fields: `file`, `language`, `prompt`, `temperature`, `response_format`
(`json` | `verbose_json` | `text`), plus extensions: `diarize` (`auto` / `off`),
`preprocessing` (`auto` / `off`), `num_speakers`, `min_speakers`, `max_speakers`.

### `GET /v1/models`  — list served model

### `GET /v1/status` — live pipeline status for the UI (polled every ~1 s)

### `GET /health`    — readiness probe

## Building from source

```bash
git clone https://github.com/nick413-bit/ct2-whisper-gfx906.git
cd ct2-whisper-gfx906
docker build -t ct2-whisper-gfx906:dev -f docker/Dockerfile .
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the request/response flow,
VAD-aware chunking, and how diarization output is merged with transcription
segments.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).

Upstream components keep their own licenses:
CTranslate2 (MIT), faster-whisper (MIT), pyannote.audio (MIT,
weights require accepting pyannote's terms of use), ROCm stack (varies).

## Credits

- [arlo-phoenix/CTranslate2-rocm](https://github.com/arlo-phoenix/CTranslate2-rocm)
  — the HIP/ROCm port of CTranslate2 that makes this possible on gfx906.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
