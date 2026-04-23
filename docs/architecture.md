# Architecture

## Request lifecycle

```
client ──► POST /v1/audio/transcriptions
              │
              ▼
       [ upload  ]  raw file written to tempfile
              │
              ▼
       [ ffmpeg  ]  → 16 kHz mono WAV
              │
              ▼
       [ preproc ]  noisereduce + butter 80–7600 Hz + RMS norm
              │           (skipped if preprocessing=off)
              ▼
       [ diarize ]  pyannote speaker-diarization-3.1 on GPU 0
              │           (skipped if HF_TOKEN absent or diarize=off)
              ▼
       [ VAD split ]  find_split_points() uses faster_whisper.vad
              │        to cut at natural silence boundaries
              ▼
       ┌──────┼──────┐   parallel_transcribe(): ThreadPoolExecutor
       ▼      ▼      ▼   one CT2 WhisperModel per GPU
     GPU0   GPU1   GPUN
       │      │      │
       └──────┼──────┘
              ▼
       [ merge segments sorted by start time ]
              │
              ▼
       [ assign_speakers ]  max-overlap matching with diarization turns
              │
              ▼
       response (json / verbose_json / text)
```

## Progress pipeline

A single `JOB_STATUS` dict is updated from the worker thread via `emit_event(stage, message, progress)`
and exposed on `GET /v1/status`. The web UI polls this endpoint every 1.2 s
and drives the 4-stage progress bar (`upload → preprocess → diarize → transcribe`)
plus the live log panel.

## Multi-GPU sharding

- `ctranslate2.get_cuda_device_count()` is queried at startup.
- One `WhisperModel(device="cuda", device_index=i)` per GPU is loaded eagerly
  so that transcription never pays cold-start cost.
- VAD is run on the CPU once; `find_split_points()` picks the nearest pause
  to each ideal cut (`total / n_gpus`) to avoid mid-word splits.
- Chunks are dispatched to workers; partial results are merged and sorted by
  absolute start time (each chunk carries its offset).

## Diarization → transcript fusion

`assign_speakers()` walks every transcript segment and picks the pyannote turn
with the **largest temporal overlap**. If no turn overlaps, the segment is
labelled `UNKNOWN`. The UI then assigns one of six stable colours per speaker.

## ROCm / gfx906 specifics

- Built against [`arlo-phoenix/CTranslate2-rocm`](https://github.com/arlo-phoenix/CTranslate2-rocm)
  with `-DCMAKE_HIP_ARCHITECTURES="gfx906"`.
- `HSA_OVERRIDE_GFX_VERSION=9.0.6` pins the ISA on newer ROCm runtimes.
- `CT2_CUDA_ALLOCATOR=cub_caching` with
  `CT2_CUDA_CACHING_ALLOCATOR_CONFIG=4,3,12,419430400` significantly reduces
  VRAM fragmentation for long-session workloads on 32 GB MI50.
- `ROCR_VISIBLE_DEVICES` is mirrored into `HIP_VISIBLE_DEVICES` at import time
  so the image works unchanged under GPUStack, which only sets the ROCR form.
