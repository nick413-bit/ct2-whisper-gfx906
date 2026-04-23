# GPUStack integration

Register the image as a **custom inference backend** (Admin → Inference Backends → New):

```yaml
backend_name: ct2-whisper-gfx906-custom
image: nickoptimal/ct2-whisper-gfx906:latest
default_run_command: >
  --model-path {{model_path}}
  --port {{port}}
  --host 0.0.0.0
  --model-name {{model_name}}
default_backend_param:
  - --beam-size=2
  - --compute-type=int8
env:
  - HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Notes:

- `HF_TOKEN` is injected via the **Env** tab in the model card — do **not**
  check it into the backend definition in production.
- `HSA_OVERRIDE_GFX_VERSION=9.0.6` is already baked into the image; no extra
  env is needed for gfx906 hosts.
- The backend does not need GPU-count arguments — `ctranslate2.get_cuda_device_count()`
  discovers all GPUs exposed via `ROCR_VISIBLE_DEVICES`.
