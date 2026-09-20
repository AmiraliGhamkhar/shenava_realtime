# Current review/status

## Current implementation

The repository now targets one runtime path:

```text
16 kHz mono audio → RMS VAD → sherpa-onnx OnlineRecognizer.from_nemo_ctc
→ one OnlineStream per VAD utterance → endpoint-safe stabilization
→ deterministic Persian/medical post-processing → overlay/injector/console
```

The intended model is the streaming CTC INT8 ONNX export:

`mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26`

pinned at revision:

`4be3d2375c98a985154122d69b43360eb8bdca5a`

The local files are `models/shenava/model.int8.onnx` and `models/shenava/tokens.txt`.  They are intentionally gitignored and must be provisioned manually.

## Invariants

- Python 3.10+
- sherpa-onnx, pinned in `requirements.txt`
- CPU inference
- 16 kHz mono audio
- one long-lived `OnlineRecognizer`
- one fresh `OnlineStream` per VAD utterance
- RMS VAD with optional adaptive noise floor
- bounded queues and explicit discontinuity handling
- endpoint-safe transcript stabilization
- deterministic post-processing only
- no LLM, embeddings, RAG, vector database, PyTorch, NeMo runtime, or new service layer

## Recent fixes in this patch

- The startup streaming gate now runs a real online lifecycle probe instead of checking method names only.
- `ASRConfig.require_streaming` defaults to `True`, so production startup fails when no valid streaming recognizer is available.
- `AudioCapture` now passes all adaptive VAD settings from `AudioConfig` into `VADConfig`; `SHENAVA_VAD_ADAPTIVE=0` and `--no-adaptive-vad` now reach the actual `EnergyVAD`.
- Microphone diagnostics distinguish frames arriving from frames arriving at an effectively silent level, without modifying audio or thresholds.
- Startup diagnostics now report model/tokens paths, sherpa-onnx version when available, streaming status, input device, sample rate, VAD thresholds, adaptive VAD state, decoder, and second-pass mode.
- Dependencies are reproducible: `sherpa-onnx==1.13.8`; `soundfile` was removed because repository-wide search showed no use; `pytest` moved to `requirements-dev.txt`.
- Documentation now describes the current sherpa-onnx CTC implementation rather than obsolete NeMo/v1.5 runtime instructions.

## Verification status

Expected local checks:

```bash
python -m pytest -q
python main.py --self-test
python tests/test_asr_backend_smoke.py
python tools/verify_pipeline.py --model models/shenava/model.int8.onnx --tokens models/shenava/tokens.txt --wav path/to/real_16khz_mono_pcm16.wav
```

`--allow-endpoint` must not be used for production streaming validation.

Real-model and real-microphone checks require the ONNX model files, `sherpa-onnx`, an input WAV or microphone, and the target desktop environment.  Synthetic tests and `--self-test` verify orchestration only; they are not accuracy or real-ASR validation.

## Known limitations

- No automatic model download.
- No hidden denoising, auto-threshold lowering, or gain correction.
- No semantic correction layer; unknown text remains unchanged.
- Overlay and injector behavior depends on the host desktop session and permissions.
- Clinical extraction is deterministic and conservative; persisted records still require review.
