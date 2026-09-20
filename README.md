# Shenava realtime

Shenava realtime is a local desktop Persian medical dictation app.  It captures 16 kHz mono microphone audio, segments speech with an RMS VAD, decodes with the Shenava-Koochik-v1.0 sherpa-onnx streaming CTC model on CPU, applies deterministic Persian/medical post-processing, then writes stable text to the console, overlay, clipboard, or keyboard injector.

No LLM, embeddings, RAG, vector database, PyTorch, NeMo runtime, or semantic/fuzzy correction layer is used.

## Architecture

```text
microphone 16 kHz mono
  → AudioCapture bounded queue
  → RMS EnergyVAD (optional adaptive noise floor)
  → SPEECH_START / audio / SPEECH_END events
  → one long-lived sherpa_onnx.OnlineRecognizer
  → one fresh OnlineStream per VAD utterance
  → endpoint-safe stabilizer
  → deterministic post-processing:
       raw ASR → Unicode normalization → terminology → numbers
       → measurements/BP → clinical context → stable final text
  → output mode: overlay / inject / both / clipboard / console
```

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For tests and development:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python main.py --self-test
```

## Exact model provisioning

The app does not download models automatically.  Download the exact sherpa-onnx export manually:

```bash
hf download \
  mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26 \
  --revision 4be3d2375c98a985154122d69b43360eb8bdca5a \
  model.int8.onnx tokens.txt \
  --local-dir models/shenava
```

Expected files:

```text
models/shenava/model.int8.onnx
models/shenava/tokens.txt
```

`tokens.txt` is expected to contain 1025 tokens.  See `models/shenava/README.md` for the provisioning contract.

## Basic run

```bash
python main.py
```

Useful desktop switches:

```bash
python main.py --output-mode console
python main.py --no-overlay --no-inject --output-mode console
python main.py --list-devices
python main.py --audio-device 1
```

## Debug run

```bash
python main.py \
  --no-overlay \
  --no-inject \
  --output-mode console \
  --require-streaming \
  --second-pass off \
  --log-level DEBUG
```

Startup logs include model path, tokens path, sherpa-onnx version if importable, streaming yes/no, input device, sample rate, VAD thresholds, adaptive VAD status, decoder name, and second-pass mode.  Runtime logs distinguish microphone delivery, low/silent input, VAD `SPEECH_START`, decoder steps at DEBUG level, endpoint finalization, and output.

## Real WAV verification

Use a real PCM16 16 kHz mono WAV.  Do not pass `--allow-endpoint` when validating the production streaming path.

```bash
python tools/verify_pipeline.py \
  --model models/shenava/model.int8.onnx \
  --tokens models/shenava/tokens.txt \
  --wav path/to/real_16khz_mono_pcm16.wav
```

`python main.py --self-test` is synthetic orchestration only; it is not a real ASR validation.

## Configuration and environment variables

Common variables:

| Variable | Meaning |
|---|---|
| `SHENAVA_MODEL_PATH` | local `model.int8.onnx` path |
| `SHENAVA_TOKENS_PATH` | local `tokens.txt` path |
| `SHENAVA_REQUIRE_STREAMING` | default `1`; set `0` only for explicit endpoint-fallback tests/tools |
| `SHENAVA_NUM_THREADS` | sherpa-onnx CPU threads |
| `SHENAVA_SECOND_PASS` | `greedy` or `off` |
| `SHENAVA_AUDIO_DEVICE` | sounddevice input device index/name |
| `SHENAVA_VAD_ADAPTIVE` | `1`/`0`; `0` disables adaptive VAD |
| `SHENAVA_OUTPUT_MODE` | `overlay`, `inject`, `both`, `clipboard`, or `console` |
| `SHENAVA_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

CLI overrides include `--model`, `--tokens`, `--threads`, `--audio-device`, `--no-adaptive-vad`, `--second-pass`, `--output-mode`, `--no-overlay`, `--no-inject`, and `--log-level`.

## Output modes

- `overlay`: show text in the overlay only.
- `inject`: type/paste stable text into the focused application only.
- `both`: overlay plus injector.
- `clipboard`: copy final utterances to the clipboard.
- `console`: print stable text to stdout.

## Limitations

- Requires the local sherpa-onnx ONNX export; no automatic downloads.
- CPU-only by design.
- The streaming model must pass the startup online lifecycle probe; otherwise production startup fails.
- RMS VAD is intentionally simple and threshold-based.  Low microphone gain is logged for diagnosis but not auto-corrected.
- Medical post-processing is deterministic and conservative.  Unknown text is preserved; there is no automatic clinical inference.
- Real accuracy, WER/CER, latency, microphone behavior, and operating-system injection behavior must be validated on the deployment machine with real audio.
