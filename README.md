# Shenava Realtime

**Shenava Realtime** is a local desktop application for **Persian medical speech-to-text**.

It captures microphone audio at **16 kHz mono**, detects speech with a lightweight **RMS-based VAD**, transcribes speech using the **Shenava-Koochik-v1.0 Streaming CTC** model through **Sherpa-ONNX**, applies deterministic Persian and medical text processing, and sends the final text to the selected output.

The application is designed to run **locally on CPU**.

It does **not** use:

* LLMs
* Embeddings
* RAG
* Vector databases
* PyTorch
* NeMo runtime
* Semantic or fuzzy correction

---

## Architecture

```text
Microphone
    │
    ▼
16 kHz Mono Audio
    │
    ▼
AudioCapture
    │
    ▼
RMS Energy VAD
    │
    ├── SPEECH_START
    ├── Audio
    └── SPEECH_END
    │
    ▼
Sherpa-ONNX Streaming CTC
    │
    ▼
Endpoint-Safe Stabilizer
    │
    ▼
Deterministic Post-Processing
    │
    ├── Unicode normalization
    ├── Medical terminology
    ├── Number normalization
    ├── Measurements / blood pressure
    └── Clinical text cleanup
    │
    ▼
Stable Final Text
    │
    ├── Overlay
    ├── Keyboard Injector
    ├── Clipboard
    └── Console
```

The application creates **one long-lived `OnlineRecognizer`** and a **new `OnlineStream` for each VAD-detected utterance**.

---

# Installation

Python **3.10 or newer** is required.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

# Development and Tests

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Run the test suite:

```powershell
python -m pytest -q
```

Run the application self-test:

```powershell
python main.py --self-test
```

> `--self-test` checks application orchestration only. It does **not** validate real speech recognition accuracy.

---

# Model Setup

The application does **not** download the model automatically.

Download the exact Shenava Sherpa-ONNX model manually.

From the **root directory of the repository**, run:

```powershell
hf download mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26 `
  --revision 4be3d2375c98a985154122d69b43360eb8bdca5a `
  model.int8.onnx tokens.txt `
  --local-dir models/shenava
```

The following files must exist:

```text
models/
└── shenava/
    ├── model.int8.onnx
    └── tokens.txt
```

`tokens.txt` is expected to contain **1025 tokens**.

For the complete model provisioning details, see:

```text
models/shenava/README.md
```

---

# Basic Run

Start the application:

```powershell
python main.py
```

Useful commands:

```powershell
python main.py --output-mode console
```

Run without the overlay and keyboard injector:

```powershell
python main.py --no-overlay --no-inject --output-mode console
```

List available audio devices:

```powershell
python main.py --list-devices
```

Use a specific microphone:

```powershell
python main.py --audio-device 1
```

---

# Debug Mode

For troubleshooting, run:

```powershell
python main.py `
  --no-overlay `
  --no-inject `
  --output-mode console `
  --require-streaming `
  --second-pass off `
  --log-level DEBUG
```

At startup, the application reports important runtime information such as:

* Model path
* Tokens path
* Sherpa-ONNX version, when available
* Streaming support
* Input device
* Sample rate
* VAD thresholds
* Adaptive VAD status
* Decoder
* Second-pass mode

Runtime logs help distinguish between:

* microphone/audio delivery problems
* silent or very low microphone input
* `SPEECH_START`
* decoder activity
* endpoint finalization
* final output

Use `DEBUG` logging when you need detailed decoder information.

---

# Real WAV Verification

For real ASR validation, use a **PCM16, 16 kHz, mono WAV** file.

Example:

```powershell
python tools/verify_pipeline.py `
  --model models/shenava/model.int8.onnx `
  --tokens models/shenava/tokens.txt `
  --wav path\to\real_16khz_mono_pcm16.wav
```

Do **not** use `--allow-endpoint` when testing the normal production streaming path.

### Important

A successful:

```powershell
python main.py --self-test
```

does not mean that the ASR model works correctly.

Real transcription must be tested with **real microphone input or real speech WAV files**.

---

# Configuration

The application can be configured through environment variables.

| Variable                    | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `SHENAVA_MODEL_PATH`        | Path to `model.int8.onnx`                              |
| `SHENAVA_TOKENS_PATH`       | Path to `tokens.txt`                                   |
| `SHENAVA_REQUIRE_STREAMING` | Require streaming mode. Default: `1`                   |
| `SHENAVA_NUM_THREADS`       | Number of Sherpa-ONNX CPU threads                      |
| `SHENAVA_SECOND_PASS`       | `greedy` or `off`                                      |
| `SHENAVA_AUDIO_DEVICE`      | Microphone device index or name                        |
| `SHENAVA_VAD_ADAPTIVE`      | `1` to enable adaptive VAD, `0` to disable             |
| `SHENAVA_OUTPUT_MODE`       | `overlay`, `inject`, `both`, `clipboard`, or `console` |
| `SHENAVA_LOG_LEVEL`         | `DEBUG`, `INFO`, `WARNING`, or `ERROR`                 |

### Streaming

`SHENAVA_REQUIRE_STREAMING` is enabled by default:

```text
SHENAVA_REQUIRE_STREAMING=1
```

Set it to `0` only when explicitly testing endpoint-fallback behavior or development tools.

---

# Command-Line Options

Important CLI options include:

```text
--model
--tokens
--threads
--audio-device
--no-adaptive-vad
--second-pass
--output-mode
--no-overlay
--no-inject
--log-level
```

CLI options override the corresponding environment variables.

---

# Output Modes

### `overlay`

Shows the final text in the desktop overlay.

### `inject`

Types or pastes stable final text into the currently focused application.

### `both`

Displays the text in the overlay and sends it through the injector.

### `clipboard`

Copies the final utterance to the system clipboard.

### `console`

Prints the final text to the terminal.

---

# Processing Pipeline

The transcription pipeline is intentionally deterministic:

```text
Raw ASR
   ↓
Unicode normalization
   ↓
Medical terminology
   ↓
Number normalization
   ↓
Measurements / blood pressure
   ↓
Clinical text cleanup
   ↓
Stable final text
```

Unknown or uncertain text is preserved rather than automatically replaced with a guessed medical term.

The application does not perform clinical reasoning or generate information that was not present in the speech.

---

# Limitations

* The Shenava model must be available locally.
* The application does not download models automatically.
* Inference is CPU-only by design.
* The selected Shenava model must pass the startup streaming lifecycle check.
* The RMS VAD is intentionally simple and threshold-based.
* Low microphone gain is reported for diagnosis but is not automatically corrected.
* Medical text processing is deterministic and conservative.
* Unknown text is preserved.
* There is no automatic clinical inference.
* Real **WER, CER, latency, microphone performance, and keyboard injection behavior** must be tested on the actual deployment machine.

---

# Recommended Validation

For a reliable deployment test, validate these components separately:

```text
1. Microphone
2. Audio format
3. VAD
4. Streaming decoder
5. Final text
6. Medical post-processing
7. Overlay
8. Keyboard injection
9. Clipboard
10. End-to-end latency
```

A successful application startup alone does not guarantee accurate real-time transcription.
