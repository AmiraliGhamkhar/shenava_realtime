# Shenava Real-time ASR

Real-time Persian speech recognition with a floating overlay, global hotkeys and
automatic text injection into the focused application. Powered by the
[Shenava-Koochik](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) NeMo
FastConformer checkpoint, decoded through its **CTC** head.

Everything downstream of the acoustic model is **deterministic**: a finite-state
(trie) rewriter for medical terms and units, a rule-based Persian normalizer and
a table-driven spoken-number converter. There is no LLM, no embedding model, no
vector database, no fuzzy matching, and no network service in the pipeline.

```text
Microphone
  → Audio capture (16 kHz mono, bounded queue)
  → RMS VAD (speech/silence hysteresis, pre-roll)
  → Shenava streaming CTC (cache-aware when the checkpoint supports it,
                           otherwise a bounded sliding window)
  → Transcript stabilization (committed_text / current_partial)
  → FST post-processing (normalization, medical terms, units, numbers, punctuation)
  → Stable text deltas
  → Clipboard (Windows, Unicode) / keyboard injection / overlay / console
```

## Install

Python 3.10+. Two dependency sets keep the install small:

```bash
# 1. runtime: capture, overlay, hotkeys, injection
pip install -r requirements.txt

# 2. the ASR model stack (torch + NeMo). Only needed to actually recognise speech.
pip install -r requirements-asr.txt

# optional: run the tests
pip install -r requirements-dev.txt
```

The bundled torch wheel may be CPU-only; the app auto-detects CUDA and uses it
when a CUDA-enabled torch is installed, otherwise it runs on CPU.

## Model

The engine prefers a **local** checkpoint and only falls back to a Hugging Face
download when the file is missing.

| Setting | Default |
|---|---|
| `SHENAVA_MODEL_PATH` | `./shenava-koochik/shenava-koochik-v1.0.nemo` |
| `SHENAVA_MODEL_NAME` | `Reza2kn/Shenava-Koochik-v1.0` |
| `SHENAVA_DEVICE` | `auto` (`cpu` / `cuda` / `cuda:1`) |
| `SHENAVA_DECODER` | `ctc` |
| `SHENAVA_NUM_THREADS` | `4` |

The repository's `.env` is read on start (a tiny built-in loader; no
`python-dotenv` dependency), and environment variables always win over it.

## Run

```bash
python main.py                      # foreground
python run_app.py                   # detached, logs to app.log
python main.py --list-devices       # enumerate input devices
python main.py --device cpu --threads 8 --output-mode inject
python main.py --no-overlay --no-inject --output-mode console
python main.py --save-config config.json
python main.py --help
```

Startup takes ~30 s while the 460 MB checkpoint loads. Speak, and the recognised
text appears in the overlay and is inserted into the focused window.

## Hotkeys

| Combo | Action |
|---|---|
| `Ctrl+Alt+R` | Pause / resume recording |
| `Ctrl+Alt+O` | Show / hide the overlay |
| `Ctrl+Alt+I` | Enable / disable text injection |
| `Ctrl+Alt+C` | Clear the current transcript |
| `Ctrl+Alt+Q` | Quit |

## Output modes (`--output-mode` / `SHENAVA_OUTPUT_MODE`)

| Mode | Behaviour |
|---|---|
| `both` | overlay + injection (default) |
| `overlay` | overlay only |
| `inject` | inject only |
| `clipboard` | copy each finished utterance to the clipboard |
| `console` | print to the terminal |

Injection mode (`SHENAVA_INJECTOR_MODE`): `auto` (default) picks the **clipboard
paste** path on Windows — the only reliable way to insert Persian text, since
simulated keystrokes go through the active keyboard layout — and the keyboard
path elsewhere. The previous clipboard content is saved and restored after each
paste.

## Post-processing

Deterministic and idempotent, in this order (see `shenava_realtime/postprocessor.py`):

1. **Persian/Arabic Unicode normalization** — `ي→ی`, `ك→ک`, `أ/إ/ٱ→ا`, `ة→ه`,
   harakat/tatweel/ZWJ removal, Arabic-Indic and Persian digits, whitespace and
   punctuation spacing, ZWNJ for clitics (`می‌رود`, `کتاب‌ها`).
2. **FST phrase rewriting** — medical terms, abbreviations and units
   (`shenava_realtime/fst.py` + `lexicon.py`).
3. **Repetition removal** — consecutive duplicated words (a common CTC artifact).
4. **Spoken numbers** — digits, `%`, `°`, blood-pressure ratios.
5. **Spacing/punctuation** tidy-up.

| Spoken | Output |
|---|---|
| `کابج` / `سی ای بی جی` | `CABG` |
| `پی سی آی` | `PCI` |
| `پنج میلی گرم` | `5 mg` |
| `سی و پنج درصد` | `35%` |
| `صد و بیست روی هشتاد` | `120/80` |
| `ضربان هفتاد و پنج بار در دقیقه` | `ضربان 75 bpm` |
| `دمای بدن سی و هشت درجه سانتیگراد` | `دمای بدن 38°C` |

Add your own vocabulary without touching code:

```python
from shenava_realtime.postprocessor import PostProcessor

PostProcessor(extra_terms={"آزیترومایسین": "azithromycin"})
```

### Why a Python trie instead of Pynini/OpenFst?

`pynini` needs an OpenFst C++ toolchain and has no reliable Windows wheels, which
conflicts with the "minimal, offline, Windows-first" goal. `fst.py` implements
the same thing the rules need — a deterministic automaton over word tokens with
longest-match semantics and an output string per accepting state — in a single
120-line module with no dependencies. The rule tables are plain data, so they
can be compiled by Pynini later without changing any logic. No fuzzy matching, no edit distance, no
embeddings.

## Configuration

Precedence: **CLI flags → environment → `--config` JSON → defaults**.

```bash
python main.py --save-config config.json   # dump every knob
python main.py --config config.json
```

Environment variables: `SHENAVA_MODEL_PATH`, `SHENAVA_MODEL_NAME`,
`SHENAVA_DEVICE`, `SHENAVA_DECODER`, `SHENAVA_NUM_THREADS`,
`SHENAVA_PARTIAL_INTERVAL_S`, `SHENAVA_OUTPUT_MODE`, `SHENAVA_INJECTOR_MODE`,
`SHENAVA_AUDIO_DEVICE`, `SHENAVA_LOG_LEVEL`, `SHENAVA_DEBUG`.

Tuning that matters in practice (`shenava_realtime/config.py`):

| Key | Default | Meaning |
|---|---|---|
| `audio.vad_onset_rms` / `vad_offset_rms` | `0.015` / `0.008` | hysteresis band; raise both in a noisy room |
| `audio.vad_min_silence_ms` | `700` | silence that ends an utterance |
| `audio.vad_pre_speech_ms` | `320` | pre-roll kept so word starts are not clipped |
| `asr.partial_interval_s` | `0.5` | how often a partial decode runs |
| `asr.left_context_s` / `max_window_s` | `2.0` / `10.0` | streaming window size |
| `asr.holdback_words` | `2` | words kept un-committed until they stop changing |
| `postprocess.digits` | `ascii` | `persian` for ۱۲۰/۸۰ |

## Architecture

| Module | Responsibility |
|---|---|
| `audio_capture.py` | PortAudio callback → bounded queue → single consumer thread → VAD |
| `vad.py` | RMS state machine (SILENCE → PENDING → SPEECH) with pre-roll and flush |
| `asr_backend.py` | lazy NeMo loading, CTC decode, score → confidence, cache-aware stream adapter |
| `streaming.py` | `CacheAwareDecoder` / `WindowedDecoder` — bounded work per step |
| `stabilizer.py` | `committed_text` / `current_partial`, prefix-only commits |
| `pipeline.py` | decoder → stabilizer → FST → **deltas** (emitted exactly once) |
| `postprocessor.py`, `fst.py`, `lexicon.py`, `fa_numbers.py`, `text_normalize.py` | deterministic text layer |
| `realtime_engine.py` | one worker thread that owns the model and the pipeline |
| `injector/` | clipboard (ctypes, CF_UNICODETEXT) and keyboard (pynput) backends |
| `overlay/` | Tk overlay; every mutation is marshalled onto the Tk thread |
| `hotkeys/` | pynput listener, callbacks dispatched off the listener thread |

Threading rules the code follows:

* the audio callback only copies and enqueues — it never blocks on inference;
* exactly one thread touches the model, the stabilizer and the post-processor;
* Tk widgets are only touched on the Tk thread;
* every thread has an explicit stop path (sentinel + `join` with a timeout), and
  an open utterance is flushed on shutdown.

## Tests

The suite runs **without** torch, NeMo, sounddevice, pynput or a GPU — the model
is replaced by stubs (`tests/fakes.py`), and one test asserts that no heavy
module leaks into `sys.modules`.

```bash
python -m compileall .
pytest
```

Covered: Persian normalization, medical terms, units, spoken numbers, blood
pressure, transcript stabilization, VAD transitions, streaming window bounds,
delta emission (no duplicates), injection dedupe, configuration and clean
shutdown.

## Development

```bash
python tools/profile_asr.py --seconds 1 2 5 10 --threads 4 --streaming
```

prints offline decode latency/RTF per segment length and the streaming window
cost, using the configured checkpoint.

## Troubleshooting

- **No text in the overlay** — the overlay auto-hides after
  `overlay.auto_hide_delay` seconds of silence and re-shows on the next result.
- **Hotkeys do nothing** — another application may own the combo; pynput is
  required and is checked at startup.
- **Garbled Persian in the target app** — use clipboard injection
  (`SHENAVA_INJECTOR_MODE=clipboard`); it is the default on Windows.
- **Console shows `?` instead of Persian** — `main.py` forces UTF-8 stdio; in
  some terminals also set `PYTHONIOENCODING=utf-8`.
- **Nothing is recognised in a noisy room** — raise
  `audio.vad_onset_rms`/`vad_offset_rms` (see `tools/profile_asr.py` and the
  RMS levels in the debug log).

## Known limitations

- Cache-aware streaming is used only when the checkpoint ships a NeMo streaming
  config; the Shenava FP32 source checkpoint does not, so the bounded sliding
  window is the default path (constant work per step, but not truly cache-aware).
- The stabilizer cannot retract text that was already injected; if the final
  decode rewrites a committed word, the divergence is logged and the new tail is
  appended.
- `holdback_words` trades latency for stability: raise it in noisy conditions,
  lower it for snappier output.
- Unit/term rewriting is exact-match by design; unseen spellings need a new
  entry in `lexicon.py`.
- The overlay and the injection backends are Windows-oriented (Tk overlay,
  Win32 clipboard); Linux/macOS work through pynput and, optionally, pyperclip.
