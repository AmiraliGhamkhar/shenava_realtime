# Shenava realtime Persian medical transcription

A small local-first Python 3.10+ desktop transcription pipeline. No LLM,
embeddings, vector database, edit-distance or semantic matching.

```text
16 kHz mono audio → RMS VAD → Shenava cache-aware greedy CTC
  → transcript stabilization → Persian normalization → medical Trie/FST
  → numbers/units/punctuation → stable text → overlay / keyboard / clipboard
                                         └→ optional dictionary NER → rules
                                             → JSONL / SQLite
```

**Clinical safety:** output is dictated text and extracted mentions, not a
verified medical record. Review negation, drug names, doses and abnormal values.
ASR and dictionary errors are possible. Clinical records always carry
`review_required: true`. This is not a validated medical device.

## Install and provision

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
python main.py --self-test
```

The core/test dependencies are NumPy and pytest. The self-test uses **synthetic
speech activity and a synthetic ASR backend**, but real capture/VAD orchestration,
ASR worker, stabilization, normalization and extraction. It does not measure
recognition accuracy or test a microphone.

For recognition, install matching PyTorch/torchaudio wheels for your platform,
then the ASR stack. The native adapter targets NeMo **2.4.0** and the requirements
specify a matching Torch/torchaudio 2.7.1 pair. Use a supported Linux/CUDA or CPU
environment; availability of NeMo's compiled dependencies on native Windows
varies. Keep the desktop process on a machine with microphone/display access.

```bash
pip install -r requirements-asr.txt
pip install -r requirements-desktop.txt
```

Desktop requirements include sounddevice/PortAudio, pynput and pyperclip. Linux
may also need the OS packages for PortAudio, Tk and an X11 clipboard utility.
Pynput/global shortcuts and injection may be restricted by Wayland or OS privacy
permissions. An Arena/browser sandbox cannot access your local microphone or
inject into your local desktop; this repository is not a browser application.

### Models

Primary: **Shenava Koochik v1.0, 114M**, hybrid FastConformer with CTC decoding.
The bundled [model card](shenava-koochik/README.md) documents contexts
`[70,13]`, `[70,6]`, `[70,1]`, `[70,0]` and model provenance.

Provision the trusted `.nemo` checkpoint from
[Reza2kn/Shenava-Koochik-v1.0](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0)
onto local storage, then set `SHENAVA_MODEL_PATH` or use `--model`. The default is
`shenava-koochik/shenava-koochik-v1.0.nemo`. **Weights are not included.**
NeMo checkpoints are executable serialization artifacts: do not load untrusted
files. Verify the publisher's checksum when provisioning.

Missing local files fail clearly **before importing the model stack**. There is
no automatic network fallback. `--allow-download` / `SHENAVA_ALLOW_DOWNLOAD=1`
explicitly enables provisioning with `ASRModel.from_pretrained(model_name)`;
normal deployed inference should leave this disabled.

For a smaller model, select a locally provisioned Shenava Rizeh (32M) or
Rizeh-Pizeh (6.9M) `.nemo` file. Streaming support is checked from the **actual
encoder metadata**, not assumed from its name. No automatic second checkpoint
is loaded, avoiding doubled RAM and surprising model changes.

## Run

```bash
python main.py --model /models/shenava-koochik-v1.0.nemo --device cpu
# Strict native-streaming mode (recommended for deployment validation):
python main.py --model /models/shenava-koochik-v1.0.nemo --require-streaming \
  --right-context 13 --output-mode console --no-overlay --no-inject
python main.py --list-devices
```

Output modes remain `both` (default), `overlay`, `inject`, `clipboard`, `console`.
The overlay shows live partials. **Injection and clinical extraction wait until
an endpoint by default**: neither repeated hypothesis agreement nor a fixed
word holdback can guarantee a number/medical phrase will not be rewritten.
For example `سی` can become `سی و پنج` or part of `سی ای بی جی`.

Hotkeys remain:

| Shortcut | Action |
|---|---|
| Ctrl+Alt+R | pause/resume (pause flushes the current segment) |
| Ctrl+Alt+O | toggle overlay |
| Ctrl+Alt+I | toggle injection |
| Ctrl+Alt+C | clear transcript/abort current segment |
| Ctrl+Alt+Q | shutdown |

Ctrl+C / SIGTERM also request orderly shutdown. `run_app.py` remains a convenience
launcher for detached desktop use; prefer the foreground command while debugging.

## Streaming behavior and tuning

- Native path uses `encoder.get_initial_cache_state()` and
  `model.conformer_stream_step()`, with encoder channel/time/length caches,
  previous CTC predictions, pre-encoder feature overlap and an explicit final
  flush (`keep_all_outputs=True`). There is **no repeated encoder window**.
- The frontend recomputes only bounded hop-aligned STFT overlap, waits for stable
  right-edge features, and applies NeMo online chunk normalization. It requires
  centered STFT with no frame splicing; unsupported frontend configuration is an
  explicit initialization error. Online normalization may differ in accuracy
  from the publisher's offline evaluation—measure on your own audio.
- Default right context is 13 for quality. Choose 6, 1 or 0 for less lookahead.
  Koochik's encoder step is approximately 80 ms; this is **not** an end-to-end
  latency guarantee. NeMo metadata determines chunk/shift/cache sizes.
- `partial_interval_s` (default 0.5 s) batches new mic blocks before handing them
  to the native session; it does not override encoder lookahead.
- Unsupported/offline checkpoints use **endpoint-only CTC**, once per segment,
  with an explicit startup warning. `require_streaming=true` rejects this mode.
  A broken native adapter does not silently fall back.
- Default VAD: onset RMS .015, offset .008, 250 ms onset confirmation, 700 ms
  endpoint silence, 320 ms pre-roll, 20 s maximum segment. Tune RMS thresholds
  to your microphone's gain/noise; this detector is not speech classification.
  Continuous speech creates explicit END/START boundaries without replaying
  audio. Phrases crossing a forced boundary may lose linguistic context.
- ASR segment cap: 22 s including pre-roll. Raw/features/caches/hypotheses are
  reset at each endpoint. Audio and ASR queues default to 32 blocks (~2 s each
  at 64 ms/block); overlay/injector queues are 64 items, clinical queue 32.
- An overrun aborts incomplete recognition instead of joining audio across a
  gap. Check logged errors and `overruns`; use faster hardware, lower lookahead
  or a smaller model rather than hiding overload with a huge queue.
- Model errors abort the affected utterance and allow the next segment to
  recover. A worker that exceeds the shutdown timeout retains its live handle,
  preventing a second worker from starting over it. Python cannot forcibly
  interrupt a hung CUDA/driver call; terminate the process if it never returns.
- In-memory session text is a bounded rolling tail (20,000 characters/200
  segment history), **not an unlimited archival record**.

### Configuration

```bash
python main.py --save-config config.json
python main.py --config config.json
```

Precedence: defaults → JSON → `.env`/exported environment → CLI. Exported values
win over `.env`. Useful environment variables:

`SHENAVA_MODEL_PATH`, `SHENAVA_MODEL_NAME`, `SHENAVA_DEVICE`,
`SHENAVA_NUM_THREADS`, `SHENAVA_RIGHT_CONTEXT`, `SHENAVA_PARTIAL_INTERVAL_S`,
`SHENAVA_REQUIRE_STREAMING`, `SHENAVA_ALLOW_DOWNLOAD`, `SHENAVA_AUDIO_DEVICE`,
`SHENAVA_OUTPUT_MODE`, `SHENAVA_INJECTOR_MODE`, `SHENAVA_LOG_LEVEL`.

`SHENAVA_DECODER` accepts only `ctc`. Invalid numeric ranges are rejected.
JSON exposes the VAD, UI and optional output settings in `config.py`.
`commit_on_endpoint=false` retains an experimental early-commit mode for API
compatibility; it cannot retract a committed prefix and is not recommended for
medical injection. Conflicting rewritten tails are suppressed, not re-injected.

Compatibility: callbacks still accept `(text, confidence)` and decode results
retain their second numeric slot, but the production backend always supplies
**0.0 = unknown**. There is no calibrated confidence estimator or display.
`confidence_threshold`, `show_confidence`, `left_context_s`, `max_window_s` are
legacy config fields with no runtime effect. Heuristic confidence functions and
the unsafe window decoder were removed. Consecutive-word deletion and injector
text-equality deduplication are disabled by default: legitimate repetitions must
not be deleted. The FST/stabilizer handle ASR output deterministically instead.

## Deterministic text processing

Rules in `lexicon.py`, longest-match token trie in `fst.py`, normalization in
`text_normalize.py`, numeric grammar in `fa_numbers.py`.

```text
سي اي بي جي                  → CABG
پنج میلی گرم در دسی لیتر     → 5 mg/dL
دو و نیم میلی گرم            → 2.5 mg
سه ممیز صفر پنج میلی گرم     → 3.05 mg
فشار خون صد و بیست روی هشتاد → BP 120/80
نه درد دارد نه تب            → نه درد دارد نه تب
```

Arabic/Persian letter variants, digit styles, Unicode marks, ZWNJ, Persian/Latin
acronyms, punctuation and explicit units are handled without fuzzy matching.
Unknown text is preserved. Dictionary replacements retain punctuation and
cannot match across sentence punctuation. Decimal punctuation is not separated.
Numbers require an ordered grammar instead of summing adjacent number words.
General `به` / `از` phrases are no longer guessed to be blood pressure. Existing
explicit numeric ratios are retained. This is deliberately a small grammar,
not full Persian language understanding; extend it with reviewed examples/tests.

Custom terminology: `PostProcessor(extra_terms={"spoken form": "canonical"})`.
Conflicting trie rules raise rather than silently replacing each other.

## Optional clinical extraction / persistence

```bash
python main.py --model /models/shenava-koochik-v1.0.nemo \
  --clinical-sqlite clinical.sqlite --clinical-jsonl clinical.jsonl
```

Without these flags/config paths, no clinical worker or database is created.
`clinical.py` is independently usable: `DictionaryNER.extract(text)` and
`extract_record(text)`. It detects only exact listed entity phrases, explicit
adjacent negation, measurements with explicit units, and BP pairs. JSON retains
source text and character offsets. It **does not infer diagnoses, medication-dose
links, unspecified units or missing values**. Unspecified assertion is not a
positive diagnosis. Add domain vocabulary through `DictionaryNER(terms=...)`.

SQLite uses parameterized inserts and one transaction per completed utterance.
JSONL appends one record per line. These independent sinks are not a distributed
atomic transaction; disk errors are explicit and do not block ASR. Queue overflow
reports an unstored record. Inspect logs and reconcile against the reviewed
transcript; neither sink promises delivery after process termination.

**Privacy:** transcript saving and clinical persistence are opt-in. Console,
clipboard and injection intentionally expose text to the selected destination.
Routine logs record lengths/timing, not transcript contents, and `app.log` rotates.
SQLite/JSONL are plaintext; use restricted permissions, encrypted storage,
retention/deletion policies and consent. Files grow on disk until you archive or
delete them; there is no hidden retention service. Runtime/model files are ignored
by Git. Third-party NeMo logs should also be audited before handling patient data.

## Verification and deployment gate

```bash
pytest -q
python main.py --self-test
python tools/verify_pipeline.py --model /models/shenava-koochik-v1.0.nemo \
  --wav /data/consented-persian-sample.wav --device cpu --right-context 13
```

Replay requires PCM16, mono, 16 kHz WAV and runs at microphone speed. It uses the
real backend unless `--self-test` is explicitly supplied. Native streaming is
required unless `--allow-endpoint` is passed. Output reports errors, overruns,
text and extracted JSON. `tools/profile_asr.py --streaming` can measure model
costs on synthetic noise, but is not an accuracy benchmark.

**Verification in this checkout:** the lightweight suite and synthetic headless
startup/replay pass; native-cache tests use a tensor/model double. Real NeMo
checkpoint compatibility, microphone/desktop behavior, Persian WER and CPU/GPU
latency have **not** been verified here: weights and NeMo are absent, and the CPU
PyTorch download attempt failed with a TLS/network error. Do not treat passing
synthetic tests as production model certification. Before deployment, run real
WAV/microphone tests on your hardware, test all chosen contexts and smaller
checkpoints, confirm no overruns, and review representative medical dictations.

See [the inspection and change notes](docs/REVIEW.md) for the original defects
and remaining limitations.
