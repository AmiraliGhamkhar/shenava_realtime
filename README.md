# Shenava realtime Persian medical transcription

A small local-first Python 3.10+ desktop transcription pipeline. No LLM,
embeddings, vector database, edit-distance or semantic matching.

```text
16 kHz mono audio → adaptive RMS VAD → bounded queue
  → Shenava-Koochik-v1.0 streaming CTC decoder (sherpa-onnx, CPU, INT8)
  → stabilization → endpoint second pass → Persian normalization → protected spans
  → token Aho–Corasick → deterministic span resolver → typed grammars
  → canonical stable text → overlay / keyboard / clipboard
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

The core/test dependencies are NumPy and pytest (`requirements-dev.txt`;
`requirements.txt` is a full-stack aggregate of all three files). The
self-test uses **synthetic
speech activity and a synthetic ASR backend**, but real capture/VAD orchestration,
ASR worker, stabilization, normalization and extraction. It does not measure
recognition accuracy or test a microphone.

For recognition, install `sherpa-onnx` (CPU) — there is no PyTorch, NeMo or
GPU dependency anywhere in the runtime path. `sherpa-onnx` bundles its own
ONNX Runtime, so nothing else is required. This runs on any platform
`sherpa-onnx` publishes CPU wheels for (Linux/macOS/Windows). Keep the desktop
process on a machine with microphone/display access.

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

**Shenava-Koochik-v1.0** (114M), FastConformer, **CTC-only streaming export**
running on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + ONNX Runtime
(CPU, INT8, greedy CTC decoding). There is no RNNT head, no second pass beam
search over raw emissions, and no GPU/CUDA path in this backend — see
`docs/REVIEW.md` / the migration report for what was intentionally dropped
when this repository moved off NeMo/PyTorch.

- **Source**: HF repo
  [`mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26`](https://huggingface.co/mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26),
  pinned revision `4be3d2375c98a985154122d69b43360eb8bdca5a`.
- **License: CC-BY-NC 4.0** (inherited from the parent
  `Reza2kn/Shenava-Koochik-v1.0` checkpoint) — **non-commercial**. Confirm this
  is compatible with your deployment before using the model.
- **Canonical local path**: `models/shenava/` (git-ignored). See
  [`models/shenava/README.md`](models/shenava/README.md) for the exact
  `hf download` command, pinned revision and sha256 checksums of both files.
  The `Shenava-Koochik-v1.5/` directory in the repo root is a leftover git
  submodule pointer from the NeMo era (a `.nemo` checkpoint location); nothing
  in this codebase reads it any more and no config defaults to it. Do
  **not** put files there — `models/shenava/` is the only path the backend
  or its documentation reference.

Provision the model once with the documented command in
`models/shenava/README.md`, then set `SHENAVA_MODEL_PATH` /
`SHENAVA_TOKENS_PATH` (defaults already point at `models/shenava/`) or use
`--model` / `--tokens`. **Weights are not included in this repository.**

Missing local files fail clearly **before importing sherpa-onnx**. There is
**no automatic network fallback anywhere in the runtime path** — provisioning
is always the manual `hf download` command in `models/shenava/README.md`.

## Run

### 1. Default: streaming CTC (sherpa-onnx, CPU, INT8)

```bash
python main.py --model models/shenava/model.int8.onnx \
  --tokens models/shenava/tokens.txt --device cpu
# Strict streaming mode (recommended for deployment validation; the GATE
# never falls back to an offline decoder regardless of this flag):
python main.py --require-streaming --output-mode console --no-overlay --no-inject
# Disable the utterance-end second-pass decoder:
python main.py --second-pass off
python main.py --list-devices
```

There is no `--decoder`/RNNT option any more: this backend is CTC-only. There
is also no `second_pass=context` (CTC beam + hotword biasing) any more —
sherpa-onnx's public Python API exposes only decoded text, not the raw
per-frame emissions or tokenizer object that mode needed; it is rejected at
startup with a message naming `greedy`/`off` as the alternatives.

### 2. Benchmark / evaluation

```bash
# Real audio (this is the one that measures recognition accuracy):
python tools/evaluate_audio.py corpus.jsonl --report ctc.json \
  --model models/shenava/model.int8.onnx --tokens models/shenava/tokens.txt
python tools/evaluate_audio.py --compare ctc.json other.json

# Deterministic text regression (NOT recognition accuracy):
python tools/evaluate_medical.py tests/corpus/medical_regression.jsonl
```

`corpus.jsonl` rows are `{"audio": "clips/0001.wav", "reference": "..."}` with
paths relative to the metadata file; WAVs must be 16 kHz mono (never resampled
silently). The evaluator reports WER, CER, S/I/D counts, medical-term,
drug-name, dose/number, BP and abbreviation error rates, forced-boundary error
rate, latency and RTF, plus an error attribution split (acoustic /
normalization / terminology / numbers / endpointing).

**No accuracy claim is made in this repository.** The numbers on the model card
are the publisher's, measured on their benchmark with their normalizer; run the
evaluator on your own corpus before drawing conclusions.

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

- The ASR backend (`shenava_realtime/asr_backend.py`) owns exactly **one**
  long-lived `sherpa_onnx.OnlineRecognizer` for the process, built once at
  startup (CPU, INT8, greedy CTC decoding). **Every VAD segment gets a fresh
  `OnlineStream`** — streams are never reused/reset across segments, since a
  new stream starts with a zero-filled decoder cache (verified state-clean
  lifecycle; see the migration report). Sherpa-onnx's own endpointer is always
  disabled (`enable_endpoint_detection=False`): the existing RMS VAD is the
  **only** endpointer.
- At a VAD end the engine appends 0.5 s of trailing silence to the stream
  (`FINALIZE_TAIL_PADDING_S` in `asr_backend.py`) before calling
  `input_finished()`, so the last word is not dropped by the streaming
  encoder's lookahead, drains all ready decode steps, reads the final text,
  then discards the stream.
- Audio is float32 mono in `[-1, 1]` at 16 kHz throughout; `int16 -> float32`
  conversion happens at most once (in the WAV replay tools; the live
  microphone path already delivers float32 via `sounddevice`).
- **GATE (hard)**: if the configured model does not load and behave as an
  online/streaming recognizer, the backend raises `StreamingUnavailable` (when
  `require_streaming` is set) or the load itself fails with `ModelLoadError`.
  There is **no fallback to `sherpa_onnx.OfflineRecognizer`** anywhere in this
  codebase.
- `partial_interval_s` (default 0.5 s) batches new mic blocks before handing
  them to the streaming recognizer; it does not change how much audio the
  encoder itself needs to look ahead (that is fixed by the ONNX export).
- **Utterance-end second pass** (config `asr.second_pass`; CLI `--second-pass`;
  env `SHENAVA_SECOND_PASS`). Live streaming greedy stays the primary path; the
  second pass runs only at **natural** endpoints (never on forced cuts), only
  for utterances of at least `second_pass_min_utterance_s` (0.5 s), and only in
  endpoint-commit mode:
  - `off` — streaming greedy only;
  - `greedy` (default) — one offline greedy re-decode of the same segment
    audio via the same recognizer (`backend.transcribe()`), full context, no
    bias.
  - `context` (CTC beam search + reviewed-terminology hotword biasing) **is
    not available on this backend and is rejected at startup.** It needed raw
    per-frame emission logits and the model's own BPE tokenizer object — both
    NeMo-specific internals that sherpa-onnx's public Python API does not
    expose (only decoded text). `ASRConfig(second_pass="context")` raises
    `ValueError` naming `greedy`/`off` as the supported alternatives.
  A second-pass result that differs from the streaming text replaces the
  unemitted final delta and adds the structured review reason
  `decoder_disagreement`. Statistics: `engine.get_statistics()["second_pass"]`
  and `second_pass_stats` (`runs`/`rewrites`/`fallbacks`).
- Default VAD: onset RMS .015, offset .008, 250 ms onset confirmation, 700 ms
  endpoint silence, 320 ms pre-roll, 20 s maximum segment. Tune RMS thresholds
  to your microphone's gain/noise; this detector is not speech classification.
  Continuous speech creates explicit END/START boundaries without replaying
  audio. Phrases crossing a forced boundary may lose linguistic context.
  There is **no denoising** — but each completed segment gets lightweight,
  deterministic audio diagnostics (RMS, peak, clipping ratio, level
  `silent|low|ok|clipped`) that are logged and exposed in
  `AudioCapture.get_statistics()` (including `clipped_segments` /
  `low_level_segments` counters) so a quiet or clipping microphone is visible.
  Diagnostics never alter the transcript.
  Forced cuts (VAD segment cap, ASR window cap) are tagged and never treated
  as natural endpoints: a number phrase left open at the cut (``سی و`` |
  ``پنج``) keeps its spoken words unparsed and logs a warning instead of being
  completed into a wrong value (``30`` + ``5``). A phrase cut *between* two
  complete halves (``سی`` | ``پنج``) can still parse as two values — no
  context model decides this; treat forced-split output as reviewable.
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
- A capture heartbeat detects an input device that stops delivering blocks
  (`dropout_timeout_s`, default 2 s; distinct from queue overflow, which means
  blocks arrive faster than they are consumed). The stall is logged as an
  error, counted in `dropouts`/`last_error`, and an open utterance is aborted
  through the explicit discontinuity path instead of stalling silently.
- In-memory session text is a bounded rolling tail (20,000 characters/200
  segment history), **not an unlimited archival record**.

### Configuration

```bash
python main.py --save-config config.json
python main.py --config config.json
```

Precedence: defaults → JSON → `.env`/exported environment → CLI. Exported values
win over `.env`. Useful environment variables:

`SHENAVA_ASR_BACKEND` (unprefixed `ASR_BACKEND` also accepted; only
`sherpa_onnx_ctc` is supported), `SHENAVA_MODEL_PATH`, `SHENAVA_TOKENS_PATH`,
`SHENAVA_DEVICE` (`cpu` only), `SHENAVA_NUM_THREADS`, `SHENAVA_SAMPLE_RATE`
(`16000` only), `SHENAVA_FEATURE_DIM` (default `80`, validated against the
loaded model), `SHENAVA_DECODING_METHOD` (`greedy_search` only),
`SHENAVA_PARTIAL_INTERVAL_S`, `SHENAVA_REQUIRE_STREAMING`,
`SHENAVA_AUDIO_DEVICE`, `SHENAVA_OUTPUT_MODE`, `SHENAVA_INJECTOR_MODE`,
`SHENAVA_LOG_LEVEL`, `SHENAVA_SECOND_PASS` (`off|greedy`),
`SHENAVA_VAD_ADAPTIVE`, `SHENAVA_BENCHMARK_MODE`.

**Removed in this migration** (each raises a clear `ValueError` naming its
replacement if set): `SHENAVA_RIGHT_CONTEXT`, `SHENAVA_MODEL_NAME`,
`SHENAVA_ALLOW_DOWNLOAD`, `SHENAVA_DECODER`, `SHENAVA_BEAM_SIZE`,
`SHENAVA_HOTWORD_ALIASES`, `SHENAVA_HOTWORD_PHONETIC`,
`SHENAVA_CUDA_GRAPH_STREAMING`. Pointing `SHENAVA_MODEL_PATH` at a legacy
`.nemo` file also raises a clear migration error naming `SHENAVA_MODEL_PATH`
(`.onnx`) and `SHENAVA_TOKENS_PATH` instead.

### Adaptive VAD

The RMS detector keeps its hysteresis, minimum speech/silence durations,
bounded pre-roll and hard segment cap. On top of those it now estimates a
local noise floor by *minimum statistics* over a bounded window of non-speech
frames, and derives onset/offset from it. Adaptation is one-directional and
clamped: while the measured floor sits below the configured `vad_offset_rms`
(a quiet room, or an unmeasured one) the configured static thresholds are used
verbatim, and thresholds can never exceed `vad_adaptive_max_gain` times them.
A quiet speaker therefore behaves exactly as before; only genuine background
noise raises the bar. Disable with `--no-adaptive-vad` / `SHENAVA_VAD_ADAPTIVE=0`.

Documented limit: noise already louder than the configured onset reads as
speech, because the floor is only measured in silence. That case still needs
measured thresholds — the adaptive layer refines tuning, it does not replace it.
JSON exposes the VAD, UI and optional output settings in `config.py`.
`commit_on_endpoint=false` retains an experimental early-commit mode for API
compatibility; it cannot retract a committed prefix and is not recommended for
medical injection — startup **refuses** that mode when injection or
`--clinical-sqlite`/`--clinical-jsonl` is enabled, instead of trusting a
warning. Conflicting rewritten tails are suppressed, not re-injected.

Compatibility: callbacks still accept `(text, confidence)` and decode results
retain their second numeric slot, but the backend always supplies **0.0 =
unknown**. There is no calibrated confidence estimator or display.
`confidence_threshold`, `show_confidence` are legacy config fields with no
runtime effect. Heuristic confidence functions and the unsafe window decoder
were removed (see `docs/REVIEW.md`). Consecutive-word deletion and injector
text-equality deduplication are disabled by default: legitimate repetitions must
not be deleted. The FST/stabilizer handle ASR output deterministically instead.

## Deterministic text processing

The production path is structured and span-based:

```text
normalized ASR → token offsets / protected spans
 → token Aho–Corasick (all overlapping candidates)
 → leftmost-longest / priority / risk / context resolver
 → Persian numbers → typed measurements / BP → medication-dose grammar
 → dictionary concepts → negation / laterality → safety validation → render
```

Reviewed structured rules live in `shenava_realtime/data/terminology.json`; each
has an ID, canonical form, explicit spoken/alias/phonetic forms, category,
specialty, priority, risk and context/sensitivity flags, plus the optional
bounded `bias` used only for decoder hotword biasing. `terminology.py`
validates this schema. `aho_corasick.py` is a dependency-free matcher abstraction,
`span_resolver.py` owns selection policy, `number_grammar.py` preserves semantic
number offsets, and `medical_grammar.py` contains measurement, medication and
conservative context grammars. The old `TrieFST` remains importable for NER/API
compatibility but is no longer the terminology normalization engine.

The matcher returns all nested/overlapping candidates and never rewrites during
scanning. Resolution is deterministic: leftmost, longest, explicit priority,
risk policy, then bounded context constraints. High-risk or context-required
rules are preserved and set `ProcessingResult.review_required`; optional
contextual/LLM resolution is an explicit disabled extension point. There is no
fuzzy matching or unrestricted generation.

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
Conflicting normalized rules raise at `PostProcessor` construction time, before
any audio runs — never lazily on first match. For metadata-rich deployments,
construct `TerminologyRule` records and `MedicalNormalizationPipeline` directly.

`PostProcessor.process()` remains string-in/string-out. Its `last_result` exposes
numbers, measurements, medication/dose/frequency fields, clinical assertions,
anatomy/laterality, protected spans and explicit review reasons. Offsets refer to
the Unicode-normalized input and remain half-open. Date/time, existing ratios,
important abbreviations, parsed numbers, measurements and medication expressions
are protected from later rewrites. A unit, dose, relationship, assertion or side
is never supplied when its required explicit structure is absent.

Negation scope is deliberately local and conservative (`بدون تب`, `تب ندارد`,
`وجود ندارد`, `مشاهده نشد`, `منفی است`); historical/possible cues and right,
left or bilateral anatomy are represented as metadata rather than rewritten.
This is not full clinical reasoning. Unknown/ambiguous text remains unchanged.

After rendering, `value_validation.py` performs deterministic plausibility
checks on the *typed* spans (percentage 0–100, body temperature 24–45 °C /
75–113 °F, pulse 20–300 bpm, non-positive doses, explicit BP connector pairs
with systolic > diastolic inside 60–260 / 30–160). An obviously malformed
value is **preserved verbatim** and adds a structured reason
`suspicious_value:<kind>=<value>` plus a `ProcessingResult.value_issues`
entry — it is never silently corrected, and abnormal-but-possible readings
(e.g. SpO2 84%) are not flagged. `ProcessingResult.review_reasons` (also
`pipeline.last_review_reasons` / `engine.last_review_reasons`) is a stable
sorted list drawn from: `rare_medical_term`, `drug_name`, `dose_value`,
`numeric_value`, `negation_sensitive`, `laterality_sensitive`,
`forced_boundary`, `decoder_disagreement`, `suspicious_value:*`, and
`unsafe_terminology:<rule id>`. Reasons are flags for review; they do not
alter the output text.

Recurring ASR misspellings of a known term can be collected for human review
with `tools/collect_term_variants.py CORPUS.jsonl --term "CABG" --suggest`:
reference → observed raw variant → reviewed candidate → terminology rule. The
suggested fragment is printed for review only and is **never** written into
`terminology.json` automatically.

## Optional clinical extraction / persistence

```bash
python main.py --model models/shenava/model.int8.onnx --tokens models/shenava/tokens.txt \
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
**Accepted risk, disclosed:** there is deliberately **no reconciliation on
restart** between what was injected/displayed and what reached the clinical
sinks. A crash (or kill) between an injection and the clinical write loses that
utterance's record permanently; a restart never replays or back-fills it. The
reviewed transcript/console output is the only complete record of the session.

**Privacy:** transcript saving and clinical persistence are opt-in. Console,
clipboard and injection intentionally expose text to the selected destination.
Routine logs record lengths/timing, not transcript contents, and `app.log` rotates.
SQLite/JSONL are plaintext; use restricted permissions, encrypted storage,
retention/deletion policies and consent. Files grow on disk until you archive or
delete them; there is no hidden retention service. Runtime/model files are ignored
by Git. Third-party sherpa-onnx/ONNX Runtime logs should also be audited
before handling patient data.

## Verification and deployment gate

```bash
pytest -q
python main.py --self-test
python tools/evaluate_medical.py tests/corpus/medical_regression.jsonl
python tools/verify_pipeline.py \
  --model models/shenava/model.int8.onnx --tokens models/shenava/tokens.txt \
  --wav /data/consented-persian-sample.wav --device cpu
```

`evaluate_medical.py` reports raw-ASR and post-processing WER/CER separately,
category error rates (medical terms, medications, numbers, units, dose,
negation and laterality), a **forced-boundary error rate** (corpus rows with
`"forced": true` run the pipeline's forced-endpoint path and must keep the
preserved spoken words), a weighted **medical essential error** (reference
characters inside reviewed entity spans cost 3×, ordinary prose 1×), and
entity precision/recall/F1. The bundled corpus covers code-switching, drugs,
abbreviations, measurements, rates, negation, laterality, radiology/ultrasound,
cardiology, nursing and forced splits; it is a small engineering regression
fixture, not a representative clinical benchmark; its scores must not be
presented as clinical accuracy or generalized improvement. Use a versioned,
consented domain corpus for deployment decisions.

Replay requires PCM16, mono, 16 kHz WAV and runs at microphone speed. It uses the
real backend unless `--self-test` is explicitly supplied. Streaming is
required unless `--allow-endpoint` is passed. Output reports errors, overruns,
text and extracted JSON. `tools/profile_asr.py --streaming` can measure model
costs on synthetic noise, but is not an accuracy benchmark.

**Verification in this checkout:** the full pytest suite passes without the
model (mock `sherpa_onnx` recognizer, see `tests/fake_sherpa_onnx.py`), and
`python main.py --self-test` passes (synthetic backend). Real model
compatibility, microphone/desktop behavior, Persian WER/CER parity against the
retired NeMo backend, and CPU RTF at 1/2/4 threads have **not** been verified
in this checkout: `model.int8.onnx` (132 MB) could not be downloaded in the
migration sandbox (network egress to huggingface.co's LFS/CDN endpoints was
blocked; `tokens.txt` alone was retrievable and its checksum recorded in
`models/shenava/README.md`). Do not treat the passing mock-based test suite as
production model certification. Before deployment: provision the real model
per `models/shenava/README.md`, run `tools/verify_pipeline.py` against a real
WAV, run `tools/evaluate_audio.py` on a consented Persian corpus, confirm no
overruns, and review representative medical dictations.

See [the inspection and change notes](docs/REVIEW.md) for the original defects
and remaining limitations.
