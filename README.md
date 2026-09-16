# Shenava realtime Persian medical transcription

A small local-first Python 3.10+ desktop transcription pipeline. No LLM,
embeddings, vector database, edit-distance or semantic matching.

```text
16 kHz mono audio → RMS VAD → Shenava cache-aware greedy CTC
  → transcript stabilization → Persian normalization → protected spans
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
# Disable (or restrict) the utterance-end second-pass decoder:
python main.py --model /models/shenava-koochik-v1.0.nemo --second-pass off
python main.py --model /models/shenava-koochik-v1.0.nemo \
  --second-pass context --hotword-specialty cardiology
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
  Values outside `{0, 1, 6, 13}` are rejected when the configuration loads.
  A streaming-capable encoder that does not list `[70, right_context]` in its
  metadata fails at startup with the supported contexts; only checkpoints with
  no native streaming support fall back to endpoint-only decoding (with an
  explicit warning, and rejected under `require_streaming`).
  Koochik's encoder step is approximately 80 ms; this is **not** an end-to-end
  latency guarantee. NeMo metadata determines chunk/shift/cache sizes.
- `partial_interval_s` (default 0.5 s) batches new mic blocks before handing them
  to the native session; it does not override encoder lookahead.
- Unsupported/offline checkpoints use **endpoint-only CTC**, once per segment,
  with an explicit startup warning. `require_streaming=true` rejects this mode.
  A broken native adapter does not silently fall back.
- **Utterance-end second pass** (config `asr.second_pass`; CLI `--second-pass`;
  env `SHENAVA_SECOND_PASS`). Live streaming greedy stays the primary path; the
  second pass runs only at **natural** endpoints (never on forced cuts), only
  for utterances of at least `second_pass_min_utterance_s` (0.5 s), and only in
  endpoint-commit mode:
  - `off` — streaming greedy only (fully disables the pass and its hotwords);
  - `greedy` (default) — one offline greedy re-decode of the same segment with
    the same model/BPE vocabulary and full context (works with any backend that
    has `transcribe`);
  - `context` — CTC beam search over the model's own emissions with decoder-time
    **hotword biasing** built from the reviewed terminology rules (no second
    dictionary; biases are small log-prob prefixes capped at 1.5 by category —
    the list is bounded to `hotword_max` phrases, restricted to a specialty
    with `--hotword-specialty` / `SHENAVA_HOTWORD_SPECIALTY`; units are never
    boosted, and categories without a default boost, such as anatomy, join
    only through an explicit reviewed `bias` on the rule). Requires the NeMo
    backend's emissions/tokenizer capability — a missing capability is a
    startup error, and a decode failure keeps the streaming text, counts a
    fallback, and never switches modes silently.
  A second-pass result that differs from the streaming text replaces the
  unemitted final delta and adds the structured review reason
  `decoder_disagreement`. Statistics: `engine.get_statistics()["second_pass"]`
  and `second_pass_stats` (`runs`/`rewrites`/`fallbacks`). In endpoint-only
  mode the single offline decode *is* the full-context pass, so the second
  pass is skipped (the decoder retains no separate streaming audio).
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

`SHENAVA_MODEL_PATH`, `SHENAVA_MODEL_NAME`, `SHENAVA_DEVICE`,
`SHENAVA_NUM_THREADS`, `SHENAVA_RIGHT_CONTEXT`, `SHENAVA_PARTIAL_INTERVAL_S`,
`SHENAVA_REQUIRE_STREAMING`, `SHENAVA_ALLOW_DOWNLOAD`, `SHENAVA_AUDIO_DEVICE`,
`SHENAVA_OUTPUT_MODE`, `SHENAVA_INJECTOR_MODE`, `SHENAVA_LOG_LEVEL`,
`SHENAVA_SECOND_PASS` (`off|greedy|context`), `SHENAVA_HOTWORD_SPECIALTY`,
`SHENAVA_HOTWORD_MAX`.

`SHENAVA_DECODER` accepts only `ctc`. Invalid numeric ranges are rejected.
JSON exposes the VAD, UI and optional output settings in `config.py`.
`commit_on_endpoint=false` retains an experimental early-commit mode for API
compatibility; it cannot retract a committed prefix and is not recommended for
medical injection — startup **refuses** that mode when injection or
`--clinical-sqlite`/`--clinical-jsonl` is enabled, instead of trusting a
warning. Conflicting rewritten tails are suppressed, not re-injected.

Compatibility: callbacks still accept `(text, confidence)` and decode results
retain their second numeric slot, but the production backend always supplies
**0.0 = unknown**. There is no calibrated confidence estimator or display.
`confidence_threshold`, `show_confidence`, `left_context_s`, `max_window_s` are
legacy config fields with no runtime effect. Heuristic confidence functions and
the unsafe window decoder were removed. Consecutive-word deletion and injector
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
by Git. Third-party NeMo logs should also be audited before handling patient data.

## Verification and deployment gate

```bash
pytest -q
python main.py --self-test
python tools/evaluate_medical.py tests/corpus/medical_regression.jsonl
python tools/verify_pipeline.py --model /models/shenava-koochik-v1.0.nemo \
  --wav /data/consented-persian-sample.wav --device cpu --right-context 13
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
