# Repository review and changes

Reviewed the application/launcher, configuration, ASR adapters, audio/VAD,
stabilization, FST/normalization/number tables, UI/hotkeys/injection, utilities,
profiling tool, bundled model card, dependency files and original tests.
The baseline lightweight suite passed 221 tests but did not cover several
critical integration defects.

## Findings and disposition

| Finding | Change |
|---|---|
| Streaming searched for `CacheAwareStreamInfer.transcribe_chunk`, checked model rather than encoder streaming metadata, and silently fell back | Replaced with native `conformer_stream_step`, encoder caches and metadata-driven lookahead. Explicit capability fallback; adapter initialization errors fail visibly. |
| Sliding decode hypotheses dropped their prefix before a reset, replayed overlapping speech and duplicated unstable text | Removed sliding decoder. Unsupported checkpoints use one endpoint decode, hard sample cap. |
| Reprocessing a committed prefix changed numbers/FST phrases; divergent text deltas re-injected words | Default endpoint commits, live partial display; early mode refuses non-prefix rewrites. Final hypothesis remains authoritative before endpoint. |
| Final streaming tail had no explicit flush | Native EOF feature flush, final `keep_all_outputs`, idempotent finalize and state reset tests. |
| Forced VAD split emitted END without START | Explicit consecutive END/START with empty continuation pre-roll; integration test checks every sample appears exactly once. |
| Pending VAD audio could accumulate, pending flush emitted an orphan END, pre-roll sizing imprecise | Bounded pending duration, validated finite thresholds/times, exact pre-roll clipping, discard unconfirmed onset on flush. |
| Mono PortAudio arrays could alias reused callback storage | Unconditional owned copy before enqueue. |
| Queue overflow discarded boundary events and inference continued over audio gaps | Explicit abort/reset on discontinuity; orphan audio ignored until next START. Bounded UI, injection and clinical queues. |
| Shutdown cleared thread handles before threads exited; pause did not endpoint | Preserve live handles on timeout; pause flush marker; capture startup cleanup closes streams. |
| Injector retry reordered text and could duplicate a partly completed paste | Removed retry; bounded nonblocking submission; disable output on overload. Drain tracks in-flight tasks. |
| Arbitrary “confidence” from word count/log-score exponentiation | Removed heuristic functions and display. Legacy numeric API slots remain unknown (zero), no filtering. |
| Implicit hub download on typo/missing checkpoint | Local-first failure before importing Torch; network provisioning requires explicit opt-in. |
| Decimal dots were separated; trie rewrites lost punctuation/crossed punctuation | Numeric punctuation protected, FST retains delimiters and stops at punctuation. |
| Number parser summed adjacent values, rewrote Persian negation `نه`, guessed BP from general connectors | Ordered value/scale grammar; conjunction restrictions; preserve negation; explicit decimal/negative/half rules; conservative BP connectors. |
| FST unit detection included every medical output, broad `انفارکتوس`→MI asserted anatomy | Unit vocabulary separated; ambiguous infarction shorthand removed; explicit longer phrase retained. |
| ASR output text logged and stored by default | Routine transcript-content logs removed, rotating logs; persistence opt-in. |
| Hotkey auto-repeat could spawn unbounded callback threads | Ignore key repeat and permit at most one active callback per binding. |
| No separate clinical extraction | Optional exact-dictionary NER, conservative rules, offsets/review flag, bounded asynchronous JSONL/SQLite worker. |
| Core dependencies pulled desktop packages into headless installs | Split core, desktop, ASR and development dependency files. Pin native NeMo API target. |

No new service, server, model download helper, database framework, language
model, fuzzy matcher or build system was introduced. The deterministic trie,
desktop UI, clipboard and hotkey architecture were retained.

## API/migration notes

- Primary model remains Koochik 114M. Smaller local models use the same loader.
- Existing app entrypoints/output modes/callback shapes remain. Callback numeric
  values no longer imply confidence. Legacy window/confidence config keys load
  but are inactive; removed helper functions and `WindowedDecoder` require
  callers importing those internals to migrate to `EndpointDecoder`.
- Only greedy CTC is accepted. RNNT selection now fails clearly.
- Output waits for endpoints by default. Automatic transcript disk saving,
  destructive repetition removal and injector text-equality dedup are now off.
- General `به` blood-pressure guessing was intentionally removed and its old
  test expectation changed. Abnormal explicit numeric ratios remain unchanged.
- Recovery discards incomplete utterances rather than manufacturing a clinical
  statement across a missing audio interval. No promise of lossless output
  during overload, disk failure, process crash or an uninterruptible driver call.

## Native implementation reference and limits

The adapter was checked against the NeMo 2.4 source interfaces:

- `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
- `nemo/collections/asr/parts/utils/streaming_utils.py`
- `nemo/collections/asr/parts/preprocessing/features.py`

Source: https://github.com/NVIDIA/NeMo/tree/v2.4.0

Feature overlap is hop-aligned and recomputed only for a bounded STFT margin.
Encoder chunk/shift/pre-encoder cache are obtained from the model. Prediction
history is bounded by utterance length. These contract tests are not substitutes
for real NeMo numerical parity or recognition evaluation. Centered STFT only is
supported; unknown frontend layouts must fail rather than guessing.

The current dictionary/NER inventory is intentionally small and manually
reviewable. Negation scope is only explicit adjacent phrasing, measurements are
not associated with medications, speaker diarization is absent, and forced
segment boundaries can split phrases. Persisted clinical records are plaintext
and always require review. Validate models, timing and medical-language accuracy
on the actual deployment machine before use.

## Checks executed in the workspace

- Baseline: `pytest -q` — 221 passed.
- After changes: `.venv/bin/pytest -q` — 245 passed.
- `python main.py --self-test` — starts/stops the threaded headless engine,
  emits `بیمار دوز 2.5 mg` once, extracts the 2.5 mg measurement, zero overruns.
- Application lifecycle test starts/stops `ShenavaApp` with injected audio/model
  doubles; it verifies final text is drained before shutdown.
- Missing checkpoint startup exits with an actionable local-file error.
- Python compilation and `git diff --check` pass.
- Real checkpoint/microphone/GPU tests: not run; no local weights/NeMo and CPU
  Torch wheel provisioning failed with TLS errors.

## Second-pass review: doc-review hypotheses checked against the source

Each hypothesis below was verified in the actual code before any change; two
were already satisfied and only needed an explicit regression pin.

| Finding | Change |
|---|---|
| A forced VAD/ASR-cap split (e.g. `سی و` \| `پنج`) was indistinguishable from a natural endpoint and each half parsed as a complete wrong value (`30 و` + `5`); `VADEvent` carried no forced flag and the pipeline treated every END identically | Forced cuts are tagged end to end (`VADEvent.forced` → `on_speech_end(duration, forced)` → engine queue item → `pipeline.end_utterance(forced=)`; ASR window resets count as forced too). New pure helpers `fa_numbers.open_number_tail` / `leading_number_span` detect an unfinished phrase; the pipeline keeps the boundary-spanning words unparsed (spoken words visible, warning logged, `forced_splits` counter) instead of parsing each half. Natural endpoints keep the documented per-utterance behavior; a cut between two complete halves (`سی` \| `پنج`) remains undecidable without context and is disclosed in the README |
| `commit_on_endpoint=false` (early-commit legacy mode) had no enforcement — only README wording — so it could run with injection or clinical persistence active | Hard startup refusal in `ShenavaApp.__init__` (`ValueError` → readable `error:` exit 1), same fail-explicit pattern as the `require_streaming`/local-checkpoint checks. The mode still loads when no medical output path is configured |
| Crash/kill between injection and clinical persistence: confirmed there is no reconciliation on restart — `ClinicalWorker` is append-only, the injector keeps no journal, nothing replays or back-fills | Documented as an explicit accepted risk in README (clinical section) and the limits list above; no reconciliation system built, per the deterministic-architecture constraint |
| `--right-context`: values outside `{0,1,6,13}` were already rejected at config load (CLI choices + `ASRConfig.__post_init__`), but a *metadata*-unsupported yet set-valid value silently degraded to endpoint-only decoding with a warning | `NeMoASR._check_streaming_context` now fails at startup (`RuntimeError` listing the encoder's supported contexts) when a streaming-capable encoder lacks `[70, right_context]`. Only checkpoints with no native streaming support at all keep the documented endpoint-only fallback |
| `PostProcessor(extra_terms=...)` conflicts: verified they already raise at construction time (`build_rewriter` → `TrieFST.add` during `__init__`), not lazily on first match | No source change. Added a regression pin: conflicting extra terms (against each other and against built-in tables) raise `ValueError` at construction; an identical restatement is not a conflict |
| Audio device dropout: confirmed no detection path — the capture consumer blocked forever on `queue.get()`, so a silent/disconnected mic stalled the app silently (distinct from queue overflow, which was handled) | Capture heartbeat: consumer polls with a timeout (`AudioConfig.dropout_timeout_s`, default 2 s, validated). A stall is reported once per episode as a logged error + `dropouts`/`last_error` statistics, an inactive stream reports immediately, an open utterance is aborted through the existing discontinuity path, and recovery logs when blocks resume. Paused/stopping/closed states never false-positive |

## Checks executed in the second pass

- Before changes: `.venv/bin/pytest -q` — 245 passed; `python main.py --self-test`
  emits `بیمار دوز 2.5 mg` and extracts the 2.5 mg measurement.
- After changes: `.venv/bin/pytest -q` — 283 passed (38 new regression tests,
  3 existing callbacks updated for the `forced` flag); `python main.py
  --self-test` passes unchanged; `python -m compileall` and `git diff --check`
  pass.
- No new services, frameworks, dependencies or fuzzy/ML matching; all
  mitigations are deterministic and fail-explicit.
- Remaining disclosed limit: a forced cut between two *complete-looking*
  number halves (e.g. `سی` \| `پنج`, no trailing connector) cannot be
  distinguished from two separate values without linguistic context, so only
  open phrases (trailing `و`/`ممیز`) and the mirrored leading run are
  suppressed; the rest stays reviewable output as before.

## Third-pass review: accuracy/robustness hardening (second pass, hotwords, validation, diagnostics)

All changes preserve the architecture, public APIs, CLI, desktop output modes,
queue/discontinuity behavior and the safety-first design. No LLM/embedding/
vector-DB/heavy NLP dependency was added; the beam decoder is a small
deterministic NumPy module over the model's own emissions and BPE vocabulary.

| Finding / request | Change |
|---|---|
| Live greedy decoding has no full-context repair at a natural utterance end | Config-optional utterance-end **second pass** (`second_pass`: `off`/`greedy`/`context`, default `greedy`; CLI `--second-pass`, env `SHENAVA_SECOND_PASS`). Runs only on natural endpoints (never forced cuts), only for utterances ≥ `second_pass_min_utterance_s`, only in endpoint-commit mode (early-commit disables it with a warning). `greedy` reuses the backend's `transcribe`; `context` uses `NeMoASR.build_second_pass` → `BeamSecondPass` (CTC beam over `model.forward` logits, blank index probed from `ctc_decoder` with the documented default of 0). Failures keep the streaming text, count a fallback, log a warning — never a silent mode switch; a missing capability in `context` mode is a startup `RuntimeError`. `GreedySecondPass`/`CTCBeamDecoder`/`BeamSecondPass` live in `second_pass.py` with a two-method interface (`decode_greedy`, `decode_with_context`) |
| Post-ASR terminology rules were ignored by the decoder | Decoder-time **hotword biasing** from the same reviewed `terminology.json` rules (no second dictionary): `hotwords.build_hotwords` (bounded by `hotword_max`, specialty subset + general, deterministic priority order, category-weighted conservative log-prob boosts, unit/anatomy excluded). Biases apply only while the beam path continues a hotword token prefix (`hotword_token_bias`); untokenizable phrases are skipped. Built only for `context` mode |
| Suspected values (SpO2 102%, temperature 50°, pulse 1000 bpm, reversed BP) were emitted without any structured flag | `value_validation.py`: deterministic plausibility checks on typed spans (explicit conservative ranges). Source text is **always preserved**; results are `ValueIssue` entries on `ProcessingResult.value_issues` plus `suspicious_value:<kind>=<value>` review reasons. BP connector pairs (`روی`/`بر روی`/`خط`) that the grammar already refuses to parse are recorded with the same mechanism. Clinical records (`extract_record`) run the same checks on persisted measurements and merge the pipeline's review reasons into `record["review_reasons"]` (additive JSON key; `review_required` stays `true`) |
| No structured review reasons beyond `review_required` | `ProcessingResult.review_reasons` / `pipeline.last_review_reasons` / `engine.last_review_reasons` expose a stable sorted list: `rare_medical_term`, `drug_name`, `dose_value`, `numeric_value`, `negation_sensitive`, `laterality_sensitive`, `forced_boundary`, `decoder_disagreement`, `suspicious_value:*`, `unsafe_terminology:<id>`. `main.py` forwards the reasons to clinical persistence. Reasons never alter text |
| A self-alias rule whose output equals the matched text (e.g. Latin `EKG` for `EKG`) was flagged `unsafe_terminology` because the protected span conflicted with its own no-op rewrite | `MedicalNormalizationPipeline` filters candidates whose output is exactly the matched text (case normalisation like `cabg` → `CABG` is **not** a no-op and still rewrites) |
| Grammar gaps found while probing realistic inputs | `medical_grammar.py`: rate keywords (`نبض`/`ضربان`/`ضربان قلب` → bpm, `نفس` → rpm) with the explicit `در دقیقه` tail (span covers number..`دقیقه` only; never rewrites an already-unitful value); `mg/kg` unit (both spellings + `mg در kg`) in `lexicon.py`/`terminology.json` (`unit.0029`); `U insulin` classified as a dose unit; medication grammar no longer double-counts the drug word inside `واحد انسولین` |
| Quiet/clipping microphones were invisible | `audio_diagnostics.py` + `AudioCapture`: per-segment RMS/peak/clipping-ratio/level (`silent|low|ok|clipped`) computed at SPEECH_END, logged, exposed via `get_statistics()` (`last_segment`, `clipped_segments`, `low_level_segments`). Metrics only — no denoising, no transcript effect |
| No evidence-driven path to collect recurring ASR misspellings of a reviewed term | `variants.py` (`align_tokens` token-level edit alignment, `project_span` with the single-token/absorb-deleted-adjacent heuristic for split spoken forms) + `tools/collect_term_variants.py` CLI (reference → observed → reviewed → rule; `--suggest` prints a review fragment, never writes `terminology.json`) |
| Frontend streaming/offline alignment was only partly covered | `tests/test_frontend.py`: faithful centered-window (rectangular STFT proxy) preprocessor double; every encoder window the stream re-encodes is asserted equal to the offline frames at its absolute position within a bounded tolerance; raw buffer bounded + hop-aligned; no lost/duplicate frames at the final flush; unsupported sample rate fails at initialization (`native_stream.py` now validates `cfg.preprocessor.sample_rate == 16000`) |
| Evaluator lacked dose/forced-boundary/high-risk weighting | `tools/evaluate_medical.py`: per-category error rates incl. **dose**; **forced-boundary error rate** (corpus rows `"forced": true` run the pipeline's forced path, reference keeps the spoken words); **medical essential error** (weighted character distance: reference characters inside entity spans cost 3×, prose 1×); trailing "engineering regression only — not clinical validation" note. Corpus expanded 6 → 33 rows (code-switching, drugs, abbreviations, rates, temperature, mg/kg, SpO2, FBS, CHF, CABG/PCI/ICU, dates, unknown-unit preservation, negation/laterality, forced splits, suspicious-value preservation) |

## Checks executed in the third pass

- Before changes: `.venv/bin/pytest -q` — 291 passed; `main.py --self-test`
  exit 0; evaluator 6 samples with 0 post-processing WER/CER.
- After changes: `.venv/bin/pytest -q` — 381 passed (90 new tests: second
  pass/beam/hotwords, value validation, audio diagnostics, variant
  collection, VAD boundary scenarios, frontend parity, native flush
  accounting, WAV replay, engine second pass, CLI flags, config env,
  post-processor grammar); `main.py --self-test` exit 0 (second pass
  `runs=1`, record carries `dose_value`/`numeric_value` reasons); evaluator
  on the 33-row corpus: post WER/CER 0, all category rates 0, forced-boundary
  rate 0, medical essential error 0.
- Real NeMo checkpoint runs, real WAV replay against the published model,
  `[70,13]` native streaming on hardware and CPU/GPU latency remain
  **unverified in this environment** (torch/NeMo unavailable; the CPU download
  attempt failed earlier with a TLS/network error). Those paths are covered by
  the tensor/model doubles and the headless WAV-replay test against a labelled
  fake backend.

## Fourth-pass review: audit against the accuracy/robustness mandate

Re-audited every item of the mandate against the actual implementation. The
streaming foundation (cache handoff, flush accounting, context gating), the
frontend parity tests, the second-pass/hotword layer, forced-boundary
protections, review reasons, the evaluator and the output/persistence
contracts were verified in code and tests; the defects below were the gaps
this pass fixes.

| Finding | Change |
|---|---|
| The persisted-record plausibility check (`clinical._record_review_reasons`) had drifted from the pipeline's `value_validation.py`: body temperature 25–45 vs the documented 24–45 °C, 77–113 vs 75–113 °F, and a `suspicious_value:nonpositive=` label that the reason vocabulary does not define; non-dose units were flagged for any non-positive value while the pipeline judged none | Clinical records now call `value_validation.plausibility_reason()`/`bp_plausible()` — the same ranges, the same `suspicious_value:<kind>=<value>` labels. The BP pair gate, previously spelled out four times (`fa_numbers`, `medical_grammar`, `value_validation`, `clinical`), is the single `bp_plausible()` |
| The README (and the two `asr_backend` error messages) named `requirements-dev.txt` / `requirements-asr.txt` / `requirements-desktop.txt` and pinned NeMo 2.4.0 + Torch/torchaudio 2.7.1, but only an unpinned combined `requirements.txt` existed | The split files exist again with exactly what the README describes; `requirements.txt` is a full-stack aggregate of the three. No new dependency was introduced |
| Hotword biasing could not express the mandate's per-rule, review-justified exception (e.g. a rare anatomy term) and the anatomy category was hard-excluded | `TerminologyRule` gains the minimal metadata field for decoder biasing: optional `bias`, schema-validated to `[0, 1.5]` (`0` = suppress a category default; untokenizable forms were already skipped). Units remain never-boosted; every other category keeps its conservative default, and categories without one (anatomy, disease-less groups) appear only via an explicit reviewed value. Shipped `terminology.json` sets no overrides — existing boost behaviour is byte-identical, now pinned by a test |
| VAD boundary coverage had no scenario for speech containing numbers or medication names (mandate §3) | `tests/test_vad_scenarios.py`: a spoken BP pair and a medication/dose phrase with short hesitations stay in exactly one segment; a long pause yields two bounded, unjoined segments; a tuned low-volume capture flows end-to-end and the natural endpoint commits `BP 120/80` once |
| `audio_diagnostics.crest_factor()` returned `False` or the ratio (annotation said `float`) | Returns `0.0` when the ratio is not finite |

`PostProcessor.last_result.review_reasons` semantics are unchanged: this pass
touched no rewriting path; only the *reasons attached to persisted records* can
differ (the drifted bounds and the unknown `nonpositive` label are gone, and
non-dose negative values are no longer judged in one layer but not the other).

## Checks executed in the fourth pass

- Before changes: `.venv/bin/pytest -q` — 382 passed; `main.py --self-test`
  exit 0; evaluator 33 rows, all post-processing metrics 0.
- After changes: `.venv/bin/pytest -q` — 394 passed (12 new tests: bias
  metadata + schema bounds, hotword opt-in/suppression, shared-range pin for
  clinical records, BP gate, VAD number/medication boundaries, clinical
  persistence opt-in); `main.py --self-test` exit 0 unchanged; evaluator
  output unchanged (post WER/CER 0, forced-boundary 0, essential 0).
- Gates: WAV replay verified through `tests/test_wav_replay.py` (real PCM16
  WAV through the replay path, labelled fake backend); `[70,13]` and the
  lower contexts `6/1/0` are pinned by the encoder-metadata tests and the
  config/CLI rejection path; `--second-pass off` and the missing-capability
  startup error are pinned in `tests/test_second_pass.py`. Real NeMo
  checkpoint execution remains impossible here (weights/NeMo absent) and is
  still declared unverified below.
