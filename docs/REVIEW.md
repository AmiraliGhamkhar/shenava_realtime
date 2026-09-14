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
