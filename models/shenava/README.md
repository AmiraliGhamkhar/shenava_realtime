# Shenava-Koochik-v1.0 — sherpa-onnx streaming CTC model

This directory is **git-ignored**: the model is provisioned locally, never
downloaded automatically by the application. Provision it once with the
command below before running `main.py` or the real-model tests/tools.

## Source

- **HF repo**: [`mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26`](https://huggingface.co/mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26)
- **Pinned revision (commit sha)**: `4be3d2375c98a985154122d69b43360eb8bdca5a`
- **Base model**: `Reza2kn/Shenava-Koochik-v1.0` (FastConformer Hybrid RNNT/CTC,
  114M params) — this export is the CTC-only streaming head, quantized to
  INT8 with ONNX Runtime dynamic quantization.
- **License: CC-BY-NC 4.0** (inherited from the parent model). This is
  **non-commercial**; verify your use case is compatible before deploying.

## Files (place directly in this directory)

| File | Size (bytes) | sha256 |
| --- | --- | --- |
| `model.int8.onnx` | 132,048,387 | `439983c95ab83c55c841e0795ba3a61d56718ec5c972a3a208548b93470b04b1` |
| `tokens.txt` | 12,236 | `8e192963f6e666dfa5721e5cbd4710bc1ef592460a45f08cefc94b2db16a6954` |

The `sha256` values above come from the HF repo's own LFS/file metadata
(`model.int8.onnx`) and from re-fetching `tokens.txt` byte-for-byte during
this migration (its exact 12,236-byte content is checked into
`tokens.txt.reference` in this directory for convenience/diffing — it is
**not** a substitute for downloading your own copy and verifying the
checksum below). `model.int8.onnx` could not be fetched in the migration
sandbox (network egress to huggingface.co's LFS/CDN endpoints was blocked);
it must be downloaded on a machine with network access using the command
below, and its checksum verified before use.

## Download (manual only — no automatic downloads anywhere in this app)

```bash
pip install -U "huggingface_hub[cli]"
hf download mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26 \
  --revision 4be3d2375c98a985154122d69b43360eb8bdca5a \
  --local-dir models/shenava
```

Then verify the checksums:

```bash
sha256sum models/shenava/model.int8.onnx models/shenava/tokens.txt
# compare against the table above
```

## Model properties (from the HF model card)

| Property | Value |
| --- | --- |
| Architecture | FastConformer Hybrid RNNT/CTC → CTC decoder export |
| Language | Persian (Farsi) |
| Parameters | 114M |
| Vocabulary size | 1025 tokens (BPE + blank) |
| Sample rate | 16 kHz |
| Subsampling | 8x |
| Streaming | Yes (cache-aware CTC export) |
| Weight type | INT8 (dynamic quantization) |

Reported benchmark (from the model card, not independently re-verified in
this repository): WER 7.90% / CER 2.60% on `golden-6669`; WER 11.20% / CER
4.10% on `FLEURS-fa`.

## Configuration

Point `shenava_realtime` at these files via `.env` or environment variables
(see `../../.env.example`):

```
SHENAVA_MODEL_PATH=./models/shenava/model.int8.onnx
SHENAVA_TOKENS_PATH=./models/shenava/tokens.txt
```

`SHENAVA_FEATURE_DIM` defaults to `80` (NeMo FastConformer's standard mel-bin
count); `main.py`/the backend validate this against the loaded ONNX model at
startup and fail clearly on a mismatch — see `shenava_realtime/asr_backend.py`.
