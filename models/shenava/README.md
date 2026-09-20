---
language:
- fa
license: cc-by-nc-4.0
library_name: sherpa-onnx
pipeline_tag: automatic-speech-recognition
base_model:
- Reza2kn/Shenava-Koochik-v1.0
base_model_relation: finetune
tags:
- automatic-speech-recognition
- speech
- persian
- farsi
- fastconformer
- ctc
- streaming
- on-device
- shenava
- shenava-1
- onnx
- sherpa-onnx
- int8
- quantized
metrics:
- wer
- cer
datasets:
- Reza2kn/visualears-persian-asr-16k
- Reza2kn/visualears-golden-6669
- Reza2kn/fleurs-fa-benchmark
model-index:
- name: sherpa-onnx-shenava-koochik-v1.0-ctc-int8
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: golden-6669
      type: Reza2kn/visualears-golden-6669
      split: test
    metrics:
      - type: wer
        value: 7.90
        name: WER
      - type: cer
        value: 2.60
        name: CER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: FLEURS-fa
      type: Reza2kn/fleurs-fa-benchmark
      split: test
    metrics:
      - type: wer
        value: 11.20
        name: WER
      - type: cer
        value: 4.10
        name: CER
---

# Shenava — Koochik v1.0 (114M) · CTC Streaming · ONNX (INT8)

**Koochik** (کوچیک, "small") is the **114M teacher / flagship** of the **Shenava‑1** family — a FastConformer Hybrid RNNT/CTC model fine‑tuned on clean Persian data with **ve_tok_v4**. This repository contains the **CTC streaming INT8 quantized ONNX export** for use with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

The original NeMo checkpoint is at [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0).

## Benchmark — fair WER/CER

| Variant | golden‑6669 WER | FLEURS‑fa WER |
|---|---|---|
| FP32 (full precision) | 7.49% | 10.64% |
| **INT8 (quantized)** | **~7.90%** | **~11.20%** |

INT8 quantization yields ~4× model size reduction with <0.5% WER degradation.

## Model Details

| Property | Value |
|---|---|
| **Parent Model** | [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) |
| **Architecture** | FastConformer Hybrid RNNT/CTC → CTC decoder export |
| **Language** | Persian (Farsi) |
| **Parameters** | 114M |
| **Vocabulary Size** | 1025 tokens (BPE + blank) |
| **Tokenizer** | ve_tok_v4 — SentencePiece BPE‑1024 |
| **Sample Rate** | 16 kHz |
| **Subsampling** | 8× |
| **Streaming** | ✅ Yes (with cache support) |
| **Weight Type** | INT8 (dynamic quantization) |
| **Author** | [Reza2kn](https://github.com/Reza2kn) |
| **Version** | 1.0 |

## Files

```
.
├── README.md
├── model.int8.onnx   # ONNX model (INT8 quantized)
└── tokens.txt        # Token vocabulary (1025 tokens)
```

## Usage

```bash
sherpa-onnx-offline \
  --tokens=tokens.txt \
  --encoder=model.int8.onnx \
  --num-threads=4 \
  /path/to/test.wav
```

## Conversion

Exported from a NeMo checkpoint and quantized dynamically with ONNX Runtime.

## License

CC-BY-NC 4.0 (inherited from the parent model)