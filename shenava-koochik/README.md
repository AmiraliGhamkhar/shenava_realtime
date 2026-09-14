---
language:
- fa
license: apache-2.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model:
- nvidia/stt_fa_fastconformer_hybrid_large
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
- visualears
- rnnt
- nemo
metrics:
- wer
- cer
datasets:
- Reza2kn/visualears-persian-asr-16k
- Reza2kn/visualears-golden-6669
- Reza2kn/fleurs-fa-benchmark
---

# 🎙️ Shenava Koochik v1.0 · شنوا کوچیک

The 114M-parameter flagship of the Shenava-1 Persian ASR family. This is the FP32 NeMo source checkpoint for evaluation, fine-tuning, and export; use one of the deployment repositories below for browser, Core ML, sherpa-onnx, or tract applications.

## ✨ At a glance | معرفی سریع

| | English | فارسی |
|---|---|---|
| 🧠 Role | 114M flagship teacher | مدل معلم و پرچم‌دار ۱۱۴M |
| 📦 Format | FP32 NeMo source | checkpoint اصلی FP32 و NeMo |
| 🎧 Input | 16 kHz mono Persian speech | گفتار فارسی تک‌کانالهٔ ۱۶ کیلوهرتز |
| 📝 Output | Persian transcription | رونویسی فارسی |
| 🛠️ Best for | Evaluation, fine-tuning, and export | ارزیابی، آموزش تکمیلی و تبدیل |

- Canonical repository: [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0)
- PersianML mirror: [`PersianML/Shenava-Koochik-v1.0`](https://huggingface.co/PersianML/Shenava-Koochik-v1.0)

## 🧠 Model

- Architecture: FastConformer Hybrid RNNT/CTC; the CTC head is used by the published deployment exports.
- Audio: mono, 16 kHz Persian speech.
- Encoder: `d_model=512`, 17 layers, 8x subsampling (about 80 ms per encoder step).
- Contexts: `[70,13]`, `[70,6]`, `[70,1]`, and `[70,0]`.
- Tokenizer: ve_tok_v4, SentencePiece BPE-1024 plus CTC blank.
- Output: Persian text. Numbers are emitted in spoken form; apply Persian inverse text normalization when digits are wanted.

The two `.nemo` filenames are compatibility aliases with identical content and SHA-256:

`f7b5124a9fd2d50c15bf070abdc0f80ec2449d64948c5657adb79a7e91dd1d16`

## 📊 Published evaluation

Decoded with context `[70,13]` and the double-benchmark ITN/Persian-digit normalization convention.

| Set | WER | CER |
|---|---:|---:|
| visualears-golden-6669 | 7.49% | 2.30% |
| FLEURS-fa | 10.64% | 3.79% |

## 🚀 Load with NeMo

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from("shenava-koochik-v1.0.nemo")
text = model.transcribe(["speech.wav"])[0].text
print(text)
```

## 🧩 Deployment variants

- [ONNX FP16](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16): fixed-window browser/WebGPU/WASM export used by Shenava.
- [sherpa-onnx](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx): offline cross-platform bundle.
- [Core ML FP16](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16): fixed-window ML Program package.
- [Core ML iOS 15](https://huggingface.co/Reza2kn/Shenava-Koochik-1.0-CoreML-iOS15-fp16): cache-aware streaming NeuralNetwork model for older Apple OS versions.
- [tract streaming](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-streaming) and [tract offline](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-offline): pure-Rust runtime exports.

## 🇮🇷 خلاصهٔ فارسی

این مخزن نسخهٔ اصلی FP32 و NeMo مدل ۱۱۴ میلیون‌پارامتری «شنوا کوچیک» است. برای اجرای مرورگر، Core ML، sherpa-onnx یا tract از مخزن مخصوص همان قالب استفاده کنید. ورودی گفتار فارسی تک‌کانالهٔ ۱۶ کیلوهرتز و خروجی متن فارسی است.

## 🌌 Explore Shenava-1

🧠 **Koochik 114M** · [⚖️ Rizeh 32M](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) · [🐣 Rizeh-Pizeh 6.9M](https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0) · [🌐 Browser](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16) · [🍎 Apple](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-CoreML-fp16) · [🦀 Rust](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-tract-streaming)

Apache-2.0. ASR quality can vary with accent, noise, overlap, recording channel, and code-switching; review meaning-critical transcripts before relying on them.
