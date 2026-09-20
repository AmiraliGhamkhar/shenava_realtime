# Shenava model provisioning

The application uses the sherpa-onnx streaming CTC ONNX export of Shenava-Koochik-v1.0.  Model weights are intentionally not committed to Git; `models/shenava/*.onnx` is gitignored.

## Exact artifact

- Hugging Face repository: `mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26`
- Revision: `4be3d2375c98a985154122d69b43360eb8bdca5a`
- Runtime loader: `sherpa_onnx.OnlineRecognizer.from_nemo_ctc`
- Inference mode: streaming CTC, CPU, 16 kHz mono, greedy search

## Download command

Install the Hugging Face CLI if needed, then run from the repository root:

```bash
hf download \
  mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26 \
  --revision 4be3d2375c98a985154122d69b43360eb8bdca5a \
  model.int8.onnx tokens.txt \
  --local-dir models/shenava
```

The application never downloads this model automatically.  If these local files are missing, startup fails with a clear error.

## Expected files

```text
models/shenava/model.int8.onnx
models/shenava/tokens.txt
```

Expected vocabulary size: `1025` tokens in `tokens.txt`.

## Checksums

No model or token SHA256 checksum was recorded in the repository history available to this checkout.  If you need an integrity pin, compute and record it after downloading the exact revision above, for example:

```bash
sha256sum models/shenava/model.int8.onnx models/shenava/tokens.txt
```

Keep the large ONNX file out of Git.
