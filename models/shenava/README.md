# Shenava Model Setup

This project uses the **Shenava-Koochik-v1.0** model with **Sherpa-ONNX Streaming CTC**.

The model files are **not included in Git**. You must download them manually before running the application.

## 1. Install Hugging Face CLI

If the `hf` command is not installed:

```powershell
pip install -U huggingface_hub
```

Verify the installation:

```powershell
hf --help
```

## 2. Download the Model

Open PowerShell in the **root folder of the repository** and run:

```powershell
hf download mah92/sherpa-onnx-nemo-ctc-fa-shenava-koochik-v1.0-streaming-int8-2026-06-26 `
  --revision 4be3d2375c98a985154122d69b43360eb8bdca5a `
  model.int8.onnx tokens.txt `
  --local-dir models/shenava
```

The command downloads the exact model revision required by the application.

## 3. Required Files

After the download, the following files must exist:

```text
models/
└── shenava/
    ├── model.int8.onnx
    └── tokens.txt
```

`tokens.txt` is expected to contain **1025 tokens**.

## 4. Model Configuration

| Setting     | Value                |
| ----------- | -------------------- |
| Model       | Shenava-Koochik-v1.0 |
| Runtime     | Sherpa-ONNX          |
| Model Type  | Streaming CTC        |
| Language    | Persian (`fa`)       |
| Sample Rate | 16 kHz               |
| Audio       | Mono                 |
| Device      | CPU                  |
| Decoder     | Greedy Search        |
| Format      | INT8 ONNX            |

## 5. Model Revision

The application is pinned to this Hugging Face revision:

```text
4be3d2375c98a985154122d69b43360eb8bdca5a
```

Using a fixed revision ensures that the same model artifact is used across environments.

## 6. SHA256 Verification (Optional)

Model checksums are not currently stored in the repository.

To calculate SHA256 hashes in PowerShell:

```powershell
Get-FileHash .\models\shenava\model.int8.onnx -Algorithm SHA256
Get-FileHash .\models\shenava\tokens.txt -Algorithm SHA256
```

You can save the resulting hashes in the project documentation if you want an additional integrity check.

## Important

The application **does not download the model automatically**.

If the required model files are missing, startup will fail with a clear error message.

The large ONNX model file is intentionally excluded from Git and should **not** be committed to the repository.
