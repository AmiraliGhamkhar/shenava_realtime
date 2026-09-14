# Shenava Real-time ASR

Real-time Persian speech recognition with a floating overlay window, global hotkeys, and automatic text injection into other applications. Powered by the [Shenava-Koochik](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) NeMo ASR model.

## Requirements

- Windows (text injection and hotkeys are Windows-oriented)
- Python 3.13 with the bundled `venv/` (already configured with all dependencies, including `nemo_toolkit[asr]`)
- A working microphone

## Model

The ASR engine loads the model **from a local `.nemo` file** — no internet download is needed at runtime.

- Default path: `C:\Users\ali\Desktop\localASR-shenava\shenava-koochik\shenava-koochik-v1.0.nemo`
- Configured in `shenava_realtime/config.py` → `ASRConfig.model_path`

If the local file is missing, the engine falls back to downloading the model from HuggingFace (`Reza2kn/Shenava-Koochik-v1.0`). Both `shenava-koochik-1.0.nemo` and `shenava-koochik-v1.0.nemo` in the local folder are identical (459 MB).

Note: the bundled torch build is CPU-only. The code auto-detects CUDA and will use it when a CUDA-enabled torch is installed; otherwise inference runs on CPU.

## Run

```bash
# Option 1: background launcher (recommended; writes app.log)
venv/Scripts/python.exe run_app.py

# Option 2: foreground in the terminal
venv/Scripts/python.exe main.py
```

Startup takes ~30 seconds (loading the 460 MB model). The overlay window appears near the bottom of the screen; speak and recognized text will show in the overlay and be typed into the focused application (output mode: `both`).

## Hotkeys

| Combo | Action |
|---|---|
| `Ctrl+Alt+R` | Toggle recording |
| `Ctrl+Alt+O` | Show/hide overlay |
| `Ctrl+Alt+I` | Toggle text injection |
| `Ctrl+Alt+C` | Clear current transcript |
| `Ctrl+Alt+Q` | Emergency stop (shuts down the app) |

Stop the app with `Ctrl+Alt+Q` or by killing the python processes; the background launcher can also be stopped with `taskkill //F //PID <pid>` (PID is printed when `run_app.py` starts).

## Output modes

Configured via `AppConfig.output_mode` in `shenava_realtime/config.py`:

- `overlay` — show transcription only
- `inject` — type transcription into the focused window
- `both` — overlay + injection (default)
- `clipboard` — copy transcription to clipboard
- `console` — print to terminal only

## Transcripts

When `save_transcripts` is enabled (default), each session's transcript is saved to `transcripts/` on shutdown.

## Troubleshooting

- **No text appears in overlay**: the overlay auto-hides after a few seconds of silence and re-shows on the next transcription; check that `overlay.enabled` is true.
- **Hotkeys do nothing**: another application may be capturing the key combo; hotkeys require the pynput listener (checked at startup).
- **Injection types wrong characters**: Persian text requires Unicode-aware injection; `use_unicode` is enabled by default in `InjectorConfig`.
- **UnicodeEncodeError / garbled Persian in console**: `main.py` forces UTF-8 stdio; if using option 2 in some terminals, also set `PYTHONIOENCODING=utf-8`.
