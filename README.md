# Transcriber-AI

A Python project for **audio → text transcription** using Whisper, with optional **2-speaker diarization** (pyannote).

## Features

- **ASR (Speech-to-Text)**: `openai/whisper-large-v3` (default)
- **Language control**: auto-detect or pin via `.env`
- **Long audio support**: automatically transcribes long audio by **chunking**
- **Speaker diarization (optional)**: 2-speaker diarization with pyannote + segment-wise transcription

## Project structure

```text
Transcriber-AI/
├── data/                 # Input audio files (.m4a, .mp3, .wav, ...)
├── exports/              # Outputs (output.txt, diarized_output.txt)
├── models/               # HF cache (recommended)
├── src/
│   ├── diarizer.py       # pyannote diarization
│   ├── transcriber.py    # Whisper transcribe + chunking
│   └── utils.py          # ffmpeg segment export, audio helpers
├── main.py               # entrypoint
└── requirements.txt
```

## Requirements

- **Python 3.11+**
- **ffmpeg** (required for segment extraction and some codecs)
- **Hugging Face account + token** (required for diarization)

## Setup

### 1) Create & activate venv (Windows PowerShell)

```powershell
cd C:\Users\busra\Projects\Transcriber-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

> If you plan to use GPU, make sure your PyTorch installation matches your CUDA setup.

## Configuration (`.env`)

Create a `.env` file in the project root:

```env
# Whisper
WHISPER_MODEL_ID=openai/whisper-large-v3
# WHISPER_LANGUAGE=english   # optional: pin language; omit to auto-detect

# Hugging Face cache (recommended on Windows to reduce symlink warnings)
HF_HOME=./models/hf_cache

# Diarization
DIARIZE=1
HF_TOKEN=hf_...
```

## Usage

### A) Plain transcription (no diarization)

Set `DIARIZE=0` in `.env` (or remove it), then:

```powershell
python main.py
```

Output: `exports/output.txt`

### B) 2-speaker diarization + transcript

In `.env`:

- `DIARIZE=1`
- `HF_TOKEN=hf_...`

Then:

```powershell
python main.py
```

Output: `exports/diarized_output.txt`

Example format:

```text
[00:00:05.120 - 00:00:07.900] SPEAKER_00: ...
[00:00:08.100 - 00:00:12.640] SPEAKER_01: ...
```

## Notes & troubleshooting

### Hugging Face 403 (gated repos)

The pyannote diarization pipeline downloads additional models at runtime. If you get a 403:

- Accept the user conditions for `pyannote/speaker-diarization-3.1`
- Also accept conditions for any other pyannote repos mentioned in the logs (e.g. `pyannote/speaker-diarization-community-1`, `pyannote/segmentation-3.0`)
- Ensure your token has **Read** access and is set as `HF_TOKEN=...` in `.env`

Reference: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### `torchcodec` warnings on Windows

`pyannote.audio` may print `torchcodec` warnings on some Windows setups. In this project, diarization is run by passing audio **from memory**, so this warning typically does not block execution.

### ffmpeg

`ffmpeg` is required for segment extraction. Verify it is available:

```powershell
ffmpeg -version
```

## Security

- Do not commit `.env` to git (it is already ignored via `.gitignore`).

