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
├── data/                 # Input media for transcription and for trim script (video / audio)
├── exports/              # Transcription outputs; also source .srt for `trim_video_srt.py` (same stem as media)
├── output/               # Trimmed media + re-synced .srt from `trim_video_srt.py` (gitignored)
├── parts/                # Optional multi-segment config (see below)
│   ├── parts.json        # per-part time ranges, titles, hooks, output suffixes
│   └── parts.txt         # human-readable copy of the same plan
├── models/               # HF cache (recommended)
├── src/
│   ├── diarizer.py       # pyannote diarization
│   ├── transcriber.py    # Whisper transcribe + chunking
│   └── utils.py          # ffmpeg segment export, audio helpers
├── main.py               # transcription entrypoint
├── trim_video_srt.py     # cut media + adjust subtitles to a time window
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

### Media trim + SRT re-sync (`trim_video_srt.py`)

Cuts a file under `data/` to a time range and writes a matching `output/` file; any cues in `exports/{same_stem}.srt` that fall in that range are kept and re-timestamped to start at `00:00:00`. Requires **`moviepy`** and **`pysrt`** (listed in `requirements.txt`).

**Time semantics:** the window is half-open `[start, end)` (start included, `end` exclusive), matching MoviePy’s subclip.

**Config**

- Prefer **`parts/parts.json`**: defines `media.source_filename`, `default_part_id`, and a `parts` array (each with `id`, `time_start`, `time_end`, `title`, `hook`, and optional `output_suffix`; default suffix is `_p01`, `_p02`, …). Output files are named `{source_stem}_{sanitized_title}{suffix}.ext` when `title` is set.
- A **`parts.json` in the project root** is used if `parts/parts.json` is missing.
- If **no** `parts.json` is found, the script falls back to `SOURCE_FILENAME`, `START_TIME`, and `END_TIME` at the top of `trim_video_srt.py`.

**Examples (from the repo root, venv active)**

```powershell
# Use default part from parts/parts.json
python trim_video_srt.py

# Choose part id 3; writes e.g. output\Name_Title_p03.m4a (Title from JSON) and matching .srt
python trim_video_srt.py --part 3

# Use another JSON file
python trim_video_srt.py --config parts\parts.json --part 2
```

**Inputs/outputs:** media in `data/`, SRT in `exports/` (filename must match the media stem, e.g. `Clip.m4a` → `exports/Clip.srt`), trimmed results in `output/`.

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

