# Transcriber-AI

A Python project for **audio → text transcription** using Whisper, optional **2-speaker diarization** (pyannote), **SRT export** for CapCut, and **media trim + subtitle re-sync**.

## Features

- **ASR (main app)**: Hugging Face `openai/whisper-large-v3` via `transformers`, with long-audio **chunking**
- **Language control**: auto-detect or pin via `.env` (`WHISPER_LANGUAGE`)
- **Speaker diarization (optional)**: 2-speaker pyannote + segment-wise transcription
- **SRT export** (`whisper_srt_export.py`): separate **openai-whisper** pipeline → CapCut-ready `.srt` in `exports/`
- **Trim workflow** (`trim_video_srt.py`): cut `data/` media by time ranges from `parts/parts.json` and re-align matching SRT cues into `output/`

## Project structure

```text
Transcriber-AI/
├── data/                 # Input media (place .m4a / .mp4 here)
├── exports/              # Transcripts, diarized text, and .srt (stem must match media for trim)
├── output/               # Trim runs: output/{source_stem}/ (gitignored)
├── parts/
│   ├── parts.json        # per-part time ranges, titles, hooks (for trim_video_srt.py)
│   └── parts.txt         # human-readable copy of the same plan
├── models/               # HF cache (recommended)
├── src/
│   ├── diarizer.py       # pyannote diarization
│   ├── transcriber.py    # Whisper transcribe + chunking
│   └── utils.py          # ffmpeg segment export, audio helpers
├── main.py               # HF Whisper transcription → exports/{stem}_output.txt
├── whisper_srt_export.py # openai-whisper → exports/{stem}.srt (CapCut)
├── trim_video_srt.py     # cut media + adjust subtitles to a time window
└── requirements.txt
```

## Requirements

- **Python 3.11+**
- **ffmpeg** (required for Whisper file loading, segment extraction, and some codecs)
- **Hugging Face account + token** (only if `DIARIZE=1` in `.env`)

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

> If you plan to use GPU, make sure your PyTorch installation matches your CUDA setup (see comment at the top of `requirements.txt`).

## Configuration (`.env`)

Create a `.env` file in the project root (used by `main.py` and diarization):

```env
# Whisper (Hugging Face — main.py)
WHISPER_MODEL_ID=openai/whisper-large-v3
# WHISPER_LANGUAGE=english   # optional: pin language; omit to auto-detect

# Hugging Face cache (recommended on Windows to reduce symlink warnings)
HF_HOME=./models/hf_cache

# Diarization (optional)
DIARIZE=0
# DIARIZE=1
# HF_TOKEN=hf_...
```

Optional for **SRT export** only (CLI override; defaults are in code — see below):

```env
# WHISPER_OPENAI_MODEL=medium   # openai-whisper model name (default in script: base)
```

## Choosing input media

Both entry scripts read from `data/` but pick the file **in code** (not via CLI path):

| Script | What to edit |
|--------|----------------|
| `main.py` | `audio_path = DATA_DIR / "Your_File.m4a"` |
| `whisper_srt_export.py` | `INPUT_FILENAME = "Your_File.m4a"` and `LANGUAGE = "en"` (or `tr`, etc.) |

Put the media file in `data/` first, then update the constant and run the script.

## Usage

### SRT export for CapCut (`whisper_srt_export.py`)

Uses the **openai-whisper** package (included in `requirements.txt`). Independent of `main.py` / `src/transcriber.py`.

1. Set `INPUT_FILENAME` and `LANGUAGE` at the top of `whisper_srt_export.py`.
2. From the repo root (venv active):

```powershell
python whisper_srt_export.py
```

Optional flags: `--model medium`, `--out exports/custom.srt`, `--device cuda`.

**Output:** `exports/{stem}.srt` (e.g. `data/Clip.m4a` → `exports/Clip.srt`).

Import in CapCut: **Subtitles → Import local subtitles**.

### Media trim + SRT re-sync (`trim_video_srt.py`)

Cuts a file under `data/` to a time range and writes matching files under `output/`; cues in `exports/{same_stem}.srt` that fall in that range are kept and re-timestamped to start at `00:00:00`. Requires **`moviepy`** and **`pysrt`**.

**Time semantics:** half-open window `[start, end)` (start included, `end` exclusive), matching MoviePy’s subclip.

**Config**

- Prefer **`parts/parts.json`**: `media.source_filename` plus a `parts` array (each with `id`, `time_start`, `time_end`, `title` **required** for the output name, `hook`, and optional `output_suffix`). **By default, every part is processed in one run**; use `--part N` for a single segment. **Output** path is `output/{source_stem}/`. Under that folder, files are `{sanitized_title}.ext` (e.g. `Dopamin_Yanilgisi.m4a` and the matching `.srt`).
- A **`parts.json` in the project root** is used if `parts/parts.json` is missing.
- If **no** `parts.json` is found, the script falls back to constants at the top of `trim_video_srt.py`.

**Examples**

```powershell
python trim_video_srt.py
python trim_video_srt.py --part 3
python trim_video_srt.py --config other\parts.json
```

**Inputs/outputs:** media in `data/`, SRT in `exports/` (stem must match media, e.g. `Clip.m4a` → `exports/Clip.srt`), trimmed results in `output/{source_stem}/`.

**Typical workflow:** run `whisper_srt_export.py` on the full file → edit `parts/parts.json` → run `trim_video_srt.py`.

### Plain transcription (`main.py`, no diarization)

Set `DIARIZE=0` in `.env` (or omit it). Update `audio_path` in `main.py`, then:

```powershell
python main.py
```

**Output:** `exports/{stem}_output.txt`

### 2-speaker diarization + transcript

In `.env`: `DIARIZE=1` and `HF_TOKEN=hf_...`. Update `audio_path` in `main.py`, then:

```powershell
python main.py
```

**Output:** `exports/{stem}_diarized_output.txt`

Example format:

```text
[00:00:05.120 - 00:00:07.900] SPEAKER_00: ...
[00:00:08.100 - 00:00:12.640] SPEAKER_01: ...
```

## Notes & troubleshooting

### Two Whisper stacks

| Component | Library | Purpose |
|-----------|---------|---------|
| `main.py` | `transformers` + HF model ID | Full transcript / diarization |
| `whisper_srt_export.py` | `openai-whisper` | Timed `.srt` segments |

They do not share the same model cache or config; set each path’s file/language in code (and `.env` for the main app).

### Hugging Face 403 (gated repos)

The pyannote diarization pipeline downloads additional models at runtime. If you get a 403:

- Accept the user conditions for `pyannote/speaker-diarization-3.1`
- Also accept conditions for any other pyannote repos mentioned in the logs (e.g. `pyannote/speaker-diarization-community-1`, `pyannote/segmentation-3.0`)
- Ensure your token has **Read** access and is set as `HF_TOKEN=...` in `.env`

Reference: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### `torchcodec` warnings on Windows

`pyannote.audio` may print `torchcodec` warnings on some Windows setups. Diarization is run by passing audio **from memory**, so this warning typically does not block execution.

### ffmpeg

```powershell
ffmpeg -version
```

## Security

- Do not commit `.env` to git (it is already ignored via `.gitignore`).
