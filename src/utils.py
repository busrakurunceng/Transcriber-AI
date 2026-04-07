"""Ses dosyası yardımcıları: yükleme, doğrulama, örnekleme."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

# Whisper / librosa için yaygın örnekleme hızı
TARGET_SR = 16000


def resolve_audio_path(path: str | Path) -> Path:
    """Yolu mutlak Path'e çevirir."""
    return Path(path).expanduser().resolve()


def validate_audio_file(path: str | Path) -> Path:
    """Dosyanın var olduğunu ve ses uzantısı taşıdığını kontrol eder."""
    p = resolve_audio_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {p}")
    suffix = p.suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise ValueError(f"Desteklenmeyen uzantı: {suffix}. Beklenen: {sorted(allowed)}")
    return p


def load_audio_mono(path: str | Path, sr: int | None = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Ses dosyasını mono olarak yükler; sr verilirse librosa ile yeniden örnekler.
    ffmpeg/libsndfile destekli formatlar librosa ile okunur.
    """
    p = validate_audio_file(path)
    y, orig_sr = librosa.load(str(p), sr=sr, mono=True)
    used_sr = sr if sr is not None else orig_sr
    return y.astype(np.float32), int(used_sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Float32 mono diziyi WAV olarak kaydeder."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="PCM_16")


def ffmpeg_available() -> bool:
    """Sistemde ffmpeg'in kurulu olup olmadığını kontrol eder."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def check_ffmpeg_or_warn() -> None:
    """ffmpeg yoksa stderr'e uyarı yazar (bazı codec'ler için gerekli olabilir)."""
    if not ffmpeg_available():
        print(
            "Uyarı: ffmpeg bulunamadı. MP3 ve bazı formatlar için "
            "https://ffmpeg.org/download.html adresinden kurmanız önerilir.",
            file=sys.stderr,
        )


def extract_audio_segment_to_wav(
    audio_path: str | Path,
    *,
    start_s: float,
    end_s: float,
    sr: int = TARGET_SR,
) -> Path:
    """
    ffmpeg ile sesin belirli bir kısmını WAV (mono, 16k) olarak geçici dosyaya çıkarır.
    Whisper/pyannote entegrasyonunda segment bazlı transkripsiyon için kullanılır.
    """
    src = validate_audio_file(audio_path)
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) start_s ({start_s}) değerinden büyük olmalı.")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg bulunamadı; segment kırpma için ffmpeg gereklidir.")

    tmp_dir = Path(tempfile.gettempdir()) / "transcriber-ai"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"segment_{src.stem}_{start_s:.2f}_{end_s:.2f}.wav"

    # -ss ... -to ... input'tan önce: hızlı seek
    # -ac 1 mono, -ar 16000 örnekleme
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_s),
        "-to",
        str(end_s),
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
