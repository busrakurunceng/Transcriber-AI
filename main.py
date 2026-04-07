"""Uygulama giriş noktası: ses dosyasını metne çevirir ve exports/ altına yazar."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.diarizer import SpeakerDiarizer
from src.transcriber import WhisperTranscriber
from src.utils import check_ffmpeg_or_warn, validate_audio_file

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"

def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def main() -> None:
    load_dotenv()
    check_ffmpeg_or_warn()

    audio_path = DATA_DIR / "Why_you_can_t_stop_scrolling_at_night.m4a"
    if not audio_path.is_file():
        print(
            f"Örnek dosya yok: {audio_path}\n"
            f"Lütfen bir ses dosyasını '{DATA_DIR}' içine koyun ve "
            f"main.py içindeki audio_path değişkenini güncelleyin."
        )
        return

    validate_audio_file(audio_path)

    print("Model yükleniyor, bu biraz zaman alabilir...")
    transcriber = WhisperTranscriber()

    diarize = os.environ.get("DIARIZE", "0").strip().lower() in {"1", "true", "yes", "on"}
    if diarize:
        print("Konuşmacı analizi yapılıyor (pyannote)...")
        diarizer = SpeakerDiarizer()
        segments = diarizer.get_segments(audio_path, num_speakers=2)

        print("Metne dönüştürülüyor (segment bazlı)...")
        lines: list[str] = []
        for seg in segments:
            text = transcriber.transcribe_segment(audio_path, start_s=seg.start, end_s=seg.end)
            line = f"[{_fmt_ts(seg.start)} - {_fmt_ts(seg.end)}] {seg.speaker}: {text}"
            print(line)
            lines.append(line)

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = EXPORTS_DIR / "diarized_output.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"Metin kaydedildi: {out_file}")
        return

    print("Deşifre işlemi başladı...")
    text = transcriber.transcribe(audio_path)

    print("-" * 30)
    print("Sonuç:", text)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXPORTS_DIR / "output.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Metin kaydedildi: {out_file}")


if __name__ == "__main__":
    main()
