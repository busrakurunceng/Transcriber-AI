"""Uygulama giriş noktası: ses dosyasını metne çevirir ve exports/ altına yazar."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.transcriber import WhisperTranscriber
from src.utils import check_ffmpeg_or_warn, validate_audio_file

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"


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
    ai_tool = WhisperTranscriber()

    print("Deşifre işlemi başladı...")
    text = ai_tool.transcribe(audio_path)

    print("-" * 30)
    print("Sonuç:", text)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXPORTS_DIR / "output.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Metin kaydedildi: {out_file}")


if __name__ == "__main__":
    main()
