"""
Eklenti: OpenAI Whisper (openai-whisper paketi) ile ses/videodan .srt üretir.

CapCut: Projede Altyazılar → Yerel altyazıları içe aktar ile bu .srt dosyasını yükleyin;
uygulamanın ücretli otomatik tanımasına gerek kalmaz.

Kullanım:
  pip install openai-whisper
  (Sistemde ffmpeg kurulu olmalı — Whisper dosyayı böyle yükler.)

  python whisper_srt_export.py

  Başka dosya/dil için aşağıdaki INPUT_FILENAME ve LANGUAGE sabitlerini güncelleyin.
  İsteğe bağlı CLI: python whisper_srt_export.py --model medium

Ana uygulama (main.py, src/transcriber.py) bu dosyadan bağımsızdır.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# İşlenecek ses/video — yeni dosya için burayı değiştirin
INPUT_FILENAME = "Why_Success_Blinds_Us_To_Luck.m4a"
LANGUAGE = "en"


def _srt_timestamp(seconds: float) -> str:
    """Whisper saniye → SubRip zaman kodu (virgülle milisaniye)."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Whisper result['segments'] listesini SubRip metnine çevirir."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _srt_timestamp(seg["start"])
        end = _srt_timestamp(seg["end"])
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Whisper ile ses/videodan SRT altyazı dosyası üretir (CapCut ile uyumlu)."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_OPENAI_MODEL", "base"),
        help='Whisper model adı: tiny, base, small, medium, large, large-v2, large-v3 (varsayılan: base)',
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Çıktı .srt yolu (varsayılan: exports/<dosya_adı>.srt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda veya cpu (boş: otomatik)",
    )
    args = parser.parse_args()

    input_path = (DATA_DIR / INPUT_FILENAME).resolve()
    if not input_path.is_file():
        print(
            f"Dosya bulunamadı: {input_path}\n"
            f"Lütfen dosyayı '{DATA_DIR}' içine koyun ve "
            f"whisper_srt_export.py içindeki INPUT_FILENAME değişkenini güncelleyin.",
            file=sys.stderr,
        )
        return 1

    try:
        import whisper
    except ImportError:
        print(
            "openai-whisper yüklü değil. Kurulum: pip install openai-whisper\n"
            "Ayrıca ffmpeg'in PATH üzerinde olması gerekir.",
            file=sys.stderr,
        )
        return 1

    out_path = args.out
    if out_path is None:
        root = Path(__file__).resolve().parent
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        out_path = exports / f"{input_path.stem}.srt"
    else:
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model yükleniyor: {args.model} (ilk seferde indirilebilir)...")
    model = whisper.load_model(args.model, device=args.device)

    transcribe_kw: dict = {"language": LANGUAGE}

    print(f"Transkripsiyon: {input_path}")
    result = model.transcribe(str(input_path), **transcribe_kw)

    segments = result.get("segments") or []
    if not segments:
        print("Uyarı: segment bulunamadı; boş veya kısa çıktı olabilir.", file=sys.stderr)

    srt_body = segments_to_srt(segments)
    out_path.write_text(srt_body, encoding="utf-8")
    print(f"SRT kaydedildi: {out_path}")
    if result.get("text"):
        print("Özet metin:", (result["text"] or "").strip()[:500] + ("..." if len((result["text"] or "")) > 500 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
