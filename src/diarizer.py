"""Pyannote ile konuşmacı diarization (kim, ne zaman konuştu)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import torch
from huggingface_hub.errors import GatedRepoError
from pyannote.audio import Pipeline


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


class SpeakerDiarizer:
    """
    Pyannote speaker diarization wrapper.

    Not: Pyannote modelleri için Hugging Face token ve model şartlarını kabul etmeniz gerekebilir.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        *,
        model_id: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        # .env'den token'ı al (öncelik: arg -> HF_TOKEN -> HUGGINGFACE_TOKEN)
        token = auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token bulunamadı. Lütfen .env içine HF_TOKEN=... veya "
                "HUGGINGFACE_TOKEN=... ekleyin."
            )

        # pyannote.audio>=4 token parametresi kullanır (use_auth_token değil)
        try:
            self.pipeline = Pipeline.from_pretrained(model_id, token=token)
        except GatedRepoError as e:
            # Not: Ana pipeline repo'su erişilebilir olsa bile, pyannote bazı alt modelleri
            # (örn. speaker-diarization-community-1) ayrıca indirir ve onlar da gated olabilir.
            raise PermissionError(
                "Hugging Face 403 (gated repo). Pyannote diarization için gerekli modellerden "
                "en az birine erişimin yok.\n\n"
                f"Detay: {e}\n\n"
                "Yapman gereken:\n"
                f"- https://hf.co/{model_id} sayfasında şartları kabul et / access iste.\n"
                "- Ayrıca log'da adı geçen diğer pyannote repoları için de aynı işlemi yap "
                "(ör. `pyannote/speaker-diarization-community-1`).\n"
                "- Sonra .env içine `HF_TOKEN=hf_...` (Read) koy ve tekrar dene."
            ) from e
        except Exception as e:
            # 401/403/token formatı gibi durumlarda daha açıklayıcı mesaj
            raise PermissionError(
                f"Pyannote pipeline yüklenemedi: {e}\n"
                "HF_TOKEN doğru mu ve bu model için izin/şartlar tamam mı kontrol et."
            ) from e
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(device)
        print(f"Pyannote model loaded on {device}")

    def get_segments(
        self,
        audio_path: str | Path,
        *,
        num_speakers: int = 2,
        min_duration: float = 0.2,
    ) -> List[SpeakerSegment]:
        """
        Diarization yapıp konuşmacı segmentlerini döndürür.

        - num_speakers=2: iki konuşmacılı senaryoda hatayı azaltır.
        - min_duration: çok kısa segmentleri (öksürük/klik) filtrelemek için.
        """
        # torchcodec/ffmpeg bağımlılığını by-pass etmek için sesi belleğe alıp pipeline'a veriyoruz.
        path = str(Path(audio_path).expanduser().resolve())
        waveform, sr = librosa.load(path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, time)
        diarization = self.pipeline(
            {"waveform": wav_tensor, "sample_rate": sr},
            num_speakers=num_speakers,
        )

        # pyannote.audio v4: Pipeline genelde DiarizeOutput döndürür.
        # Annotation alanları: .speaker_diarization / .exclusive_speaker_diarization
        annotation = diarization
        if not hasattr(annotation, "itertracks"):
            if hasattr(diarization, "speaker_diarization"):
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, "exclusive_speaker_diarization"):
                annotation = diarization.exclusive_speaker_diarization
            elif hasattr(diarization, "diarization"):
                # eski sürümlerle uyumluluk
                annotation = diarization.diarization
        if not hasattr(annotation, "itertracks"):
            raise TypeError(
                "Beklenmeyen diarization çıktısı: Annotation bulunamadı. "
                f"Tip: {type(diarization)}"
            )

        segments: List[SpeakerSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end - start < min_duration:
                continue
            segments.append(SpeakerSegment(start=start, end=end, speaker=str(speaker)))

        # Zaman sırasına göre
        segments.sort(key=lambda s: (s.start, s.end))
        return segments

