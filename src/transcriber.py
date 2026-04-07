"""Hugging Face Transformers ile Whisper konuşma tanıma."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from src.utils import TARGET_SR, extract_audio_segment_to_wav, load_audio_mono


def _default_cache_dir() -> str | None:
    """Proje içi models/ klasörünü önbellek olarak kullanmak için."""
    root = Path(__file__).resolve().parent.parent
    cache = root / "models" / "hf_cache"
    if cache.parent.exists():
        cache.mkdir(parents=True, exist_ok=True)
        return str(cache)
    return None


class WhisperTranscriber:
    def __init__(
        self,
        model_id: str | None = None,
        *,
        language: str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "WHISPER_MODEL_ID", "openai/whisper-large-v3"
        )
        # None => Whisper otomatik dil algılama
        self.language = language or os.environ.get("WHISPER_LANGUAGE") or None

        hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
        if not hf_home:
            default = _default_cache_dir()
            if default:
                os.environ.setdefault("HF_HOME", default)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _transcribe_array(self, audio: np.ndarray, *, language: str | None = None) -> str:
        """
        Numpy waveform (mono) -> text.
        transformers pipeline yerine doğrudan model+processor kullanır (torchcodec bağımlılığını by-pass eder).
        """
        lang = language if language is not None else self.language

        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        generate_kwargs: dict = {"task": "transcribe"}
        if lang:
            generate_kwargs["language"] = lang

        with torch.inference_mode():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return (text or "").strip()

    def transcribe(self, file_path: str | Path, *, language: str | None = None) -> str:
        """
        Ses dosyasını metne çevirir.

        Not: Whisper doğal olarak ~30 sn pencerelerle çalışır. Uzun dosyaları otomatik chunk'layıp birleştiriyoruz.
        """
        audio, sr = load_audio_mono(file_path, sr=TARGET_SR)
        if sr != TARGET_SR:
            # load_audio_mono sr parametresi verildiğinde zaten TARGET_SR üretir; bu kontrol emniyet amaçlı.
            raise ValueError(f"Beklenmeyen sample rate: {sr}")

        # ~28 sn chunk + 2 sn overlap (cümle bölünmelerini azaltmak için)
        chunk_s = 28.0
        overlap_s = 2.0
        chunk_len = int(chunk_s * TARGET_SR)
        step = int((chunk_s - overlap_s) * TARGET_SR)

        if len(audio) <= chunk_len:
            return self._transcribe_array(audio, language=language)

        parts: list[str] = []
        for start in range(0, len(audio), step):
            end = min(len(audio), start + chunk_len)
            chunk = audio[start:end]
            if len(chunk) < int(0.5 * TARGET_SR):
                break
            parts.append(self._transcribe_array(chunk, language=language))
            if end >= len(audio):
                break

        return " ".join(p for p in parts if p).strip()

    def transcribe_segment(
        self,
        file_path: str | Path,
        *,
        start_s: float,
        end_s: float,
        language: str | None = None,
    ) -> str:
        """Ses dosyasının [start_s, end_s] aralığını transkribe eder."""
        segment_wav = extract_audio_segment_to_wav(
            file_path, start_s=start_s, end_s=end_s, sr=TARGET_SR
        )
        return self.transcribe(segment_wav, language=language)
