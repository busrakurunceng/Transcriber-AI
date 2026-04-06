"""Hugging Face Transformers ile Whisper konuşma tanıma."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
        language: str = "turkish",
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "WHISPER_MODEL_ID", "openai/whisper-large-v3"
        )
        self.language = language

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

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, file_path: str | Path, *, language: str | None = None) -> str:
        """Ses dosyasını metne çevirir (>30 sn için zaman damgası modu gerekir)."""
        lang = language or self.language
        path = str(Path(file_path).expanduser().resolve())
        result = self.pipe(
            path,
            return_timestamps=True,
            generate_kwargs={"language": lang},
        )
        if not isinstance(result, dict):
            return str(result).strip()
        text = (result.get("text") or "").strip()
        if not text and result.get("chunks"):
            parts = [c.get("text", "") for c in result["chunks"] if isinstance(c, dict)]
            text = " ".join(p for p in parts if p).strip()
        return text
