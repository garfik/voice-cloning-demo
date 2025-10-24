import os
import io
import uuid
import soundfile as sf
from typing import List

# NeuTTS Air imports
try:
    from neutts_worker.neuttsair import NeuTTSAir
except ImportError:
    print("Warning: neuttsair module not found")
    NeuTTSAir = None

class NeuTTSService:
    def __init__(self):
        self.tts = None
        self.speakers: List[str] = []

    def get_model_name(self) -> str:
        # Use GGUF quantized model by default for better CPU performance
        # neuphonic/neutts-air better for GPU
        return "neuphonic/neutts-air-q8-gguf"
    
    def get_service_name(self) -> str:
        return "NeuTTS"

    def load_models(self):
        if NeuTTSAir is None:
            raise ImportError("NeuTTS Air module not found")
        
        print(f"Loading {self.get_service_name()} model: {self.get_model_name()} (CPU)…")
        try:
            self.tts = NeuTTSAir(
                backbone_repo=self.get_model_name(),
                backbone_device="cpu",
                codec_repo="neuphonic/neucodec",
                codec_device="cpu"
            )
            print(f"{self.get_service_name()} ready.")
        except Exception as e:
            print(f"Error loading NeuTTS Air: {e}")
            raise

    def get_available_speakers(self) -> List[str]:
        return self.speakers
    
    def get_supported_languages(self) -> List[str]:
        return ["en"]

    def synthesize_by_speaker(self, text: str, speaker_id: str, language: str = "en") -> bytes:
        raise NotImplementedError("NeuTTS Air requires custom audio reference")

    def synthesize_by_audio(self, text: str, speaker_wav_path: str, language: str = "en") -> bytes:
        try:
            ref_codes = self.tts.encode_reference(speaker_wav_path)
            wav = self.tts.infer(text, ref_codes)
            
            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            print(f"Error in NeuTTS custom voice synthesis: {e}")
            raise

neutts_service = NeuTTSService()
