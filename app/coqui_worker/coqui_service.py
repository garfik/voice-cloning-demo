import os, io, base64, re, numpy as np, soundfile as sf, uuid
from typing import Dict, Any, List
from pydub import AudioSegment
from abc import ABC, abstractmethod
from TTS.api import TTS  # Coqui TTS API

SAMPLE_RATE = 16000

def _load_to_mono16k_any(path: str) -> str:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(SAMPLE_RATE)
            y = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                y = y.reshape(-1, audio.channels).mean(axis=1)
        tmp = io.BytesIO()
        sf.write(tmp, y, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        tmp.seek(0)
        tmp_path = f"/tmp/coqui_ref_{uuid.uuid4()}.wav"
        with open(tmp_path, "wb") as f:
            f.write(tmp.read())
        return tmp_path
    except Exception:
        pass

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    out_path = f"/tmp/coqui_ref_{uuid.uuid4()}.wav"
    audio.export(out_path, format="wav")
    return out_path

class CoquiTTSService(ABC):
    def __init__(self):
        self.tts = None
        self.speakers: List[str] = []

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        pass

    def load_models(self):
        model_name = self.get_model_name()
        service_name = self.get_service_name()
        print(f"Loading {service_name} model: {model_name} (CPU)…")
        self.tts = TTS(
            model_name=model_name, 
            progress_bar=False, 
            gpu=False
        )
        print(f"{service_name} ready.")
        
        self._load_builtin_speakers()

    def _load_builtin_speakers(self):
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            self.speakers = list(self.tts.speakers)

    def get_available_speakers(self) -> List[str]:
        return self.speakers
    
    def get_supported_languages(self) -> List[str]:
        if hasattr(self.tts, 'languages') and self.tts.languages:
            return list(self.tts.languages)
        return ['en']

    def _read_and_cleanup_audio_file(self, out_path: str) -> bytes:
        try:
            y, sr = sf.read(out_path, dtype="float32")
            buf = io.BytesIO()
            sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.getvalue()
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def synthesize_by_speaker(self, text: str, speaker_id: str, language: str = "en") -> bytes:
        if not self.speakers:
            raise ValueError("No speakers available. Please check if models are loaded.")
        
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}. Available: {list(self.speakers.keys())}")
        
        out_path = f"/tmp/{self.get_service_name().lower()}_out_{uuid.uuid4()}.wav"
        try:
            self.tts.tts_to_file(
                text=text,
                speaker=speaker_id,
                language=language,
                file_path=out_path
            )
        except Exception as e:
            print(f"Failed with speaker parameter, trying without: {e}")
            self.tts.tts_to_file(
                text=text,
                language=language,
                file_path=out_path
            )
        
        return self._read_and_cleanup_audio_file(out_path)

    def synthesize_by_audio(self, text: str, speaker_wav_path: str, language: str = "en") -> bytes:
        ref_path = _load_to_mono16k_any(speaker_wav_path)
        
        out_path = f"/tmp/{self.get_service_name().lower()}_out_{uuid.uuid4()}.wav"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path
        )
        
        return self._read_and_cleanup_audio_file(out_path)


