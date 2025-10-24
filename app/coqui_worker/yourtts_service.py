from .coqui_service import CoquiTTSService

class YourTTSService(CoquiTTSService):
    def get_model_name(self) -> str:
        return "tts_models/multilingual/multi-dataset/your_tts"
    
    def get_service_name(self) -> str:
        return "YourTTS"
    
    def get_supported_languages(self):
        return ["en", "fr-fr", "pt-br"]

yourtts_service = YourTTSService()
