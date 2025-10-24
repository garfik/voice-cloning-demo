from .coqui_service import CoquiTTSService

class XTTSService(CoquiTTSService):
    def get_model_name(self) -> str:
        return "tts_models/multilingual/multi-dataset/xtts_v2"
    
    def get_service_name(self) -> str:
        return "XTTS"

xtts_service = XTTSService()
