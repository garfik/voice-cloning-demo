#!/usr/bin/env python3

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.append('/app')

from coqui_worker.xtts_service import xtts_service
from coqui_worker.yourtts_service import yourtts_service

def write_worker_info():
    try:
        print("Loading XTTS model...")
        xtts_service.load_models()
        
        print("Loading YourTTS model...")
        yourtts_service.load_models()
        
        xtts_languages = xtts_service.get_supported_languages()
        xtts_speakers = xtts_service.get_available_speakers()
        
        yourtts_languages = yourtts_service.get_supported_languages()
        yourtts_speakers = yourtts_service.get_available_speakers()
        
        info = {
            "engine": "coqui",
            "models": {
                "xtts": {
                    "name": "tts_models/multilingual/multi-dataset/xtts_v2",
                    "languages": xtts_languages,
                    "speakers": xtts_speakers,
                    "supports_custom_voice": True,
                    "notes": "Works great with Russian language even if you record your voice sample in English. No sentence limitations."
                },
                "yourtts": {
                    "name": "tts_models/multilingual/multi-dataset/your_tts",
                    "languages": yourtts_languages,
                    "speakers": yourtts_speakers,
                    "supports_custom_voice": True,
                    "notes": "Really bad at voice cloning, but works nice and fast with existing samples."
                }
            }
        }
        
        with open("/tmp/coqui.info", 'w') as f:
            json.dump(info, f, indent=2)
        
        print("Coqui worker ready.")
        print(f"XTTS: {len(xtts_languages)} languages, {len(xtts_speakers)} speakers")
        print(f"YourTTS: {len(yourtts_languages)} languages, {len(yourtts_speakers)} speakers")
        
    except Exception as e:
        print(f"Error initializing Coqui worker: {e}")
        error_info = {
            "engine": "coqui",
            "error": str(e),
            "models": {}
        }
        with open("/tmp/coqui.info", 'w') as f:
            json.dump(error_info, f, indent=2)
        raise

def process_job(job_data: Dict[str, Any]) -> tuple[bool, str]:
    try:
        job_id = job_data["id"]
        text = job_data["text"]
        language = job_data.get("language", "en")
        model = job_data.get("model", "xtts")
        
        print(f"Processing Coqui job {job_id} with {model}: '{text[:50]}...'")
        
        if "your_tts" in model:
            service = yourtts_service
        else:
            service = xtts_service
        
        if "speaker" in job_data:
            speaker = job_data["speaker"]
            audio_bytes = service.synthesize_by_speaker(text, speaker, language)
        elif "ref_wav" in job_data:
            ref_wav_path = job_data["ref_wav"]
            audio_bytes = service.synthesize_by_audio(text, ref_wav_path, language)
            Path(ref_wav_path).unlink(missing_ok=True)
        else:
            return False, "No speaker or reference audio provided"
        
        output_path = f"/tmp/tts_out/{job_id}.wav"
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        with open(f"/tmp/tts_done/{job_id}.ok", 'w') as f:
            f.write("Success")
        
        print(f"Job {job_id} completed successfully")
        return True, "Success"
        
    except Exception as e:
        error_msg = f"Coqui synthesis failed: {str(e)}"
        print(f"Job {job_data.get('id', 'unknown')} failed: {error_msg}")
        
        job_id = job_data.get("id", "unknown")
        with open(f"/tmp/tts_done/{job_id}.err", 'w') as f:
            f.write(error_msg)
        
        return False, error_msg

def main():
    print("Starting Coqui Worker...")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in /app: {os.listdir('/app')}")
    
    try:
        write_worker_info()
    except Exception as e:
        print(f"Failed to initialize Coqui worker: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    queue_dir = Path("/tmp/tts_queue/coqui")
    queue_dir.mkdir(parents=True, exist_ok=True)
    
    print("Coqui Worker ready. Monitoring for jobs...")
    
    while True:
        try:
            job_files = list(queue_dir.glob("job_*.json"))
            
            for job_file in job_files:
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    success, message = process_job(job_data)
                    
                    job_file.unlink()
                    
                    if success:
                        print(f"Completed job {job_data['id']}")
                    else:
                        print(f"Failed job {job_data['id']}: {message}")
                        
                except Exception as e:
                    print(f"Error processing job file {job_file}: {e}")
                    try:
                        job_file.unlink()
                    except:
                        pass
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("Coqui Worker shutting down...")
            break
        except Exception as e:
            print(f"Unexpected error in Coqui worker: {e}")
            time.sleep(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
