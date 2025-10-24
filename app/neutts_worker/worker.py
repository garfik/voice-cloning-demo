#!/usr/bin/env python3

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.append('/app')

from neutts_worker.neutts_service import neutts_service

def write_worker_info():
    try:
        print("Loading NeuTTS Air model...")
        neutts_service.load_models()
        
        languages = neutts_service.get_supported_languages()
        speakers = neutts_service.get_available_speakers()
        
        model_name = neutts_service.get_model_name()
        info = {
            "engine": "neutts",
            "models": {
                "neutts-air": {
                    "name": model_name,
                    "languages": languages,
                    "speakers": speakers,
                    "supports_custom_voice": True,
                    "notes": "Best with single sentences. Example: 'Hello and tell me your name' works better than 'Hello! What is your name?'. Context window: 2048 tokens (~30s audio including prompt). Reference audio automatically limited to first 10s."
                }
            }
        }
        
        with open("/tmp/neutts.info", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"NeuTTS worker ready. Languages: {languages}")
        print("NeuTTS requires custom reference audio")
        
    except Exception as e:
        print(f"Error initializing NeuTTS worker: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_job(job_data: Dict[str, Any]) -> tuple[bool, str]:
    try:
        job_id = job_data["id"]
        text = job_data["text"]
        language = job_data.get("language", "en")
        
        print(f"Processing NeuTTS job {job_id}: '{text[:50]}...'")
        
        if "ref_wav" not in job_data:
            return False, "NeuTTS requires reference audio (ref_wav)"
        
        ref_wav_path = job_data["ref_wav"]
        
        if not os.path.exists(ref_wav_path):
            return False, f"Reference audio file not found: {ref_wav_path}"
        
        audio_bytes = neutts_service.synthesize_by_audio(text, ref_wav_path, language)
        
        output_path = f"/tmp/tts_out/{job_id}.wav"
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        with open(f"/tmp/tts_done/{job_id}.ok", 'w') as f:
            f.write("Success")
        
        print(f"Job {job_id} completed successfully")
        return True, "Success"
        
    except Exception as e:
        error_msg = f"NeuTTS synthesis failed: {str(e)}"
        print(f"Job {job_data.get('id', 'unknown')} failed: {error_msg}")
        
        job_id = job_data.get("id", "unknown")
        with open(f"/tmp/tts_done/{job_id}.err", 'w') as f:
            f.write(error_msg)
        
        return False, error_msg

def main():
    print("Starting NeuTTS Worker...")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in /app: {os.listdir('/app')}")
    
    try:
        write_worker_info()
    except Exception as e:
        print(f"Failed to initialize NeuTTS worker: {e}")
        return 1
    
    queue_dir = Path("/tmp/tts_queue/neutts")
    queue_dir.mkdir(parents=True, exist_ok=True)
    
    print("NeuTTS Worker ready. Monitoring for jobs...")
    
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
            print("NeuTTS Worker shutting down...")
            break
        except Exception as e:
            print(f"Unexpected error in NeuTTS worker: {e}")
            time.sleep(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
