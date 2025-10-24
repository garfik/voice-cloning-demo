from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import uuid
import time
import os
import io
import sys
import asyncio
from pathlib import Path

app = FastAPI(title="TTS Demo")

worker_capabilities = {}

ENABLED_WORKERS = os.getenv("ENABLED_WORKERS", "coqui,neutts").split(",")
WORKER_INFO_FILES = {
    "coqui": "/tmp/coqui.info",
    "neutts": "/tmp/neutts.info"
}

@app.on_event("startup")
async def startup_event():
    print("Starting TTS Gateway...")
    print(f"ENABLED_WORKERS: {os.getenv('ENABLED_WORKERS')}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Waiting for workers to initialize...")
    
    timeout = 10 * 60 # 10 minutes
    start_time = time.time()
    
    print(f"Waiting for workers: {ENABLED_WORKERS}")
    
    while time.time() - start_time < timeout:
        all_ready = True
        for worker in ENABLED_WORKERS:
            if worker in WORKER_INFO_FILES:
                info_path = Path(WORKER_INFO_FILES[worker])
                if not info_path.exists():
                    all_ready = False
                    break
        
        if all_ready:
            try:
                for worker in ENABLED_WORKERS:
                    if worker in WORKER_INFO_FILES:
                        info_path = WORKER_INFO_FILES[worker]
                        with open(info_path, 'r') as f:
                            worker_capabilities[worker] = json.load(f)
                
                print("All workers initialized successfully!")
                print(f"Available engines: {list(worker_capabilities.keys())}")
                return
            except Exception as e:
                print(f"Error loading worker info: {e}")
                break
        
        await asyncio.sleep(1)
    
    print("Warning: Some workers may not be ready. Available engines:", list(worker_capabilities.keys()))

app.mount("/static", StaticFiles(directory="gateway/static"), name="static")

class TTSRequest(BaseModel):
    text: str
    engine: str = "coqui"
    model: Optional[str] = None
    submodel: Optional[str] = None
    language: str = "en"
    speaker: Optional[str] = None

class TTSResponse(BaseModel):
    audio_data: str
    engine: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("gateway/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/models")
async def get_models():
    """Get available TTS models and their capabilities"""
    models = []
    
    for engine, capabilities in worker_capabilities.items():
        for submodel, subcapabilities in capabilities.get("models", {}).items():
            models.append({
                "engine": engine,
                "model": subcapabilities["name"],
                "languages": subcapabilities["languages"],
                "speakers": subcapabilities["speakers"],
                "supports_custom_voice": subcapabilities["supports_custom_voice"],
                "notes": subcapabilities.get("notes", "")
            })
    
    return models

@app.get("/api/health")
async def health_check():
    return {"ok": True}

async def wait_for_job_completion(job_id: str, timeout: int = 600) -> tuple[bool, str]:
    """Wait for job completion and return (success, message)"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ok_path = Path(f"/tmp/tts_done/{job_id}.ok")
        err_path = Path(f"/tmp/tts_done/{job_id}.err")
        
        if ok_path.exists():
            return True, "Success"
        elif err_path.exists():
            try:
                with open(err_path, 'r') as f:
                    error_msg = f.read().strip()
                return False, error_msg
            except:
                return False, "Unknown error"
        
        await asyncio.sleep(0.1)
    
    return False, "Timeout"

def cleanup_job_files(job_id: str):
    files_to_clean = [
        f"/tmp/tts_in/{job_id}.wav",
        f"/tmp/tts_out/{job_id}.wav",
        f"/tmp/tts_done/{job_id}.ok",
        f"/tmp/tts_done/{job_id}.err"
    ]
    
    for file_path in files_to_clean:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
        
        if request.engine not in worker_capabilities:
            raise HTTPException(status_code=400, detail=f"Engine '{request.engine}' not available")
        
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "text": request.text,
            "language": request.language
        }
        
        if request.engine == "coqui" and (request.model or request.submodel):
            job_data["model"] = request.submodel or request.model
        elif request.engine == "neutts":
            job_data["model"] = "neuphonic/neutts-air"
        
        if request.speaker:
            job_data["speaker"] = request.speaker
        
        job_path = f"/tmp/tts_queue/{request.engine}/job_{job_id}.json"
        with open(job_path, 'w') as f:
            json.dump(job_data, f)
        
        success, message = await wait_for_job_completion(job_id)
        
        if not success:
            cleanup_job_files(job_id)
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {message}")
        
        output_path = f"/tmp/tts_out/{job_id}.wav"
        if not os.path.exists(output_path):
            cleanup_job_files(job_id)
            raise HTTPException(status_code=500, detail="Output file not found")
        
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        
        cleanup_job_files(job_id)
        
        import base64
        audio_data = base64.b64encode(audio_bytes).decode("utf-8")
        
        return TTSResponse(
            audio_data=audio_data,
            engine=request.engine
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error synthesizing speech: {str(e)}")

@app.post("/tts_with_audio")
async def synthesize_with_audio(
    text: str = Form(...), 
    language: str = Form("en"), 
    engine: str = Form(...),
    model: str = Form(...), 
    submodel: str = Form(None),
    file: UploadFile = File(...)
):
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
        
        if engine not in worker_capabilities:
            raise HTTPException(status_code=400, detail=f"Engine '{engine}' not available")
        
        job_id = str(uuid.uuid4())
        
        ref_audio_path = f"/tmp/tts_in/{job_id}.wav"
        with open(ref_audio_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        job_data = {
            "id": job_id,
            "text": text,
            "language": language,
            "ref_wav": ref_audio_path
        }
        
        if engine == "coqui":
            job_data["model"] = submodel or model
        elif engine == "neutts":
            job_data["model"] = "neuphonic/neutts-air"
        
        job_path = f"/tmp/tts_queue/{engine}/job_{job_id}.json"
        with open(job_path, 'w') as f:
            json.dump(job_data, f)
        
        success, message = await wait_for_job_completion(job_id)
        
        if not success:
            cleanup_job_files(job_id)
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {message}")
        
        output_path = f"/tmp/tts_out/{job_id}.wav"
        if not os.path.exists(output_path):
            cleanup_job_files(job_id)
            raise HTTPException(status_code=500, detail="Output file not found")
        
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        
        cleanup_job_files(job_id)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes), 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=speech_{engine}.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error synthesizing speech: {str(e)}")
