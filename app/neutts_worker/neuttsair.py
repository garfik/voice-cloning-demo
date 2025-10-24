"""
NeuTTS Air implementation
Code copied from https://github.com/neuphonic/neutts-air/blob/main/neuttsair/neutts.py
"""

import torch
import torchaudio
import numpy as np
from typing import Union, List, Optional
import os
from pydub import AudioSegment
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from phonemizer.backend import EspeakBackend
from neucodec import NeuCodec, DistillNeuCodec
import re
from llama_cpp import Llama

class NeuTTSAir:
    MAX_CONTEXT = 2048
    
    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air",
        backbone_device: str = "cpu",
        codec_repo: str = "neuphonic/neucodec", 
        codec_device: str = "cpu"
    ):
        self.backbone_repo = backbone_repo
        self.backbone_device = backbone_device
        self.codec_repo = codec_repo
        self.codec_device = codec_device
        
        self._is_quantized_model = False
        self.backbone_model = None
        self.backbone_tokenizer = None
        
        self._load_models()
    
    def _load_models(self):
        try:
            print("Loading phonemizer...")
            self.phonemizer = EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )
            
            if self.backbone_repo.lower().endswith("gguf"):
                print(f"Loading GGUF model: {self.backbone_repo} (device: CPU)")
                self.backbone_model = Llama.from_pretrained(
                    repo_id=self.backbone_repo,
                    filename="*.gguf",
                    verbose=False,
                    n_gpu_layers=0,  # CPU only
                    n_ctx=self.MAX_CONTEXT,
                    mlock=True,
                    flash_attn=False,
                )
                self._is_quantized_model = True
                print("GGUF model loaded successfully")
            else:
                print(f"Loading transformer model: {self.backbone_repo} (device: {self.backbone_device})")
                self.backbone_tokenizer = AutoTokenizer.from_pretrained(self.backbone_repo)
                self.backbone_model = AutoModelForCausalLM.from_pretrained(
                    self.backbone_repo,
                    torch_dtype=torch.float32
                ).to(torch.device(self.backbone_device))
                print("Transformer model loaded successfully")
            
            print(f"Loading NeuCodec: {self.codec_repo} (device: {self.codec_device})")
            self.codec_model = NeuCodec.from_pretrained(self.codec_repo)
            self.codec_model.eval().to(self.codec_device)
            print("NeuCodec loaded successfully")
            
            print("NeuTTS Air models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading NeuTTS Air models: {e}")
            raise
    
    def _convert_audio_to_wav(self, input_path: str) -> str:
        try:
            output_path = input_path.replace('.wav', '_converted.wav')
            
            audio = AudioSegment.from_file(input_path)
            
            # IMPORTANT: Limit reference audio to 10 seconds to avoid token limit issues
            max_duration_ms = 10 * 1000  # 10 seconds
            if len(audio) > max_duration_ms:
                print(f"Warning: Reference audio too long ({len(audio)/1000:.1f}s), truncating to 10s")
                audio = audio[:max_duration_ms]
            
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(24000)  # 24kHz
            
            audio.export(output_path, format="wav")
            
            print(f"Audio converted: {input_path} -> {output_path} (duration: {len(audio)/1000:.1f}s)")
            return output_path
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            raise
    
    def encode_reference(self, audio_path: str) -> torch.Tensor:
        try:
            print("Encoding reference audio...")
            
            try:
                converted_path = self._convert_audio_to_wav(audio_path)
                audio_path = converted_path
            except Exception as e:
                print(f"Warning: Could not convert audio: {e}")
                print("Trying to load original file...")
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            wav_tensor = waveform.unsqueeze(0)
            
            with torch.no_grad():
                ref_codes = self.codec_model.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
                print("Reference audio encoded")
                return ref_codes
            
        except Exception as e:
            print(f"Error encoding reference audio: {e}")
            raise
    
    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones
    
    def _apply_chat_template(self, ref_codes: list, input_text: str) -> list:
        input_text_phones = self._to_phones(input_text)
        
        speech_replace = self.backbone_tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.backbone_tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.backbone_tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.backbone_tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.backbone_tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        
        input_ids = self.backbone_tokenizer.encode(input_text_phones, add_special_tokens=False)
        
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.backbone_tokenizer.encode(chat)
        
        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1:]
        )
        
        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.backbone_tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        
        if len(ids) > self.MAX_CONTEXT - 512:
            print(f"Warning: Prompt too long ({len(ids)} tokens, max safe is {self.MAX_CONTEXT - 512})")
            print(f"This should not happen with 10s reference audio. Consider shortening input text.")
        
        return ids
    
    def _infer_torch(self, prompt_ids: list) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone_model.device)
        speech_end_id = self.backbone_tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        with torch.no_grad():
            output_tokens = self.backbone_model.generate(
                prompt_tensor,
                max_length=self.MAX_CONTEXT,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        
        input_length = prompt_tensor.shape[-1]
        output_str = self.backbone_tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), 
            add_special_tokens=False
        )
        return output_str
    
    def _infer_ggml(self, ref_codes: list, input_text: str) -> str:
        input_text_phones = self._to_phones(input_text)
        
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        
        output = self.backbone_model(
            prompt,
            max_tokens=self.MAX_CONTEXT,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str
    
    def _decode(self, codes: str):
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        
        if len(speech_ids) > 0:
            with torch.no_grad():
                codes_tensor = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                    self.codec_model.device
                )
                recon = self.codec_model.decode_code(codes_tensor).cpu().numpy()
            
            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")
    
    def infer(
        self, 
        text: str, 
        ref_codes: list
    ) -> np.ndarray:
        try:
            print("Generating TTS with reference audio...")
            
            if self._is_quantized_model:
                output_str = self._infer_ggml(ref_codes, text)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, text)
                output_str = self._infer_torch(prompt_ids)
            
            print("Encoding to WAV...")
            wav = self._decode(output_str)
            
            print("Done")
            return wav
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
