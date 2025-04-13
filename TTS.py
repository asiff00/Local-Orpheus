import os
import requests
import json
import time
import wave
import numpy as np
import sounddevice as sd
import torch
import asyncio
import argparse
from typing import Optional

MODEL_NAME_OLLAMA = "hf.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
MODEL_NAME_LMSTUDIO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/orpheus-3b-0.1-ft-q4_k_m.gguf"
MODEL_NAME_LOCAL = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
MODEL_FILENAME = "orpheus-3b-0.1-ft-q4_k_m.gguf"

BACKENDS = {
    "lmstudio": {
        "api_url": "http://127.0.0.1:1234/v1/completions",
        "headers": {"Content-Type": "application/json"},
    },
    "ollama": {
        "api_url": "http://localhost:11434/api/generate",
        "headers": {"Content-Type": "application/json"},
    },
    "local": {
        "model": None  
    }
}

ACTIVE_BACKEND = "lmstudio"

MAX_TOKENS = 1200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"

SPECIAL_START = "<|audio|>"
SPECIAL_END = "<|eot_id|>"
CUSTOM_TOKEN_PREFIX = "<custom_token_"

session = requests.Session()
model_initialized = False

try:
    from snac import SNAC
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using SNAC device: {snac_device}")
    snac_model = snac_model.to(snac_device)
    snac_loaded = True
except Exception as e:
    print(f"Warning: Could not load SNAC model: {e}")
    print("Please install SNAC with: pip install snac")
    snac_loaded = False

def convert_to_audio(multiframe, count):
    if not snac_loaded:
        print("Error: SNAC model not loaded")
        return None
        
    if len(multiframe) < 7:
        return None
    
    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        
        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return None

    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None

def initialize_model() -> bool:
    """Initialize connection to the model."""
    global model_initialized
    if not model_initialized:
        print(f"Initializing model connection using {ACTIVE_BACKEND} backend...")
        try:
            if ACTIVE_BACKEND == "local":
                from llama_cpp import Llama
                try:
                    from llama_cpp.llama_cpp import llama_supports_gpu_offload
                    GPU_AVAILABLE = llama_supports_gpu_offload()
                    if GPU_AVAILABLE:
                        print("GPU acceleration is available for llama.cpp")
                    else:
                        print("WARNING: llama-cpp-python is installed but without GPU support!")
                except Exception as e:
                    print(f"Could not determine GPU support: {e}")
                    GPU_AVAILABLE = False
                    print("llama-cpp-python not available. To use local inference, install with: pip install llama-cpp-python")
                try:
                    n_gpu_layers = 99 
                    if not GPU_AVAILABLE:
                        print("WARNING: Running without GPU acceleration. Inference will be much slower.")
                        n_gpu_layers = 0  
                    print(f"Attempting to offload {n_gpu_layers} layers to GPU...")
                    BACKENDS["local"]["model"] = Llama.from_pretrained(
                        repo_id="isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
                        filename=MODEL_FILENAME,
                        n_ctx=2048,         
                        n_gpu_layers=n_gpu_layers,  
                        n_batch=512,        
                        verbose=False        
                    )
                    model_initialized = True
                    print("Local model initialized successfully")
                    
                    if hasattr(BACKENDS["local"]["model"], "model") and hasattr(BACKENDS["local"]["model"].model, "n_gpu_layers"):
                        actual_gpu_layers = BACKENDS["local"]["model"].model.n_gpu_layers
                        if actual_gpu_layers > 0:
                            print(f"SUCCESS: Model is using GPU acceleration with {actual_gpu_layers} layers offloaded")
                        else:
                            print("WARNING: Model is NOT using GPU acceleration!")
                    
                except Exception as e:
                    print(f"Error initializing local model: {e}")
                    if "CUDA" in str(e) or "GPU" in str(e):
                        print("\nGPU error detected. Please check your CUDA installation and make sure llama-cpp-python was compiled with CUDA support.")
                        print("Installation command: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install --force-reinstall llama-cpp-python")
                    return False
            elif ACTIVE_BACKEND == "lmstudio":
                payload = {
                    "model": MODEL_NAME_LMSTUDIO,
                    "prompt": f"{SPECIAL_START}{DEFAULT_VOICE}: test{SPECIAL_END}",
                    "max_tokens": 1,
                    "stream": False
                }
                response = session.post(
                    BACKENDS[ACTIVE_BACKEND]["api_url"], 
                    headers=BACKENDS[ACTIVE_BACKEND]["headers"], 
                    json=payload
                )
                if response.status_code == 200:
                    model_initialized = True
                    print("Model connection initialized successfully")
                else:
                    print(f"Warning: Failed to initialize model connection. Status code: {response.status_code}")
            else:  # ollama 
                payload = {
                    "model": MODEL_NAME_OLLAMA,
                    "prompt": f"{SPECIAL_START}{DEFAULT_VOICE}: test{SPECIAL_END}",
                    "options": {
                        "num_predict": 1
                    },
                    "stream": False
                }
                response = session.post(
                    BACKENDS[ACTIVE_BACKEND]["api_url"], 
                    headers=BACKENDS[ACTIVE_BACKEND]["headers"], 
                    json=payload
                )
                if response.status_code == 200:
                    model_initialized = True
                    print("Model connection initialized successfully")
                else:
                    print(f"Warning: Failed to initialize model connection. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error initializing model connection: {e}")
    return model_initialized

def format_prompt(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Format the prompt with voice and special tokens."""
    if voice not in VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
    
    formatted_prompt = f"{voice}: {text}"
    return f"{SPECIAL_START}{formatted_prompt}{SPECIAL_END}"

async def tokens_to_audio(tokens):
    """Convert tokens to audio segments."""
    audio_segments = []
    buffer = []
    count = 0
    
    for token_text in tokens:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    audio_segments.append(audio_samples)
    
    return audio_segments

async def generate(text: str, voice: str = DEFAULT_VOICE, 
                  output_file: Optional[str] = None,
                  play_audio: bool = True,
                  temperature: float = TEMPERATURE,
                  top_p: float = TOP_P,
                  max_tokens: int = MAX_TOKENS,
                  repetition_penalty: float = REPETITION_PENALTY) -> bytes:
    """Generate speech from text."""
    initialize_model()
    
    formatted_prompt = format_prompt(text, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    start_time = time.time()
    collected_tokens = []
    
    try:
        if ACTIVE_BACKEND == "local":
            llm = BACKENDS["local"]["model"]
            
            output = llm(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                stream=True
            )
            for chunk in output:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    token_text = chunk['choices'][0].get('text', '')
                    if token_text:
                        collected_tokens.append(token_text)
        else:
            if ACTIVE_BACKEND == "lmstudio":
                payload = {
                    "model": MODEL_NAME_LMSTUDIO,
                    "prompt": formatted_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repetition_penalty,
                    "stream": True
                }
            else:  # ollama
                payload = {
                    "model": MODEL_NAME_OLLAMA  ,
                    "prompt": formatted_prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "repeat_penalty": repetition_penalty
                    },
                    "stream": True
                }
            
            response = session.post(
                BACKENDS[ACTIVE_BACKEND]["api_url"], 
                headers=BACKENDS[ACTIVE_BACKEND]["headers"], 
                json=payload, 
                stream=True
            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                return b''
            
            if ACTIVE_BACKEND == "lmstudio":
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    token_text = data['choices'][0].get('text', '')
                                    if token_text:
                                        collected_tokens.append(token_text)
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON: {e}")
                                continue
            else:  # ollama
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                token_text = data['response']
                                if token_text:
                                    collected_tokens.append(token_text)
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
    except Exception as e:
        print(f"Error generating tokens: {e}")
        return b''
    
    token_time = time.time()
    print(f"Token generation took {token_time - start_time:.2f} seconds")
    
    audio_segments = await tokens_to_audio(collected_tokens)
    
    audio_time = time.time()
    print(f"Audio conversion took {audio_time - token_time:.2f} seconds")
    
    if output_file is None and audio_segments:
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{voice}_{timestamp}.wav"
    
    if output_file and audio_segments:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            for segment in audio_segments:
                wav_file.writeframes(segment)
        
        print(f"Audio saved to {output_file}")
    
    audio_buffer = b''.join(audio_segments) if audio_segments else b''
    
    if play_audio and audio_buffer:
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32767.0
        print("Playing audio...")
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()
    
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds")
    
    return audio_buffer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech")
    backend_choices = ["lmstudio", "ollama", "local"]
    parser.add_argument("--backend", type=str, choices=backend_choices, default="lmstudio", 
                        help="Backend service to use (lmstudio, ollama, or local if available)")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=VOICES, 
                        help="Voice to use for synthesis")
    parser.add_argument("--output", type=str, help="Output file path (optional)")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio after generation")
    return parser.parse_args()

if __name__ == "__main__":
    print("TTS Module - Orpheus Text-to-Speech")
    
    args = parse_arguments()
    ACTIVE_BACKEND = args.backend
    print(f"Using {ACTIVE_BACKEND} backend")
    
    async def main():
        if args.text:
            await generate(
                text=args.text,
                voice=args.voice,
                output_file=args.output,
                play_audio=not args.no_play
            )
        else:
            while True:
                text = input("\nEnter text to synthesize (or 'exit' to quit): ")
                if text.lower() in ('exit', 'quit'):
                    break
                    
                if not text:
                    text = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
                
                await generate(text=text, voice=args.voice, output_file=args.output, play_audio=not args.no_play)
    
    asyncio.run(main()) 