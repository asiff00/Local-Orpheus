# Local Orpheus

A local text-to-speech system using the Orpheus AI model.

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Backend Options [lmstudio, ollama, local]

- **LM Studio** (default): [Download here](https://lmstudio.ai/download)
- **Ollama**: [Download here](https://ollama.com/download)
- **Local**: Uses llama-cpp-python
  - Default runs on CPU
  - For CUDA: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`
  - Windows/Cuda Issue: https://github.com/abetlen/llama-cpp-python/issues/2001#issuecomment-2801082928

## Base Model

```isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF```

## Usage

```bash
# Default CLI Loop (LM Studio)
python TTS.py

# Change backend
python TTS.py --backend local

# Specify text
python TTS.py --text "Hello, this is Orpheus."

# Change voice (options: tara, leah, jess, leo, dan, mia, zac, zoe)
python TTS.py --voice leo

# Save to file without playing
python TTS.py --text "Save this to a file" --output "my_speech.wav" --no-play
```

Output files are saved to the "outputs" directory by default. 

