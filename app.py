from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
import io
import os 
import tempfile
import subprocess
import soundfile as sf
import librosa
import numpy as np
import resampy
from model import DenoisingExecutive

app = FastAPI(title="STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = []
SUPPORTING_MODELS = ["clean.full", "clean.high", "dc.n2c", "dc.n2n", "wave.unet", "wave.resnet"]
SAMPLE_RATE = 16000

# TODO: load your model/decoder once at startup
@app.on_event("startup")
def _load():
    # Example:
    # global asr
    # asr = YourASR.load_from_checkpoint("/models/ckpt.pt")
    global MODELS
    MODELS = {name : DenoisingExecutive(model_name=name) for name in SUPPORTING_MODELS}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

def ffmpeg_to_wav16k(in_path: str, out_path: str):
    # Convert anything → 16k mono s16 WAV
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16", out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def convert_to_mono_16k_wav(input_path: str, output_path: str):
    """
    Convert an audio file of any format to a mono, 16 kHz WAV file.
    
    Args:
        input_path (str): Path to the input audio (any format readable by librosa or soundfile).
        output_path (str): Path to save the converted audio (should end with .wav).
    """
    # --- Load audio (librosa handles most formats via ffmpeg) ---
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    # sr=None preserves original sample rate

    # --- Convert to mono if stereo or multi-channel ---
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # --- Resample to 16 kHz if needed ---
    target_sr = 16000
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr

    # --- Normalize to float32 for safety ---
    audio = np.asarray(audio, dtype=np.float32)

    # --- Ensure output directory exists ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Save as WAV ---
    sf.write(output_path, audio, sr, subtype="PCM_16")

    print(f"✅ Converted {input_path} → {output_path} [mono, 16 kHz WAV]")


@app.post("/inference")
def inference(file: UploadFile = File(...), output: str = "output.wav", model: str = "clean.full"):
    # Save upload to temp, convert to wav16k if needed
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename)
        with open(src_path, "wb") as f:
            f.write(file.file.read())

        wav_path = os.path.join(td, "pcm16k.wav")
        print("Converting input to 16k WAV...")
        try:
            convert_to_mono_16k_wav(src_path, wav_path)
        except Exception as e:
            raise HTTPException(400, f"ffmpeg failed to decode input: {e}")
        print("Converted input to 16k WAV")
        selected_model = MODELS[model]
        selected_model.inference(wav_path, output)
        print("Result is saved at: ", os.path.abspath(output))
        return {"output_path": os.path.abspath(output)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8041, reload=True)
