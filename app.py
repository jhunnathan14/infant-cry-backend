from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import subprocess
import json
import time
import os
import traceback
import tensorflow as tf
import imageio_ffmpeg

app = FastAPI(title="Infant Cry Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If labels.json exists (produced by the multi-class training script), use its
# model + class names. Otherwise fall back to the original binary model, where
# training used 0 = Physiological, 1 = Pathological.
CONFIG_PATH = os.path.join(BASE_DIR, "labels.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        _cfg = json.load(f)
    MODEL_FILE = _cfg["model"]
    LABELS = _cfg["labels"]
else:
    MODEL_FILE = "infant_cry_pathology_int8.tflite"
    LABELS = ["Physiological", "Pathological"]

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

# Must match training exactly
SR = 16000
DURATION = 3.0
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
TARGET_LEN = int(SR * DURATION)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite model loaded: {MODEL_FILE} | labels: {LABELS}")
except Exception as e:
    print(f"Error loading TFLite model: {e}")


def decode_audio(contents: bytes) -> np.ndarray:
    """Decode any browser audio format (webm/opus, ogg, mp4, wav) to mono float32 @ 16 kHz."""
    proc = subprocess.run(
        [FFMPEG_EXE, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        input=contents, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Decoded audio is empty.")
    return y


def preprocess(y: np.ndarray) -> np.ndarray:
    """Identical to training: first 3 s, pad, 64-mel log spectrogram, min-max to 0-1."""
    y = y[:TARGET_LEN]
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode="constant")
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    db = librosa.power_to_db(mel, ref=np.max)
    norm = (db - db.min()) / (db.max() - db.min() + 1e-6)
    return np.expand_dims(norm, axis=(0, -1)).astype(np.float32)  # (1, 64, T, 1)


@app.get("/")
def home():
    return {"status": "Online", "model": MODEL_FILE, "labels": LABELS}


@app.post("/predict")
async def predict_cry(file: UploadFile = File(...)):
    start = time.perf_counter()
    try:
        if interpreter is None:
            raise RuntimeError("Model is not loaded.")

        contents = await file.read()
        input_data = preprocess(decode_audio(contents))

        in_info = input_details[0]
        if in_info["dtype"] == np.int8:
            scale, zp = in_info["quantization"]
            input_data = np.clip(np.round(input_data / scale + zp), -128, 127).astype(np.int8)

        interpreter.set_tensor(in_info["index"], input_data)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details[0]["index"])

        out_info = output_details[0]
        out = raw.astype(np.float32)
        if out_info["dtype"] == np.int8:
            scale, zp = out_info["quantization"]
            out = (out - zp) * scale
        out = out.flatten()

        # Outputs are already probabilities (sigmoid / softmax baked into the model)
        if out.shape[0] == 1:                       # binary sigmoid: P(class 1)
            p1 = float(np.clip(out[0], 0.0, 1.0))
            probs = np.array([1.0 - p1, p1], dtype=np.float32)
        else:
            probs = out
            if np.any(probs < 0) or abs(float(probs.sum()) - 1.0) > 0.1:
                e = np.exp(probs - probs.max())
                probs = e / e.sum()
            else:
                probs = probs / probs.sum()

        idx = int(np.argmax(probs))
        label = LABELS[idx] if idx < len(LABELS) else "Unknown"

        return {
            "prediction": label,
            "confidence": float(probs[idx]),
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(min(len(LABELS), len(probs)))},
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
