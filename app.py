from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import subprocess
import time
import os
import traceback
import tensorflow as tf
import imageio_ffmpeg

app = FastAPI(title="Infant Cry Classification API")

# Enable CORS for Hostinger frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust path resolution for Render environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "infant_cry_pathology_int8.tflite")

# Bundled ffmpeg binary (installed via pip, no apt needed on Render)
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite Model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")


def decode_audio(contents: bytes, sr: int = 16000) -> np.ndarray:
    """Decode any browser audio format (webm/opus, ogg, mp4, wav) to mono float32 @ sr."""
    proc = subprocess.run(
        [
            FFMPEG_EXE, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sr),
            "pipe:1",
        ],
        input=contents,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Decoded audio is empty.")
    return y


@app.get("/")
def home():
    return {"status": "Online", "message": "Infant Cry Classification Backend is running."}


@app.post("/predict")
async def predict_cry(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        if interpreter is None:
            raise RuntimeError("Model is not loaded.")

        contents = await file.read()
        sr = 16000
        y = decode_audio(contents, sr)

        # Extract Log-Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Padding / Truncation
        expected_frames = input_details[0]['shape'][2]

        if log_mel_spec.shape[1] < expected_frames:
            pad_width = expected_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :expected_frames]

        # [Batch, Height, Width, Channels]
        input_data = np.expand_dims(log_mel_spec, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)

        # INT8 input quantization
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            if input_scale > 0:
                input_data = np.clip(
                    np.round(input_data / input_scale + input_zero_point), -128, 127
                ).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        # INT8 output dequantization
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        logits = output_data[0].astype(np.float32).flatten()

        if logits.shape[0] > 1:
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            prediction_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            p = float(1.0 / (1.0 + np.exp(-logits[0])))
            prediction_idx = 1 if p >= 0.5 else 0
            confidence = p if prediction_idx == 1 else 1.0 - p

        labels = ["Pathological", "Physiological"]
        predicted_label = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"

        latency_ms = (time.time() - start_time) * 1000

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
