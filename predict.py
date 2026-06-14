


import sys
import numpy as np
import librosa
import onnxruntime as ort
import json
import os

TARGET_SR       = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS          = 64
HOP_LENGTH      = 256

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.onnx")
THRESH_PATH = os.path.join(BASE_DIR, "threshold.json")

def load_and_standardize(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
        return None
    if len(audio) > TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) 
        audio = audio[start:start + TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        repeats = TARGET_SAMPLES // len(audio) + 1
        audio = np.tile(audio, repeats)[:TARGET_SAMPLES]
    return audio

def extract_features(audio, sr):
    audio = load_and_standardize(audio, sr)
    if audio is None:
        return None

    def norm(x):
        std = x.std()
        if std < 1e-6:
            return x - x.mean()
        return (x - x.mean()) / std

    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=512,
        fmin=20, fmax=8000
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = norm(mel_db)

    mfcc       = librosa.feature.mfcc(
        y=audio, sr=TARGET_SR, n_mfcc=40, hop_length=HOP_LENGTH
    )
    mfcc_delta = librosa.feature.delta(mfcc)

    T            = mel_norm.shape[1]
    pad          = N_MELS - 40
    mfcc_padded  = np.pad(norm(mfcc[:, :T]),       ((0,pad),(0,0)))
    delta_padded = np.pad(norm(mfcc_delta[:, :T]), ((0,pad),(0,0)))

    features = np.stack(
        [mel_norm, mfcc_padded, delta_padded], axis=-1
    ).astype(np.float32)
    return features[np.newaxis, ...]

def predict(file_path):
    session = ort.InferenceSession(MODEL_PATH)
    with open(THRESH_PATH) as f:
        threshold = json.load(f)["threshold"]

    audio, sr = librosa.load(file_path, sr=None, mono=True)
    features  = extract_features(audio, sr)

    if features is None:
        print(" Could not process this audio file (empty/silent).")
        return

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prob = float(session.run([output_name], {input_name: features})[0][0][0])

    is_fake    = prob > threshold
    confidence = prob if is_fake else (1 - prob)
    label      = "DEEPFAKE (AI-Generated)" if is_fake else "GENUINE (Human)"

    print(f"\nFile: {file_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Real probability: {(1-prob)*100:.2f}%")
    print(f"Fake probability: {prob*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        sys.exit(1)

    predict(sys.argv[1])
print("Script started...")
print(f"Looking for model at: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")