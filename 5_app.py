import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import json
import tempfile
import os
import matplotlib.pyplot as plt
import librosa.display
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────
TARGET_SR       = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS          = 64
HOP_LENGTH      = 256

# ─── PAGE SETUP ───────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Deepfake Audio Detector")
st.markdown("Upload an audio file to detect if it is "
            "**Genuine (Human)** or **Deepfake (AI-Generated)**")
st.markdown("---")

# ─── REGISTER CUSTOM FUNCTION ─────────────────────────
@tf.keras.utils.register_keras_serializable()
def mfm(x):
    shape = tf.shape(x)
    x1 = x[:, :, :, :shape[3]//2]
    x2 = x[:, :, :, shape[3]//2:]
    return tf.maximum(x1, x2)

# ─── LOAD MODEL ───────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_lcnn_final.keras",
        custom_objects={"mfm": mfm}
    )
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold

model, threshold = load_model()

# ─── FEATURE EXTRACTION ───────────────────────────────
def load_and_standardize(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
        return None
    if len(audio) > TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
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

    mfcc       = librosa.feature.mfcc(y=audio, sr=TARGET_SR,
                                       n_mfcc=40, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)

    T            = mel_norm.shape[1]
    pad          = N_MELS - 40
    mfcc_padded  = np.pad(norm(mfcc[:, :T]),       ((0,pad),(0,0)))
    delta_padded = np.pad(norm(mfcc_delta[:, :T]), ((0,pad),(0,0)))

    features = np.stack([mel_norm, mfcc_padded, delta_padded],
                        axis=-1).astype(np.float32)
    return features[np.newaxis, ...]

# ─── VISUALIZE AUDIO ──────────────────────────────────
def plot_audio(audio, sr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    librosa.display.waveshow(audio, sr=sr,
                              ax=axes[0], color='steelblue')
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")

    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, sr=sr,
                                       hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='mel',
                                       ax=axes[1], cmap='magma')
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    return fig

# ─── MAIN UI ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac", "ogg"],
    help="Supported formats: WAV, MP3, FLAC, OGG"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    audio, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration  = len(audio) / sr

    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Rate", f"{sr} Hz")
    col2.metric("Duration",    f"{duration:.2f}s")
    col3.metric("File Size",
                f"{os.path.getsize(tmp_path)/1024:.1f} KB")

    st.markdown("### 📊 Audio Visualization")
    fig = plot_audio(audio, sr)
    st.pyplot(fig)
    plt.close()

    st.markdown("### 🔍 Analysis")
    with st.spinner("Analyzing audio..."):
        features = extract_features(audio, sr)

        if features is None:
            st.error("❌ Could not process this audio file. "
                     "Please try another.")
        else:
            prob       = model.predict(features, verbose=0)[0][0]
            is_fake    = prob > threshold
            confidence = prob if is_fake else (1 - prob)

            st.markdown("---")
            if is_fake:
                st.error("## 🤖 DEEPFAKE (AI-Generated)")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                st.progress(float(confidence))
                st.markdown(
                    "> ⚠️ This audio appears to be **artificially "
                    "generated**. It may have been created using a "
                    "text-to-speech or voice cloning system."
                )
            else:
                st.success("## ✅ GENUINE (Human)")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                st.progress(float(confidence))
                st.markdown(
                    "> 🎙️ This audio appears to be a **genuine human "
                    "voice** recording."
                )

            st.markdown("### 📈 Probability Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("🟢 Real Probability",
                        f"{(1-prob)*100:.1f}%")
            col2.metric("🔴 Fake Probability",
                        f"{prob*100:.1f}%")

            fig2, ax = plt.subplots(figsize=(8, 1.5))
            ax.barh([''], [(1-prob)*100],
                    color='green', alpha=0.7, label='Real')
            ax.barh([''], [prob*100], left=[(1-prob)*100],
                    color='red', alpha=0.7, label='Fake')
            ax.axvline(x=threshold*100, color='black',
                       linestyle='--', alpha=0.7,
                       label=f'Threshold ({threshold*100:.0f}%)')
            ax.set_xlim([0, 100])
            ax.set_xlabel('Probability (%)')
            ax.legend(loc='upper right')
            ax.set_title('Real vs Fake Probability')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    os.unlink(tmp_path)

# ─── FOOTER ───────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Model:** LCNN | "
    "**Features:** Mel Spectrogram + MFCC + Delta | "
    f"**Threshold:** {threshold:.2f} | "
    "**Accuracy:** 88.90%"
)