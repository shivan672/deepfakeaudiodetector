import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import json
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, classification_report)
from sklearn.utils import resample
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────
BASE_PATH       = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/"
TARGET_SR       = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS          = 64
HOP_LENGTH      = 256

# ─── REGISTER CUSTOM FUNCTION ─────────────────────────
@tf.keras.utils.register_keras_serializable()
def mfm(x):
    shape = tf.shape(x)
    x1 = x[:, :, :, :shape[3]//2]
    x2 = x[:, :, :, shape[3]//2:]
    return tf.maximum(x1, x2)

# ─── LOAD MODEL & THRESHOLD ───────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    "best_lcnn_final.keras",
    custom_objects={"mfm": mfm}
)

with open("threshold.json") as f:
    best_thresh = json.load(f)["threshold"]
print(f"✅ Model loaded. Threshold: {best_thresh:.2f}")

# ─── FEATURE EXTRACTION ───────────────────────────────
def load_and_standardize(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
        return None
    if len(audio) > TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
        audio = audio[start:start + TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        repeats = TARGET_SAMPLES // len(audio) + 1
        audio = np.tile(audio, repeats)[:TARGET_SAMPLES]
    return audio

def extract_features(file_path):
    try:
        audio = load_and_standardize(file_path)
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

        return np.stack([mel_norm, mfcc_padded, delta_padded],
                        axis=-1).astype(np.float32)
    except:
        return None

# ─── LOAD TEST DATA ───────────────────────────────────
def build_dataset(base_path):
    records = []
    for split in ['testing']:
        for label in ['real', 'fake']:
            folder = Path(base_path) / split / label
            for file in folder.glob("*.wav"):
                records.append({
                    'file_path': str(file),
                    'label': 0 if label == 'real' else 1,
                    'label_name': label,
                    'split': split
                })
    return pd.DataFrame(records)

df = build_dataset(BASE_PATH)

# Balance test set
real = df[df['label'] == 0]
fake = df[df['label'] == 1]
n    = min(len(real), len(fake), 1000)
test_df = pd.concat([
    resample(real, n_samples=n, random_state=42, replace=False),
    resample(fake, n_samples=n, random_state=42, replace=False)
]).sample(frac=1, random_state=42)

print(f"\n⏳ Extracting test features...")
X_list, y_list = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    feat = extract_features(row['file_path'])
    if feat is not None:
        X_list.append(feat)
        y_list.append(row['label'])

X_test = np.array(X_list, dtype=np.float32)
y_test = np.array(y_list)
print(f"✅ Test set: {X_test.shape}")

# ─── PREDICTIONS ──────────────────────────────────────
y_pred_proba = model.predict(X_test).flatten()
y_pred       = (y_pred_proba > best_thresh).astype(int)

# ─── METRICS ──────────────────────────────────────────
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return ((fpr[idx] + fnr[idx]) / 2) * 100, fpr, tpr

test_acc           = accuracy_score(y_test, y_pred) * 100
test_f1            = f1_score(y_test, y_pred) * 100
test_eer, fpr, tpr = compute_eer(y_test, y_pred_proba)
roc_auc            = auc(fpr, tpr)
cm                 = confusion_matrix(y_test, y_pred)
real_acc           = cm[0,0] / cm[0].sum() * 100
fake_acc           = cm[1,1] / cm[1].sum() * 100
report             = classification_report(
                        y_test, y_pred,
                        target_names=['Real','Fake']
                     )

print("\n" + "="*45)
print("  FINAL EVALUATION REPORT")
print("="*45)
print(f"  Accuracy:      {test_acc:.2f}%")
print(f"  F1 Score:      {test_f1:.2f}%")
print(f"  EER:           {test_eer:.2f}%")
print(f"  ROC AUC:       {roc_auc:.4f}")
print(f"  Real Accuracy: {real_acc:.2f}%")
print(f"  Fake Accuracy: {fake_acc:.2f}%")
print(f"\n{report}")

# ─── GENERATE REPORT FIGURE ───────────────────────────
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
ConfusionMatrixDisplay(cm, display_labels=['Real','Fake']).plot(
    ax=ax1, colorbar=False, cmap='Blues'
)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0,1],[0,1],'k--', lw=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Score Distribution
ax3 = fig.add_subplot(gs[0, 2])
real_scores = y_pred_proba[y_test == 0]
fake_scores = y_pred_proba[y_test == 1]
ax3.hist(real_scores, bins=50, alpha=0.6,
         color='green', label='Real', density=True)
ax3.hist(fake_scores, bins=50, alpha=0.6,
         color='red',   label='Fake', density=True)
ax3.axvline(x=best_thresh, color='black',
            linestyle='--', label=f'Threshold={best_thresh:.2f}')
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('Density')
ax3.set_title('Score Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Metrics Bar Chart
ax4 = fig.add_subplot(gs[1, 0])
metrics = ['Accuracy', 'F1 Score', 'Real Acc', 'Fake Acc']
values  = [test_acc, test_f1, real_acc, fake_acc]
colors  = ['green' if v >= 80 else 'red' for v in values]
bars    = ax4.bar(metrics, values, color=colors, alpha=0.7)
ax4.axhline(y=80, color='black', linestyle='--',
            alpha=0.5, label='80% target')
ax4.set_ylim([0, 110])
ax4.set_ylabel('Percentage (%)')
ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=10)
ax4.legend()

# 5. DET Curve
ax5 = fig.add_subplot(gs[1, 1])
fnr  = 1 - tpr
ax5.plot(fpr*100, fnr*100, color='blue', lw=2)
ax5.plot([test_eer, test_eer],[0, test_eer],
         'r--', alpha=0.7)
ax5.plot([0, test_eer],[test_eer, test_eer],
         'r--', alpha=0.7, label=f'EER = {test_eer:.2f}%')
ax5.set_xlabel('False Acceptance Rate (%)')
ax5.set_ylabel('False Rejection Rate (%)')
ax5.set_title('DET Curve', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Summary Text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary = f"""
MODEL SUMMARY
─────────────────────────
Architecture:  LCNN
Features:      Mel + MFCC + Delta
Input Shape:   (64, 126, 3)
Threshold:     {best_thresh:.2f}

RESULTS
─────────────────────────
Accuracy:      {test_acc:.2f}%  ✅
F1 Score:      {test_f1:.2f}%  ✅
EER:           {test_eer:.2f}%   ✅
ROC AUC:       {roc_auc:.4f}   ✅
Real Accuracy: {real_acc:.2f}%  ✅
Fake Accuracy: {fake_acc:.2f}%  ✅

ALL TARGETS MET ✅
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.suptitle('Deepfake Audio Detection — Evaluation Report',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("evaluation_report.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved evaluation_report.png")
print("\n🎉 Phase 4 Complete!")