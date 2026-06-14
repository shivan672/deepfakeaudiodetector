import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve)
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import json
import warnings
warnings.filterwarnings('ignore')

BASE_PATH       = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/"
TARGET_SR       = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS          = 64
HOP_LENGTH      = 256

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
            y=audio, sr=TARGET_SR,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=512,
            fmin=20, fmax=8000
        )
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_norm = norm(mel_db)

        mfcc = librosa.feature.mfcc(
            y=audio, sr=TARGET_SR,
            n_mfcc=40,
            hop_length=HOP_LENGTH
        )
        mfcc_delta = librosa.feature.delta(mfcc)

        T = mel_norm.shape[1]
        mfcc_r  = mfcc[:, :T]
        delta_r = mfcc_delta[:, :T]

        pad = N_MELS - 40
        mfcc_padded  = np.pad(norm(mfcc_r),  ((0, pad), (0, 0)))
        delta_padded = np.pad(norm(delta_r), ((0, pad), (0, 0)))

        combined = np.stack([mel_norm, mfcc_padded, delta_padded], axis=-1)
        return combined.astype(np.float32)

    except Exception as e:
        return None

def build_dataset(base_path):
    records = []
    for split in ['training', 'validation', 'testing']:
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
print(f"Total files: {len(df)}")

def balance_split(df, split_name, max_per_class=5000):
    split_df = df[df['split'] == split_name]
    real = split_df[split_df['label'] == 0]
    fake = split_df[split_df['label'] == 1]
    n = min(len(real), len(fake), max_per_class)
    print(f"{split_name}: using {n} per class ({n*2} total)")
    real_b = resample(real, n_samples=n, random_state=42, replace=False)
    fake_b = resample(fake, n_samples=n, random_state=42, replace=False)
    return pd.concat([real_b, fake_b]).sample(frac=1, random_state=42)

train_df = balance_split(df, 'training',   max_per_class=5000)
val_df   = balance_split(df, 'validation', max_per_class=1000)
test_df  = balance_split(df, 'testing',    max_per_class=1000)

def extract_split(split_df, name):
    X_list, y_list = [], []
    failed = 0
    print(f"\n Extracting {name}...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        feat = extract_features(row['file_path'])
        if feat is not None:
            X_list.append(feat)
            y_list.append(row['label'])
        else:
            failed += 1
    print(f"  Done. Failed: {failed}")
    return np.array(X_list, dtype=np.float32), np.array(y_list)


print("\nTesting single file...")
sample_file = df[df['label_name'] == 'real']['file_path'].iloc[0]
test_feat = extract_features(sample_file)
if test_feat is not None:
    print(f" Feature shape: {test_feat.shape}")
else:
    print(" Failed"); exit()

X_train, y_train = extract_split(train_df, 'training')
X_val,   y_val   = extract_split(val_df,   'validation')
X_test,  y_test  = extract_split(test_df,  'testing')

print(f"\nShapes:")
print(f"  X_train: {X_train.shape} — "
      f"Real:{(y_train==0).sum()} Fake:{(y_train==1).sum()}")
print(f"  X_val:   {X_val.shape}   — "
      f"Real:{(y_val==0).sum()}  Fake:{(y_val==1).sum()}")
print(f"  X_test:  {X_test.shape}  — "
      f"Real:{(y_test==0).sum()}  Fake:{(y_test==1).sum()}")

def mfm(x):
    shape = tf.shape(x)
    x1 = x[:, :, :, :shape[3]//2]
    x2 = x[:, :, :, shape[3]//2:]
    return tf.maximum(x1, x2)

def build_lcnn(input_shape):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (5,5), padding='same')(inp)
    x = layers.Lambda(mfm)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (1,1), padding='same')(x)
    x = layers.Lambda(mfm)(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.Lambda(mfm)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (1,1), padding='same')(x)
    x = layers.Lambda(mfm)(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.Lambda(mfm)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inp, out)

model = build_lcnn(X_train.shape[1:])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


class_weights = {
    0: 1.0,   
    1: 2.5    
}

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, verbose=1),
    callbacks.ModelCheckpoint('best_lcnn.keras',
                              monitor='val_accuracy',
                              save_best_only=True, verbose=1)
]

print("\n Training LCNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=cb
)
print("\n🔍 Finding best threshold on validation set...")
val_pred_proba = model.predict(X_val).flatten()
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh = 0.5
best_f1 = 0

for t in thresholds:
    preds = (val_pred_proba > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f" Best threshold: {best_thresh:.2f} "
      f"(val F1: {best_f1*100:.2f}%)")

def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return ((fpr[idx] + fnr[idx]) / 2) * 100

y_pred_proba = model.predict(X_test).flatten()
y_pred       = (y_pred_proba > best_thresh).astype(int)

test_acc = accuracy_score(y_test, y_pred) * 100
test_f1  = f1_score(y_test, y_pred) * 100
test_eer = compute_eer(y_test, y_pred_proba)
cm       = confusion_matrix(y_test, y_pred)
real_acc = cm[0,0] / cm[0].sum() * 100
fake_acc = cm[1,1] / cm[1].sum() * 100

print("\n" + "="*45)
print("  LCNN RESULTS")
print("="*45)
print(f"  Accuracy:      {test_acc:.2f}%  (target ≥ 80%)")
print(f"  F1 Score:      {test_f1:.2f}%  (target ≥ 80%)")
print(f"  EER:           {test_eer:.2f}%  (target ≤ 12%)")
print(f"  Real Accuracy: {real_acc:.2f}%  (target ≥ 75%)")
print(f"  Fake Accuracy: {fake_acc:.2f}%  (target ≥ 75%)")
print(f"\n TARGETS MET:")
print(f"  Accuracy ≥ 80%: {'YES' if test_acc >= 80 else 'NO'}")
print(f"  EER ≤ 12%:      {'YES' if test_eer <= 12 else 'NO'}")
print(f"  F1 ≥ 80%:       {'YES' if test_f1  >= 80 else 'NO'}")
print(f"  Per-class ≥75%: {'YES' if real_acc>=75 and fake_acc>=75 else 'NO'}")

model.save("best_lcnn_final.keras")
print("\n Model saved as best_lcnn_final.keras")

with open("threshold.json", "w") as f:
    json.dump({"threshold": float(best_thresh)}, f)
print(f" Threshold saved: {best_thresh:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss')
axes[1].legend()

ConfusionMatrixDisplay(cm, display_labels=['Real','Fake']).plot(
    ax=axes[2], colorbar=False, cmap='Blues'
)
axes[2].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig("lcnn_results.png")
plt.show()

