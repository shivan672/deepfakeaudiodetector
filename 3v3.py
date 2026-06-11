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
import warnings
warnings.filterwarnings('ignore')

BASE_PATH       = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/"
TARGET_SR       = 16000
TARGET_DURATION = 1.5
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS          = 128
HOP_LENGTH      = 256

def load_and_standardize(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    return audio

def extract_mel(file_path):
    try:
        audio = load_and_standardize(file_path)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db
    except:
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

print("Class distribution per split:")
print(df.groupby(['split','label_name']).size())

def balance_split(df, split_name):
    split_df = df[df['split'] == split_name]
    real = split_df[split_df['label'] == 0]
    fake = split_df[split_df['label'] == 1]
    
    min_count = min(len(real), len(fake))
    print(f"\n{split_name}: real={len(real)}, "
          f"fake={len(fake)}, using={min_count} each")
    
    real_balanced = resample(real, n_samples=min_count, 
                             random_state=42, replace=False)
    fake_balanced = resample(fake, n_samples=min_count, 
                             random_state=42, replace=False)
    
    return pd.concat([real_balanced, fake_balanced]).sample(
        frac=1, random_state=42
    )

train_df = balance_split(df, 'training')
val_df   = balance_split(df, 'validation')
test_df  = balance_split(df, 'testing')

def extract_split(split_df, split_name):
    X_list, y_list = [], []
    failed = 0
    
    print(f"\n Extracting {split_name}...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        mel = extract_mel(row['file_path'])
        if mel is not None:
            X_list.append(mel)
            y_list.append(row['label'])
        else:
            failed += 1
    
    print(f"  Done. Failed: {failed}")
    return np.array(X_list), np.array(y_list)

X_train, y_train = extract_split(train_df, 'training')
X_val,   y_val   = extract_split(val_df,   'validation')
X_test,  y_test  = extract_split(test_df,  'testing')

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

print(f"\nShapes:")
print(f"Train: {X_train.shape}, "
      f"Real: {(y_train==0).sum()}, Fake: {(y_train==1).sum()}")
print(f"Val:   {X_val.shape},   "
      f"Real: {(y_val==0).sum()},  Fake: {(y_val==1).sum()}")
print(f"Test:  {X_test.shape},  "
      f"Real: {(y_test==0).sum()},  Fake: {(y_test==1).sum()}")

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_cnn(X_train.shape[1:])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5,
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, verbose=1),
    callbacks.ModelCheckpoint('best_cnn_model.h5',
                              monitor='val_accuracy',
                              save_best_only=True, verbose=1)
]

print("\n Training CNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=cb
)

def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return ((fpr[idx] + fnr[idx]) / 2) * 100

y_pred_proba = model.predict(X_test).flatten()
y_pred       = (y_pred_proba > 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred) * 100
test_f1  = f1_score(y_test, y_pred) * 100
test_eer = compute_eer(y_test, y_pred_proba)
cm       = confusion_matrix(y_test, y_pred)
real_acc = cm[0,0] / cm[0].sum() * 100
fake_acc = cm[1,1] / cm[1].sum() * 100

print("\n" + "="*45)
print("  CNN RESULTS")
print("="*45)
print(f"  Accuracy:      {test_acc:.2f}%  (target ≥ 80%)")
print(f"  F1 Score:      {test_f1:.2f}%  (target ≥ 80%)")
print(f"  EER:           {test_eer:.2f}%  (target ≤ 12%)")
print(f"  Real Accuracy: {real_acc:.2f}%  (target ≥ 75%)")
print(f"  Fake Accuracy: {fake_acc:.2f}%  (target ≥ 75%)")
print(f"\n TARGETS MET:")
print(f"  Accuracy ≥ 80%: {'✅' if test_acc >= 80 else '❌'}")
print(f"  EER ≤ 12%:      {'✅' if test_eer <= 12 else '❌'}")
print(f"  F1 ≥ 80%:       {'✅' if test_f1  >= 80 else '❌'}")
print(f"  Per-class ≥75%: {'✅' if real_acc>=75 and fake_acc>=75 else '❌'}")

# ─── STEP 9: PLOTS ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
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
plt.savefig("cnn_results.png")
plt.show()
