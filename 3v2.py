import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa
import os
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')

BASE_PATH      = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/"
TARGET_SR      = 16000
TARGET_DURATION = 1.5
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)
N_MELS         = 128
HOP_LENGTH     = 256

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
    except Exception as e:
        print(f"Error: {e}")
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
                    'split': split
                })
    import pandas as pd
    return pd.DataFrame(records)

import pandas as pd
df = build_dataset(BASE_PATH)
print(f"Total files: {len(df)}")

print("\n Extracting Mel Spectrograms (15-25 mins)...")

X_list, y_list, split_list = [], [], []
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    mel = extract_mel(row['file_path'])
    if mel is not None:
        X_list.append(mel)
        y_list.append(row['label'])
        split_list.append(row['split'])
    else:
        failed += 1

print(f" Done! Failed: {failed}")

X = np.array(X_list)
y = np.array(y_list)
splits = np.array(split_list)

X = X[..., np.newaxis]
print(f"X shape: {X.shape}")

X_train = X[splits == 'training']
y_train = y[splits == 'training']
X_val   = X[splits == 'validation']
y_val   = y[splits == 'validation']
X_test  = X[splits == 'testing']
y_test  = y[splits == 'testing']

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

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

input_shape = X_train.shape[1:]  # (128, 94, 1)
model = build_cnn(input_shape)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
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
    eer = (fpr[np.argmin(np.abs(fnr - fpr))] +
           fnr[np.argmin(np.abs(fnr - fpr))]) / 2
    return eer * 100

y_pred_proba = model.predict(X_test).flatten()
y_pred       = (y_pred_proba > 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred) * 100
test_f1  = f1_score(y_test, y_pred) * 100
test_eer = compute_eer(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
real_acc = cm[0,0] / cm[0].sum() * 100
fake_acc = cm[1,1] / cm[1].sum() * 100

print("\n" + "="*45)
print("  CNN RESULTS")
print("="*45)
print(f"  Accuracy:       {test_acc:.2f}%  (target ≥ 80%)")
print(f"  F1 Score:       {test_f1:.2f}%  (target ≥ 80%)")
print(f"  EER:            {test_eer:.2f}%  (target ≤ 12%)")
print(f"  Real Accuracy:  {real_acc:.2f}%  (target ≥ 75%)")
print(f"  Fake Accuracy:  {fake_acc:.2f}%  (target ≥ 75%)")
print(f"\n TARGETS MET:")
print(f"  Accuracy ≥ 80%: {'✅' if test_acc >= 80 else '❌'}")
print(f"  EER ≤ 12%:      {'✅' if test_eer <= 12 else '❌'}")
print(f"  F1 ≥ 80%:       {'✅' if test_f1  >= 80 else '❌'}")
print(f"  Per-class ≥75%: {'✅' if real_acc >= 75 and fake_acc >= 75 else '❌'}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training history
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].legend()

ConfusionMatrixDisplay(cm, display_labels=['Real','Fake']).plot(
    ax=axes[2], colorbar=False, cmap='Blues'
)
axes[2].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig("cnn_results.png")
plt.show()
print(" Saved cnn_results.png")
