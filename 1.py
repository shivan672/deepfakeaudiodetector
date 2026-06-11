import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

real_audio, sr = librosa.load("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/real/file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav", sr=None)
 
fake_audio, sr = librosa.load("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/fake/file1000.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav", sr=None)

print(f"Sample Rate: {sr}")
print(f"Duration: {len(real_audio)/sr:.2f} seconds")
print(f"Total Samples: {len(real_audio)}")
print(f"Duration: {len(fake_audio)/sr:.2f} seconds")
print(f"Total Samples: {len(fake_audio)}")

import os

real_train = os.listdir("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/real/")
fake_train = os.listdir("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/fake/")

print(f"Real files: {len(real_train)}")
print(f"Fake files: {len(fake_train)}")



fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].set_title("Real Audio - Waveform")
librosa.display.waveshow(real_audio, sr=sr, ax=axes[0, 0], color='green')

axes[0, 1].set_title("Fake Audio - Waveform")
librosa.display.waveshow(fake_audio, sr=sr, ax=axes[0, 1], color='red')

real_mel = librosa.feature.melspectrogram(y=real_audio, sr=sr)
fake_mel = librosa.feature.melspectrogram(y=fake_audio, sr=sr)

real_mel_db = librosa.power_to_db(real_mel, ref=np.max)
fake_mel_db = librosa.power_to_db(fake_mel, ref=np.max)

img1 = librosa.display.specshow(real_mel_db, sr=sr, ax=axes[1,0])
axes[1, 0].set_title("Real Audio - Mel Spectrogram")

img2 = librosa.display.specshow(fake_mel_db, sr=sr, ax=axes[1,1])
axes[1, 1].set_title("Fake Audio - Mel Spectrogram")

plt.tight_layout()
plt.savefig("audio_visualization.png")
plt.show()


import librosa
import numpy as np

import librosa
import os

durations = []
folder = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/real/"

for f in os.listdir(folder)[:50]:  # check first 50 files
    audio, sr = librosa.load(folder + f, sr=None)
    durations.append(len(audio)/sr)

print(f"Min: {min(durations):.2f}s")
print(f"Max: {max(durations):.2f}s")
print(f"Average: {sum(durations)/len(durations):.2f}s")

TARGET_SR = 16000     
TARGET_DURATION = 1.5  
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  

def load_and_standardize(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    elif len(audio) < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    
    return audio

load_and_standardize("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/training/fake/file1000.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav")


import pandas as pd
from pathlib import Path

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
df = build_dataset("C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/")
print(df.head())
print(df['label_name'].value_counts())
print(df['split'].value_counts())
from sklearn.utils import resample

train_df = df[df['split'] == 'training']

real_df = train_df[train_df['label'] == 0]
fake_df = train_df[train_df['label'] == 1]

if len(real_df) > len(fake_df):
    fake_df = resample(fake_df, 
                        replace=True,
                        n_samples=len(real_df),
                        random_state=42)
else:
    real_df = resample(real_df,
                       replace=True, 
                       n_samples=len(fake_df),
                       random_state=42)

train_balanced = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42)
print(train_balanced['label_name'].value_counts())

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
from tqdm import tqdm         


audio, sr = librosa.load("file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav", sr=16000)

mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
print(f"MFCC shape: {mfccs.shape}")
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
print(f"Mel shape: {mel_db.shape}")

