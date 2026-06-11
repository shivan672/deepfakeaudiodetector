import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

BASE_PATH    = "C:/Users/tempadmin/Downloads/archive/for-norm/for-norm/"
TARGET_SR    = 16000
TARGET_DURATION = 1.5
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)  
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
print(f"Total files: {len(df)}")
print(df['split'].value_counts())

def load_and_standardize(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    return audio

def extract_features(file_path):
    try:
        audio = load_and_standardize(file_path)

        mfccs       = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=40)
        delta_mfcc = librosa.feature.delta(mfccs)

        delta2_mfcc = librosa.feature.delta(mfccs,order=2)

        delta_mean = np.mean(delta_mfcc, axis=1)
        delta_std = np.std(delta_mfcc, axis=1)
        delta2_mean = np.mean(delta2_mfcc, axis=1)
        delta2_std = np.std(delta2_mfcc, axis=1)
        mfccs_mean  = np.mean(mfccs, axis=1)
        mfccs_std   = np.std(mfccs, axis=1)

        mel         = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=128)
        mel_db      = librosa.power_to_db(mel, ref=np.max)
        mel_mean    = np.mean(mel_db, axis=1)
        mel_std     = np.std(mel_db, axis=1)

        chroma      = librosa.feature.chroma_stft(y=audio, sr=TARGET_SR)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma, axis=1)

        zcr         = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean    = np.mean(zcr)
        zcr_std     = np.std(zcr)

        rms         = librosa.feature.rms(y=audio)
        centroid = librosa.feature.spectral_centroid(y=audio,sr=TARGET_SR)

        bandwidth = librosa.feature.spectral_bandwidth(y=audio,sr=TARGET_SR)

        rolloff = librosa.feature.spectral_rolloff(y=audio,sr=TARGET_SR)

        contrast = librosa.feature.spectral_contrast(y=audio,sr=TARGET_SR)

        tonnetz = librosa.feature.tonnetz(y=audio,sr=TARGET_SR)
        rms_mean    = np.mean(rms)
        rms_std     = np.std(rms)

        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)

        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)

        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)

        #feature_vector = np.concatenate([
            #mfccs_mean, mfccs_std,
            #mel_mean,   mel_std,
            #chroma_mean, chroma_std,
            #[zcr_mean, zcr_std],
            #[rms_mean, rms_std]
        #])
        feature_vector = np.concatenate([

            mfccs_mean,
            mfccs_std,

            delta_mean,
            delta_std,

            delta2_mean,
            delta2_std,

            mel_mean,
            mel_std,

            chroma_mean,
            chroma_std,

            contrast_mean,
            contrast_std,

            tonnetz_mean,
            tonnetz_std,

            [zcr_mean, zcr_std],

            [rms_mean, rms_std],

            [centroid_mean, centroid_std],

            [bandwidth_mean, bandwidth_std],

            [rolloff_mean, rolloff_std]
        ])
        return feature_vector

    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None
test_file = df[df['label_name'] == 'real']['file_path'].iloc[0]
result = extract_features(test_file)

if result is not None:
    print(f"\n Single file test passed!")
    print(f"Feature vector shape: {result.shape}")
else:
    print(" Feature extraction failed — fix before continuing")
    exit()

print("\n Extracting features from all files (10-20 mins)...")

features_list, labels_list, paths_list = [], [], []
failed = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
    feat = extract_features(row['file_path'])
    if feat is not None:
        features_list.append(feat)
        labels_list.append(row['label'])
        paths_list.append(row['file_path'])
    else:
        failed += 1

print(f"\n Extracted: {len(features_list)}")
print(f" Failed:    {failed}")

X = np.array(features_list)
y = np.array(labels_list)

with open("features.pkl", "wb") as f:
    pickle.dump({'X': X, 'y': y, 
                 'paths': paths_list, 
                 'splits': df['split'].tolist()}, f)

print(f" Saved features.pkl")

split_col = df['split'].values

X_train = X[split_col == 'training']
y_train = y[split_col == 'training']

X_val   = X[split_col == 'validation']
y_val   = y[split_col == 'validation']

X_test  = X[split_col == 'testing']
y_test  = y[split_col == 'testing']

print(f"\nSplits:")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Normalized + scaler saved!")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_val_scaled:   {X_val_scaled.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
