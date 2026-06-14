# Deepfake Audio Detection

A machine learning system for classifying speech recordings as Genuine (Human) or Deepfake (AI-Generated). Built as part of a problem statement on detecting AI-generated voice content.

Live app: https://shivan672-deepfake-audio-detector.hf.space/

## Background

With the rise of generative AI, synthetic speech has become difficult to distinguish from real human voices. This creates risks around impersonation, fraud, and misinformation. This project trains a model to flag whether a given audio clip is real or AI-generated.

## Dataset

The Fake-or-Real Dataset (Kaggle) was used, specifically the for-norm directory with its training, validation, and testing splits. A total of 69,300 audio files were processed. Each split was balanced between real and fake samples to avoid the model learning a bias toward one class.

## Approach

Audio files are first resampled to 16kHz mono and standardized to 2-second clips - longer clips are center-cropped and shorter ones are tiled to fill the duration. Files that are silent or empty are discarded during preprocessing.

For each clip, three feature representations are computed and stacked together as a 3-channel input:

- Log Mel Spectrogram (64 mel bands)
- MFCCs (40 coefficients)
- MFCC delta (rate of change over time)

The final input shape passed to the model is 64 x 126 x 3.

The model itself is a Light CNN (LCNN), which uses Max Feature Map activations across four convolutional blocks, followed by batch normalization, dropout, and a sigmoid output layer.

During training, the fake class was weighted higher (2.5x) to correct for the model's initial tendency to over-predict "real". The classification threshold was also tuned on the validation set to maximize F1 score, landing at 0.57.

## Results

Evaluated on a held-out, balanced test set:
Accuracy=88.90%
F1 Score=87.53% 
Equal Error Rate=3.70%
Real class accuracy=99.90%
Fake class accuracy=77.90%

A full breakdown including the confusion matrix, ROC curve, DET curve, and prediction score distributions is available in evaluation_report.png.

##Demo Video
Link:https://drive.google.com/file/d/1CnWCybdgB7GZWpODiF_hV3XUI5hZa3WX/view?usp=sharing
