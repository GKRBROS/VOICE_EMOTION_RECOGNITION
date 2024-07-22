import librosa
import numpy as np
from keras.models import load_model

model_path = 'my_model.h5'  
model = load_model(model_path)

def extract_features(file_path):
    # Load audio file
    data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)

    # Feature extraction
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    # Stack features horizontally
    features = np.hstack((zcr, chroma_stft, mfcc, rms, mel))

    return features

    

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    # Convert prediction to readable emotion
    emotion = np.argmax(prediction)
    return emotion

# You might need to adjust the function names and logic according to how your model expects inputs and provides outputs
