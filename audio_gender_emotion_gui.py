import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import joblib
from keras.models import load_model
import os

# Feature extraction for gender (20 features, order must match training)
def extract_gender_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    features = {}
    features['meanfreq'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['sd'] = np.std(y)
    features['median'] = np.median(y)
    features['Q25'] = np.percentile(y, 25)
    features['Q75'] = np.percentile(y, 75)
    features['IQR'] = features['Q75'] - features['Q25']
    features['skew'] = np.mean((y - np.mean(y))**3) / (np.std(y)**3 + 1e-6)
    features['kurt'] = np.mean((y - np.mean(y))**4) / (np.std(y)**4 + 1e-6)
    features['sp.ent'] = np.sum(-np.abs(y) * np.log2(np.abs(y) + 1e-6))
    features['sfm'] = np.mean(librosa.feature.spectral_flatness(y=y))
    y_int = ((y - y.min()) * 100).astype(int)
    if len(y_int) > 0 and np.min(y_int) >= 0:
        features['mode'] = np.argmax(np.bincount(y_int)) / 100
    else:
        features['mode'] = 0.0
    features['centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    # The following are best-effort approximations for meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx
    zc = librosa.zero_crossings(y, pad=False)
    features['meanfun'] = np.mean(zc)
    features['minfun'] = np.min(zc)
    features['maxfun'] = np.max(zc)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['meandom'] = np.mean(sb)
    features['mindom'] = np.min(sb)
    features['maxdom'] = np.max(sb)
    features['dfrange'] = np.ptp(y)
    features['modindx'] = np.mean(np.abs(np.diff(y)))
    feature_order = ['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx']
    return np.array([[features[f] for f in feature_order]])

# For emotion, use MFCCs as in your emotion model
def extract_emotion_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfcc, axis=0)

# Load models and encoders
gender_model = joblib.load('svm_gender_model.pkl')
gender_encoder = joblib.load('gender_label_encoder.pkl')
emotion_model = load_model('emotion_detection_model.keras')
import pickle
with open('label_encoder.pkl', 'rb') as f:
    emotion_encoder = pickle.load(f)

# GUI
def predict_audio():
    file_path = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')])
    if not file_path:
        return
    try:
        features = extract_gender_features(file_path)
        gender_pred = gender_model.predict(features)[0]
        gender_label = gender_encoder.inverse_transform([gender_pred])[0]
        if gender_label == 'male':
            messagebox.showinfo('Result', 'Please give female voice')
            return
        # If female, extract MFCCs for emotion detection
        mfcc = extract_emotion_features(file_path)
        emotion_pred = emotion_model.predict(mfcc)
        emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
        messagebox.showinfo('Result', f'Gender: Female\nEmotion: {emotion_label}')
    except Exception as e:
        messagebox.showerror('Error', f'Error processing file: {e}')

def main():
    root = tk.Tk()
    root.title('Audio Gender & Emotion Classifier')
    root.geometry('400x200')
    tk.Label(root, text='Upload a WAV file for Gender & Emotion Detection', font=('Arial', 12)).pack(pady=20)
    tk.Button(root, text='Select Audio File', command=predict_audio, font=('Arial', 12)).pack(pady=20)
    root.mainloop()

if __name__ == '__main__':
    main()
