# Female Emotion Detection from Audio

This project provides a complete pipeline for gender and emotion detection from audio files using machine learning and deep learning models.

## Features
- Gender classification using SVM on extracted audio features
- Emotion classification for female voices using a Keras deep learning model
- GUI for easy audio file selection and prediction

## Project Structure
- `audio_gender_emotion_gui.py` — GUI application for gender and emotion prediction
- `gender_classification.ipynb` — Notebook for training the gender SVM model
- `Emotionaudio.ipynb` — Notebook for training the emotion detection model
- `svm_gender_model.pkl` — Trained SVM model for gender classification
- `gender_label_encoder.pkl` — Label encoder for gender classes
- `emotion_detection_model.keras` — Trained Keras model for emotion detection
- `label_encoder.pkl` — Label encoder for emotion classes
- `voice.csv` — CSV file with extracted features for gender classification
- `requirements.txt` — List of required Python packages

## Datasets Used
- **Gender Classification:**
  - Features extracted from a primary object voice dataset from Kaggle (link: [add your Kaggle dataset link here])
- **Emotion Classification:**
  - [Toronto emotional speech set (TESS)](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

## How to Use
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Download model and encoder files:**
   - Download the `.keras` and `.pkl` files from the provided Google Drive link:
     - [Download Models and Encoders](YOUR_GOOGLE_DRIVE_LINK_HERE)
   - Place them in the project folder.
3. **Run the GUI:**
   ```
   python audio_gender_emotion_gui.py
   ```
4. **Select a WAV audio file:**
   - The GUI will extract features and predict gender. If the voice is female, it will also predict emotion.

## Prediction Process
- When you select an audio file in the GUI:
  1. The app extracts 20 features from the audio and uses the SVM model to predict gender.
  2. If the voice is detected as male, you are prompted to provide a female voice.
  3. If the voice is detected as female, MFCC features are extracted and passed to the emotion detection model to predict the emotion.

## Notes
- The gender model expects 20 extracted features in the same order as in `voice.csv`.
- The emotion model expects MFCC features as input.
- If you want to retrain the models, use the provided notebooks.

## Requirements
See `requirements.txt` for all dependencies.

## Credits
- Gender dataset: [https://www.kaggle.com/datasets/primaryobjects/voicegender]
- Emotion dataset: [Toronto emotional speech set (TESS)](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

---




