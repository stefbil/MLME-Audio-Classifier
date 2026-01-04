# MLME Audio Classifier

This project is part of the Machine Learning for Media Experiences (MLME) course of the Medialogy MSc program at Aalborg University in Copenhagen, Denmark.

[Colab Notebook](https://colab.research.google.com/drive/1BqLk5yibEToMCvh4GHibGGQPoz7P2tkX?usp=sharing)

## 🎵 Project Overview

A Machine Learning project for classifying audio  into **Music**, **Speech**, and **Noise**. This project uses a Convolutional Neural Network (CNN) trained on Mel Spectrograms from the [MUSAN dataset](https://www.openslr.org/17/).

This classifier captures audio (either from files or real-time microphone input), converts it into a Mel Spectrogram, and uses a trained TensorFlow/Keras model to predict the category.

- **Classes**: Music, Speech, Noise
- **Input**: 3-second audio clips (resampled to 16kHz)
- **Features**: Log Mel Spectrograms (128 Mel bands)
- **Model**: Custom CNN with 3 convolutional blocks
- **Frameworks**: TensorFlow, Keras, Librosa

## 📂 Project Structure

- `MLME_Mini-Project.ipynb`: The main Jupyter Notebook. It handles:
    - Downloading and extracting the MUSAN dataset.
    - Preprocessing and Data Augmentation (Time shift, Gain, SpecAugment).
    - Training the CNN model.
    - Evaluating performance.
    - Converting the model to TFLite format (`model.tflite`).
- `inference.py`: A Python script for real-time inference using your microphone and the trained TFLite model.
- `model.tflite`: The trained and optimized TFLite model.
- `requirements.txt`

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/stefbil/MLME-Audio-Classifier.git
   cd MLME-Audio-Classifier
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment. You can install the necessary packages using `pip`.
   
   **Core requirements:**
   ```bash
   pip install tensorflow numpy librosa matplotlib seaborn scikit-learn tqdm requests sounddevice
   ```

   *(Optional) If you want to install exactly the same environment as used in development, use:*
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

### 1. Training the Model
Open `MLME_Mini-Project.ipynb` in Jupyter Notebook or Google Colab.
Run the cells sequentially to:
1.  Download the dataset (approx. 15-20 mins).
2.  Process the audio files.
3.  Train the model.
4.  Generate `model.tflite`.

### 2. Real-Time Inference
Once you have `model.tflite` in the project directory, you can run the real-time classifier:

```bash
python inference.py
```

- The script will start capturing audio from your default microphone.
- It predicts the class every ~0.5 seconds based on the last 3 seconds of audio.
- Press `Ctrl+C` to stop.

## 🧠 Model Architecture

The model is a Sequential CNN constructed with Keras:
- **Input Content**: (128 Mel bands, Time steps, 1 Channel)
- **Layers**:
    - 3x Convolutional Blocks (Conv2D + MaxPooling2D + ReLU) to extract spectral features.
    - GlobalAveragePooling2D to reduce dimensionality.
    - Dense layers for classification.
    - Output: Softmax activation (3 classes).

## 📊 Dataset
The project automatically downloads the **MUSAN** dataset (A Music, Speech, and Noise Corpus) if it's not present in the `./data` folder.
