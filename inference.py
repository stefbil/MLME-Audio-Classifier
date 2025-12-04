import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time

MODEL_PATH = 'model.tflite'
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
# Labels match training order.
# label_map = le.inverse_transform([0, 1, 2])
LABELS = ['Music', 'Noise', 'Speech'] 

# Calculation
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
BUFFER_SIZE = SAMPLES_PER_TRACK

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

print(f"Model Loaded. Expecting input shape: {input_details[0]['shape']}")

# Initialize with zeros
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def preprocess_audio(y):
    # Ensure audio is the correct length
    # Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = np.maximum(mel, 1e-10)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize (-80dB to 0dB, 0.0 to 1.0)
    spec = (log_mel + 80) / 80.0
    spec = np.clip(spec, 0, 1)
    
    # Batch and Channel dimensions (1, 128, 94, 1)
    spec = spec[..., np.newaxis]
    spec = np.expand_dims(spec, axis=0)
    return spec.astype(np.float32)

def audio_callback(indata, frames, time, status):

    global audio_buffer
    if status:
        print(status)
    
    # Flatten input data
    new_data = indata.flatten()
    
    # Roll buffer and append new data
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data

# run every 0.5 seconds
BLOCK_SIZE = int(SAMPLE_RATE * 0.5) 

print("Starting Stream... Press Ctrl+C to stop.")

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            # Create copy of buffer to avoid threading issues
            current_audio = audio_buffer.copy()
            
            # Preprocess
            input_tensor = preprocess_audio(current_audio)

            # Inference
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)[0]

            # Result
            prediction = np.argmax(output_data)
            confidence = output_data[prediction]
            
            # Visual Output
            label = LABELS[prediction]
            print(f"Pred: {label: <10} | Conf: {confidence:.2f} | {output_data}")
            
            # Reduce CPU usage slightly
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")