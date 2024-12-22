import numpy as np
import librosa
import soundfile as sf

# Load stereo audio file
audio_path = "/home/ZA/Music/Media project/AMF_MP/new wav files/M1F1-Alaw-AFsp.wav"
y, sr = sf.read(audio_path)

# Ensure it’s stereo
if y.ndim != 2 or y.shape[1] != 2:
    raise ValueError("The audio file must be stereo (2 channels).")

# Normalize the audio
y = y / np.max(np.abs(y))

# Print audio information
print(f"Sample Rate: {sr}")
print(f"Audio Shape: {y.shape}")  # (num_samples, 2 channels)
