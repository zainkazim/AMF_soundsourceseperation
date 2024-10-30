# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import soundfile as sf

# # Define path to your audio file
# audio_path = '/home/ZA/Music/Media project expose files/AMF_MP/audiofiles/new/07003055.wav'

# # Load audio data using soundfile for better compatibility
# audio, sample_rate = sf.read(audio_path)

# # Ensure audio is properly formatted: If stereo, convert to mono
# if audio.ndim > 1:
#     audio = audio.mean(axis=1)

# # Ensure audio is a 1D array for further processing
# if audio.ndim != 1:
#     raise ValueError("Loaded audio data is not in a valid format. Expected 1D array.")

# # Print sample rate and audio shape
# print(f"Sample Rate: {sample_rate}, Audio Shape: {audio.shape}")

# # Set STFT parameters
# nperseg = 512
# noverlap = nperseg // 2  # Half overlap
# nfft = 1024

# # Compute STFT
# f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

# # Convert to dB scale
# magnitude_spectrum = 20 * np.log10(np.abs(Zxx) + 1e-10)

# # Plot the spectrogram
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', vmin=-80, vmax=0, cmap='viridis')
# plt.colorbar(label='Magnitude (dB)')
# plt.title('Spectrogram')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import soundfile as sf

# Define path to your audio file
audio_path = '/home/ZA/Music/Media project expose files/AMF_MP/07002136.wav'

# Load audio data using soundfile for better compatibility
audio, sample_rate = sf.read(audio_path)

# Ensure audio is properly formatted: If stereo, convert to mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Ensure audio is a 1D array for further processing
if audio.ndim != 1:
    raise ValueError("Loaded audio data is not in a valid format. Expected 1D array.")

# Print sample rate and audio shape
print(f"Sample Rate: {sample_rate}, Audio Shape: {audio.shape}")

# Set STFT parameters
nperseg = 512
noverlap = nperseg // 2  # Half overlap
nfft = 1024

# Compute STFT
f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

# Convert to dB scale
magnitude_spectrum = 20 * np.log10(np.abs(Zxx) + 1e-10)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', vmin=-80, vmax=0, cmap='viridis')
plt.colorbar(label='Magnitude (dB)')
plt.title('Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# Define parameters for FastMNMF
num_components = 2  # Number of components to separate
max_iterations = 500  # Maximum number of iterations
convergence_threshold = 1e-5  # Convergence criteria

# Print FastMNMF parameters
print(f"FastMNMF Parameters:\nNumber of Components: {num_components}\nMax Iterations: {max_iterations}\nConvergence Threshold: {convergence_threshold}")
