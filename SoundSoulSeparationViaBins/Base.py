#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
EPS = 1e-10
MIC_INDEX = 1
Total_Index=0


def MultiSTFT(wav_TM: "np.ndarray", n_fft=1024, hop_length=None) -> np.ndarray:
    """
    Multichannel STFT

    Parameters
    ---------
    wav_TM: ndarray (T x M) or (T)
    n_fft: int
        The window size (default 1024)
    hop_length: int
        The shift length (default None)
        If None, n_fft // 4

    Returns
    -------
    spec_FTM: np.ndarray (F x T x M) or (F x T)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if wav_TM.ndim == 1:
        wav_TM = wav_TM[:, None]

    _, M = wav_TM.shape

    for m in range(M):
        spec = librosa.core.stft(wav_TM[:, m], n_fft=n_fft, hop_length=hop_length)
        if m == 0:
            spec_FTM = np.zeros([*spec.shape, M], dtype=spec.dtype)
        spec_FTM[..., m] = spec

    return spec_FTM.squeeze()


def MultiISTFT(spec, hop_length=None, shape="FTM"):
    """
    Multichannel inverse STFT

    Parameters
    ---------
    spec: np.ndarray (F x T x M) or (F x T)
    hop_length: int
        The shift length (default None)
        If None, (F-1) * 4
    shape: str
        Shape of the spec. FTM or MFT

    Returns
    -------
    wav_TM: np.ndarray ((T' x M) or T')
    """
    if spec.ndim == 2:
        spec = spec[..., None]
        shape = "FTM"

    if shape == "MFT":
        spec = spec.transpose(1, 2, 0)

    F, _, M = spec.shape

    if hop_length is None:
        n_fft = (F - 1) * 2
        hop_length = n_fft // 4

    for m in range(M):
        wav = librosa.core.istft(spec[..., m], hop_length=hop_length)
        if m == 0:
            wav_TM = np.zeros([len(wav), M], dtype=wav.dtype)
        wav_TM[:, m] = wav

    return wav_TM.squeeze()
def estimate_steering_vector(azimuths, frequency, mic_distance):
    """
    Estimate steering vectors using azimuths and frequency.

    Parameters:
    ----------
    azimuths : np.ndarray (T,)
        Azimuth angles.
    frequency : float
        Frequency (Hz).
    mic_distance : float
        Distance between microphones (meters).

    Returns:
    -------
    steering_vector : np.ndarray (M,)
        Steering vector for a specific frequency and azimuth.
    """
    speed_of_sound = 343.0
    return np.exp(-1j * 2 * np.pi * frequency * mic_distance * np.sin(azimuths) / speed_of_sound)
def calculate_time_delays(phase_differences, frequencies):
    """
    Calculate time delays from phase differences.

    Parameters:
    ----------
    phase_differences : np.ndarray (F x T)
        Phase differences between microphones.
    frequencies : np.ndarray (F,)
        Frequencies corresponding to FFT bins.

    Returns:
    -------
    time_delays : np.ndarray (F x T)
        Time delays between microphone pairs.
    """
    return phase_differences / (2 * np.pi * frequencies[:, None])

def calculate_phase_differences(wav_TM: np.ndarray, n_fft=1024, hop_length=None) -> np.ndarray:
    """
    Calculate phase differences between all pairs of channels from multichannel STFT.

    Parameters
    ----------
    wav_TM : np.ndarray (T x M)
        Multichannel audio signal, where T is the number of time samples and M is the number of channels.
    n_fft : int
        The FFT size for STFT (default: 1024).
    hop_length : int or None
        Hop length for STFT. If None, defaults to n_fft // 4.

    Returns
    -------
    phase_differences : np.ndarray (F x T x M x M)
        Phase differences between all pairs of channels for all frequency bins and time frames.
        phase_differences[f, t, i, j] = phase difference between channel i and j at frequency bin f and time frame t.
    """
    # Validate input dimensions
    if wav_TM.ndim == 1:
        raise ValueError("Input signal must have at least two dimensions (T x M). For mono, add a channel dimension.")
    if wav_TM.ndim != 2:
        raise ValueError(f"Expected input shape (T x M), got {wav_TM.shape}.")
    if wav_TM.shape[1] < 2:
        raise ValueError("Phase difference requires at least two channels.")

    # Perform multichannel STFT
   # spec_FTM = MultiSTFT(wav_TM, n_fft=n_fft, hop_length=hop_length)
    spec_FTM = MultiSTFT(wav_TM[:, :2], n_fft=1024, hop_length=256)
    magnitude_channel_1 = np.abs(spec_FTM[..., 0])
    magnitude_channel_2 = np.abs(spec_FTM[..., 1])
    phase_channel_1 = np.angle(spec_FTM[..., 0])
    phase_channel_2 = np.angle(spec_FTM[..., 1])
    difference = np.abs(magnitude_channel_1 - magnitude_channel_2)
    raw_phase_diff = phase_channel_1 - phase_channel_2
    _, _, M = spec_FTM.shape

    # Precompute phases for all channels
    phases_FTM = np.angle(spec_FTM)

    # Initialize array to store phase differences
    phase_differences = np.zeros((spec_FTM.shape[0], spec_FTM.shape[1], M, M))

    phase_channel_1 = np.angle(spec_FTM[..., 0])
    phase_channel_2 = np.angle(spec_FTM[..., 1])
    phase_differences = np.unwrap(phase_channel_1 - phase_channel_2, axis=0)
    print("Sum of Magnitude Differences:", np.sum(difference))
    print(f"STFT shape: {spec_FTM.shape}")
    print("Raw Phase Differences Shape:", raw_phase_diff.shape)
    print("Raw Phase Differences (First 5 bins, 5 frames):")
    print(raw_phase_diff[:5, :5])
    plt.figure(figsize=(10, 6))
    plt.imshow(raw_phase_diff, aspect='auto', origin='lower', cmap='twilight', interpolation='nearest')
    plt.colorbar(label="Phase Difference (radians)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title("Phase Differences - Channel Pair 0-1")
    plt.tight_layout()
    plt.show()
    return phase_differences, spec_FTM[..., 0], spec_FTM[..., 1]
def cluster_frequency_bins_single_pair(phase_diff, n_clusters=2):
    """
    Cluster frequency bins for a single pair of channels based on phase differences.

    Parameters
    ----------
    phase_diff : np.ndarray (F x T)
        Phase differences between a single channel pair.
    n_clusters : int
        Number of clusters for k-means (default: 2).

    Returns
    -------
    labels : np.ndarray (F x T)
        Cluster labels for each frequency bin and time frame.
    """
    # Flatten phase differences for clustering
    phase_diff_flat = phase_diff.reshape(-1, 1)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(phase_diff_flat)

    # Reshape cluster labels back to (F, T)
    labels = cluster_labels.reshape(phase_diff.shape)

    return labels


def plot_phase_difference(phase_diff):
    """
    Plot phase differences for stereo or multichannel audio.

    Parameters
    ----------
    phase_diff : np.ndarray
        Phase differences array.
        Shape can be (F x T) for stereo or (F x T x P) for multichannel, where P = number of channel pairs.
    """
    if phase_diff.ndim == 2:  # Stereo case
        plt.figure(figsize=(12, 6))
        plt.imshow(phase_diff, aspect='auto', origin='lower', cmap='twilight', interpolation='nearest')
        plt.colorbar(label="Phase Difference (radians)")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        plt.title("Phase Differences - Channel Pair 0-1")
        plt.tight_layout()
        plt.show()
    elif phase_diff.ndim == 3:  # Multichannel case
        num_pairs = phase_diff.shape[2]  # Number of channel pairs
        plt.figure(figsize=(12, 6 * num_pairs))
        for pair in range(num_pairs):
            plt.subplot(num_pairs, 1, pair + 1)
            plt.imshow(phase_diff[..., pair], aspect='auto', origin='lower', cmap='twilight', interpolation='nearest')
            plt.colorbar(label="Phase Difference (radians)")
            plt.xlabel("Time Frames")
            plt.ylabel("Frequency Bins")
            plt.title(f"Phase Differences - Channel Pair {pair}")
        plt.tight_layout()
        plt.show()

def reconstruct_sources(stft, labels, n_clusters):
    """
    Reconstruct sources from STFT using clustering labels.

    Parameters:
    ----------
    stft : np.ndarray (F x T x M or F x T)
        STFT of the multichannel signal.
    labels : np.ndarray (F x T or F x T x M)
        Clustering labels for each frequency-time pair.
    n_clusters : int
        Number of clusters.

    Returns:
    -------
    sources : list of np.ndarray
        Reconstructed sources.
    """
    sources = []
    for cluster_id in range(n_clusters):
        # Create mask based on cluster labels
        mask = (labels == cluster_id).astype(float)
        if mask.ndim < stft.ndim:  # Match dimensions if labels are 2D
            mask = mask[..., np.newaxis]
        stft_source = stft * mask
        # Reconstruct source using inverse STFT
        source = MultiISTFT(stft_source)
        sources.append(source)
    print(f"Mask for cluster {cluster_id}: {mask.shape}")
    plt.imshow(mask[..., 0], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f"Mask for Cluster {cluster_id}")
    plt.show()
    return sources

def calculate_azimuth(phase_differences, frequencies, mic_distance=0.2, sample_rate=16000):
    """
    Calculate azimuth angles from phase differences.

    Parameters:
    ----------
    phase_differences : np.ndarray
        Phase differences between microphone pairs (shape: F x T x M-1).
    frequencies : np.ndarray
        Array of frequencies (shape: F).
    mic_distance : float
        Distance between microphones in meters.
    sample_rate : int
        Sampling rate in Hz.

    Returns:
    -------
    azimuths : np.ndarray
        Estimated azimuth angles (shape: F x T x M-1).
    """

    # Ensure frequencies and wavelengths are numpy arrays
    if isinstance(frequencies, tuple):
        frequencies = np.array(frequencies)

    # Avoid division by zero for DC component (frequencies[0])
    wavelengths = np.zeros_like(frequencies)
    non_zero_mask = frequencies > 0
    wavelengths[non_zero_mask] = sample_rate / frequencies[non_zero_mask]  # Shape: (F,)

    # Reshape wavelengths for broadcasting
    wavelengths = wavelengths[:, None]  # Shape: (F, 1)

    # Replace NaN and Inf values in phase differences
    phase_diff_cleaned = np.nan_to_num(phase_differences, nan=0.0, posinf=0.0, neginf=0.0)

    # Time differences of arrival (TDOA) computation
    tdoas = (phase_diff_cleaned / (2 * np.pi)) * wavelengths  # Shape: (F, T)

    # Convert TDOAs to azimuth angles
    azimuths = np.arcsin(np.clip(tdoas / mic_distance, -1, 1))  # Shape: (F, T)

    # Debug outputs
    print(f"TDOA Range: {tdoas.min()} to {tdoas.max()}")
    print(f"Azimuth Range: {azimuths.min()} to {azimuths.max()}")
    return azimuths


def analyze_azimuths(azimuths):
    """
    Analyze the azimuth data.

    Parameters:
    ----------
    azimuths : np.ndarray
        Array of azimuth angles, shape should be (F, T) or (F, T, M-1).

    Returns:
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Check the dimensionality of azimuths
    if azimuths.ndim == 2:
        # Azimuths are (F, T)
        azimuths_2d = azimuths
    elif azimuths.ndim == 3:
        # Azimuths are (F, T, M-1)
        azimuths_2d = azimuths[:, :, 0]  # Select the first mic pair for analysis
    else:
        raise ValueError("Azimuth array must be 2D or 3D.")

    # Plot azimuth distribution
    plt.figure(figsize=(10, 4))
    plt.hist(azimuths_2d.flatten(), bins=100, color='blue', alpha=0.7)
    plt.xlabel('Azimuth (radians)')
    plt.ylabel('Frequency')
    plt.title('Azimuth Distribution')
    plt.show()

    # Plot azimuths over time
    plt.figure(figsize=(10, 6))
    plt.imshow(azimuths_2d, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(label='Azimuth (radians)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title('Azimuths Over Time')
    plt.show()

    # Mean azimuth over time
    mean_azimuths = np.mean(azimuths_2d, axis=0)  # Mean over frequency bins
    plt.figure(figsize=(10, 4))
    plt.plot(mean_azimuths, label='Mean Azimuth (Mic Pair 1)')
    plt.xlabel('Time Frames')
    plt.ylabel('Azimuth (radians)')
    plt.title('Mean Azimuth Over Time')
    plt.legend()
    plt.show()

    # Return analysis results
    azimuth_range = (np.min(azimuths_2d), np.max(azimuths_2d))
    nan_count = np.sum(np.isnan(azimuths_2d))
    inf_count = np.sum(np.isinf(azimuths_2d))

    print(f"Azimuth Range: {azimuth_range[0]:.2f} to {azimuth_range[1]:.2f} radians")
    print(f"NaN Count: {nan_count}, Inf Count: {inf_count}")


class Base:
    " Base Class for Source Separation Methods"

    def __init__(self, xp=np, seed=0, n_bit=64):
        self.xp = xp
        np.random.seed(seed)
        self.xp.random.seed(seed)

        self.n_bit = n_bit
        if self.n_bit == 64:
            self.TYPE_FLOAT = self.xp.float64
            self.TYPE_COMPLEX = self.xp.complex128
        elif self.n_bit == 32:
            self.TYPE_FLOAT = self.xp.float32
            self.TYPE_COMPLEX = self.xp.complex64
        self.method_name = "Base"
        self.save_param_list = ["n_bit"]

    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: np.ndarray F x T x M
                Spectrogram of observed signals
        """
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.TYPE_COMPLEX)
        self.sample_rate = sample_rate
        self.start_idx = 0
    

    def solve(
        self,
        n_iter=100,
        save_dir="./output/",
        save_wav=True,  # Only save final output, not intermediate
        save_wav_all=False,  # No need to save intermediate files
        interval_save=30,
        mic_index=MIC_INDEX,
        init=True,
        base_name=None  # Use base_name to define final output file name
    ):
        """
        Parameters:
            n_iter: int
            save_dir: str
            save_wav: bool
                Save the separated signals only after the last iteration 
            save_wav_all: bool
                Save the separated signals at every 'interval_save' iterations
            interval_save: int
                interval of saving wav
            mic_index: int
                Index of the microphone
            init: bool
                Whether to initialize the models
            base_name: str or None
                Base name for the saved files (optional)
        """
        self.n_iter = n_iter
        if init:
            self.init_source_model()
            self.init_spatial_model()

        print(f"Update {self.method_name}-{self}  {self.n_iter-self.start_idx} times ...")

        for self.it in tqdm(range(self.start_idx, n_iter)):
            self.update()

            # Skip saving intermediate files to avoid saving undesired names
            if save_wav_all and ((self.it + 1) % interval_save == 0) and ((self.it + 1) != n_iter):
                continue  # Skip intermediate file saving

        # Final separation and saving the .wav file with the desired base_name
        self.separate(mic_index=mic_index)
        save_fname = f"{save_dir}/{base_name}.wav"
        if save_wav or save_wav_all:
            if base_name is None:
               print('extra run')
                # Use the default name for final output
                # save_fname = f"{save_dir}/{self.method_name}-sep-final.wav"
            else:
                # Save final output with the specified base_name
                
             self.save_to_wav(self.separated_spec, save_fname=save_fname, shape="FTM")

    def save_to_wav(self, spec, save_fname="./sample.wav", shape="FTM"):
        spec = self.convert_to_NumpyArray(spec)
        assert not np.isnan(spec).any(), "spec includes NaN"
        assert spec.ndim <= 3, f"shape of spec is wrong : {spec.shape}"
       
        separated_signal = MultiISTFT(spec.transpose(1, 2, 0), shape=shape)
        sf.write(save_fname, separated_signal, self.sample_rate)

    def save_param(self, fname):
        import h5py
        with h5py.File(fname, "w") as f:
            for param in self.save_param_list:
                data = getattr(self, param)
                if type(data) is self.xp.ndarray:
                    data = self.convert_to_NumpyArray(data)
                f.create_dataset(param, data=data)
            f.flush()

    def load_param(self, fname):
        import h5py
        with h5py.File(fname, "r") as f:
            for key in f.keys():
                data = f[key]
                if (type(data) is self.xp) and (self.xp is not np):
                    data = self.xp.asarray(data)
                setattr(self, key, data)

            if "n_bit" in f.keys():
                if self.n_bit == 64:
                    self.TYPE_COMPLEX = self.xp.complex128
                    self.TYPE_FLOAT = self.xp.float64
                else:
                    self.TYPE_COMPLEX = self.xp.complex64
                    self.TYPE_FLOAT = self.xp.float32
