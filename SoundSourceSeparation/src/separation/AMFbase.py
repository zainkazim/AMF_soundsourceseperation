import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Base import MultiSTFT, MultiISTFT

class EnhancedSourceSeparator:
    """
    Improved sound source separation using phase differences and magnitude weighting.
    """
    def __init__(self, X_FTM, n_sources):
        self.X_FTM = X_FTM
        self.n_sources = n_sources
        self.phase_diff_FT = None
        self.magnitude = None
        self.freq_bins_per_source = None

    def compute_phase_differences(self):
        phase_left = np.angle(self.X_FTM[:, :, 0])  # Channel 0: Left
        phase_right = np.angle(self.X_FTM[:, :, 1])  # Channel 1: Right
        self.phase_diff_FT = phase_left - phase_right
        self.magnitude = np.abs(self.X_FTM)  # Magnitude for weighting
        print(f"Phase differences calculated. Range: Min={self.phase_diff_FT.min()}, Max={self.phase_diff_FT.max()}")

    def create_masks(self):
        phase_min, phase_max = self.phase_diff_FT.min(), self.phase_diff_FT.max()
        phase_segment_width = (phase_max - phase_min) / self.n_sources
        self.freq_bins_per_source = {}

        for source_idx in range(self.n_sources):
            lower_bound = phase_min + source_idx * phase_segment_width
            upper_bound = lower_bound + phase_segment_width
            mask = (self.phase_diff_FT >= lower_bound) & (self.phase_diff_FT < upper_bound)
            weighted_mask = mask * self.magnitude[:, :, 0]  # Weight with magnitude of left channel
            self.freq_bins_per_source[source_idx] = weighted_mask
            print(f"Source {source_idx}: Frequency bins contributing = {np.sum(mask)}")

    def reconstruct_sources(self):
        separated_sources = []
        for source_idx, mask in self.freq_bins_per_source.items():
            masked_spec = self.X_FTM * mask[:, :, None]  # Apply mask across all microphones
            separated_audio = MultiISTFT(masked_spec)
            separated_sources.append(separated_audio)
            print(f"Source {source_idx} reconstructed. Non-zero elements: {np.count_nonzero(masked_spec)}")
        return separated_sources

def calculate_metrics(reference_path, separated_sources):
    """
    Calculate SDR, SIR, and SAR for validation.

    Parameters:
    - reference_path: str
      Path to the reference stereo audio file.
    - separated_sources: list
      List of separated source signals.

    Returns:
    - dict: Metrics for each source.
    """
    reference_audio, sample_rate = sf.read(reference_path)
    if reference_audio.ndim != 2:
        raise ValueError("Reference audio must be stereo (2 channels).")

    metrics = {}
    for i, sep in enumerate(separated_sources):
        ref_channel = reference_audio[:, i] if i < reference_audio.shape[1] else np.zeros_like(sep[:, 0])

        # Handle length mismatch
        min_length = min(len(ref_channel), len(sep))
        ref_channel = ref_channel[:min_length]
        sep = sep[:min_length]

        # If separated source is stereo, take the first channel for metrics
        if sep.ndim == 2:
            sep = sep[:, 0]

        # Compute SDR, SIR, and SAR
        sdr, sir, sar, _ = bss_eval_sources(ref_channel[np.newaxis, :], sep[np.newaxis, :])
        metrics[f"Source {i}"] = {"SDR": sdr[0], "SIR": sir[0], "SAR": sar[0]}
        print(f"Source {i} Metrics - SDR: {sdr[0]:.2f}, SIR: {sir[0]:.2f}, SAR: {sar[0]:.2f}")

    return metrics


if __name__ == "__main__":
    # Paths
    input_file = "/home/ZA/Music/Media project/AMF_MP/new wav files/M1F1-Alaw-AFsp.wav"
    output_dir = "/home/ZA/Music/Media project/AMF_MP/output/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load audio
    wav, sample_rate = sf.read(input_file)
    if wav.ndim != 2:
        raise ValueError("Input file must be stereo (2 channels).")

    # Compute STFT
    spec_FTM = MultiSTFT(wav[:, :2], n_fft=1024)

    # Initialize and run the separator
    separator = EnhancedSourceSeparator(X_FTM=spec_FTM, n_sources=2)
    separator.compute_phase_differences()
    separator.create_masks()
    separated_sources_audio = separator.reconstruct_sources()

    # Save separated sources
    for i, source in enumerate(separated_sources_audio):
        output_file = f"{output_dir}/source_{i}.wav"
        sf.write(output_file, source, sample_rate)
        print(f"Saved separated source to {output_file}")

    # Validate separation using metrics
    metrics = calculate_metrics(reference_path=input_file, separated_sources=separated_sources_audio)

    # Print summary of metrics
    print("\nSummary of Metrics:")
    for source, values in metrics.items():
        print(f"{source}: SDR={values['SDR']:.2f}, SIR={values['SIR']:.2f}, SAR={values['SAR']:.2f}")
