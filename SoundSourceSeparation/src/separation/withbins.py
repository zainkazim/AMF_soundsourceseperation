import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf

import FastMNMF2 
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT, Total_Index, MultiISTFT

class FastMNMF2Bin(Base):
    """
    Blind source separation using FastMNMF2 for stereo signals.

    Attributes:
    -----------
    X_FTM: np.ndarray
        Observed complex spectrogram.
    Q_FMM: np.ndarray
        Spatial model (diagonalizer that converts SCMs to diagonal matrices).
    G_NM: np.ndarray
        Spatial gain matrix.
    W_NFK: np.ndarray
        Basis vectors.
    H_NKT: np.ndarray
        Activations.
    PSD_NFT: np.ndarray
        Power spectral densities.
    Qx_power_FTM: np.ndarray
        Power spectra of Q_FMM times X_FTM.
    Y_FTM: np.ndarray
        Combined model estimate.
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        init_SCM="twostep",
        algo="IP",
        n_iter_init=30,
        n_iter=30,
        g_eps=5e-2,
        interval_norm=10,
        n_bit=64,
        xp=np,
        seed=0,
        n_mic=2,  # Default: stereo signals
    ):
        """Initialize FastMNMF2Bin."""
        super().__init__(xp=xp, n_bit=n_bit, seed=seed)
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.n_iter = n_iter
        self.n_mic = n_mic

        if self.n_mic != 2:
            raise ValueError("This implementation is restricted to stereo signals (n_mic must be 2).")

        self.save_param_list += ["W_NFK", "H_NKT", "G_NM", "Q_FMM"]

    def init_spatial_model(self):
    # Randomly initialize Q_FMM with small perturbations
        self.Q_FMM = (
            self.xp.tile(self.xp.eye(self.n_mic), (self.n_freq, 1, 1)) 
            + self.xp.random.normal(scale=0.01, size=(self.n_freq, self.n_mic, self.n_mic))
        ).astype(self.TYPE_COMPLEX)

        # Initialize G_NM with diverse spatial gains
        self.G_NM = self.xp.random.dirichlet([1] * self.n_mic, size=self.n_source).astype(self.TYPE_FLOAT)

        # Normalize spatial gains and parameters
        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()

        # Calculate phase differences and azimuth after normalization
        self.calculate_Qx()  # Ensure Qx_FTM reflects the normalized Q_FMM
        self.calculate_azimuth()  # Ensure azimuth is based on the updated Qx_FTM

        # Debugging phase and azimuth ranges
        print(f"Phase Difference Range: {self.phase_diff_FT.min()}, {self.phase_diff_FT.max()}")
        print(f"Azimuth Range: {self.azimuth_FT.min()}, {self.azimuth_FT.max()}")

        # Debugging final spatial model
        print("Spatial model initialized and normalized.")

    def normalize(self):
        phi_F = self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
        self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.W_NFK /= phi_F[None, :, None]

        mu_N = self.G_NM.sum(axis=1)
        self.G_NM /= mu_N[:, None]
        self.W_NFK *= mu_N[:, None, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.calculate_Qx()
        self.calculate_PSD()
        self.calculate_Y()

    def calculate_Qx(self):
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    # def calculate_Qx(self):
    #     self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
    #     self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    #     phase_left = self.xp.angle(self.X_FTM[:, :, 0])
    #     phase_right = self.xp.angle(self.X_FTM[:, :, 1])
    #     self.phase_diff_FT = phase_left - phase_right
    #     print(f"Phase Differences Min/Max: {self.phase_diff_FT.min()}, {self.phase_diff_FT.max()}")

    #     self.azimuth_FT = self.xp.arcsin(self.phase_diff_FT / (2 * self.xp.pi))
    #     print(f"Azimuth Min/Max: {self.azimuth_FT.min()}, {self.azimuth_FT.max()}")
    def calculate_Qx(self):
        # Compute the spatially transformed spectrogram
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

        # Calculate phase differences between the left and right channels
        phase_left = self.xp.angle(self.X_FTM[:, :, 0])
        phase_right = self.xp.angle(self.X_FTM[:, :, 1])
        self.phase_diff_FT = phase_left - phase_right

        # Wrap phase differences to [-π, π]
        self.phase_diff_FT = self.xp.mod(self.phase_diff_FT + self.xp.pi, 2 * self.xp.pi) - self.xp.pi

        # Normalize phase differences and clip to [-1, 1] for arcsin
        normalized_phase_diff = self.phase_diff_FT / (2 * self.xp.pi)
        normalized_phase_diff = self.xp.clip(normalized_phase_diff, -1, 1)

        # Calculate azimuth
        self.azimuth_FT = self.xp.arcsin(normalized_phase_diff)

        # Debugging outputs
        print(f"Phase Differences Min/Max: {self.phase_diff_FT.min()}, {self.phase_diff_FT.max()}")
        print(f"Azimuth Min/Max: {self.azimuth_FT.min()}, {self.azimuth_FT.max()}")


    def calculate_azimuth(self):
        if not hasattr(self, "azimuth_FT"):
            raise ValueError("Azimuth has not been calculated. Run calculate_Qx first.")
        return self.azimuth_FT


    def create_azimuth_mask(self, desired_azimuth, azimuth_tolerance):
        # Create the azimuth mask for a given source
        mask = (self.azimuth_FT >= desired_azimuth - azimuth_tolerance) & (
            self.azimuth_FT <= desired_azimuth + azimuth_tolerance
        )
        return mask

    def separate(self):
        separated_sources = []

        azimuth_min, azimuth_max = self.azimuth_FT.min(), self.azimuth_FT.max()
        azimuth_tolerance = (azimuth_max - azimuth_min) / self.n_source * 1.5  # Increase tolerance for better separation

        total_mask = np.zeros_like(self.azimuth_FT, dtype=bool)  # Track mask coverage

        for source_idx in range(self.n_source):
            # Calculate the desired azimuth for the current source
            desired_azimuth = azimuth_min + (source_idx / self.n_source) * (azimuth_max - azimuth_min)

            # Add a small bias to spread sources further apart
            desired_azimuth += source_idx * 0.05

            # Create the mask for the current source
            mask = (self.azimuth_FT >= desired_azimuth - azimuth_tolerance) & (
                self.azimuth_FT <= desired_azimuth + azimuth_tolerance
            )
            mask &= ~total_mask  # Ensure disjoint masks
            total_mask |= mask  # Update total mask

            print(f"Source {source_idx}: Mask Non-Zero Count: {np.sum(mask)}")
            if np.sum(mask) == 0:
                print(f"Warning: Empty mask for Source {source_idx}.")
                continue

            # Apply the mask to the spectrogram
            separated_sources.append(self.X_FTM * mask[:, :, None])

        # Debugging total mask coverage
        print(f"Total Mask Coverage: {np.sum(total_mask)} out of {self.azimuth_FT.size}")

        # Stack separated sources
        self.separated_spec = np.stack(separated_sources, axis=0)
        return self.separated_spec

    def init_source_model(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)

    def solve(self, n_iter=100, save_dir="./output/", save_wav=True, base_name="final_separation"):
        os.makedirs(save_dir, exist_ok=True)

        # Ensure models are initialized
        self.init_source_model()
        self.init_spatial_model()

        # Perform separation
        separated_spec = self.separate()
        for idx, source_spec in enumerate(separated_spec):
            save_fname = os.path.join(save_dir, f"{base_name}_source{idx}.wav")
            self.save_to_wav(source_spec, save_fname)

    def save_to_wav(self, spec, save_fname):
        separated_signal = MultiISTFT(spec)
        sf.write(save_fname, separated_signal, self.sample_rate)
        print(f"Saved .wav file to {save_fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_source", type=int, default=2, help="Number of sources")
    parser.add_argument("--n_fft", type=int, default=1024, help="Number of FFT bins")
    args = parser.parse_args()

    input_file = input("Enter the path to the input stereo audio file: ")
    wav, sample_rate = sf.read(input_file)

    if wav.ndim == 1:
        raise ValueError("Input file is mono. Azimuth-based separation requires stereo input.")

    spec_FTM = MultiSTFT(wav[:, :2], n_fft=args.n_fft)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("STFT Magnitude")
    plt.imshow(np.abs(spec_FTM[:, :, 0]), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Magnitude")
    plt.subplot(2, 1, 2)
    plt.title("STFT Phase")
    plt.imshow(np.angle(spec_FTM[:, :, 0]), aspect="auto", origin="lower", cmap="twilight")
    plt.colorbar(label="Phase")
    plt.show()

    separater = FastMNMF2Bin(n_source=args.n_source)
    separater.load_spectrogram(spec_FTM, sample_rate)

    separater.solve(
        n_iter=100,
        save_dir="./output",
        save_wav=True,
        base_name="final_separation"
    )
