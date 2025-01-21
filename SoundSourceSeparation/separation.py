#! /usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT,Total_Index,calculate_phase_differences, calculate_azimuth,MultiISTFT,calculate_phase_differences,cluster_frequency_bins_single_pair,reconstruct_sources,plot_phase_difference,estimate_steering_vector,analyze_azimuths
import soundfile as sf
import argparse

class FastMNMF2(Base):
    """
    The blind souce separation using FastMNMF2

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
    G_NM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities
    Qx_power_FTM: power spectra of Q_FMM times X_FTM
    Y_FTM: sum of (PSD_NFT x G_NM) over all sources
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        init_SCM="twostep",
        algo="IP",
        n_iter_init=30,
        g_eps=5e-2,
        interval_norm=10,
        n_bit=64,
        xp=np,
        seed=0,
    ):
        """Initialize FastMNMF2

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases for the NMF-based source model.
            init_SCM: str ('circular', 'obs', 'twostep')
                How to initialize SCM.
                'obs' is for the case that one speech is dominant in the mixture.
            algo: str (IP, ISS)
                How to update Q.
            n_iter_init: int
                The number of iteration for the first step in 'twostep' initialization.
            xp : numpy or cupy
        """
        super().__init__(xp=xp, n_bit=n_bit, seed=seed)
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.save_param_list += ["W_NFK", "H_NKT", "G_NM", "Q_FMM"]

        if self.algo == "IP":
            self.method_name = "FastMNMF2_IP"
        elif "ISS" in algo:
            self.method_name = "FastMNMF2_ISS"
        else:
            raise ValueError("algo must be IP or ISS")

    def __str__(self):
        init = f"twostep_{self.n_iter_init}it" if self.init_SCM == "twostep" else self.init_SCM
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={init}-g={self.g_eps}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM, sample_rate=sample_rate)
        if self.algo == "IP":
            self.XX_FTMM = self.xp.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())
            print("Calculating phase differences...")
            print("Calculating phase differences...")
        phase_diff, stft_left, stft_right = calculate_phase_differences(wav[:, :M], n_fft=args.n_fft)
        print(f"Phase differences shape: {phase_diff.shape}")
        # Calculate azimuth from phase differences
        frequencies = np.fft.rfftfreq(1024, d=1/sample_rate)
        self.azimuths = calculate_azimuth(phase_diff, frequencies, mic_distance=0.2, sample_rate=sample_rate)
        print(f"Azimuths shape: {self.azimuths.shape}")
        analysis_results = analyze_azimuths(
        azimuths=self.azimuths 
)



    def init_source_model(self):
        
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
   
    def init_spatial_model(self):
 
        self.Q_FMM = (
            self.xp.tile(self.xp.eye(self.n_mic), (self.n_freq, 1, 1))
            + self.xp.random.normal(scale=0.01, size=(self.n_freq, self.n_mic, self.n_mic))
        ).astype(self.TYPE_COMPLEX)


        self.G_NM = self.xp.random.dirichlet([1] * self.n_mic, size=self.n_source).astype(self.TYPE_FLOAT)


        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()
        self.calculate_Qx()


    def calculate_Qx(self):
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_WH()
        self.update_G()
        if self.algo == "IP":
            self.update_Q_IP()
        else:
            self.update_Q_ISS()
        if self.it % self.interval_norm == 0:
            self.normalize()
        else:
            self.calculate_Qx()

    def update_WH(self):
        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)
        self.W_NFK *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp1_NFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp2_NFT)
        self.H_NKT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, self.Qx_power_FTM / (self.Y_FTM**2))
        denominator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NM *= self.xp.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_Q_IP(self):
        for m in range(self.n_mic):
            V_FMM = self.xp.einsum("ftij, ft -> fij", self.XX_FTMM, 1 / self.Y_FTM[..., m]) / self.n_time
            tmp_FM = self.xp.linalg.inv(self.Q_FMM @ V_FMM)[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / self.xp.sqrt(self.xp.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()

    def update_Q_ISS(self):
        for m in range(self.n_mic):
            QxQx_FTM = self.Qx_FTM * self.Qx_FTM[:, :, m, None].conj()
            V_tmp_FxM = (QxQx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
            V_FxM = (QxQx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
            V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
            self.Qx_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Qx_FTM[:, :, m])
            self.Q_FMM -= self.xp.einsum("fi, fj -> fij", V_FxM, self.Q_FMM[:, m])

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

    def separate(self, mic_index=MIC_INDEX):
        Y_NFTM = self.xp.einsum("nft, nm -> nftm", self.PSD_NFT, self.G_NM)
        self.Y_FTM = Y_NFTM.sum(axis=0)
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        Qinv_FMM = self.xp.linalg.inv(self.Q_FMM)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / self.Y_FTM, Y_NFTM
        )

    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.Qx_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()))).sum()
        ).real
        return log_likelihood

    def load_param(self, filename):
        super().load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=30, help="nujmber of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="twostep",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--g_eps", type=float, default=5e-2, help="minumum value used for initializing G_NM")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q")
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        try:
            import cupy as xp

            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU.\n")
            import numpy as xp

    separater = FastMNMF2(
        n_source=args.n_source,
        n_basis=args.n_basis,
        xp=xp,
        init_SCM=args.init_SCM,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init,
        g_eps=args.g_eps,
    )


    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2

    if wav.ndim > 1:  # Multichannel audio
        print(f"Number of microphones (channels): {wav.shape[1]}")
        Total_Index=wav.shape[1]
        M = min(wav.shape[1], args.n_mic)  # Limit M to the actual number of channels in the audio
    else:  # Mono audio
        print("Number of microphones (channels): 1")
        M = 1  # For mono audio, only one channel exists

    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)
    print(f"STFT shape: {spec_FTM.shape}")
    separater.file_id = args.input_fname.split("/")[-1].split(".")[0]
    separater.load_spectrogram(spec_FTM, sample_rate)
    print("Calculating phase differences...")
    print(f"wav shape: {wav.shape}")
    if np.allclose(wav[:, 0], wav[:, 1]):
     print("The two channels are identical!")
    else:
     print("Channels are different.")
    print(f"WAV shape: {wav.shape}")
    print(f"Variance of Channel 1: {np.var(wav[:, 0])}")
    print(f"Variance of Channel 2: {np.var(wav[:, 1])}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(wav[:, 0])
    plt.title("Waveform - Channel 1")
    plt.subplot(2, 1, 2)
    plt.plot(wav[:, 1])
    plt.title("Waveform - Channel 2")
    plt.tight_layout()
    plt.show()


    phase_diff, stft_left, stft_right = calculate_phase_differences(wav[:, :M], n_fft=args.n_fft)
    print(f"Phase differences shape: {phase_diff.shape}")
    # Calculate and plot phase difference
    
    plot_phase_difference(phase_diff)


    output_dir = "./output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for mic_index in range(Total_Index):
        separater.solve(
            n_iter=args.n_iter,
            save_dir="./output/",
            # save_likelihood=False,
            # save_param=False,
            save_wav_all=False,
            save_wav=True,
            mic_index=mic_index,  # Pass the loop index
            interval_save=5,
            base_name= f"{mic_index}_channel"

        )
 
