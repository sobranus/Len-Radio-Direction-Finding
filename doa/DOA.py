from functools import lru_cache

import numba as nb
from numba import float32, njit, vectorize

# Math
import numpy as np
import numpy.linalg as lin

import scipy
from scipy import fft, signal
from pyargus import directionEstimation as de


class DOA:
    def __init__(self,
                 signals,
                 sampling_freq = 2_400_000,
                 vfo_bw = 12_500,
                 freq_offset = 50_000,
                 center_freq = 433_920_000,
                 default_fir_order_factor = 4,
                 dsp_decimation = 1,
                 DOA_inter_elem_space = 0.5,
                 array_offset = 0.0,
                 DOA_expected_num_of_sources = 1
        ):
        self.signals = signals
        self.processed_signal = np.empty(0)
        self.sampling_freq = sampling_freq
        self.vfo_bw = vfo_bw
        self.freq_offset = freq_offset
        self.center_freq = center_freq
        self.vfo_freq = self.freq_offset + self.center_freq
        self.default_fir_order_factor = default_fir_order_factor
        self.dsp_decimation = dsp_decimation
        self.DOA_inter_elem_space = DOA_inter_elem_space
        self.array_offset = array_offset
        self.DOA_expected_num_of_sources = DOA_expected_num_of_sources
        self.DOA = np.ones(181)
        self.DOA_theta = np.linspace(0, 359, 360)
    
    def run(self):
        
        self.processed_signal = np.ascontiguousarray(self.signals)
        sampling_freq = self.sampling_freq
        freq_offset = self.freq_offset
        decimation_factor = max((sampling_freq // self.vfo_bw), 1)
        fir_order_factor = self.default_fir_order_factor

        global_decimation_factor = max(
            int(self.dsp_decimation), 1
        )

        if global_decimation_factor > 1:
            self.processed_signal = signal.decimate(
                self.processed_signal,
                global_decimation_factor,
                n=global_decimation_factor * 5,
                ftype="fir",
            )
            sampling_freq = sampling_freq // global_decimation_factor
            
        vfo_channel = channelize(
            self.processed_signal,
            freq_offset,
            decimation_factor,
            fir_order_factor,
            sampling_freq,
        )
        
        print("Calculating DOA...\n")
        theta_0 = self.estimate_DOA(vfo_channel, self.vfo_freq)
        print(f"Angle of Arrival: {int(theta_0)}°\n")
    
    
    def estimate_DOA(self, processed_signal, vfo_freq):
        """
        Estimates the direction of arrival of the received RF signal
        """
        R = corr_matrix(processed_signal)
        M = R.shape[0]

        frq_ratio = vfo_freq / self.center_freq
        inter_element_spacing = self.DOA_inter_elem_space * frq_ratio

        scanning_vectors = gen_scanning_vectors(
            M, inter_element_spacing, int(self.array_offset)
        )
    
        DOA_MUSIC_res = DOA_MUSIC(
            R, scanning_vectors, signal_dimension=self.DOA_expected_num_of_sources
        )
        self.DOA = DOA_MUSIC_res
        
        print(f"Highest correlation angles: {np.argpartition(self.DOA, -5)[-5:]}")

        theta_0 = self.DOA_theta[np.argmax(self.DOA)]

        return theta_0



# NUMBA optimized MUSIC function. About 100x faster on the Pi 4
# @njit(fastmath=True, cache=True, parallel=True)
@njit(fastmath=True, cache=True)
def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution=1):
    # --> Input check
    if R[:, 0].size != R[0, :].size:
        print("ERROR: Correlation matrix is not quadratic")
        return np.ones(1, dtype=np.complex64) * -1

    if R[:, 0].size != scanning_vectors[:, 0].size:
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return np.ones(1, dtype=np.complex64) * -2

    ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
    M = R[:, 0].size

    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    sigmai = np.abs(sigmai)

    idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
    vi = vi[:, idx]

    noise_dimension = M - signal_dimension

    E = np.empty((M, noise_dimension), dtype=np.complex64)
    for i in range(noise_dimension):
        E[:, i] = vi[:, i]

    E_ct = E @ E.conj().T
    theta_index = 0
    for i in range(scanning_vectors[0, :].size):
        S_theta_ = scanning_vectors[:, i]
        S_theta_ = np.ascontiguousarray(S_theta_.T)
        ADORT[theta_index] = 1 / np.abs(S_theta_.conj().T @ E_ct @ S_theta_)
        theta_index += 1

    return ADORT


# Get the FIR filter
@lru_cache(maxsize=32)
def get_fir(n, q, padd):
    return signal.dlti(signal.firwin(n, 1.0 / (q * padd), window="hann"), 1.0)


# Get the frequency rotation exponential
@lru_cache(maxsize=32)
def get_exponential(freq, sample_freq, sig_len):
    # Auto shift peak frequency center of spectrum, this frequency will be decimated:
    # https://pysdr.org/content/filters.html
    f0 = -freq  # +10
    Ts = 1.0 / sample_freq
    t = np.arange(0.0, Ts * sig_len, Ts)
    exponential = np.exp(2j * np.pi * f0 * t)  # this is essentially a complex sine wave

    return np.ascontiguousarray(exponential)


@njit(fastmath=True, cache=True)
def numba_mult(a, b):
    return a * b

        
# Memoize the total shift filter
@lru_cache(maxsize=32)
def shift_filter(decimation_factor, fir_order_factor, freq, sampling_freq, padd):
    fir_order = decimation_factor * fir_order_factor
    fir_order = fir_order + (fir_order - 1) % 2
    system = get_fir(fir_order, decimation_factor, padd)
    b = system.num
    a = system.den
    exponential = get_exponential(-freq, sampling_freq, len(b))
    b = numba_mult(b, exponential)
    return signal.dlti(b, a)


# This function takes the full data, and efficiently returns only a filtered and decimated requested channel
# Efficient method: Create BANDPASS Filter for frequency of interest, decimate with that bandpass filter, then do the final shift
def channelize(processed_signal, freq, decimation_factor, fir_order_factor, sampling_freq):
    system = shift_filter(
        decimation_factor, fir_order_factor, freq, sampling_freq, 1.1
    )  # Decimate with a BANDPASS filter
    
    decimated = signal.decimate(processed_signal, decimation_factor, ftype=system)
    
    exponential = get_exponential(
        freq, sampling_freq / decimation_factor, len(decimated[0, :])
    )  # Shift the signal AFTER to get back to normal decimate behaviour
    
    return numba_mult(decimated, exponential)
    
    
# Numba optimized version of pyArgus corr_matrix_estimate with "fast". About 2x faster on Pi4
# @njit(fastmath=True, cache=True)
def corr_matrix(X: np.ndarray) -> np.ndarray:
    N = X[0, :].size
    R = np.dot(X, X.conj().T)
    R = np.divide(R, N)
    return R


# LRU cache memoize about 1000x faster.
@lru_cache(maxsize=32)
def gen_scanning_vectors(M, DOA_inter_elem_space, offset):
    thetas = np.linspace(
        0, 359, 360
    )
    
    # convert UCA inter element spacing back to its radius
    to_r = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2.0 * np.pi / M)))
    r = DOA_inter_elem_space * to_r
    x = r * np.cos(2 * np.pi / M * np.arange(M))
    y = -r * np.sin(2 * np.pi / M * np.arange(M))

    scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
    for i in range(thetas.size):
        scanning_vectors[:, i] = np.exp(
            1j * 2 * np.pi * (x * np.cos(np.deg2rad(thetas[i] + offset)) + y * np.sin(np.deg2rad(thetas[i] + offset)))
        )

    return np.ascontiguousarray(scanning_vectors)


if __name__ == "__main__":
    pass