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
    def __init__(self):
        
        self.iq_samples = 
        self.processed_signal = np.empty(0)
        self.vfo_bw = 
        self.sampling_freq = 

        # self.sampling_freq = 2,000,000 → 2 MS/s

        # What it tells Kraken:
        # How much spectrum you can “see”
        # How fast phase changes from the real world map into digital samples
        # The bandwidth of usable DOA estimation
        # Think of it as how fast your radio “takes pictures” of the RF world.
        
        self.freq = 
        self.daq_center_freq = 
        self.vfo_freq = 
        
        # self.vfo_freq = freq + self.daq_center_freq
        # This converts baseband frequency → RF frequency.

        # Example:
        # self.daq_center_freq = 433.920 MHz
        # self.freq = FFT peak returned = –50 kHz
        # self.vfo_freq = actual peak freq location = 433.870 MHz

        # So Kraken calculates the absolute RF frequency of the incoming signal.
        # This is important when multiple signals are present.
        # If there is multiple signals present, there will be multiple peak freqs around 1 daq center freq.
        
        self.default_fir_order_factor = int(2)
        self.DOA_ant_alignment = "UCA"
        self.dsp_decimation = 
        self.DOA_decorrelation_method = "FBA"
        self.DOA_inter_elem_space = 0.5
        self.array_offset = 0.0
        self.DOA_algorithm = "MUSIC"
        self.DOA_expected_num_of_sources = 1
        self.DOA = np.ones(181)
        self.DOA_theta = np.linspace(0, 359, 360)
    
    def run(self):
        
        self.processed_signal = np.ascontiguousarray(self.iq_samples)
        sampling_freq = self.sampling_freq
        freq = self.freq
        decimation_factor = max((sampling_freq // self.vfo_bw), 1)
        fir_order_factor = self.default_fir_order_factor

        global_decimation_factor = max(
            int(self.dsp_decimation), 1
        )  # max(int(self.phasetest[0]), 1) #ps_len // 65536 #int(self.phasetest[0]) + 1

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
            freq,
            decimation_factor,
            fir_order_factor,
            sampling_freq,
        )
        
        theta_0 = self.estimate_DOA(vfo_channel, self.vfo_freq)
    
    
    def estimate_DOA(self, processed_signal, vfo_freq):
        """
        Estimates the direction of arrival of the received RF signal
        """
        # Calculating spatial correlation matrix
        antennas_alignment = self.DOA_ant_alignment
        R = corr_matrix(processed_signal)
        M = R.shape[0]

        if self.DOA_decorrelation_method == "FBA":
            R = de.forward_backward_avg(R)

        M = R.shape[0]

        frq_ratio = vfo_freq / self.daq_center_freq
        inter_element_spacing = self.DOA_inter_elem_space * frq_ratio

        if antennas_alignment == "UCA":
            scanning_vectors = gen_scanning_vectors(
                M, inter_element_spacing, antennas_alignment, int(self.array_offset)
            )

        if self.DOA_algorithm == "MUSIC":  # self.en_DOA_MUSIC:
            DOA_MUSIC_res = DOA_MUSIC(
                R, scanning_vectors, signal_dimension=self.DOA_expected_num_of_sources
            )  # de.DOA_MUSIC(R, scanning_vectors, signal_dimension = 1)
            self.DOA = DOA_MUSIC_res

        # ULA Array, choose bewteen the full omnidirecitonal 360 data, or forward/backward data only

        if self.DOA_algorithm != "ROOT-MUSIC":
            theta_0 = self.DOA_theta[np.argmax(self.DOA)]

        return theta_0



# NUMBA optimized MUSIC function. About 100x faster on the Pi 4
# @njit(fastmath=True, cache=True, parallel=True)
@njit(fastmath=True, cache=True)
def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution=1):
    # --> Input check
    if R[:, 0].size != R[0, :].size:
        print("ERROR: Correlation matrix is not quadratic")
        return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

    if R[:, 0].size != scanning_vectors[:, 0].size:
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return np.ones(1, dtype=np.complex64) * -2

    ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
    M = R[:, 0].size  # np.size(R, 0)

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    sigmai = np.abs(sigmai)

    idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
    vi = vi[:, idx]

    # Generate noise subspace matrix
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

    # Old Method
    # Auto shift peak frequency center of spectrum, this frequency will be decimated:
    # https://pysdr.org/content/filters.html
    # f0 = -freq #+10
    # Ts = 1.0/sample_freq
    # t = np.arange(0.0, Ts*len(processed_signal[0, :]), Ts)
    # exponential = np.exp(2j*np.pi*f0*t) # this is essentially a complex sine wave

    # Decimate down to BW
    # decimation_factor = max((sample_freq // bw), 1)
    # decimated_signal = signal.decimate(processed_signal, decimation_factor, n = decimation_factor * 2, ftype='fir')

    # return decimated_signal        
    
    
# Numba optimized version of pyArgus corr_matrix_estimate with "fast". About 2x faster on Pi4
# @njit(fastmath=True, cache=True)
def corr_matrix(X: np.ndarray) -> np.ndarray:
    N = X[0, :].size
    R = np.dot(X, X.conj().T)
    R = np.divide(R, N)
    return R


# LRU cache memoize about 1000x faster.
@lru_cache(maxsize=32)
def gen_scanning_vectors(M, DOA_inter_elem_space, type, offset):
    thetas = np.linspace(
        0, 359, 360
    )  # Remember to change self.DOA_thetas too, we didn't include that in this function due to memoization cannot work with arrays
    if type == "UCA":
        # convert UCA inter element spacing back to its radius
        to_r = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2.0 * np.pi / M)))
        r = DOA_inter_elem_space * to_r
        x = r * np.cos(2 * np.pi / M * np.arange(M))
        y = -r * np.sin(2 * np.pi / M * np.arange(M))  # For this specific array only

    scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
    for i in range(thetas.size):
        scanning_vectors[:, i] = np.exp(
            1j * 2 * np.pi * (x * np.cos(np.deg2rad(thetas[i] + offset)) + y * np.sin(np.deg2rad(thetas[i] + offset)))
        )

    return np.ascontiguousarray(scanning_vectors)