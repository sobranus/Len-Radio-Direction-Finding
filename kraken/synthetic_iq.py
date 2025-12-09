import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Parameters (you can tweak these)
# ==============================
fs = 2_400_000              # Sample rate (Hz)
duration = 0.01             # seconds
num_antennas = 5            # KrakenSDR channels
center_freq = 433_920_000   # RF center freq (for phase physics)
signal_offset = 100_000     # signal offset from DC (Hz)

c = 3e8
wavelength = c / center_freq
radius = 0.5 * wavelength

aoa_deg = 30
noise_power = 0.01

# ==============================
# Derived values
# ==============================
n_samples = int(fs * duration)
t = np.arange(n_samples) / fs
theta_signal = np.deg2rad(aoa_deg)

# ==============================
# Generate baseband tone (signal)
# ==============================
signal = np.exp(1j * 2 * np.pi * signal_offset * t)

# ==============================
# UCA geometry
# ==============================
antenna_angles = np.linspace(0, 2*np.pi, num_antennas, endpoint=False)
x = radius * np.cos(antenna_angles)
y = radius * np.sin(antenna_angles)

# ==============================
# Compute phase shifts (UCA steering vector)
# ==============================
phase_shifts = np.exp(
    1j * 2 * np.pi *
    (x * np.cos(theta_signal) + y * np.sin(theta_signal))
)

# ==============================
# Create multi-channel IQ
# ==============================
iq = np.zeros((num_antennas, n_samples), dtype=np.complex64)
for i in range(num_antennas):
    iq[i, :] = phase_shifts[i] * signal

# ==============================
# Add noise
# ==============================
noise = (np.random.randn(*iq.shape) + 1j*np.random.randn(*iq.shape))
iq += np.sqrt(noise_power / 2) * noise

# ==============================
# ----------- PLOTS -------------
# ==============================

# 1) Time domain (antenna 0)
plt.figure()
plt.plot(t[:400], iq[0, :400].real)
plt.plot(t[:400], iq[0, :400].imag)
plt.title("Antenna 0 - I/Q (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["I", "Q"])
plt.grid()
plt.show()

# 2) Constellation (antenna 0)
plt.figure()
pts = iq[0, ::max(1, n_samples // 5000)]
plt.scatter(pts.real, pts.imag, s=3)
plt.title("Antenna 0 - IQ Constellation")
plt.xlabel("I")
plt.ylabel("Q")
plt.axis("equal")
plt.grid()
plt.show()

# 3) Spectrum (antenna 0)
plt.figure()
fft = np.fft.fftshift(np.fft.fft(iq[0]))
freqs = np.fft.fftshift(np.fft.fftfreq(len(fft), 1/fs))
plt.plot(freqs, 20*np.log10(np.abs(fft) + 1e-12))
plt.title("Antenna 0 - Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()

# 4) Phase across antennas (snapshot)
snapshot = n_samples // 2
phases = np.angle(iq[:, snapshot])
plt.figure()
plt.plot(range(num_antennas), phases, marker="o")
plt.title("Phase Across Antennas (Single Snapshot)")
plt.xlabel("Antenna Index")
plt.ylabel("Phase (radians)")
plt.grid()
plt.show()

# 5) All antennas (real part, short view)
plt.figure()
for i in range(num_antennas):
    plt.plot(t[:300], iq[i, :300].real + i * 1.2)
plt.title("All Antennas - Real Part (Offset for visibility)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset")
plt.grid()
plt.show()

print("Done: Synthetic KrakenSDR-style IQ generated and plotted.")
print("IQ array shape:", iq.shape)
print(iq[:, 0])
