import numpy as np

class IQ:
    def __init__(
                self,
                sampling_freq = 2_400_000,
                duration = 0.1,
                num_antennas = 5,
                center_freq = 433_920_000,
                freq_offset = 50_000,
                aoa = 60,
                noise_power = 0.001
        ):
        
        self.sampling_freq = sampling_freq      # Sample rate (Hz)
        self.duration = duration                # seconds
        self.num_antennas = num_antennas        # KrakenSDR channels
        self.center_freq = center_freq          # RF center freq (for phase physics)
        self.freq_offset = freq_offset          # signal offset from DC (Hz)
        self.aoa = aoa                          # Angle of Arrival (degree)
        self.noise_power = noise_power          # Noise

        self.theta_signal = np.deg2rad(self.aoa)
        self.c = 3e8
        self.wavelength = self.c / center_freq
        self.radius = 0.5 * self.wavelength
        
        self.iq = self.generate_iq()
    
    def generate_iq(self):

        self.n_samples = int(self.sampling_freq * self.duration)
        self.t = np.arange(self.n_samples) / self.sampling_freq

        signal = np.exp(1j * 2 * np.pi * self.freq_offset * self.t)

        antenna_angles = np.linspace(0, 2*np.pi, self.num_antennas, endpoint=False)
        x = self.radius * np.cos(antenna_angles)
        y = -self.radius * np.sin(antenna_angles)

        self.phase_shifts = np.exp(
            1j * 2 * np.pi *
            (x * np.cos(self.theta_signal) + y * np.sin(self.theta_signal))
        )

        iq = np.zeros((self.num_antennas, self.n_samples), dtype=np.complex64)
        for i in range(self.num_antennas):
            iq[i, :] = self.phase_shifts[i] * signal

        noise = (np.random.randn(*iq.shape) + 1j*np.random.randn(*iq.shape))
        iq += np.sqrt(self.noise_power / 2) * noise
        
        return iq



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    iq_signal = IQ()
    
    # 1) Time domain (antenna 0)
    plt.figure()
    plt.plot(iq_signal.t[:200], iq_signal.iq[0, :200].real)
    plt.plot(iq_signal.t[:200], iq_signal.iq[0, :200].imag)
    plt.title("Antenna 0 - I/Q (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(["I", "Q"])
    plt.grid()
    plt.show()

    # 2) Spectrum (antenna 0)
    plt.figure()
    fft = np.fft.fftshift(np.fft.fft(iq_signal.iq[0]))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fft), 1/iq_signal.sampling_freq))
    plt.plot(freqs, 20*np.log10(np.abs(fft) + 1e-12))
    plt.title("Antenna 0 - Frequency Spectrum")
    plt.xlabel(f"Frequency (Hz) ({iq_signal.center_freq} Hz Center Freq)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.show()

    # 3) Phase across antennas (snapshot)
    snapshot = iq_signal.n_samples // 2
    phases = np.angle(iq_signal.iq[:, snapshot])
    plt.figure()
    plt.plot(range(iq_signal.num_antennas), phases, marker="o")
    plt.title("Phase Across Antennas (Single Snapshot)")
    plt.xlabel("Antenna Index")
    plt.ylabel("Phase (radians)")
    plt.grid()
    plt.show()

    # 4) All antennas (real part, short view)
    plt.figure()
    for i in range(iq_signal.num_antennas):
        plt.plot(iq_signal.t[:300], iq_signal.iq[i, :300].real + i * 1.2)
    plt.title("All Antennas - Real Part (Offset for visibility)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset")
    plt.grid()
    plt.show()

    print("Done: Synthetic KrakenSDR-style IQ generated and plotted.")
    print("IQ array shape:", iq_signal.iq.shape)
    print(iq_signal.iq[:, 0])
    print(iq_signal.phase_shifts)
