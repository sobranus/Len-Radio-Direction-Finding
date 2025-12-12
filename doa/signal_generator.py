import numpy as np
import matplotlib.pyplot as plt

class IQ:
    def __init__(
                self,
                sampling_freq = 2_400_000,
                duration = 0.1,
                num_antennas = 5,
                ant_radius_wavelength = 0.5,
                center_freq = 433_920_000,
                freq_offset = 50_000,
                aoa = 60,
                noise_power = 0.001,
                plot_signal = False
        ):
        
        self.sampling_freq = sampling_freq      # Sample rate (Hz)
        self.duration = duration                # seconds
        self.num_antennas = num_antennas        # KrakenSDR channels
        self.ant_radius_wavelength = ant_radius_wavelength
        self.center_freq = center_freq          # RF center freq (for phase physics)
        self.freq_offset = freq_offset          # signal offset from DC (Hz)
        self.aoa = aoa                          # Angle of Arrival (degree)
        self.noise_power = noise_power          # Noise
        
        self.plot_signal = plot_signal

        self.theta_signal = np.deg2rad(self.aoa)
        self.c = 3e8
        self.wavelength = self.c / center_freq
        self.radius = self.ant_radius_wavelength * self.wavelength
        
        self.iq = self.generate_iq()
        if self.plot_signal:
            self.signal_plot()
    
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
    
    def signal_plot(self):
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.canvas.manager.set_window_title("Generated IQ Signal")

        # 1) Time domain (antenna 0)
        axes[0, 0].plot(self.t[:200], self.iq[0, :200].real, label="I")
        axes[0, 0].plot(self.t[:200], self.iq[0, :200].imag, label="Q")
        axes[0, 0].set_title("Antenna 0 - I/Q (Time Domain)", fontsize=10, pad=8)
        axes[0, 0].set_xlabel("Time (s)", fontsize=9)
        axes[0, 0].set_ylabel("Amplitude", fontsize=9)
        axes[0, 0].tick_params(axis='both', labelsize=8)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid()

        # 2) Spectrum (antenna 0)
        fft = np.fft.fftshift(np.fft.fft(self.iq[0]))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(fft), 1/self.sampling_freq))
        axes[0, 1].plot(freqs, 20*np.log10(np.abs(fft) + 1e-12))
        axes[0, 1].set_title("Antenna 0 - Frequency Spectrum", fontsize=10, pad=8)
        axes[0, 1].set_xlabel(f"Frequency (Hz) ({self.center_freq} Hz Center)\n", fontsize=9)
        axes[0, 1].set_ylabel("Magnitude (dB)", fontsize=9)
        axes[0, 1].tick_params(axis='both', labelsize=8)
        axes[0, 1].grid()

        # 3) Phase across antennas (snapshot)
        snapshot = self.n_samples // 2
        phases = np.angle(self.iq[:, snapshot])
        axes[1, 0].plot(range(self.num_antennas), phases, marker="o")
        axes[1, 0].set_title("Phase Across Antennas (Single Snapshot)", fontsize=10, pad=8)
        axes[1, 0].set_xlabel("Antenna Index", fontsize=9)
        axes[1, 0].set_ylabel("Phase (radians)", fontsize=9)
        axes[1, 0].tick_params(axis='both', labelsize=8)
        axes[1, 0].grid()

        # 4) All antennas (real part, short view)
        for i in range(self.num_antennas):
            axes[1, 1].plot(self.t[:300], self.iq[i, :300].real + i * 1.2)
        axes[1, 1].set_title("All Antennas - Real Part (Offset)", fontsize=10, pad=8)
        axes[1, 1].set_xlabel("Time (s)", fontsize=9)
        axes[1, 1].set_ylabel("Amplitude + offset", fontsize=9)
        axes[1, 1].tick_params(axis='both', labelsize=8)
        axes[1, 1].grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    
    iq_signal = IQ(
        sampling_freq = 2_400_000,
        duration = 0.1,
        num_antennas = 5,
        center_freq = 433_920_000,
        freq_offset = 50_000,
        aoa = 60,
        noise_power = 0.001,
        plot_signal = True
    )
