
from signal_generator import IQ
import DOA


def run():
    
    # IQ GENERATOR INPUT
    sampling_freq = 2_400_000               # Sample rate (Hz)
    duration = 0.1                          # seconds
    num_antennas = 5                        # KrakenSDR channels
    center_freq = 433_920_000               # RF center freq (for phase physics)
    freq_offset = 50_000                    # signal offset from DC (Hz)
    aoa = 130
    noise_power = 0.001
    
    signal = IQ(
        sampling_freq = sampling_freq,
        duration = duration,
        num_antennas = num_antennas,
        center_freq = center_freq,
        freq_offset = freq_offset,
        aoa = aoa,
        noise_power = noise_power,
        plot_signal = True
    )

    
    # DAQ INPUT
    sampling_freq = 2_400_000
    vfo_bw = 12_500
    freq_offset = 50_000
    center_freq = 433_920_000
    default_fir_order_factor = 4
    dsp_decimation = 1
    DOA_inter_elem_space = 0.5
    array_offset = 0.0
    DOA_expected_num_of_sources = 1
    
    doa = DOA.DOA(
        signals = signal.iq,
        sampling_freq = sampling_freq,
        vfo_bw = vfo_bw,
        freq_offset = freq_offset,
        center_freq = center_freq
    )
    
    doa.run()
    
    
run()
    
    