import math

def calculate_aoa(freq_hz, baseline_m, phase1_deg, phase2_deg):
    """
    Menghitung arah kedatangan sinyal (Angle of Arrival / AoA)
    menggunakan dua antenna interferometer.
    """

    # Kecepatan cahaya
    c = 3e8  

    # Panjang gelombang
    wavelength = c / freq_hz

    # Selisih fase antar antena (normalized to -180 .. +180)
    delta_phi_deg = (phase2_deg - phase1_deg + 180) % 360 - 180
    delta_phi_rad = math.radians(delta_phi_deg)

    # Rumus interferometri
    # sin(theta) = (Δφ * λ) / (2π d)
    sin_theta = (delta_phi_rad * wavelength) / (2 * math.pi * baseline_m)

    # Hindari domain error arcsin
    sin_theta = max(-1, min(1, sin_theta))

    # Hasil sudut dalam derajat
    theta_rad = math.asin(sin_theta)
    theta_deg = math.degrees(theta_rad)

    return theta_deg, delta_phi_deg, wavelength


# === Contoh penggunaan ===
if __name__ == "__main__":
    freq = 2.4e9        # 2.4 GHz
    d = 0.0625          # 6.25 cm
    phase1 = 0          # fase antena 1
    phase2 = 1       # fase antena 2

    aoa_deg, dphi, lam = calculate_aoa(freq, d, phase1, phase2)

    print(f"Panjang gelombang : {lam*100:.2f} cm")
    print(f"Selisih fase      : {dphi:.2f} derajat")
    print(f"Angle of Arrival  : {aoa_deg:.4f} derajat")
