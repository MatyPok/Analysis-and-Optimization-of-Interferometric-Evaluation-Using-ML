import numpy as np
from scipy.ndimage import gaussian_filter

from time import time

from polynomials import Zernikes
from coefficients import ZernikesCoefficients
from noise import NoiseGeneration

import json
import os

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


#Loading paramters from extern json config file
def load_config(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)

        return config
    

wdir = os.getcwd()
#config = load_confing(os.path.join(wdir, "src", "data_generation", "config_0.json"))
config = load_config(os.path.join(wdir,"config_0.json"))


#wavelength = 632.8e-9  # Vlnová délka (He-Ne laser) v metrech
wavelength = config["wavelength"]
wavelength = 1
#size = 1024  # Velikost snímku (101x101 pixelů)
size = config["image_size"]
#phase_max = 6 * np.pi  # Maximální fázový posun
phase_max = config["max_phase_times_pi"]

# Parametry šumu z konfiguračního souboru
noise_params = config.get("noise_params", {})
apply_noise = noise_params.get("apply_noise")
sigma1_gaussian = noise_params.get("sigma1_gaussian")
sigma2_poisson = noise_params.get("sigma2_poisson")
mu3_dark_current = noise_params.get("mu3_dark_current")
sigma3_dark_current = noise_params.get("sigma3_dark_current")
sigma_speckle = noise_params.get("sigma_speckle")
sigma_phase = noise_params.get("sigma_phase")

snr_sensor_db = noise_params.get("snr_sensor_db")
snr_interf_db = noise_params.get("snr_interf_db")
var_snr = noise_params.get("var_snr")

dust_count = (0, 6)
dust_radius = (1, 5)
dust_strength = 0.5

num_of_samples = config["num_samples"]

# Výpočet vlnového čísla
#k = 2 * np.pi / wavelength
k = 1

# Vytvoření souřadnicové mřížky s počátkem ve středu
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)


# Vzdálenost od středu (polární souřadnice)
R = np.sqrt(X**2 + Y**2)

# Polární úhel
theta = np.arctan2(Y,X)
 

# Normalizace vzdálenosti pro rozsah fáze 0 až 6*pi

#R = R / np.max(R)  # Normalizujeme na interval [0,1]
R = np.clip(R / np.max(R), 0, 1)

# Výpočet radiální části Zernikeho polynomu R_n^m(r)
zernike_polynomials = Zernikes(R, theta).zernike_array()
# Konec statické části
###############################################################

def GaussianIntensity(size, variance):
    sigma = np.sqrt(variance)
    a = np.ones((size, size))

    return gaussian_filter(a, sigma=sigma)


def apply_random_carrier_frequency_tilt(coefficients):
    # Random carrier frequency tilt (FTM setup).
        # Magnitude and angle are randomized per sample so the model cannot
        # overfit to a specific fringe orientation or spacing.
        carrier_mag = np.random.uniform(40,60) * wavelength
        #carrier_mag = np.random.uniform(0,0) * wavelength
        carrier_angle = np.random.uniform(0, 2 * np.pi)
        carrier_z2 = carrier_mag * np.sin(carrier_angle)  # tilt-y (OSA mode 1)
        carrier_z3 = carrier_mag * np.cos(carrier_angle)  # tilt-x (OSA mode 2)

        # Total tilt = carrier + aberration tilt. Passed to model so it can
        # resolve the cos(phase) = cos(-phase) sign ambiguity.
        total_z2 = coefficients[0] + carrier_z2
        total_z3 = coefficients[1] + carrier_z3

        # Build phase with total tilt included
        phase_coefficients = coefficients.copy()
        phase_coefficients[0] = total_z2
        phase_coefficients[1] = total_z3

        return phase_coefficients

def apply_fft_on_intens(intens):

    fft = np.fft.fftshift(np.fft.fft2(intens))
    fft_mag = np.abs(fft)
    fft_log = np.log(fft_mag + 1e-8)

    fft_log_minus_dc = fft_log.copy()

    h, w = fft_log.shape
    fft_log_minus_dc[h//2, w//2] = 0

    fft_norm = (fft_log - np.mean(fft_log)) / (np.std(fft_log) + 1e-8)
    fft_norm_minus_dc = (fft_log_minus_dc - np.mean(fft_log_minus_dc)) / (np.std(fft_log_minus_dc) + 1e-8)
    
    return fft_norm, fft_norm_minus_dc


def Generate(coefficients):
    """
    Generuje dvojici interferogramů (piston=0 a piston=1) pro danou sadu koeficientů.
    Vrací numpy pole o tvaru (výška, šířka, 2).
    """

    coefficients = apply_random_carrier_frequency_tilt(coefficients)

    highest_zernike_index = len(coefficients) + 1  # +1 protože nezahrnujeme piston (Z_0^0)

    # Zde změna, abychom počítali vlnoplochu bez pistonu, nepočítáme tedy první Zernikeho polynom (piston term, Z_0^0)
    W = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_polynomials[1:highest_zernike_index], axis=0)


    phase_circular = k * W

    #intensity_circular = GaussianIntensity(size=size, variance=0.2) + 0.5 * np.cos(phase_circular)
    intensity_circular = 0.5 + 0.5 * np.cos(phase_circular)

    # 5. Aplikace šumu na každý obrázek zvlášť (pokud je zapnuto)
    if apply_noise:
        noise_generator = NoiseGeneration(
            dust_count_range=dust_count,
            dust_radius_range=dust_radius,
            dust_strength=dust_strength,
            snr_sensor_db=snr_sensor_db,
            snr_interf_db=snr_interf_db,
            var_snr=var_snr,
            sigma_read=sigma1_gaussian,
            sigma_poisson=sigma2_poisson,
            mu_dark=mu3_dark_current,
            sigma_dark=sigma3_dark_current,
            sigma_speckle=sigma_speckle,
            sigma_phase=sigma_phase
        )

        intensity_circular = noise_generator.apply_noise(intensity_circular)

    # 6. Maskování intenzity mimo kruh pro oba obrázky
    mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)
    intensity_circular *= mask

    #fft_norm, fft_norm_minus_dc = apply_fft_on_intens(intensity_circular)

    #return intensity_circular, phase_circular, fft_norm, fft_norm_minus_dc, [coefficients[0], coefficients[1]], coefficients[2:]
    return intensity_circular, [coefficients[0], coefficients[1]], coefficients[2:]


def main(n):
    folder = "output/"
    os.makedirs(folder, exist_ok=True)

    start = time()
    
    # Generování koeficientů, kde je piston implicitně nastaven na 0
    # dle vaší poznámky v původním kódu.
    coefficients = [ZernikesCoefficients(max_value=wavelength).generate() for _ in range(n)]

    # Paralelní generování dat. Funkce Generate nyní vrací dvojice obrázků.
    with ProcessPoolExecutor() as executor:
        intens_phases_and_such = list(tqdm(executor.map(Generate, coefficients), total=len(coefficients)))

    # Rozdělení výsledků na intenzity, fáze, carrier frekvence a koeficienty)
    #intens, phases, fft_norm, fft_norm_minus_dc, tilt, target_coefficients = map(np.array, zip(*intens_phases_and_such))
    intens, tilt, target_coefficients = map(np.array, zip(*intens_phases_and_such))

    end = time()
    print(f"Execution time: {end - start} seconds")
    
    # Uložení pole interferogramů do souboru
    intens_filename = config["intens_filename"]
    np.savez_compressed(folder + intens_filename, np.array(intens)) # Použití komprese je vhodné pro velká data
    os.chmod(folder + intens_filename + ".npz", 0o664)

    # Uložení fází do souboru
    phases_filename = config["phases_filename"]
    #np.savez_compressed(folder + phases_filename, np.array(phases))
    #os.chmod(folder + phases_filename + ".npz", 0o664)

    # Uložení fft do souboru
    #fft_filename = config["fft_filename"]
    #np.savez_compressed(folder + fft_filename, np.array(fft_norm))
    #os.chmod(folder + fft_filename + ".npz", 0o664)

    # Uložení fft bez dc do souboru
    #fft_no_dc_filename = config["fft_no_dc_filename"]
    #np.savez_compressed(folder + fft_no_dc_filename, np.array(fft_norm_minus_dc))
    #os.chmod(folder + fft_no_dc_filename + ".npz", 0o664)

    # Uložení carrier frekvencí do souboru
    tilt_filename = config["tilt_filename"]
    np.savez_compressed(folder + tilt_filename, np.array(tilt))
    os.chmod(folder + tilt_filename + ".npz", 0o664)

    # Uložení koeficientů do souboru
    coeff_filename = config["coeff_filename"]
    np.savez_compressed(folder + coeff_filename, np.array(target_coefficients))
    os.chmod(folder + coeff_filename + ".npz", 0o664)

    print("Data saved.")
    print(f"Intens shape: {intens.shape} - {np.array(intens).shape}")
    #print(f"Phases shape: {phases.shape} - {np.array(phases).shape}")
    #print(f"FFT shape: {fft_norm.shape} - {np.array(fft_norm).shape}")
    #print(f"FFT no dc shape: {fft_norm_minus_dc.shape} - {np.array(fft_norm_minus_dc).shape}")
    print(f"Tilt shape: {tilt.shape} - {np.array(tilt).shape}")
    print(f"Target coeffs. shape: {target_coefficients.shape} - {np.array(target_coefficients).shape}")

    


if __name__ == "__main__":
    main(num_of_samples)

