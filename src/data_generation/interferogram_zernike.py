import numpy as np
import math

from time import time

import matplotlib.pyplot as plt

from polynomials import Zernikes
from coefficients import ZernikesCoefficients
from noise import NoiseGeneration

import json
import os

#Loading paramters from extern json config file
def load_confing(path: str ="config.json") -> dict:
    with open(path, 'r') as file:
        config = json.load(file)

        return config
    
start = time()

wdir = os.getcwd()
config = load_confing(os.path.join(wdir, "src", "data_generation", "config_0.json"))

#wavelength = 632.8e-9  # Vlnová délka (He-Ne laser) v metrech
wavelength = config["wavelength"]
#size = 1024  # Velikost snímku (101x101 pixelů)
size = config["image_size"]
#phase_max = 6 * np.pi  # Maximální fázový posun
phase_max = config["max_phase_times_pi"]

# snr_db = config["snr_db"]  # SNR v dB, None pokud není potřeba přidávat šum
snr_db = config["snr_db"]
# Parametry šumu z konfiguračního souboru
noise_params = config.get("noise_params", {})
apply_noise = noise_params.get("apply_noise", True)
sigma1_gaussian = noise_params.get("sigma1_gaussian", 0.03)
sigma2_poisson = noise_params.get("sigma2_poisson", 0.03)
mu3_dark_current = noise_params.get("mu3_dark_current", 0.02)
sigma3_dark_current = noise_params.get("sigma3_dark_current", 0.03)

# Výpočet vlnového čísla
k = 2 * np.pi / wavelength


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

# Koeficienty pro jednotlivé Zernikeho polynomy
coefficients = ZernikesCoefficients(max_value=wavelength).generate()

# Dráhový posun jako součet všech polynomů násobených jejich koeficientem
# poznámka: np.dot je zde ve skutečnosti asi 2x pomalejší než np.sum
W = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_polynomials[1:], axis=0)

W_max = np.max(W)

# Výpočet fáze a intenzity
phase_circular = k * W

# Výpočet intenzity
#intensity_circular = 0.5 + 0.5*np.cos(2*math.pi * W)
intensity_circular = 0.5 + 0.5*np.cos(phase_circular)

# Pokud je zadán SNR, přidáme šum
"""
if snr_db is not None:
    snr_db *= np.random.uniform(0.5, 1.5)  # Náhodné vylepšení SNR pro variabilitu
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(intensity_circular ** 2)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), intensity_circular.shape)
    intensity_circular = np.clip(intensity_circular + noise, 0, 1)
"""

if apply_noise:
    intensity_circular = NoiseGeneration(
        snr_db=snr_db,
        sigma1_gaussian=sigma1_gaussian,
        sigma2_poisson=sigma2_poisson,
        mu3_dark_current=mu3_dark_current,
        sigma3_dark_current=sigma3_dark_current
    ).add_realistic_noise(intensity_circular)

# Maskování intenzity mimo kruh
mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)
intensity_circular *= mask

end = time()

# Uložení dráhového posunu do souboru
#intens_filename = "intensity_circular.npy"
intens_filename = config["intens_filename"]
np.save(intens_filename, intensity_circular)

# Uložení koeficientů do souboru
#coeff_filename = "coefficients.npy"
coeff_filename = config["coeff_filename"]
np.save(coeff_filename, coefficients)

# Uložení času výpočtu a dalších parametrů do souboru
#calculation_params_filename = "calculation_params.txt"
calculation_params_filename = config["calculation_params_filename"]

with open(calculation_params_filename, "w") as f:
    f.write(f"Execution time: {end - start} seconds\n")
    f.write(f"Image size: {size}x{size}\n")
    f.write(f"Wavelength: {wavelength} m\n")
    f.write(f"Max phase shift: {phase_max} rad\n")
    f.write(f"Maximum wavefront displacement (W_max): {W_max} m\n")
    f.write(f"Max Zernike coefficient value: {np.max(coefficients)}\n")
    f.write(f"Min Zernike coefficient value: {np.min(coefficients)}\n")



print("Execution time: ", end - start)



# Zobrazení fáze
plt.figure(figsize=(6,6))
plt.imshow(phase_circular, cmap='hsv', extent=[-1, 1, -1, 1])
plt.colorbar(label="Phase (radians)")
plt.title("Synthetic Circular Interferogram Phase")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()




# Zobrazení interferogramu

plt.figure(figsize=(6,6))
plt.imshow(intensity_circular, cmap='gray', extent=[-1, 1, -1, 1])
plt.colorbar(label="Intensity")
plt.title("Synthetic Circular Interferogram")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

 

# Vrátíme maximální hodnotu dráhového posunu W

#W_max = np.max(W)
#print(W_max)