import numpy as np
import math

from time import time

from polynomials import Zernikes
from coefficients import ZernikesCoefficients
from noise import NoiseGeneration

import json
import os

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


#Loading paramters from extern json config file
def load_confing(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)

        return config
    

wdir = os.getcwd()
#config = load_confing(os.path.join(wdir, "src", "data_generation", "config_0.json"))
config = load_confing(os.path.join(wdir,"config_0.json"))


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


num_of_samples = config["num_samples"]

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
# Konec statické části
###############################################################


def Generate(coefficients):
    """
    Generuje dvojici interferogramů (piston=0 a piston=1) pro danou sadu koeficientů.
    Vrací numpy pole o tvaru (výška, šířka, 2).
    """
    # 1. Výpočet základní vlnoplochy W (odpovídá piston=0)
    # Zde změna, abychom počítali vlnoplochu bez pistonu, nepočítáme tedy první Zernikeho polynom (piston term, Z_0^0)
    W_piston0 = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_polynomials[1:], axis=0)

    # 2. Vytvoření vlnoplochy pro piston=1
    # Přičteme první Zernikeho polynom (piston term, Z_0^0), který je polem jedniček.
    # Toto je efektivnější než měnit koeficient a počítat celý součet znovu.
    W_piston1 = W_piston0 + zernike_polynomials[0]

    # 3. Výpočet fáze pro oba případy
    phase_piston0 = k * W_piston0
    phase_piston1 = k * W_piston1

    # 4. Výpočet intenzity pro oba případyzernike_polynomials[0]
    intensity_piston0 = 0.5 + 0.5 * np.cos(phase_piston0)
    intensity_piston1 = 0.5 + 0.5 * np.cos(phase_piston1)

    # 5. Aplikace šumu na každý obrázek zvlášť (pokud je zapnuto)
    if apply_noise:
        noise_generator = NoiseGeneration(
            snr_db=snr_db,
            sigma1_gaussian=sigma1_gaussian,
            sigma2_poisson=sigma2_poisson,
            mu3_dark_current=mu3_dark_current,
            sigma3_dark_current=sigma3_dark_current
        )
        # Je důležité aplikovat šum na každý obrázek nezávisle,
        # protože šum je náhodný proces.
        intensity_piston0 = noise_generator.add_realistic_noise(intensity_piston0)
        intensity_piston1 = noise_generator.add_realistic_noise(intensity_piston1)

    # 6. Maskování intenzity mimo kruh pro oba obrázky
    mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)
    intensity_piston0 *= mask
    intensity_piston1 *= mask

    # 7. Spojení obou obrázků do jednoho pole s tvarem (výška, šířka, 2)
    interferogram_pair = np.stack([intensity_piston0, intensity_piston1], axis=-1)

    return interferogram_pair


def main(n):
    folder = "output/"
    os.makedirs(folder, exist_ok=True)

    start = time()
    
    # Generování koeficientů, kde je piston implicitně nastaven na 0
    # dle vaší poznámky v původním kódu.
    coefficients = [ZernikesCoefficients(max_value=wavelength).generate() for _ in range(n)]

    # Paralelní generování dat. Funkce Generate nyní vrací dvojice obrázků.
    with ProcessPoolExecutor() as executor:
        intensity_pairs = list(tqdm(executor.map(Generate, coefficients), total=len(coefficients)))

    end = time()
    print(f"Execution time: {end - start} seconds")

    # Převod seznamu výsledků na jedno velké numpy pole.
    # Výsledný tvar bude (num_of_samples, height, width, 2)
    final_intensity_array = np.array(intensity_pairs)
    
    # Uložení pole interferogramů do souboru
    intens_filename = config["intens_filename"]
    np.savez_compressed(folder + intens_filename, final_intensity_array) # Použití komprese je vhodné pro velká data
    
    # Uložení koeficientů do souboru
    coeff_filename = config["coeff_filename"]
    np.savez_compressed(folder + coeff_filename, np.array(coefficients))

    print(f"Data uložena. Tvar pole s interferogramy: {final_intensity_array.shape}")


if __name__ == "__main__":
    main(num_of_samples)

