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



def Generate(coefficients):
    """
    Generuje dvojici interferogramů (piston=0 a piston=1) pro danou sadu koeficientů.
    Vrací numpy pole o tvaru (výška, šířka, 2).
    """
    # Zde změna, abychom počítali vlnoplochu bez pistonu, nepočítáme tedy první Zernikeho polynom (piston term, Z_0^0)
    W = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_polynomials[1:], axis=0)


    phase_circular = k * W

    #intensity_circular = GaussianIntensity(size=size, variance=0.2) + 0.5 * np.cos(phase_circular)
    intensity_circular = 0.5 + 0.5 * np.cos(phase_circular)

    # 5. Aplikace šumu na každý obrázek zvlášť (pokud je zapnuto)
    if apply_noise:
        noise_generator = NoiseGeneration(

            snr_db=snr_db,
            sigma1_gaussian=sigma1_gaussian,
            sigma2_poisson=sigma2_poisson,
            mu3_dark_current=mu3_dark_current,
            sigma3_dark_current=sigma3_dark_current,

            dust_count_range = (0, 0),
            dust_radius_range = (1, 5)
        )

        intensity_circular = noise_generator.add_realistic_noise(intensity_circular)

    # 6. Maskování intenzity mimo kruh pro oba obrázky
    mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)
    intensity_circular *= mask


    return intensity_circular, phase_circular


def main(n):
    folder = "output/"
    os.makedirs(folder, exist_ok=True)

    start = time()
    
    # Generování koeficientů, kde je piston implicitně nastaven na 0
    # dle vaší poznámky v původním kódu.
    coefficients = [ZernikesCoefficients(max_value=wavelength).generate() for _ in range(n)]

    # Paralelní generování dat. Funkce Generate nyní vrací dvojice obrázků.
    with ProcessPoolExecutor() as executor:
        intens_phases = list(tqdm(executor.map(Generate, coefficients), total=len(coefficients)))

    # Rozdělení výsledků na intenzity a fáze)
    
    # Rozdělení výsledků na intenzity, fáze, carrier frekvence a koeficienty)
    intens, phases = map(np.array, zip(*intens_phases))

    end = time()
    print(f"Execution time: {end - start} seconds")
    
    # Uložení pole interferogramů do souboru
    intens_filename = config["intens_filename"]
    np.savez_compressed(folder + intens_filename, np.array(intens)) # Použití komprese je vhodné pro velká data

    # Uložení fází do souboru (pokud je potřeba)
    phases_filename = config["phases_filename"]
    np.savez_compressed(folder + phases_filename, np.array(phases))
    
    # Uložení koeficientů do souboru
    coeff_filename = config["coeff_filename"]
    np.savez_compressed(folder + coeff_filename, np.array(coefficients))

    print(f"Data uložena. Tvar pole s interferogramy: {intens.shape}")


if __name__ == "__main__":
    main(num_of_samples)

