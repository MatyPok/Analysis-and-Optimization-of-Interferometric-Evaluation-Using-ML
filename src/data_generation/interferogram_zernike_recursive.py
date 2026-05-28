import numpy as np
import math

import matplotlib.pyplot as plt


wavelength = 632.8e-9  # Vlnová délka (He-Ne laser) v metrech
size = 101  # Velikost snímku (101x101 pixelů)
phase_max = 6 * np.pi  # Maximální fázový posun

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


# Funkce pro výpočet radiální části Zernikeho polynomu R_n^m(r)
def Rmk(n, m, r):
    """
    Vypočítá radiální část Zernikeho polynomu R_n^m(r).
    """
    if m < 0:
        m = -m  # Zernikeho polynomy pro záporné m jsou stejné jako pro kladné m
    Rnm = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        # Výpočet koeficientů pro radiální část podle vzorce pro Zernikeho polynomy
        coeff = ((-1)**k * math.factorial(n-k) /
                 (math.factorial(k) * math.factorial((n+m)//2 - k) * math.factorial((n-m)//2 - k)))
        Rnm += coeff * r**(n - 2*k)
    return Rnm


def W(n, m, r, theta):
    Cmk = Dmk = 0.2
    W = Cmk*Rmk(n,m,r)*np.cos(m*theta) + Dmk*Rmk(n,m,r)*np.sin(m*theta)
    return W



# W jako rozvoj Zernikeho polynomů
#W = (phase_max / k) * R_norm  # Dráhový posun

W = W(2,0,R,theta)

# Výpočet fáze a intenzity

phase_circular = phase_max * W

intensity_circular = 0.5 * (1 + np.cos(phase_circular))


# Zobrazení interferogramu

plt.figure(figsize=(6,6))

plt.imshow(intensity_circular, cmap='gray', extent=[-1, 1, -1, 1])

plt.colorbar(label="Intensity")

plt.title("Synthetic Circular Interferogram")

plt.xlabel("X")

plt.ylabel("Y")

plt.show()

 

# Vrátíme maximální hodnotu dráhového posunu W

W_max = np.max(W)

print(W_max)