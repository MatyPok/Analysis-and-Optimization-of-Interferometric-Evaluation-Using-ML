import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os

# Definice Zernikových polynomů
def zernike_radial(n, m, rho):
    R = 0
    m = abs(m)
    for s in range((n - m) // 2 + 1):
        num = (-1)**s * sp.factorial(n - s)
        denom = sp.factorial(s) * sp.factorial((n + m)//2 - s) * sp.factorial((n - m)//2 - s)
        R += (num / denom) * rho**(n - 2*s)
    return sp.simplify(R)


def zernike_polynomial(n, m, rho, theta):
    if m == 0:
        return zernike_radial(n, m, rho)
    elif m > 0:
        return zernike_radial(n, m, rho) * sp.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * sp.sin(-m * theta)

# Definici vlnoplochy
def generate_wavefront(size, index, coefficient):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    wavefront = np.zeros_like(X)

    n, m = zernike_modes[index]
    zernike_func = sp.lambdify((sp.Symbol('rho'), sp.Symbol('theta')),
                               zernike_polynomial(n, m, sp.Symbol('rho'), sp.Symbol('theta')),
                               'numpy')
    wavefront += coefficient * np.where(rho <= 1, zernike_func(rho, theta), 0)
    return wavefront

# Generace interferenčních záznamů s gauss šumem
def generate_interferogram(wavefront, wavelength=0.6328, snr_db=None):
    phase = (2 * np.pi / wavelength) * wavefront
    interferogram = 0.5 * (1 + np.cos(phase))
    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(interferogram ** 2)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), interferogram.shape)
        interferogram += noise
        interferogram = np.clip(interferogram, 0, 1)
    return interferogram

# Řezení zerniků a parametry interferogramů
zernike_modes = [
    (0, 0),      # Z1  Piston
    (1, -1),     # Z2  Tilt y
    (1, 1),      # Z3  Tilt x
    (2, 0),      # Z4  Defocus
    (2, -2),     # Z5  Astig y
    (2, 2),      # Z6  Astig x
    (3, -1),     # Z7  Koma y
    (3, 1),      # Z8  Koma x
    (4, 0),      # Z9  Spherical
    (3, -3),     # Z10 Trefoil y
    (3, 3),      # Z11 Trefoil x
    (4, -2),     # Z12 2nd order Astig y
    (4, 2),      # Z13 2nd order Astig x
    (5, -1),     # Z14 2nd order Koma y
    (5, 1),      # Z15 2nd order Koma x
    (6, 0),      # Z16 Higher-order spherical
    (5, -3),     # Z17 2nd order trefoil y
    (5, 3),      # Z18 2nd order trefoil x
    (6, -2),     # Z19 Astig 3rd y
    (6, 2),      # Z20 Astig 3rd x
    (7, -1),     # Z21 Koma 3rd y
    (7, 1),      # Z22 Koma 3rd x
    (8, 0),      # Z23 Spherical 3rd
    (6, -4),     # Z24 Tetrafoil y
    (6, 4),      # Z25 Tetrafoil x
    (7, -3),     # Z26 Trefoil 3rd y
    (7, 3),      # Z27 Trefoil 3rd x
    (8, -2),     # Z28 Astig 4th y
    (8, 2),      # Z29 Astig 4th x
    (9, -1),     # Z30 Koma 4th y
    (9, 1),      # Z31 Koma 4th x
    (10, 0),     # Z32 Spherical 4th
    (7, -5),     # Z33 Pentafoil y
    (7, 5),      # Z34 Pentafoil x
    (8, -4),     # Z35 Tetrafoil 2nd y
    (8, 4),      # Z36 Tetrafoil 2nd x
]


output_dir = "interferograms"
os.makedirs(output_dir, exist_ok=True)

zernike_index = 20
min_val, max_val, step_size = 0.1, 1.0, 0.1
snr_db = 40
size = 1000

# generace a uložení interfegramu (bez pistonu)
for value in np.arange(min_val, max_val + step_size, step_size):
    wf = generate_wavefront(size, zernike_index-1, value)
    interferogram = generate_interferogram(wf, snr_db=snr_db)
    filename = os.path.join(output_dir, f"interferogram_{zernike_index}_{value:.2f}_SNR{snr_db}dB.png")
    plt.imsave(filename, interferogram, cmap='gray')