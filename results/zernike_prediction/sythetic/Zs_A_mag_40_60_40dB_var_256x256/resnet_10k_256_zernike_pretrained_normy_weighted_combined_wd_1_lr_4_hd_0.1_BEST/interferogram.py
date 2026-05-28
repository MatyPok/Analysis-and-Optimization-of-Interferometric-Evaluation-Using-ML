
import numpy as np

class Interferogram():
    # Generates an interferogram from Zernike coefficients

    def __init__(self, zernikes: np.ndarray, size: int=1024, wavelength: float=632.8e-9, show: bool=False, tilt=True, piston=False) -> None:
        # Zernikes = numpy array of Zernike coefficients
        self.zernikes = zernikes
        self.size = size
        self.wavelength = wavelength
        self.show = show
        self.tilt = tilt # If tilt is false - polynomials and coefficients counted 3:N
        self.piston = piston # Primary not counted

    def wavefront(self) -> np.ndarray:
        # Generate the interferogram based on Zernike coefficients
        import numpy as np

        import matplotlib.pyplot as plt

        from interferogram import Zernikes

        wavelength = self.wavelength
        # Výpočet vlnového čísla
        k = 2 * np.pi / wavelength


        # Vytvoření souřadnicové mřížky s počátkem ve středu
        size = self.size
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
        coefficients = self.zernikes

        ###
        # Kontrola, zda je vstupní pole 1D nebo 2D
        if coefficients.ndim == 1:
            # 1D pole (jeden set Zerniků)
            coefficients_reshaped = coefficients[np.newaxis, :]
        elif coefficients.ndim == 2:
            # 2D pole (více setů Zerniků)
            coefficients_reshaped = coefficients
        else:
            raise ValueError("Input zernikes must be a 1D or 2D numpy array.")
        ###
        # Zde se provádí hlavní výpočet
        # Přidání rozměrů pro broadcasting
        # Rovnou předpokláme, že pokud nepočítáme tilt nepočítáme ani piston, proto není možnost (piston==true & tilt==false)
        if self.piston:
            W = np.sum(coefficients_reshaped[:, :, np.newaxis, np.newaxis] * zernike_polynomials, axis=1)
        else:
            if self.tilt:
                W = np.sum(coefficients_reshaped[:, :, np.newaxis, np.newaxis] * zernike_polynomials[1:], axis=1)
            else:
                W = np.sum(coefficients_reshaped[:, :, np.newaxis, np.newaxis] * zernike_polynomials[2:], axis=1)


        # Maskování intenzity mimo kruh
        #mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)
        #W *= mask

        # Zobrazení vlnoplochy pokud je show=True
        if self.show:

            # Zobrazení fáze
            plt.figure(figsize=(6,6))
            plt.imshow(W[0], cmap='hsv', extent=[-1, 1, -1, 1])
            plt.colorbar(label="Phase (radians)")
            plt.title("Synthetic Circular Interferogram Phase")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()


        return W
    
    def phase(self) -> np.ndarray:
        W = self.wavefront()
        k = 2 * np.pi / self.wavelength
        phase_circular = k * W
        return phase_circular
    
    
    def generate_interferogram(self) -> np.ndarray:
        # Generate the interferogram based on Zernike coefficients
        import numpy as np

        import matplotlib.pyplot as plt

        from interferogram import Zernikes


        # Vytvoření souřadnicové mřížky s počátkem ve středu
        size = self.size
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)

        mask = (np.sqrt(X**2 + Y**2) <= 1).astype(float)

        # Výpočet fáze a intenzity
        phase_circular = self.phase()

        # Výpočet intenzity
        intensity_circular = 0.5 + 0.5*np.cos(phase_circular)

        # Maskování intenzity mimo kruh
        intensity_circular *= mask
        phase_circular *= mask


        # Zobrazení interferogramu pokud je show=True
        if self.show:

            # Zobrazení fáze
            plt.figure(figsize=(6,6))
            plt.imshow(phase_circular[0], cmap='hsv', extent=[-1, 1, -1, 1])
            plt.colorbar(label="Phase (radians)")
            plt.title("Synthetic Circular Interferogram Phase")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()


            # Zobrazení interferogramu

            plt.figure(figsize=(6,6))
            plt.imshow(intensity_circular[0], cmap='gray', extent=[-1, 1, -1, 1])
            plt.colorbar(label="Intensity")
            plt.title("Synthetic Circular Interferogram")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()


        return intensity_circular
    



class Zernikes():
    def __init__(self, R, theta):
        """
        Zernike polynomials for circular aperture.
        :param R: radius of the circle
        :param theta: angle in radians
        """

        shape = R.shape

        self.R = R
        self.theta = theta

        self.R1 = np.ones(shape)
        self.R2 = R * np.cos(theta)
        self.R3 = R * np.sin(theta)
        self.R4 = R**2 - 1
        self.R5 = R**2 * np.cos(2 * theta)
        self.R6 = R**2 * np.sin(2 * theta)
        self.R7 = (3 * R**3 - 2 * R) * np.cos(theta)
        self.R8 = (3 * R**3 - 2 * R) * np.sin(theta)
        self.R9 = 6 * R**4 - 6 * R**2 + 1
        self.R10 = R**3 * np.cos(3 * theta)
        self.R11 = R**3 * np.sin(3 * theta)
        self.R12 = (4 * R**4 - 3 * R**2) * np.cos(2 * theta)
        self.R13 = (4 * R**4 - 3 * R**2) * np.sin(2 * theta)
        self.R14 = (10 * R**5 - 12 * R**3 + 3 * R) * np.cos(theta)
        self.R15 = (10 * R**5 - 12 * R**3 + 3 * R) * np.sin(theta)
        self.R16 = 20 * R**6 - 30 * R**4 + 12 * R**2 - 1
        self.R17 = R**4 * np.cos(4 * theta)
        self.R18 = R**4 * np.sin(4 * theta)
        self.R19 = (5 * R**5 - 4 * R**3) * np.cos(3 * theta)
        self.R20 = (5 * R**5 - 4 * R**3) * np.sin(3 * theta)
        self.R21 = (15 * R**6 - 20 * R**4 + 6 * R**2) * np.cos(2 * theta)
        self.R22 = (15 * R**6 - 20 * R**4 + 6 * R**2) * np.sin(2 * theta)
        self.R23 = (35 * R**7 - 60 * R**5 + 30 * R**3 - 4 * R) * np.cos(theta)
        self.R24 = (35 * R**7 - 60 * R**5 + 30 * R**3 - 4 * R) * np.sin(theta)
        self.R25 = 70 * R**8 - 140 * R**6 + 90 * R**4 - 20 * R**2 + 1
        self.R26 = R**5 * np.cos(5 * theta)
        self.R27 = R**5 * np.sin(5 * theta)
        self.R28 = (6 * R**6 - 5 * R**4) * np.cos(4 * theta)
        self.R29 = (6 * R**6 - 5 * R**4) * np.sin(4 * theta)
        self.R30 = (21 * R**7 - 30 * R**5 + 10 * R**3) * np.cos(3 * theta)
        self.R31 = (21 * R**7 - 30 * R**5 + 10 * R**3) * np.sin(3 * theta)
        self.R32 = (56 * R**8 - 105 * R**6 + 60 * R**4 - 10 * R**2) * np.cos(2 * theta)
        self.R33 = (56 * R**8 - 105 * R**6 + 60 * R**4 - 10 * R**2) * np.sin(2 * theta)
        self.R34 = (126 * R**9 - 280 * R**7 + 210 * R**5 - 60 * R**3 + 5 * R) * np.cos(theta)
        self.R35 = (126 * R**9 - 280 * R**7 + 210 * R**5 - 60 * R**3 + 5 * R) * np.sin(theta)

    def zernike_array(self):
        """
        Zernike polynomials for circular aperture.
        :return: array of Zernike polynomials
        """
        return np.array([
            self.R1, self.R2, self.R3, self.R4, self.R5, self.R6, self.R7, self.R8,
            self.R9, self.R10, self.R11, self.R12, self.R13, self.R14, self.R15,
            self.R16, self.R17, self.R18, self.R19, self.R20, self.R21, self.R22,
            self.R23, self.R24, self.R25, self.R26, self.R27, self.R28, self.R29,
            self.R30, self.R31, self.R32, self.R33, self.R34, self.R35
        ])
    
    """
    Example of use:
    import numpy as np
    from src.data_generation.polynomials import Zernikes

    # Definujte hodnoty R a theta
    R = 0.5  # Poloměr (např. 0.5)
    theta = np.pi / 4  # Úhel v radiánech (např. 45 stupňů)

    # Vytvořte instanci třídy Zernikes
    zernike = Zernikes(R, theta)

    # Získejte pole Zernikeových polynomů
    zernike_polynomials = zernike.zernike_array()

    # Výstup
    print("Zernike polynomials:", zernike_polynomials)
    """



# Testování třídy
# Testovací numpy pole 
"""
import random

max_values = np.array([5, 5, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

test_array = np.array([random.uniform(-max_values[i]*632.8e-9, max_values[i]*632.8e-9) for i in range(31)])

if __name__ == "__main__":
    # Testování třídy
    Interferogram(test_array[:31],show=True).generate()

"""

if __name__ == "__main__":

    import random

    # Testovací numpy pole s 3 sadami Zernikeho koeficientů
    num_sets = 3
    num_coefficients = 31

    # Pole pro nastavení limitů pro náhodné hodnoty, aby nevznikaly extrémní výsledky
    max_values = np.array([5, 5, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                          0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

    # Generování 3 sad Zernikeho koeficientů
    multi_set_zernikes = np.zeros((num_sets, num_coefficients))
    for i in range(num_sets):
        multi_set_zernikes[i, :] = [random.uniform(-max_values[j] * 632.8e-9, max_values[j] * 632.8e-9) for j in range(num_coefficients)]

    print("Generování interferogramů pro 3 sady Zernikeho koeficientů...")
    
    # Vytvoření instance třídy s 2D polem koeficientů a zobrazení výsledku
    # 'show=True' zajistí zobrazení všech tří interferogramů
    generated_interferograms = Interferogram(multi_set_zernikes,show=True).generate_interferogram()

    # Kontrola tvaru (shape) výsledného pole
    expected_shape = (num_sets, 1024, 1024)
    print(f"Tvar výsledného pole: {generated_interferograms.shape}")
    print(f"Očekávaný tvar: {expected_shape}")
    
    # Ověření, zda je tvar správný
    assert generated_interferograms.shape == expected_shape, f"Tvar pole je špatný! Očekávaný {expected_shape}, získaný {generated_interferograms.shape}"
    print("Test úspěšně dokončen. Tvar pole je správný!")

    # Zobrazení prvního interferogramu z generovaných sad
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.imshow(generated_interferograms[0], cmap='gray', extent=[-1, 1, -1, 1])
    plt.colorbar(label="Intensity")
    plt.title("First Generated Synthetic Circular Interferogram")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show() 