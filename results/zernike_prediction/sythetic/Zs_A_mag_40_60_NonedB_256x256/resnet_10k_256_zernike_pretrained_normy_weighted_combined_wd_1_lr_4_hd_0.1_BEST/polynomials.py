import numpy as np

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
        self.R36 =  252* R**10 - 630 * R**8 + 560 * R**6 - 210 * R**4 + 30 * R**2 - 1

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
            self.R30, self.R31, self.R32, self.R33, self.R34, self.R35, self.R36
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