import numpy as np
import random

class ZernikesCoefficients():
    def __init__(self, n_coefficients: int=35, max_value: float=1, positive_only: bool=False) -> None:
        """
        Generate random Zernike coefficients.
        :param n_coefficients: number of coefficients to generate
        :param max_value: maximum value for the coefficients
        """
        self.n_coefficients = n_coefficients
        self.max_value = max_value
        self.positive_only = positive_only

    def generate(self) -> np.ndarray:
        """
        Generate random Zernike coefficients.
        :return: list of random Zernike coefficients
        """

        # Max values for Zernike coefficients up to 36 excluding piston
        # 2nd-3rd: [-4,4], 4th-6th: [-0.2,0.2], 7th-36th: [-0.02,0.02]
        #max_values = [4, 4, 0.2, 0.2, 0.2] + 30*[0.02]
        max_values = [4, 4] + [2] + 5*[1] + 6*[0.5] + 21*[0.25] # A
        #max_values = 35*[1]
        #max_values = [4, 4, 2] + 8*[0.03]
        #min_values = [-4, -4, -2] + 8*[-0.03]
        
        if self.positive_only:
            coefficients = np.array([random.uniform(0, max_values[i]*self.max_value) for i in range(self.n_coefficients)])
        else:
            coefficients = np.array([random.uniform(-max_values[i]*self.max_value, max_values[i]*self.max_value) for i in range(self.n_coefficients)])

        #coefficients = np.array([random.uniform(min_values[i]*self.max_value, max_values[i]*self.max_value) for i in range(len(max_values))])

        
        print(f"Coefficients shape (including Z2, Z3) from coefficients.py: {coefficients.shape}")
        return coefficients