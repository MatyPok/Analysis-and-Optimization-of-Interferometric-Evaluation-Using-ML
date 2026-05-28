"""FFT-based tilt (Z2, Z3) extraction from interferograms.

Convention (OSA/zernikepy):
  - Z2 = index 1 = tilt-y: fringes run vertically, frequency in y-direction
  - Z3 = index 2 = tilt-x: fringes run horizontally, frequency in x-direction
"""

import numpy as np


class TiltExtractor:
    """Extract tilt coefficients (Z2, Z3) from an interferogram using FFT.

    This is a non-learned module. It finds the dominant off-center peak in the
    2D FFT spectrum to determine carrier frequency, then converts to tilt
    Zernike coefficients.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def extract(self, interferogram: np.ndarray) -> tuple[float, float]:
        """Extract tilt coefficients from a single interferogram.

        Args:
            interferogram: 2D array of shape (H, W), intensity in [0, 1].

        Returns:
            (z2, z3): Tilt-y and tilt-x Zernike coefficients (OSA order).
        """
        spectrum = np.fft.fft2(interferogram)
        spectrum_shifted = np.fft.fftshift(spectrum)
        magnitude = np.abs(spectrum_shifted)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Zero out DC component and a small region around it
        dc_radius = 3
        yy, xx = np.ogrid[:h, :w]
        dc_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= dc_radius**2
        magnitude[dc_mask] = 0.0

        # Find the dominant peak
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        peak_y, peak_x = peak_idx

        # Frequency offsets from center (in pixels)
        dy = peak_y - cy  # row offset → y-frequency (tilt-y / Z2)
        dx = peak_x - cx  # col offset → x-frequency (tilt-x / Z3)

        # The interferogram I = 0.5*(1 + cos(phase)) produces symmetric peaks.
        # cos(phase) has equal peaks at +f and -f. We resolve ambiguity by
        # picking a canonical half-plane (positive y, or positive x if dy==0).
        if dy < 0 or (dy == 0 and dx < 0):
            dy = -dy
            dx = -dx

        # Convert pixel frequency to Zernike tilt coefficient.
        # Phase from tilt: phi = coeff * Z(rho, theta).
        # Z2 = 2*rho*sin(theta), Z3 = 2*rho*cos(theta) in OSA convention.
        # The maximum phase gradient from Z2 with coeff=a is 2*a across diameter.
        # This creates fringes with frequency = a/pi across the aperture.
        # Pixel frequency f_pix = a/pi * (N/2) / N = a/(2*pi).
        # But the aperture only covers the unit disk within the image.
        # Empirically: dy/N ≈ coeff / (2*pi), so coeff ≈ 2*pi * dy/N.
        z2 = 2.0 * np.pi * dy / h  # tilt-y
        z3 = 2.0 * np.pi * dx / w  # tilt-x

        return float(z2), float(z3)

    def extract_batch(self, interferograms: np.ndarray) -> np.ndarray:
        """Extract tilts for a batch of interferograms.

        Args:
            interferograms: Array of shape (B, H, W).

        Returns:
            Array of shape (B, 2) with [z2, z3] per sample.
        """
        results = np.array([self.extract(img) for img in interferograms])
        return results
