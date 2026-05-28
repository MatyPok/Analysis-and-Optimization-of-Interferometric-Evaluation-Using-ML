import numpy as np

class NoiseGeneration():
    def __init__(self,
                 dust_count_range,
                 dust_radius_range,
                 dust_strength,

                 # SNR controls
                 snr_sensor_db: float = None,
                 snr_interf_db: float = None,
                 var_snr: bool = False,

                 # sensor noise params
                 sigma_read: float = 0.03,
                 sigma_poisson: float = 0.03,
                 mu_dark: float = 0.02,
                 sigma_dark: float = 0.03,

                 # interferometric noise params
                 sigma_speckle: float = 0.1,
                 sigma_phase: float = 0.05
                 ) -> None:

        self.dust_count_range = dust_count_range
        self.dust_radius_range = dust_radius_range
        self.dust_strength = dust_strength

        if var_snr:
            self.snr_sensor_db = np.random.uniform(snr_sensor_db-10, snr_sensor_db+10)
            self.snr_interf_db = np.random.uniform(snr_interf_db-10, snr_interf_db+10)
        else:
            self.snr_sensor_db = snr_sensor_db
            self.snr_interf_db = snr_interf_db

        self.sigma_read = sigma_read
        self.sigma_poisson = sigma_poisson
        self.mu_dark = mu_dark
        self.sigma_dark = sigma_dark

        self.sigma_speckle = sigma_speckle
        self.sigma_phase = sigma_phase

    # -----------------------------
    # Interferometric noise
    # -----------------------------
    def generate_interferometric_noise(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape

        # speckle (multiplicative)
        speckle = 1 + np.random.normal(0, self.sigma_speckle, (h, w))
        I_speckle = image * speckle

        # phase-like noise (multiplicative perturbation)
        phase_noise = np.random.normal(0, self.sigma_phase, (h, w))
        I_phase = I_speckle + image * phase_noise

        return I_phase - image  # return ONLY noise component

    # -----------------------------
    # Sensor noise
    # -----------------------------
    def generate_sensor_noise(self, image: np.ndarray) -> np.ndarray:

        h, w = image.shape

        # read noise
        G1 = np.random.normal(0, self.sigma_read, (h, w))

        # Poisson approx
        I_pos = np.maximum(image, 0)
        G2 = np.random.normal(0, self.sigma_poisson, (h, w))
        poisson = np.sqrt(I_pos + 1e-12) * G2

        # dark current
        G3 = np.random.normal(self.mu_dark, self.sigma_dark, (h, w))

        return G1 + poisson + G3

    # -----------------------------
    # SNR scaling utility
    # -----------------------------
    def scale_noise_to_snr(self, signal: np.ndarray, noise: np.ndarray, snr_db: float):

        if snr_db is None:
            return noise

        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)

        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear

        scale = np.sqrt(target_noise_power / (noise_power + 1e-12))

        return noise * scale

    # -----------------------------
    # Dust
    # -----------------------------
    def add_dust_particles(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        h, w = image.shape
        num_particles = rng.integers(self.dust_count_range[0],
                                    self.dust_count_range[1] + 1)

        result = image.copy()
        yy, xx = np.ogrid[:h, :w]

        for _ in range(num_particles):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)
            r = rng.integers(self.dust_radius_range[0],
                             self.dust_radius_range[1] + 1)
            # Scaling radius by size of image to maintain relative size across different resolutions
            r = int(r * (h / 100))
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = dist_sq <= r**2
            result[mask] *= rng.uniform(1 - self.dust_strength, 1.0)

        return result

    # -----------------------------
    # Full pipeline
    # -----------------------------
    def apply_noise(self, image: np.ndarray) -> np.ndarray:

        clean = image.copy()

        snr_interf_db = self.snr_interf_db
        snr_sensor_db = self.snr_sensor_db

        if snr_interf_db is not None:
            # --- interferometric noise ---
            interf_noise = self.generate_interferometric_noise(clean)
            interf_noise = self.scale_noise_to_snr(clean, interf_noise, self.snr_interf_db)

            I_after_interf = clean + interf_noise
        else:
            I_after_interf = clean

        if snr_sensor_db is not None:
            # --- sensor noise ---
            sensor_noise = self.generate_sensor_noise(I_after_interf)
            sensor_noise = self.scale_noise_to_snr(I_after_interf, sensor_noise, self.snr_sensor_db)

            noisy = I_after_interf + sensor_noise
        else:
            noisy = I_after_interf

        # --- dust ---
        noisy = self.add_dust_particles(noisy)

        return np.clip(noisy, 0, 1)
    

if __name__ == "__main__":

    import json

    #Loading paramters from extern json config file
    def load_config(path="config.json"):
        with open(path, 'r') as file:
            config = json.load(file)

        return config

    config = load_config("src/data_generation/config_0.json")

    noise_params = config.get("noise_params", {})
    apply_noise = noise_params.get("apply_noise")
    sigma1_gaussian = noise_params.get("sigma1_gaussian")
    sigma2_poisson = noise_params.get("sigma2_poisson")
    mu3_dark_current = noise_params.get("mu3_dark_current")
    sigma3_dark_current = noise_params.get("sigma3_dark_current")
    sigma_speckle = noise_params.get("sigma_speckle")
    sigma_phase = noise_params.get("sigma_phase")

    var_snr = noise_params.get("var_snr")
    snr_sensor_db = noise_params.get("snr_sensor_db")
    snr_interf_db = noise_params.get("snr_interf_db")

    # Příklad použití
    noise_gen = NoiseGeneration(
        dust_count_range=(2, 6),
        dust_radius_range=(2, 7),
        dust_strength=0.5,
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

    # Vytvoření čistého testovacího obrázku (kruh s gradientem)
    clean_image = np.ones((256, 256), dtype=np.float32)

    noisy_image = noise_gen.apply_noise(clean_image)

    # Zobrazit výsledky (vyžaduje matplotlib)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5,5))
    plt.title("Noisy Image")
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.show()