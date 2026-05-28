import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from pathlib import Path
import PIL.Image as Image


# vstupní a výstupní adresář
input_dir = Path("data/interferogramy_meopta/Zygo/Ctverec_obdelnik")
output_dir = Path("data/pom/images/Ctverec_obdelnik")

# vytvořit výstupní adresář pokud neexistuje
output_dir.mkdir(parents=True, exist_ok=True)


def process_dat_file(file_path: Path, output_dir: Path, show_plot=False):
    print(f"\nZpracovávám: {file_path.name}")

    try:
        # čtení hlavičky
        with open(file_path, "rb") as f:
            f.seek(52)
            width = struct.unpack(">H", f.read(2))[0]
            height = struct.unpack(">H", f.read(2))[0]
            n_buckets = struct.unpack(">H", f.read(2))[0]
            ac_range = struct.unpack(">H", f.read(2))[0]
            ac_n_bytes = struct.unpack(">I", f.read(4))[0]

        print(f"Width: {width}, Height: {height}, Buckets: {n_buckets}, Bytes: {ac_n_bytes}, Range: {ac_range}")

        # velikost hlavičky
        header_size = 834 if "-" in file_path.name else 4096

        # načíst data
        with open(file_path, "rb") as f:
            f.seek(header_size)
            data = np.frombuffer(f.read(ac_n_bytes), dtype='>u2')

        # reshape
        if n_buckets > 1:
            intensity = data.reshape((height, width, n_buckets))
            # pokud chceš jen první bucket:
            intensity = intensity[:, :, 0]
        else:
            intensity = data.reshape((height, width))

        # převod na float a NaN maska
        intensity = intensity.astype(np.float32)
        intensity[intensity >= 65535] = np.nan

        # normalizace
        min_val = np.nanmin(intensity)
        max_val = np.nanmax(intensity)
        normalized = (intensity - min_val) / (max_val - min_val)

        print(f"Min: {min_val}, Max: {max_val}")
        print(f"Shape: {intensity.shape}")

        # uložit PNG
        normalized_uint8 = (normalized * 255).astype(np.uint8)

        img = Image.fromarray(normalized_uint8)

        output_path = output_dir / (file_path.stem + ".png")

        img.save(output_path)

        print(f"Uloženo: {output_path}")

        # volitelné vykreslení
        if show_plot:
            plt.figure(figsize=(width/100, height/100))
            plt.imshow(normalized, cmap='gray', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(label='Normalized intensity')
            plt.axis('off')
            plt.title(file_path.name)
            plt.show()

    except Exception as e:
        print(f"CHYBA u {file_path.name}: {e}")


# projít všechny .dat soubory
dat_files = sorted(input_dir.glob("*.dat"))

print(f"Nalezeno {len(dat_files)} souborů")

for file_path in dat_files:
    process_dat_file(file_path, output_dir, show_plot=False)

print("\nHotovo.")
