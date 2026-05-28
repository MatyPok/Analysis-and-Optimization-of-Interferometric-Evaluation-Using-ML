import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import PIL.Image as Image
import os


input_dir = Path("data/interferogramy_meopta/Zygo/Kruh")
output_dir = Path("data/pom/extra_float32")

output_dir.mkdir(parents=True, exist_ok=True)


def read_header(file_path):

    with open(file_path, "rb") as f:

        f.seek(52)

        width = struct.unpack(">H", f.read(2))[0]
        height = struct.unpack(">H", f.read(2))[0]
        buckets = struct.unpack(">H", f.read(2))[0]
        ac_range = struct.unpack(">H", f.read(2))[0]
        ac_n_bytes = struct.unpack(">I", f.read(4))[0]

    header_size = 834 if "-" in file_path.name else 4096

    return width, height, ac_n_bytes, header_size


def extract_extra_float32(file_path, show_plot=False):

    print(f"\nSoubor: {file_path.name}")

    width, height, ac_n_bytes, header_size = read_header(file_path)

    offset = header_size + ac_n_bytes

    file_size = os.path.getsize(file_path)

    remaining = file_size - offset

    if remaining <= 0:

        print("Žádná extra data")

        return

    if remaining % 4 != 0:

        print("Extra data nejsou float32")

        return

    n_pixels = remaining // 4

    with open(file_path, "rb") as f:

        f.seek(offset)

        data = np.frombuffer(f.read(), dtype=">f4")

    print(f"float32 pixelů: {n_pixels}")

    # uložit raw
    np.save(output_dir / (file_path.stem + "_float32.npy"), data)

    # pokus vytvořit square image
    approx_size = int(np.sqrt(n_pixels))

    if approx_size * approx_size == n_pixels:

        image = data.reshape((approx_size, approx_size))

        save_image(image, file_path)

    else:

        # sparse visualization
        visualize_sparse(data, file_path)


def save_image(image, file_path):

    norm = normalize(image)

    if norm is None:
        return

    img = Image.fromarray(norm)

    path = output_dir / (file_path.stem + "_float32.png")

    img.save(path)

    print(f"Uloženo PNG: {path}")


def visualize_sparse(data, file_path):

    plt.figure()

    plt.plot(data)

    plt.title(file_path.name + " float32 data")

    plt.savefig(output_dir / (file_path.stem + "_float32_plot.png"))

    plt.close()

    print("Uložen plot")


def normalize(image):

    min_val = np.nanmin(image)

    max_val = np.nanmax(image)

    if max_val == min_val:

        return None

    norm = (image - min_val) / (max_val - min_val)

    return (norm * 255).astype(np.uint8)


# main

files = sorted(input_dir.glob("*.dat"))

print(f"Nalezeno {len(files)} souborů")

for f in files:

    extract_extra_float32(f)

print("\nHotovo")