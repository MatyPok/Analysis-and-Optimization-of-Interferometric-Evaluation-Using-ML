import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import PIL.Image as Image

# cesta
file_path = "data/interferogramy_meopta/Zygo/Oval/RWE-56311-PLANPLATE 45.dat"

file = os.path.basename(file_path).replace('.dat', '.png')
output_path = "data/pom/images/Oval/" + file

# parametry hlavičky - popis je v metropro reference guide, popř. na forech
with open(file_path, "rb") as f:
    f.seek(52)
    width = struct.unpack(">H", f.read(2))[0]
    height = struct.unpack(">H", f.read(2))[0]
    n_buckets = struct.unpack(">H", f.read(2))[0]
    ac_range = struct.unpack(">H", f.read(2))[0]
    ac_n_bytes = struct.unpack(">I", f.read(4))[0]

print(f"Width: {width}, Height: {height}, Buckets: {n_buckets}, Bytes: {ac_n_bytes}, Range: {ac_range}")


""" header_size je v závislosti na konkrétním souboru a jeho formátu buď 834 nebo 4096.
Zdá se že u souborů které mají "-" v názvu je to 834, u ostatních 4096."""
# import dat
#header_size = 834 if "-" in os.path.basename(file_path) else 4096
header_size = 4096
with open(file_path, "rb") as f:
    f.seek(header_size)
    data = np.frombuffer(f.read(ac_n_bytes), dtype='>u2')

#reshaping
if n_buckets > 1:
    intensity = data.reshape((height, width, n_buckets))
else:
    intensity = data.reshape((height, width))

# neplatné hodnnoty - nan
intensity = intensity.astype(np.float32)
intensity[intensity >= 65535] = np.nan

# norma
valid_mask = ~np.isnan(intensity)
min_val = np.nanmin(intensity)
max_val = np.nanmax(intensity)
normalized = (intensity - min_val) / (max_val - min_val)

# debug
print(f"Min value: {min_val}, Max value: {max_val}")
print(f"Min normalized: {np.nanmin(normalized)}, Max normalized: {np.nanmax(normalized)}")
print(f"Data type: {intensity.dtype}, Shape: {intensity.shape}")
print(f"Valid pixels: {np.sum(valid_mask)} / {intensity.size}")


# Uložit do PNG pro diagnostiku
normalized_uint8 = (normalized * 255).astype(np.uint8)
img = Image.fromarray(normalized_uint8)
img.save(output_path)
print(f"Obrázek uložen do: {output_path}")

# vykresleni - MUSÍŠ POUŽÍT normalized, ne intensity!
plt.figure(figsize=(width/100, height/100))

plt.imshow(normalized, cmap='gray', aspect='equal', vmin=0, vmax=1)
#plt.imshow(intensity, cmap='gray', aspect='equal')

plt.colorbar(label='Normalized intensity')
plt.axis('off')
plt.title(f"Interferogram ({width}x{height})")
plt.tight_layout()
#plt.savefig(file_path.replace('.dat', '_matplotlib.png'), dpi=100, bbox_inches='tight')
#print(f"Matplotlib graf uložen do: {file_path.replace('.dat', '_matplotlib.png')}")
plt.show()



