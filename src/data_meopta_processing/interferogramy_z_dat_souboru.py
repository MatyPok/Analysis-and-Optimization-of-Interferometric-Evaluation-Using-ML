import numpy as np
import matplotlib.pyplot as plt
import struct
import os

# cesta (problematic file)
file_path = "data/interferogramy_meopta/Zygo/Kruh/1.dat"

def detect_and_load(file_path, candidate_header_sizes=(834, 4096)):
    """
    Robustně detekuje header_size a načte data.
    Pokusí se pro každý kandidát header_size:
      - přečíst metadata na offsetu 52
      - spočítat očekávaný počet vzorků width*height*n_buckets
      - použít reálný počet bajtů v souboru (file_size - header_size) a pokusit se data načíst
      - zkusit obě endian varianty pro data ('>u2' a '<u2')
    Vrací tuple: (header_size, metadata_dict, intensity_numpy, data_endian)
    nebo (None, None, None, None) pokud se nic nenašlo.
    """
    file_size = os.path.getsize(file_path)

    for header_size in candidate_header_sizes:
        try:
            with open(file_path, 'rb') as f:
                f.seek(52)
                width = struct.unpack('>H', f.read(2))[0]
                height = struct.unpack('>H', f.read(2))[0]
                n_buckets = struct.unpack('>H', f.read(2))[0]
                ac_range = struct.unpack('>H', f.read(2))[0]
                ac_n_bytes = struct.unpack('>I', f.read(4))[0]

            # basic sanity checks
            if not (1 <= width <= 20000 and 1 <= height <= 20000):
                continue
            if not (1 <= n_buckets <= 1000):
                continue

            # Prefer using ac_n_bytes reported in header (more reliable)
            data_bytes_to_read = int(ac_n_bytes) if ac_n_bytes > 0 else (file_size - header_size)

            actual_data_bytes = file_size - header_size
            if actual_data_bytes < 2:
                continue

            # If file doesn't contain as many bytes as header reports, skip this header_size
            if actual_data_bytes < data_bytes_to_read:
                continue

            # Read reported data bytes (use reported ac_n_bytes)
            with open(file_path, 'rb') as f:
                f.seek(header_size)
                raw = f.read(data_bytes_to_read)

            # Number of uint16 samples available
            if data_bytes_to_read % 2 != 0:
                # not even number of bytes -> invalid
                continue

            samples = data_bytes_to_read // 2

            # Try both endianness variants
            for dtype, endian_name in (('>u2', 'big'), ('<u2', 'little')):
                try:
                    arr = np.frombuffer(raw, dtype=dtype)
                except Exception:
                    continue

                if arr.size != samples:
                    continue

                # Determine n_buckets from actual sample count if possible
                if width * height == 0:
                    continue

                if samples % (width * height) != 0:
                    # not divisible -> suspicious, skip this endian
                    continue

                inferred_buckets = samples // (width * height)
                if inferred_buckets < 1 or inferred_buckets > 1000:
                    continue

                # reshape
                if inferred_buckets > 1:
                    intensity = arr.reshape((height, width, inferred_buckets))
                else:
                    intensity = arr.reshape((height, width))

                # basic content validation: check variance and non-trivial range
                if np.nanstd(intensity) == 0:
                    # flat image -> suspicious
                    continue

                metadata = {
                    'header_size': header_size,
                    'width': width,
                    'height': height,
                    'n_buckets': inferred_buckets,
                    'ac_range': ac_range,
                    'ac_n_bytes': data_bytes_to_read,
                }
                return header_size, metadata, intensity, endian_name

        except Exception:
            continue

    return None, None, None, None


# Note: tile-based reconstruction and geometric transforms removed per user request.
# We focus only on values and robust normalization; any pixel rearrangements are not attempted.


def detect_with_diagnostics(file_path, candidate_header_sizes=(834, 4096), offsets=(52, 128, 256, 512)):
    """
    Rozšířená detekce která zkouší více offsetů pro metadata a vytiskne diagnózu.
    Vrátí první rozumný výsledek nebo None.
    """
    file_size = os.path.getsize(file_path)
    candidates = []

    for offset in offsets:
        for header_size in candidate_header_sizes:
            try:
                with open(file_path, 'rb') as f:
                    f.seek(offset)
                    w = struct.unpack('>H', f.read(2))[0]
                    h = struct.unpack('>H', f.read(2))[0]
                    nb = struct.unpack('>H', f.read(2))[0]
                    ar = struct.unpack('>H', f.read(2))[0]
                    ac_nb = struct.unpack('>I', f.read(4))[0]

                # quick sanity
                if not (1 <= w <= 20000 and 1 <= h <= 20000 and 1 <= nb <= 1000):
                    continue

                data_bytes_to_read = int(ac_nb) if ac_nb > 0 else (file_size - header_size)
                if data_bytes_to_read <= 0 or (file_size - header_size) < 2:
                    continue

                # Read small sample of data for diagnostics
                with open(file_path, 'rb') as f:
                    f.seek(header_size)
                    sample = f.read(min(1024, data_bytes_to_read))

                if len(sample) < 2:
                    continue

                # test both endians
                for dtype, endian_name in (('>u2', 'big'), ('<u2', 'little')):
                    try:
                        arr = np.frombuffer(sample, dtype=dtype)
                    except Exception:
                        continue

                    stats = {
                        'offset': offset,
                        'header_size': header_size,
                        'width': w,
                        'height': h,
                        'n_buckets': nb,
                        'ac_n_bytes': ac_nb,
                        'endian': endian_name,
                        'sample_len': arr.size,
                        'sample_min': int(arr.min()) if arr.size>0 else None,
                        'sample_max': int(arr.max()) if arr.size>0 else None,
                        'sample_std': float(arr.std()) if arr.size>0 else None,
                        'sample_unique': int(len(np.unique(arr))) if arr.size>0 else 0,
                    }
                    candidates.append(stats)

            except Exception:
                continue

    # Print diagnostics sorted by sample_std (descending)
    if not candidates:
        print('❌ Žádní kandidáti pro detekci (zkuste jiné offsety/kandidáty)')
        return None, None, None, None

    candidates_sorted = sorted(candidates, key=lambda x: (x['sample_std'] if x['sample_std'] is not None else 0), reverse=True)
    print('\nDiagnostika kandidátů (top 10 podle směrodatné odchylky):')
    for c in candidates_sorted[:10]:
        print(f" offset={c['offset']:>4}, hdr={c['header_size']:>4}, endian={c['endian']:>5}, w={c['width']:>4}, h={c['height']:>4}, nb={c['n_buckets']:>3}, ac_bytes={c['ac_n_bytes']:>8}, sample_len={c['sample_len']:>4}, min={c['sample_min']:>6}, max={c['sample_max']:>6}, std={c['sample_std']:.2f}, unique={c['sample_unique']}")

    # Try to fully load using the best candidate(s)
    for c in candidates_sorted:
        try:
            hdr = c['header_size']
            with open(file_path, 'rb') as f:
                f.seek(hdr)
                raw = f.read(c['ac_n_bytes'])

            if len(raw) < 2:
                continue

            samples = len(raw) // 2
            if samples % (c['width'] * c['height']) != 0:
                continue

            inferred_buckets = samples // (c['width'] * c['height'])
            dtype = '>u2' if c['endian'] == 'big' else '<u2'
            arr = np.frombuffer(raw, dtype=dtype)
            if inferred_buckets > 1:
                intensity = arr.reshape((c['height'], c['width'], inferred_buckets))
            else:
                intensity = arr.reshape((c['height'], c['width']))

            # validate not flat
            if np.nanstd(intensity) == 0:
                continue

            # Simple acceptance: reshape according to inferred_buckets and validate
            if inferred_buckets > 1:
                intensity = arr.reshape((c['height'], c['width'], inferred_buckets))
            else:
                intensity = arr.reshape((c['height'], c['width']))

            # validate not flat
            if np.nanstd(intensity) == 0:
                continue

            metadata = {
                'header_size': hdr,
                'width': c['width'],
                'height': c['height'],
                'n_buckets': inferred_buckets,
                'ac_range': None,
                'ac_n_bytes': len(raw),
            }
            return hdr, metadata, intensity, c['endian']

        except Exception:
            continue

    return None, None, None, None


# Detekce a načtení (diagnostika)
header_size, metadata, intensity, data_endian = detect_with_diagnostics(file_path)
if header_size is None:
    raise RuntimeError(f"Nepodařilo se detekovat header_size pro {file_path}")

width = metadata['width']
height = metadata['height']
n_buckets = metadata['n_buckets']
ac_range = metadata['ac_range']
ac_n_bytes = metadata['ac_n_bytes']

print(f"Detected header_size={header_size} (data endian: {data_endian})")
print(f"Width: {width}, Height: {height}, Buckets: {n_buckets}, Bytes: {ac_n_bytes}, Range: {ac_range}")

# neplatné hodnnoty - nan (bez transformací/posunů)
intensity = intensity.astype(np.float32)
intensity[intensity >= 65535] = np.nan

# norma (robustní - použijeme percentily, aby jediné outliery nezpůsobily černý obrázek)
valid_mask = ~np.isnan(intensity)
min_val = np.nanmin(intensity)
max_val = np.nanmax(intensity)

# robustní rozsah: použijeme 0.1 a 99.9 percentil (přizpůsobitelné)
p_low = float(np.nanpercentile(intensity, 0.1))
p_high = float(np.nanpercentile(intensity, 99.9))
if p_high <= p_low:
    # fallback na min/max
    p_low = min_val
    p_high = max_val

# Ořízneme a normalizujeme podle robustního rozsahu
clipped = np.clip(intensity, p_low, p_high)
normalized = (clipped - p_low) / (p_high - p_low)

# debug
print(f"Min value: {min_val}, Max value: {max_val}")
print(f"Min normalized: {np.nanmin(normalized)}, Max normalized: {np.nanmax(normalized)}")
print(f"Data type: {intensity.dtype}, Shape: {intensity.shape}")
print(f"Valid pixels: {np.sum(valid_mask)} / {intensity.size}")

# Additional diagnostics
pcts = np.nanpercentile(intensity, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
print("Percentiles (0,0.1,1,5,25,50,75,95,99,99.9,100):", pcts)
unique_vals = np.unique(intensity[~np.isnan(intensity)])
print(f"Unique non-nan values: {len(unique_vals)} (showing up to 20):", unique_vals[:20])
zero_frac = np.sum(intensity==0) / float(np.prod(intensity.shape))
nans_frac = np.sum(np.isnan(intensity)) / float(np.prod(intensity.shape))
print(f"Fraction zeros: {zero_frac:.6f}, Fraction NaNs: {nans_frac:.6f}")

# show small sample block
try:
    h0 = min(8, intensity.shape[0])
    w0 = min(8, intensity.shape[1])
    print("Top-left 8x8 block (raw intensity):")
    print(intensity[:h0, :w0])
except Exception:
    pass

# Save debug PNG of normalized (uint8) for external inspection
from PIL import Image
os.makedirs('tmp', exist_ok=True)
debug_png = os.path.join('tmp', os.path.basename(file_path).replace('.dat', '_debug.png'))
try:
    # robust normalized image
    normalized_uint8 = (np.nan_to_num(normalized) * 255).astype(np.uint8)
    img = Image.fromarray(normalized_uint8)
    img.save(debug_png)
    print(f"Saved robust debug PNG: {debug_png}")

    # also save raw min/max normalized image for comparison
    try:
        raw_minmax = (np.nan_to_num((intensity - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
        img_raw = Image.fromarray(raw_minmax)
        raw_png = os.path.join('tmp', os.path.basename(file_path).replace('.dat', '_raw_debug.png'))
        img_raw.save(raw_png)
        print(f"Saved raw min-max debug PNG: {raw_png}")

        # side-by-side composite
        comp = Image.new('L', (img.width * 2, img.height))
        comp.paste(img_raw, (0, 0))
        comp.paste(img, (img.width, 0))
        comp_png = os.path.join('tmp', os.path.basename(file_path).replace('.dat', '_compare_debug.png'))
        comp.save(comp_png)
        print(f"Saved comparison PNG: {comp_png}")
    except Exception as e:
        print(f"Failed to save raw/comparison PNGs: {e}")
except Exception as e:
    print(f"Failed to save debug PNG: {e}")


# Uložit do PNG pro diagnostiku
#from PIL import Image
#normalized_uint8 = (normalized * 255).astype(np.uint8)
#img = Image.fromarray(normalized_uint8)
#output_path = file_path.replace('.dat', '_output.png')
#img.save(output_path)
#print(f"Obrázek uložen do: {output_path}")

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



