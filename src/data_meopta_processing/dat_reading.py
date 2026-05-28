import numpy as np
import struct

file_path = "data/interferogramy_meopta/Zygo/Kruh/1-7.dat"

with open(file_path, "rb") as f:
    # Metadata
    f.seek(52)
    width, height, n_buckets, ac_range = struct.unpack(">HHHH", f.read(8))
    ac_n_bytes = struct.unpack(">I", f.read(4))[0]
    
    # Snímek (intensity)
    f.seek(4096)
    intensity = np.frombuffer(f.read(ac_n_bytes), dtype='>u2')
    
    # Co je za snímkem?
    offset_after_image = 4096 + ac_n_bytes
    f.seek(offset_after_image)
    remaining = f.read()
    
    # Pokus se dekódovat jako float32 (fáze)
    print(f"Zbývajících bytů: {len(remaining)}")
    
    if len(remaining) % 4 == 0:
        phase = np.frombuffer(remaining, dtype='>f4')
        print(f"Možně fáze (float32): {phase.shape}")
        print(f"Min/Max: {phase.min():.4f} / {phase.max():.4f}")
        print(f"Prvních 10 hodnot: {phase[:10]}")