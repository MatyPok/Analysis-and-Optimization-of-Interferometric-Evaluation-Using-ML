"""
Analýza struktury všech .dat souborů - zjišťuje, co se v nich nachází
"""
import numpy as np
import struct
import os
from pathlib import Path

def analyze_dat_file(file_path):
    """Analyzuje strukturu jednoho .dat souboru"""
    
    try:
        file_size = os.path.getsize(file_path)
        
        with open(file_path, "rb") as f:
            # Metadata
            f.seek(52)
            width = struct.unpack(">H", f.read(2))[0]
            height = struct.unpack(">H", f.read(2))[0]
            n_buckets = struct.unpack(">H", f.read(2))[0]
            ac_range = struct.unpack(">H", f.read(2))[0]
            ac_n_bytes = struct.unpack(">I", f.read(4))[0]
            
            # Data type
            f.seek(86)
            data_type = struct.unpack(">H", f.read(2))[0]
            
            # Offset po snímku
            header_size = 4096
            offset_after_image = header_size + ac_n_bytes
            remaining_bytes = file_size - offset_after_image
            
            # Analýza zbývajících dat
            extra_info = ""
            if remaining_bytes > 0:
                f.seek(offset_after_image)
                test_data = f.read(min(50000, remaining_bytes))
                unique_bytes = len(set(test_data))
                
                # Pokud je málo unikátních bajtů, je to pravděpodobně padding
                if unique_bytes <= 5:
                    extra_info = f"PADDING ({unique_bytes} unikátní bajty)"
                else:
                    # Pokus se detekovat data typ
                    if remaining_bytes % 4 == 0:
                        extra_info = f"FLOAT32 data? ({remaining_bytes // 4} pixelů)"
                    elif remaining_bytes % 2 == 0:
                        extra_info = f"INT16 data? ({remaining_bytes // 2} pixelů)"
                    else:
                        extra_info = f"NEZNÁMÁ data ({unique_bytes} unikátních bajtů)"
            else:
                extra_info = "NIČEHO"
            
            return {
                'file': os.path.basename(file_path),
                'size_mb': file_size / 1024 / 1024,
                'width': width,
                'height': height,
                'buckets': n_buckets,
                'data_type': data_type,
                'image_bytes': ac_n_bytes,
                'remaining_bytes': remaining_bytes,
                'extra_data': extra_info,
                'valid': True
            }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'valid': False,
            'error': str(e)
        }

def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "interferogramy_meopta/Zygo/Kruh"
    
    # Hledáme všechny .dat soubory
    dat_files = sorted(data_dir.rglob("*.dat"))
    
    if not dat_files:
        print("Nebyly nalezeny žádné .dat soubory")
        return
    
    print(f"Nalezeno {len(dat_files)} .dat souborů\n")
    print("=" * 140)
    print(f"{'Soubor':<50} {'Velikost':<10} {'Rozměry':<15} {'Buckets':<8} {'Za snímkem':<40}")
    print("=" * 140)
    
    summary_padding = 0
    summary_extra = 0
    
    for file_path in dat_files:
        result = analyze_dat_file(file_path)
        
        if result['valid']:
            print(f"{result['file']:<50} {result['size_mb']:>8.2f} MB  "
                  f"{result['width']:>4}×{result['height']:<4}  "
                  f"{result['buckets']:>7}  "
                  f"{result['extra_data']:<40}")
            
            if "PADDING" in result['extra_data']:
                summary_padding += 1
            elif "NIČEHO" not in result['extra_data']:
                summary_extra += 1
        else:
            print(f"{result['file']:<50} CHYBA: {result['error']}")
    
    print("=" * 140)
    print(f"\nSOUHRN:")
    print(f"  - Soubory jen s padding za snímkem: {summary_padding}")
    print(f"  - Soubory s potenciálními extra daty: {summary_extra}")
    print(f"  - Soubory bez ničeho navíc: {len(dat_files) - summary_padding - summary_extra}")

if __name__ == "__main__":
    main()
