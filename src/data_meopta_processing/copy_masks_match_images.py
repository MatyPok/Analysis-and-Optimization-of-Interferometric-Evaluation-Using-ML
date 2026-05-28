#!/usr/bin/env python3
"""Copy images (optional) and copy mask files renamed to match image filenames.

Usage examples:
  python src/data_meopta_processing/copy_masks_match_images.py \
    --source-root data/interferogramy_meopta/Interferogramy_kontrast \
    --image-dirs Kruh Obdelnik \
    --out-images out/images --out-masks out/masks

Behavior:
  - If no mask files are provided, the script will try to autodetect files
    in the source root containing 'maska' in their name.
  - If the number of mask files equals the number of image dirs, they are
    mapped by index order. Otherwise a single mask is reused for all images
    unless an explicit mapping is provided with --map.
  - --map entries have the form DirName=maskfile and may be repeated.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def find_masks_in_root(root: Path) -> List[Path]:
    masks = [p for p in root.iterdir() if p.is_file() and 'maska' in p.name.lower()]
    return masks


def list_images(dirp: Path) -> List[Path]:
    if not dirp.exists() or not dirp.is_dir():
        return []
    return [p for p in dirp.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def parse_map_entries(entries: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for e in entries:
        if '=' in e:
            k, v = e.split('=', 1)
            mapping[k] = v
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description='Copy images and duplicate masks renamed to match images')
    p.add_argument('--source-root', default='data/interferogramy_meopta/Interferogramy_kontrast')
    p.add_argument('--image-dirs', nargs='+', required=True,
                   help='Subdirectories (under source-root) or absolute dirs with images')
    p.add_argument('--mask-files', nargs='*', help='Mask file paths (absolute or relative to source-root)')
    p.add_argument('--map', nargs='*', default=[],
                   help='Optional mappings like DirName=maskfile to assign specific mask to a dir')
    p.add_argument('--out-images', default=None, help='Destination directory for images (optional)')
    p.add_argument('--out-masks', required=True, help='Destination directory for mask copies')
    args = p.parse_args()

    source_root = Path(args.source_root)
    image_dirs = [Path(d) if Path(d).is_absolute() else source_root / d for d in args.image_dirs]
    out_images = Path(args.out_images) if args.out_images else None
    out_masks = Path(args.out_masks)
    out_masks.mkdir(parents=True, exist_ok=True)
    if out_images:
        out_images.mkdir(parents=True, exist_ok=True)

    # Resolve mask files: provided or autodetect
    mask_paths: List[Path] = []
    if args.mask_files:
        for m in args.mask_files:
            mp = Path(m)
            if not mp.is_absolute():
                cand = source_root / m
                mp = cand if cand.exists() else Path(m)
            if not mp.exists():
                raise FileNotFoundError(f"Mask file not found: {m}")
            mask_paths.append(mp)
    else:
        mask_paths = find_masks_in_root(source_root)

    if not mask_paths:
        raise SystemExit('No mask files found; provide --mask-files or place mask files in source root')

    # Build mapping from directory basename -> mask path
    user_map = parse_map_entries(args.map)

    dir_to_mask: Dict[Path, Path] = {}
    # First try user map by basename
    for d in image_dirs:
        key = d.name
        if key in user_map:
            mp = Path(user_map[key])
            if not mp.is_absolute():
                mp = source_root / mp
            if not mp.exists():
                raise FileNotFoundError(f"Mapped mask not found: {mp}")
            dir_to_mask[d] = mp

    # If not mapped, use 1:1 by index if counts match
    unmapped_dirs = [d for d in image_dirs if d not in dir_to_mask]
    if len(unmapped_dirs) and len(mask_paths) == len(unmapped_dirs) and not dir_to_mask:
        for d, m in zip(image_dirs, mask_paths):
            dir_to_mask[d] = m

    # If still unmapped, but single mask available, reuse it
    if len(unmapped_dirs) and len(mask_paths) >= 1:
        default_mask = mask_paths[0]
        for d in unmapped_dirs:
            if d not in dir_to_mask:
                dir_to_mask[d] = default_mask

    # Now process
    for d in image_dirs:
        images = list_images(d)
        print(f'Processing dir: {d} -> {len(images)} images')
        if not images:
            continue
        mask_for_dir = dir_to_mask.get(d)
        if mask_for_dir is None:
            print(f'  Warning: no mask assigned for {d}; skipping')
            continue
        for img in images:
            # copy image if requested
            if out_images:
                dest_img = out_images / img.name
                shutil.copy2(img, dest_img)
            # copy and rename mask: use image base name + mask extension
            img_base = img.stem
            mask_ext = mask_for_dir.suffix
            dest_mask = out_masks / (img_base + mask_ext)
            shutil.copy2(mask_for_dir, dest_mask)

    print('Done.')


if __name__ == '__main__':
    main()
