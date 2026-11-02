"""Apply the fitted Chromagraph .cube LUT to photos.

Usage:
  uv run --no-project --python 3.12 --with numpy --with scipy --with pillow --with pillow-heif \
    python -I apply_lut.py chromagraph_approx_33.cube photo1.heic [photo2.jpg ...]

Each photo is converted from its embedded color profile (Display P3 for iPhone photos) to sRGB,
graded with the LUT, and written next to the original as <name>_chromagraph.jpg.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageCms
from scipy.interpolate import RegularGridInterpolator

pillow_heif.register_heif_opener()
SRGB = ImageCms.createProfile("sRGB")


def read_cube(path: Path) -> np.ndarray:
    size, rows = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif line[0].isdigit() or line[0] in "-.":
            rows.append([float(v) for v in line.split()])
    lut = np.asarray(rows, dtype=np.float64)
    # .cube lists red fastest, so reshape to [blue][green][red] and reorder to [red][green][blue].
    return lut.reshape(size, size, size, 3).transpose(2, 1, 0, 3)


def to_srgb(img: Image.Image) -> Image.Image:
    icc = img.info.get("icc_profile")
    img = img.convert("RGB")
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        img = ImageCms.profileToProfile(img, src, SRGB, outputMode="RGB")
    return img


def main() -> None:
    cube, photos = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    lut = read_cube(cube)
    grid = np.linspace(0.0, 1.0, lut.shape[0])
    interp = RegularGridInterpolator((grid, grid, grid), lut)
    srgb_icc = ImageCms.ImageCmsProfile(SRGB).tobytes()
    for photo in photos:
        img = to_srgb(Image.open(photo))
        rgb = np.asarray(img, dtype=np.float64) / 255.0
        out = interp(rgb.reshape(-1, 3)).reshape(rgb.shape)
        out_img = Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))
        dest = photo.with_name(f"{photo.stem}_chromagraph.jpg")
        out_img.save(dest, quality=95, icc_profile=srgb_icc)
        print(dest)


if __name__ == "__main__":
    main()
