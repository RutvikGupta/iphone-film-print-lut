"""Write a brighter copy of a .cube LUT by lifting shadows and midtones while keeping white at white.

Usage:
  uv run --no-project --python 3.12 --with numpy python -I brighten_lut.py IN.cube OUT.cube STOPS "TITLE"

STOPS is the shadow lift in stops; midtones rise by a bit less and white stays at 1.0.
"""

import sys
from pathlib import Path

import numpy as np


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def main() -> None:
    src, dest, stops, title = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    gain = 2.0**stops
    header, rows = [], []
    for line in src.read_text().splitlines():
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] in "-."):
            rows.append([float(v) for v in stripped.split()])
        elif stripped.startswith("TITLE"):
            header.append(f'TITLE "{title}"')
        else:
            header.append(line)
    lin = srgb_to_linear(np.clip(np.asarray(rows), 0.0, 1.0))
    # Reinhard-style lift: slope `gain` at black, exactly 1.0 at white, so highlights compress instead of clipping.
    lifted = lin * gain / (1.0 + (gain - 1.0) * lin)
    out = np.clip(linear_to_srgb(lifted), 0.0, 1.0)
    header.insert(1, f"# Brightened copy: shadows +{stops:g} stops, white unchanged.")
    body = "\n".join(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in out)
    dest.write_text("\n".join(header) + "\n" + body + "\n")
    print(dest)


if __name__ == "__main__":
    main()
