"""Write a copy of a .cube LUT whose neutral inputs stay neutral: greys and whites lose any colour cast.

Usage:
  uv run --no-project --python 3.12 --with numpy --with colour-science python -I neutralize_lut.py IN.cube OUT.cube "TITLE"

Coloured inputs keep the LUT's colour rendering; the cast is removed in proportion to how grey the input is.
"""

import sys
from pathlib import Path

import colour
import numpy as np

# Input chroma (CIELab C*) at which the neutralizing pull has faded to about 37%.
_NEUTRAL_RADIUS = 8.0


def main() -> None:
    src, dest, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    header, rows = [], []
    size = None
    for line in src.read_text().splitlines():
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] in "-."):
            rows.append([float(v) for v in stripped.split()])
        elif stripped.startswith("TITLE"):
            header.append(f'TITLE "{title}"')
        else:
            if stripped.startswith("LUT_3D_SIZE"):
                size = int(stripped.split()[1])
            header.append(line)
    out = np.clip(np.asarray(rows), 0.0, 1.0)
    # .cube rows list red fastest, then green, then blue.
    grid = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")
    inp = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    def to_lab(rgb: np.ndarray) -> np.ndarray:
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))

    in_lab, out_lab = to_lab(inp), to_lab(out)
    in_chroma = np.hypot(in_lab[:, 1], in_lab[:, 2])
    pull = np.exp(-((in_chroma / _NEUTRAL_RADIUS) ** 2))
    out_lab[:, 1:] *= (1.0 - pull)[:, None]
    fixed = np.clip(colour.XYZ_to_sRGB(colour.Lab_to_XYZ(out_lab)), 0.0, 1.0)
    header.insert(1, "# Neutral copy: grey and white inputs come out without a colour cast.")
    body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in fixed)
    dest.write_text("\n".join(header) + "\n" + body + "\n")
    print(dest)


if __name__ == "__main__":
    main()
