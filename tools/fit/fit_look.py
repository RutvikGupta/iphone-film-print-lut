#!/usr/bin/env python3
"""Fit an approximate "Chromagraph" look as 33-point .cube 3D LUTs from public photo pairs.

Each pair is a stock iPhone image and a Chromagraph image of the same scene, listed in
pairs.json (add pairs there; no code changes are needed). For every pair marked "fit":

1. register the Chromagraph image onto the stock frame (SIFT + RANSAC homography),
2. decode both images through their ICC profiles into linear sRGB / Rec.709 light, blur them
   slightly and mask clipped, misregistered and mismatched pixels,
3. fit one shared look: a monotonic neutral tone curve with a per-pair exposure gain (fitted on
   luminance), then a smooth 3D residual (Gaussian RBF that decays to zero away from the data),
4. bake it with the median exposure gain, make output lightness monotonic, and (v2) anchor the
   white point to the highlight level the real Chromagraph images reach and constrain neutral
   inputs to the tint the data supports for near-neutral pixels,
5. validate every variant leave-one-pair-out with CIEDE2000, and
6. write the .cube files, previews, a user-photo check and a Markdown report.

Variants: v1 (as shipped: round-1 pairs, v1 method), each v2 change on its own, and v2 with all
changes, plus two brighter renderings of v2. The images are untrusted data: they are only read
through Path arguments. See report.md (written by this script) for the exact command used.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import scipy.linalg  # noqa: F401  (fails to load for Python 3.10 wheels on recent macOS)
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"scipy failed to import: {exc}\n"
        "Run with `uv run --no-project --python 3.12 ...` (the command is at the top of report.md)."
    )

import matplotlib

matplotlib.use("Agg")
import colour  # noqa: E402
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps  # noqa: E402
from scipy.interpolate import RBFInterpolator  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

COMMAND_PREFIX = (
    "uv run --no-project --python 3.12 --with numpy --with scipy --with pillow --with pillow-heif "
    "--with opencv-python-headless --with colour-science --with matplotlib "
    "python -I fit/fit_look.py"
)

# ----------------------------------------------------------------------------- settings
SEED = 20261102
LUT_SIZE = 33

DETECT_LONG = 1600        # long side for feature detection (px)
LOWE_RATIO = 0.8
RANSAC_PX = 3.0           # RANSAC threshold at detection resolution (px)
WORK_LONG = 1024          # long side of the stock frame used for colour fitting (px)
BLUR_SIGMA = 2.0          # blur applied to both images at working resolution (px)
CLIP_LO, CLIP_HI = 2, 253  # 8-bit code values counted as clipped (any channel, either source)
CLIP_FRAC_MAX = 0.2       # exclude a working pixel if > 20% of the source pixels under its blur footprint are clipped

HP_SIGMA = 4.0            # structure check: high-pass scale (working px)
NCC_SIGMA = 4.0           # structure check: local window (working px)
TEX_MIN = 0.01            # local high-pass std (sRGB-encoded luma) that counts as texture
NCC_MIN = 0.4             # minimum local correlation where both images are textured
LUM_ABS = 0.10            # luma-residual threshold floor (sRGB-encoded units)
LUM_MAD_K = 4.0           # luma-residual threshold in robust sigmas

MIN_INLIERS = 40          # registration acceptance
MIN_STRUCT_AGREE = 0.5
MIN_VALID_FRAC = 0.15

BAL_BINS = 16             # RGB bins for balancing (per axis)
BAL_CAP = 24              # samples kept per bin per pair (stage 1)
STAGE1_MAX = 8000         # samples per pair (stage 1)
STAGE2_MAX = 60000        # samples per unit pair weight aggregated for the residual (stage 2)
KNOTS = np.arange(-14.0, 4.0 + 1e-9, 0.5)  # tone-curve knots in log2 linear input
LAM_CURV = 2e-5           # tone curve: penalty on 2nd differences of the log-slope
LAM_SLOPE = 1e-6          # tone curve: penalty on 1st differences of the log-slope
LAM_GAIN = 1e-6           # weak prior keeping log2 gains near the anchor pair
HUBER_K = 2.0             # IRLS Huber threshold in robust sigmas
TOE_LO, TOE_HI = 0.30, 0.46  # v2 toe weighting: full toe_weight below this stock luma (sRGB), none above

RBF_BINS = 16             # bins (per axis) for aggregating residual samples
RBF_MIN_COUNT = 3
RBF_REF_COUNT = 30
RBF_WIDTH = 0.2           # Gaussian kernel width in sRGB-encoded units
RBF_SMOOTH = 0.3          # smoothing for a well-populated bin (noise/signal variance)

NEUTRAL_C_MAX = 6.0       # stock pixels with C* below this are the near-neutral evidence
NEUTRAL_W_C = 6.0         # constraint weight for an input of chroma C: 2^-(C/NEUTRAL_W_C)^2
NEUTRAL_L_EDGES = np.arange(0.0, 100.0 + 1e-9, 5.0)  # input L* bins for the evidence
NEUTRAL_MIN_PX = 150      # pixels per pair and bin to count as evidence
NEUTRAL_N0 = 400          # evidence pixels for half confidence
NEUTRAL_SAMPLE = 150000   # pixels sampled per pair for the evidence
FALLBACK_A = (-1.0, 1.0)  # neutral tint allowed where there is no evidence
FALLBACK_B = (-1.0, 3.0)
WHITE_STOCK_L = 97.0      # stock pixels at or above this L* count as "stock white"
WHITE_MIN_PX = 50         # stock-white pixels a pair needs to vote on the white target
WHITE_PCT = 99.9          # percentile of Chromagraph L* reported alongside
WHITE_KNEE_L = 65.0       # white anchor leaves output below this L* unchanged
BRIGHT = {"v2_bright": 0.75, "v2_brighter": 1.5}  # shadow lift (stops) at fixed white

COVER_MIN = 5             # samples within one lattice step for a node to count as covered
EVAL_MAX = 100000         # pixels per held-out pair for CIEDE2000
GAIN_SEARCH = np.arange(-4.0, 4.0 + 1e-9, 0.1)  # log2 exposure grid (exposure-matched table)
GAIN_SEARCH_PX = 10000

PREVIEW_W = 1600
USER_PANEL_W = 420
USER_LONG = 1600
PREVIEW_GAP = 6
LABEL_H = 30

TINY = 2.0 ** -20
WHITE = colour.RGB_COLOURSPACES["sRGB"].whitepoint
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA64 = LUMA.astype(np.float64)
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    data: str      # "round1" (pairs with round <= 1) or "all"
    toe: bool
    neutral: bool
    white: bool
    white_L: float | None = None  # fixed white target instead of the measured one


WHITE_ASKED_L = 93.0  # the user's estimate of the real white level; scored for comparison only
VARIANTS = [
    Variant("v1", "v1 as shipped (round-1 pairs, v1 method)", "round1", False, False, False),
    Variant("data", "v1 method + new pairs", "all", False, False, False),
    Variant("toe", "+ toe down-weighting only", "all", True, False, False),
    Variant("neutral", "+ neutral-axis constraint only", "all", False, True, False),
    Variant("white", "+ white anchor only", "all", False, False, True),
    Variant("v2", "v2 (all three changes)", "all", True, True, True),
    Variant("w93", f"v2 with white at L* {WHITE_ASKED_L:g} instead of the measured level (comparison only)", "all",
            True, True, True, WHITE_ASKED_L),
]
VARIANT = {v.key: v for v in VARIANTS}
CUBES = {
    "v1": ("chromagraph_approx_33.cube", "v1 (as shipped; kept for comparison)"),
    "v2": ("chromagraph_v2_33.cube", "v2, fitted exposure (closest to the data)"),
    "v2_bright": ("chromagraph_v2_bright_33.cube", "v2 with shadows lifted 0.75 stop, white fixed"),
    "v2_brighter": ("chromagraph_v2_brighter_33.cube", "v2 with shadows lifted 1.5 stops, white fixed"),
}
SHORT = {"identity": "identity", "tone": "tone curve (v2)", "v1": "v1", "data": "+data", "toe": "+toe",
         "neutral": "+neutral", "white": "+white", "v2": "v2", "v2_bright": "v2 bright", "v2_brighter": "v2 brighter",
         "w93": f"v2, white {WHITE_ASKED_L:g}"}


# ----------------------------------------------------------------------------- colour maths
def enc(lin):
    """Linear -> sRGB-encoded (colour-science; values above 1 follow the same power law)."""
    return colour.cctf_encoding(np.maximum(lin, 0.0), function="sRGB")


def dec(s):
    """sRGB-encoded -> linear (colour-science)."""
    return colour.cctf_decoding(np.maximum(s, 0.0), function="sRGB")


LIN_TABLE = dec(np.arange(256) / 255.0).astype(np.float32)  # 8-bit sRGB code -> linear


def enc_to_lab(s):
    xyz = colour.RGB_to_XYZ(dec(np.clip(s, 0.0, 1.0)), "sRGB")
    return colour.XYZ_to_Lab(xyz, WHITE)


def lab_to_enc(lab):
    lin = colour.XYZ_to_RGB(colour.Lab_to_XYZ(lab, WHITE), "sRGB")
    return np.clip(enc(np.clip(lin, 0.0, 1.0)), 0.0, 1.0)


def delta_e(s_pred, s_true):
    return colour.delta_E(enc_to_lab(s_pred), enc_to_lab(s_true), method="CIE 2000")


def luma_enc(lin):
    return enc(np.clip(lin @ LUMA, 0.0, 1.0)).astype(np.float32)


def L_to_Y(L):
    return colour.luminance(np.asarray(L, dtype=np.float64)) / 100.0


def Y_to_L(Y):
    return colour.lightness(np.clip(np.asarray(Y, dtype=np.float64), 0.0, None) * 100.0)


def wmedian(values, weights) -> float:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


# ----------------------------------------------------------------------------- data classes
@dataclass
class PairSpec:
    name: str
    stock: str
    chroma: str
    use: str
    round: int = 1
    weight: float = 1.0
    toe_weight: float = 1.0
    neutral_weight: float | None = None
    same_exposure: bool = False
    deliberate_exposure: str = ""
    mask_top_fraction: float = 0.0
    stock_source: str = "stock_camera"
    scene: str = ""
    alignment: str = ""
    notes: str = ""
    probes: list = field(default_factory=list)


@dataclass
class Decoded:
    raw8: np.ndarray    # code values in the file's own colour space
    srgb8: np.ndarray   # converted to sRGB through the embedded ICC profile
    profile: str


@dataclass
class Registration:
    n_kp_stock: int = 0
    n_kp_chroma: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    reproj_median: float = float("nan")
    reproj_rms: float = float("nan")
    H_full: np.ndarray | None = None
    scale: float = float("nan")      # stock px per Chromagraph px
    overlap_frac: float = 0.0        # share of the stock frame covered by the warped image
    struct_agree: float = 0.0
    excluded: dict = field(default_factory=dict)
    valid_frac: float = 0.0          # share of the overlap that survives all masks
    n_valid: int = 0
    accepted: bool = False
    reason: str = ""


@dataclass
class PairData:
    name: str
    x: np.ndarray       # stock, linear Rec.709, blurred, working res (H, W, 3)
    y: np.ndarray       # Chromagraph registered onto x, same encoding
    x0: np.ndarray      # stock before the blur
    y0: np.ndarray      # registered Chromagraph before the blur
    valid: np.ndarray   # all masks applied
    loose: np.ndarray   # overlap & unclipped & label mask only (no residual exclusions)
    bright: np.ndarray  # overlap & not label (clipped pixels kept): for highlight levels
    weight: float = 1.0
    toe_weight: float = 1.0
    neutral_weight: float = 1.0
    deliberate: bool = False


@dataclass
class ToneCurve:
    values: np.ndarray  # sRGB-encoded output at KNOTS (input = log2 linear)

    def __call__(self, lz):
        return np.interp(lz, KNOTS, self.values)


@dataclass
class LookModel:
    curve: ToneCurve
    log2_gains: dict          # pair name -> log2 exposure gain (anchor pair = 0)
    rbf: RBFInterpolator | None
    n_rbf_centres: int = 0
    support: list = field(default_factory=list)  # near-neutral evidence by input L* bin
    white: dict = field(default_factory=dict)    # highlight levels and white target
    _baked: dict = field(default_factory=dict, repr=False)

    def median_gain(self) -> float:
        return float(np.median(list(self.log2_gains.values())))

    def bake(self, log2_gain: float, residual: bool = True, project: bool = True,
             neutral: bool = False, white: bool = False, size: int = LUT_SIZE, white_L: float | None = None) -> np.ndarray:
        key = (float(log2_gain), residual, project, size)
        if key not in self._baked:
            s = lattice(size).reshape(-1, 3)
            u = self.curve(np.log2(np.maximum(dec(s), TINY)) + log2_gain)
            out = u + self.rbf(np.clip(u, 0.0, 1.0)) if (residual and self.rbf is not None) else u
            out = np.clip(out, 0.0, 1.0).reshape(size, size, size, 3)
            if residual and project:
                out = monotone_lightness(out)
            self._baked[key] = out
        out = self._baked[key].copy()
        if white:
            out = anchor_white(out, self.white["target_L"] if white_L is None else white_L)
        if neutral:
            out = neutral_constrain(out, self.support)
        return out

    def bake_variant(self, v: Variant) -> np.ndarray:
        return self.bake(self.median_gain(), neutral=v.neutral, white=v.white, white_L=v.white_L)


# ----------------------------------------------------------------------------- LUT post-processing
def monotone_lightness(lut: np.ndarray, iters: int = 3) -> np.ndarray:
    """Make output luminance non-decreasing along each input axis, so brightening any input
    channel never darkens the output (no dark rings around highlights). Luminance is replaced
    by the mean of its smallest monotone majorant and largest monotone minorant, and each
    node's linear RGB is rescaled to it (keeping its chromaticity)."""
    lin = dec(lut)
    for _ in range(iters):
        Y = lin @ LUMA64
        up, lo = Y.copy(), Y.copy()
        for ax in range(3):
            up = np.maximum.accumulate(up, axis=ax)
            lo = np.flip(np.minimum.accumulate(np.flip(lo, axis=ax), axis=ax), axis=ax)
        target = 0.5 * (up + lo)
        lin = np.clip(lin * ((target + 1e-5) / (Y + 1e-5))[..., None], 0.0, 1.0)
    return np.clip(enc(lin), 0.0, 1.0)


def anchor_white(lut: np.ndarray, target_L: float) -> np.ndarray:
    """Move the white node to target_L with a smooth, monotonic highlight remap of output
    luminance (unchanged below WHITE_KNEE_L, chromaticity kept)."""
    lin = dec(lut)
    Y = lin @ LUMA64
    Yw, Yt, Yk = float(Y[-1, -1, -1]), float(L_to_Y(target_L)), float(L_to_Y(WHITE_KNEE_L))
    if Yw <= Yk + 1e-6:
        return lut
    d = max(Yt - Yw, -0.45 * (Yw - Yk))  # a lowering remap must stay monotonic
    t = np.clip((Y - Yk) / (Yw - Yk), 0.0, 1.0)
    Ynew = Y + d * t**2
    lin = np.clip(lin * ((Ynew + 1e-6) / (Y + 1e-6))[..., None], 0.0, 1.0)
    return np.clip(enc(lin), 0.0, 1.0)


def neutral_ramp(lut: np.ndarray, L_grid: np.ndarray) -> np.ndarray:
    """Output Lab of neutral inputs with the given input L*."""
    s = enc(L_to_Y(L_grid))
    return enc_to_lab(apply_lut(lut, np.repeat(s[:, None], 3, axis=1)))


def neutral_targets(lut: np.ndarray, support: list) -> dict:
    """Target a*, b* for neutral inputs by input L*: the near-neutral evidence where there is
    enough of it, otherwise the LUT's own tint clamped to FALLBACK_A / FALLBACK_B."""
    grid = np.linspace(0.0, 100.0, 201)
    lab = neutral_ramp(lut, grid)
    centres = np.array([(b["lo"] + b["hi"]) / 2 for b in support])
    alpha_b = np.array([b["n"] / (b["n"] + NEUTRAL_N0) for b in support])
    has = alpha_b > 0
    alpha = np.interp(grid, centres, alpha_b)
    if has.any():
        a_d = np.interp(grid, centres[has], [b["a"] for b, h in zip(support, has) if h])
        b_d = np.interp(grid, centres[has], [b["b"] for b, h in zip(support, has) if h])
    else:
        a_d = b_d = np.zeros_like(grid)
    fa, fb = np.clip(lab[:, 1], *FALLBACK_A), np.clip(lab[:, 2], *FALLBACK_B)
    return {"L": grid, "a_cur": lab[:, 1], "b_cur": lab[:, 2], "alpha": alpha,
            "a_t": alpha * a_d + (1 - alpha) * fa, "b_t": alpha * b_d + (1 - alpha) * fb}


def neutral_constrain(lut: np.ndarray, support: list) -> np.ndarray:
    """Shift each node's a*, b* (L* unchanged) so that neutral inputs land on the target tint.
    The shift fades with input chroma (weight 2^-(C/NEUTRAL_W_C)^2), so bluish whites such as
    skies keep their own fitted shift."""
    tg = neutral_targets(lut, support)
    lab_in = enc_to_lab(lattice(lut.shape[0]))
    L_in, C_in = lab_in[..., 0], np.hypot(lab_in[..., 1], lab_in[..., 2])
    w = 2.0 ** (-((C_in / NEUTRAL_W_C) ** 2))
    lab = enc_to_lab(lut)
    lab[..., 1] -= w * (np.interp(L_in, tg["L"], tg["a_cur"]) - np.interp(L_in, tg["L"], tg["a_t"]))
    lab[..., 2] -= w * (np.interp(L_in, tg["L"], tg["b_cur"]) - np.interp(L_in, tg["L"], tg["b_t"]))
    return lab_to_enc(lab)


def brighten(lut: np.ndarray, stops: float) -> np.ndarray:
    """Lift shadows by `stops` at fixed white: lin*g/(1+(g-1)*lin) per channel in linear light,
    with lin normalised to the LUT's white node so the anchored white stays where it is."""
    g = 2.0**stops
    lin = dec(lut)
    W = max(float(lin[-1, -1, -1] @ LUMA64), 1e-6)
    x = lin / W
    return np.clip(enc(np.clip(W * g * x / (1.0 + (g - 1.0) * x), 0.0, 1.0)), 0.0, 1.0)


# ----------------------------------------------------------------------------- LUT helpers
def lattice(n: int) -> np.ndarray:
    g = np.linspace(0.0, 1.0, n)
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([r, gg, b], axis=-1)  # indexed [r, g, b]


def apply_lut(lut: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of an [r, g, b, 3] LUT at sRGB-encoded inputs."""
    n = lut.shape[0]
    p = np.clip(s, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), n - 2)
    f = p - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = np.zeros(s.shape, dtype=np.float64)
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            for db, wb in ((0, 1 - fb), (1, fb)):
                out += wr * wg * wb * lut[r0 + dr, g0 + dg, b0 + db]
    return out


def write_cube(path: Path, lut: np.ndarray, title: str, comments: list[str]) -> None:
    n = lut.shape[0]
    lines = [f'TITLE "{title}"']
    lines += [f"# {c}" for c in comments]
    lines += [f"LUT_3D_SIZE {n}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    data = lut.transpose(2, 1, 0, 3).reshape(-1, 3)  # blue slowest, red fastest
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in data]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def read_cube(path: Path) -> np.ndarray:
    size, rows = None, []
    for line in path.read_text(encoding="ascii").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        key = t.split()[0]
        if key == "LUT_3D_SIZE":
            size = int(t.split()[1])
        elif key in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX"):
            continue
        else:
            rows.append([float(v) for v in t.split()])
    data = np.asarray(rows)
    if size is None or data.shape != (size**3, 3):
        raise ValueError(f"bad cube file {path}: size={size}, rows={data.shape}")
    return data.reshape(size, size, size, 3).transpose(2, 1, 0, 3)  # back to [r, g, b]


# ----------------------------------------------------------------------------- loading
def load_specs(path: Path) -> list[PairSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs, seen = [], set()
    for p in raw["pairs"]:
        spec = PairSpec(
            name=p["name"], stock=p["stock"], chroma=p["chromagraph"], use=p["use"],
            round=int(p.get("round", 1)), weight=float(p.get("weight", 1.0)),
            toe_weight=float(p.get("toe_weight", 1.0)),
            neutral_weight=float(p["neutral_weight"]) if "neutral_weight" in p else None,
            same_exposure=bool(p.get("same_exposure", False)),
            deliberate_exposure=p.get("deliberate_exposure", ""),
            mask_top_fraction=float(p.get("mask_top_fraction", 0.0)),
            stock_source=p.get("stock_source", "stock_camera"),
            scene=p.get("scene", ""), alignment=p.get("alignment", ""), notes=p.get("notes", ""),
            probes=list(p.get("probes", [])),
        )
        if spec.use not in ("fit", "preview"):
            raise ValueError(f"pair {spec.name}: use must be 'fit' or 'preview'")
        if spec.name in seen:
            raise ValueError(f"duplicate pair name {spec.name}")
        seen.add(spec.name)
        specs.append(spec)
    return specs


def load_image(path: Path) -> Decoded:
    """Decode an image and convert it to sRGB through its embedded ICC profile (if any)."""
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    raw8 = np.asarray(rgb, dtype=np.uint8).copy()
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = ImageCms.getProfileDescription(src).strip()
        xform = ImageCms.buildTransform(
            src, SRGB_PROFILE, "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
        )
        srgb8 = np.asarray(ImageCms.applyTransform(rgb, xform), dtype=np.uint8).copy()
        profile = f"embedded '{desc}' -> sRGB (LittleCMS, relative colorimetric)"
    else:
        srgb8 = raw8
        profile = "no embedded profile; decoded as sRGB"
    return Decoded(raw8=raw8, srgb8=srgb8, profile=profile)


def clipped(a: np.ndarray) -> np.ndarray:
    return np.any((a <= CLIP_LO) | (a >= CLIP_HI), axis=-1)


def top_mask(shape, frac: float) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=bool)
    if frac > 0:
        m[: int(math.ceil(frac * shape[0]))] = True
    return m


# ----------------------------------------------------------------------------- registration
def feature_gray(d: Decoded, top_frac: float):
    h, w = d.srgb8.shape[:2]
    sc = min(1.0, DETECT_LONG / max(h, w))
    g = cv2.cvtColor(d.srgb8, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(g, [1.0, 99.5])
    g = np.clip((g.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    mask = np.where(top_mask(g.shape, top_frac), 0, 255).astype(np.uint8)
    return g, sc, mask


def local_scale(H: np.ndarray, x: float, y: float) -> float:
    pts = np.float64([[x, y], [x + 1, y], [x, y + 1]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    J = np.stack([q[1] - q[0], q[2] - q[0]], axis=1)
    return float(math.sqrt(abs(np.linalg.det(J))))


def register(spec: PairSpec, stock: Decoded, chroma: Decoded) -> Registration:
    reg = Registration()
    gs, ss, ms = feature_gray(stock, spec.mask_top_fraction)
    gc, sc, mc = feature_gray(chroma, spec.mask_top_fraction)
    sift = cv2.SIFT_create(nfeatures=12000)
    ks, ds = sift.detectAndCompute(gs, ms)
    kc, dc = sift.detectAndCompute(gc, mc)
    reg.n_kp_stock, reg.n_kp_chroma = len(ks), len(kc)
    if ds is None or dc is None or len(ks) < 8 or len(kc) < 8:
        reg.reason = "too few features"
        return reg
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 96})
    knn = flann.knnMatch(dc, ds, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]
    reg.n_matches = len(good)
    if len(good) < 8:
        reg.reason = "too few feature matches"
        return reg
    pc = np.float32([kc[m.queryIdx].pt for m in good])
    ps = np.float32([ks[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pc, ps, cv2.RANSAC, RANSAC_PX, maxIters=20000, confidence=0.9995)
    if H is None:
        reg.reason = "RANSAC found no homography"
        return reg
    inl = inl.ravel().astype(bool)
    reg.n_inliers = int(inl.sum())
    H_full = np.linalg.inv(np.diag([ss, ss, 1.0])) @ H @ np.diag([sc, sc, 1.0])
    proj = cv2.perspectiveTransform((pc[inl] / sc).reshape(-1, 1, 2).astype(np.float64), H_full)
    err = np.linalg.norm(proj.reshape(-1, 2) - ps[inl] / ss, axis=1)
    reg.reproj_median = float(np.median(err))
    reg.reproj_rms = float(np.sqrt(np.mean(err**2)))
    hc, wc = chroma.srgb8.shape[:2]
    reg.H_full = H_full
    reg.scale = local_scale(H_full, wc / 2, hc / 2)
    return reg


def local_ncc(a: np.ndarray, b: np.ndarray):
    ha = a - cv2.GaussianBlur(a, (0, 0), HP_SIGMA)
    hb = b - cv2.GaussianBlur(b, (0, 0), HP_SIGMA)
    saa = cv2.GaussianBlur(ha * ha, (0, 0), NCC_SIGMA)
    sbb = cv2.GaussianBlur(hb * hb, (0, 0), NCC_SIGMA)
    sab = cv2.GaussianBlur(ha * hb, (0, 0), NCC_SIGMA)
    ncc = sab / np.sqrt(saa * sbb + 1e-12)
    textured = (np.sqrt(saa) > TEX_MIN) & (np.sqrt(sbb) > TEX_MIN)
    return ncc, textured


def monotone_map_residual(ls: np.ndarray, lc: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Residual of the Chromagraph luma against a per-pair monotonic luma map."""
    edges = np.linspace(0.0, 1.0, 49)
    idx = np.clip(np.digitize(ls[m], edges) - 1, 0, 47)
    centres, meds = [], []
    lcm = lc[m]
    for k in range(48):
        sel = idx == k
        if sel.sum() >= 20:
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            meds.append(np.median(lcm[sel]))
    if len(centres) < 2:
        return np.zeros_like(ls)
    meds = np.maximum.accumulate(np.asarray(meds))
    return lc - np.interp(ls, centres, meds).astype(np.float32)


def build_pair(spec: PairSpec, stock: Decoded, chroma: Decoded, reg: Registration,
               diag_path: Path | None = None) -> PairData:
    """Warp, blur and mask one registered pair at working resolution; fill QA fields in reg."""
    hs, ws = stock.srgb8.shape[:2]
    sw = min(1.0, WORK_LONG / max(hs, ws))
    Ww, Hw = round(ws * sw), round(hs * sw)
    area = dict(interpolation=cv2.INTER_AREA)
    x0 = cv2.resize(LIN_TABLE[stock.srgb8], (Ww, Hw), **area)
    clip_s = cv2.resize((clipped(stock.raw8) | clipped(stock.srgb8)).astype(np.float32), (Ww, Hw), **area)
    label_s = cv2.resize(top_mask(stock.srgb8.shape, spec.mask_top_fraction).astype(np.float32), (Ww, Hw), **area) > 0

    hc, wc = chroma.srgb8.shape[:2]
    cpre = min(1.0, reg.scale * sw)
    Wc, Hc = max(1, round(wc * cpre)), max(1, round(hc * cpre))
    lin_c = cv2.resize(LIN_TABLE[chroma.srgb8], (Wc, Hc), **area)
    clip_c = cv2.resize(clipped(chroma.raw8).astype(np.float32), (Wc, Hc), **area)
    label_c = cv2.resize(top_mask(chroma.srgb8.shape, spec.mask_top_fraction).astype(np.float32), (Wc, Hc), **area)
    Hwk = np.diag([Ww / ws, Hw / hs, 1.0]) @ reg.H_full @ np.linalg.inv(np.diag([Wc / wc, Hc / hc, 1.0]))
    warp = dict(dsize=(Ww, Hw), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    y0 = cv2.warpPerspective(lin_c, Hwk, **warp)
    cover = cv2.warpPerspective(np.ones((Hc, Wc), np.float32), Hwk, **warp) > 0.999
    clip_c = cv2.warpPerspective(clip_c, Hwk, **warp)
    label_c = cv2.warpPerspective(label_c, Hwk, **warp) > 1e-3

    # A blurred working pixel is "clipped" when clipped source pixels make up more than
    # CLIP_FRAC_MAX of its blur footprint, in either image. Labels are excluded outright.
    clip_frac = np.maximum(cv2.GaussianBlur(clip_s, (0, 0), BLUR_SIGMA), cv2.GaussianBlur(clip_c, (0, 0), BLUR_SIGMA))
    r_er = int(math.ceil(3 * BLUR_SIGMA)) + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_er + 1,) * 2)
    overlap = cv2.erode(cover.astype(np.uint8), disk) > 0
    label = cv2.dilate((label_s | label_c).astype(np.uint8), disk) > 0
    bad = (clip_frac > CLIP_FRAC_MAX) | label
    reg.overlap_frac = float(cover.mean())

    ncc, textured = local_ncc(luma_enc(x0), luma_enc(y0))
    tex_ov = textured & overlap & ~bad
    reg.struct_agree = float((ncc[tex_ov] >= NCC_MIN).mean()) if tex_ov.any() else 0.0
    struct_bad = textured & (ncc < NCC_MIN)

    x = cv2.GaussianBlur(x0, (0, 0), BLUR_SIGMA)
    y = cv2.GaussianBlur(y0, (0, 0), BLUR_SIGMA)
    loose = overlap & ~bad
    base = loose & ~struct_bad
    ls, lc = luma_enc(x), luma_enc(y)
    resid = np.abs(monotone_map_residual(ls, lc, base))
    bm = base.astype(np.float32)
    local = cv2.GaussianBlur(resid * bm, (0, 0), 3.0) / np.maximum(cv2.GaussianBlur(bm, (0, 0), 3.0), 1e-6)
    thr = max(LUM_ABS, LUM_MAD_K * 1.4826 * float(np.median(resid[base]))) if base.any() else LUM_ABS
    lum_bad = local > thr
    valid = base & ~lum_bad

    n_ov = max(int(overlap.sum()), 1)
    reg.excluded = {
        "clipped_or_label": float((overlap & bad).sum() / n_ov),
        "structure": float((overlap & ~bad & struct_bad).sum() / n_ov),
        "luma_residual": float((base & lum_bad).sum() / n_ov),
        "luma_threshold": float(thr),
    }
    reg.valid_frac = float(valid.sum() / n_ov)
    reg.n_valid = int(valid.sum())
    pair = PairData(name=spec.name, x=x, y=y, x0=x0, y0=y0, valid=valid, loose=loose, bright=overlap & ~label,
                    weight=spec.weight, toe_weight=spec.toe_weight,
                    neutral_weight=spec.weight if spec.neutral_weight is None else spec.neutral_weight,
                    deliberate=bool(spec.deliberate_exposure))
    if diag_path is not None:
        registration_diagnostic(pair, x0, y0, overlap, diag_path)
    return pair


def accept(reg: Registration) -> None:
    if reg.H_full is None:
        reg.accepted = False
        return
    problems = []
    if reg.n_inliers < MIN_INLIERS:
        problems.append(f"only {reg.n_inliers} RANSAC inliers (< {MIN_INLIERS})")
    if not (0.2 < reg.scale < 5.0):
        problems.append(f"implausible scale {reg.scale:.2f}")
    if reg.struct_agree < MIN_STRUCT_AGREE:
        problems.append(f"only {reg.struct_agree:.0%} of textured pixels agree after warping (< {MIN_STRUCT_AGREE:.0%})")
    if reg.valid_frac < MIN_VALID_FRAC:
        problems.append(f"only {reg.valid_frac:.0%} of the overlap is usable (< {MIN_VALID_FRAC:.0%})")
    reg.accepted = not problems
    reg.reason = "; ".join(problems) if problems else "passed"


def registration_diagnostic(pair: PairData, x0: np.ndarray, y0: np.ndarray, overlap: np.ndarray, path: Path) -> None:
    """Three panels: checkerboard of stock / registered Chromagraph (exposure-matched),
    red-cyan overlay of locally normalised luma (fringes = misregistration), and the
    valid mask (excluded pixels shown dark grey)."""
    m = overlap if overlap.any() else np.ones(overlap.shape, bool)
    k = np.median(x0[m] @ LUMA) / max(np.median(y0[m] @ LUMA), 1e-6)
    s, c = enc(np.clip(x0, 0, 1)), enc(np.clip(y0 * k, 0, 1))
    h, w = s.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    comp = np.where(((yy // 64 + xx // 64) % 2 == 0)[..., None], s, c)

    def norm(l):
        hp = l - cv2.GaussianBlur(l, (0, 0), 12)
        return np.clip(0.5 + hp / (4 * np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 12)) + 1e-3), 0, 1)

    a, b = norm(luma_enc(x0)), norm(luma_enc(y0))
    ana = np.stack([a, b, b], axis=-1) * m[..., None]
    tint = s.copy()
    tint[~pair.valid] = 0.15 + 0.2 * (tint[~pair.valid] @ LUMA)[:, None]  # excluded -> dark grey
    gap = np.ones((h, 8, 3))
    img = np.concatenate([comp, gap, ana, gap, tint], axis=1)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=88)


def prepare_pairs(specs: list[PairSpec], data_dir: Path, diag: Path | None):
    """Load every pair, register the fit candidates and build their working-resolution data."""
    images, regs, pdata = {}, {}, {}
    for spec in specs:
        stock, chroma = load_image(data_dir / spec.stock), load_image(data_dir / spec.chroma)
        images[spec.name] = (stock, chroma)
        log(f"[load] {spec.name}: stock {stock.srgb8.shape[1]}x{stock.srgb8.shape[0]} ({stock.profile}); "
            f"chroma {chroma.srgb8.shape[1]}x{chroma.srgb8.shape[0]} ({chroma.profile})")
        if spec.use != "fit":
            continue
        reg = register(spec, stock, chroma)
        if reg.H_full is not None:
            dpath = diag / f"registration_{spec.name}.jpg" if diag is not None else None
            pdata[spec.name] = build_pair(spec, stock, chroma, reg, dpath)
        accept(reg)
        regs[spec.name] = reg
        log(f"[register] {spec.name}: matches {reg.n_matches}, inliers {reg.n_inliers}, reproj med {reg.reproj_median:.2f}px, "
            f"scale {reg.scale:.3f}, overlap {reg.overlap_frac:.2f}, struct agree {reg.struct_agree:.2f}, "
            f"valid {reg.valid_frac:.2f} -> {reg.reason}")
    fit_names = [s.name for s in specs if s.use == "fit" and regs[s.name].accepted]
    # The first fitted same-exposure pair anchors the reported gains (no exposure difference there).
    anchor = next((s.name for s in specs if s.name in fit_names and s.same_exposure), None)
    if anchor is not None:
        fit_names.remove(anchor)
        fit_names.insert(0, anchor)
    return images, regs, pdata, fit_names


# ----------------------------------------------------------------------------- fitting
def bin_key(s: np.ndarray, bins: int) -> np.ndarray:
    q = np.clip((s * bins).astype(np.int64), 0, bins - 1)
    return (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]


def balanced_indices(s: np.ndarray, rng, bins: int, cap: int, max_total: int) -> np.ndarray:
    """Random subset keeping at most `cap` samples per RGB bin (large flat regions are capped)."""
    key = bin_key(s, bins)
    perm = rng.permutation(len(key))
    order = np.argsort(key[perm], kind="stable")
    idx, ks = perm[order], key[perm][order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    rank = np.arange(len(ks)) - np.repeat(start, np.diff(np.r_[start, len(ks)]))
    sel = idx[rank < cap]
    if len(sel) > max_total:
        sel = rng.choice(sel, max_total, replace=False)
    return sel


def fit_tone_curve(log_x: np.ndarray, target: np.ndarray, pid: np.ndarray, gamma0: np.ndarray,
                   base_w: np.ndarray | None = None):
    """Joint fit of a monotonic neutral curve (log2 linear in -> sRGB-encoded out) and
    per-pair log2 exposure gains (pair 0 is the anchor at 0). Huber IRLS, analytic Jacobian.
    log_x and target have shape (N, C); the same curve applies to every column. base_w are
    per-sample weights (pair weight x toe weight)."""
    K, P, C = len(KNOTS), len(gamma0), target.shape[1]
    dk = float(KNOTS[1] - KNOTS[0])
    v0 = np.clip(enc(2.0**KNOTS), 0.0, 1.5)
    p0 = np.concatenate([[v0[0]], np.log(np.maximum(np.diff(v0) / dk, 1e-3)), gamma0[1:] - gamma0[0]])
    D2 = np.diff(np.eye(K - 1), 2, axis=0)
    D1 = np.diff(np.eye(K - 1), 1, axis=0)
    scale = 1.0 / math.sqrt(target.size)
    midx = np.arange(K - 1)

    def unpack(p):
        th = p[1:K]
        vals = p[0] + np.concatenate([[0.0], np.cumsum(np.exp(th)) * dk])
        g = np.concatenate([[0.0], p[K:]])
        return th, vals, g

    def geom(p):
        th, vals, g = unpack(p)
        lz = log_x + g[pid][:, None]
        j = np.clip(np.searchsorted(KNOTS, lz, side="right") - 1, 0, K - 2)
        f = (lz - KNOTS[j]) / dk
        inside = (f >= 0) & (f <= 1)
        f = np.clip(f, 0.0, 1.0)
        pred = vals[j] * (1 - f) + vals[j + 1] * f
        return th, vals, g, j, f, inside, pred

    def fun(p, w):
        th, vals, g, j, f, inside, pred = geom(p)
        r = ((pred - target) * w).ravel() * scale
        reg = [math.sqrt(LAM_CURV) * (D2 @ th), math.sqrt(LAM_SLOPE) * (D1 @ th), math.sqrt(LAM_GAIN) * g[1:]]
        return np.concatenate([r] + reg)

    def jac(p, w):
        th, vals, g, j, f, inside, pred = geom(p)
        wj = (w * scale).ravel()
        jr, fr = j.ravel(), f.ravel()
        A = (midx[None, :] < jr[:, None]).astype(np.float64)
        A[np.arange(len(jr)), jr] += fr
        A *= (np.exp(th) * dk)[None, :]
        slope = np.where(inside, (vals[j + 1] - vals[j]) / dk, 0.0).ravel()
        G = np.zeros((len(jr), P - 1))
        pr = np.repeat(pid, C)
        for q in range(1, P):
            G[:, q - 1] = np.where(pr == q, slope, 0.0)
        Jd = np.concatenate([np.ones((len(jr), 1)), A, G], axis=1) * wj[:, None]
        nreg = D2.shape[0] + D1.shape[0] + (P - 1)
        Jr = np.zeros((nreg, len(p)))
        Jr[: D2.shape[0], 1:K] = math.sqrt(LAM_CURV) * D2
        Jr[D2.shape[0]: D2.shape[0] + D1.shape[0], 1:K] = math.sqrt(LAM_SLOPE) * D1
        Jr[D2.shape[0] + D1.shape[0]:, K:] = math.sqrt(LAM_GAIN) * np.eye(P - 1)
        return np.concatenate([Jd, Jr], axis=0)

    bw = np.ones_like(target) if base_w is None else np.sqrt(base_w)
    w = bw
    p = p0
    for _ in range(3):
        res = least_squares(fun, p, jac=jac, args=(w,), method="trf", x_scale="jac", max_nfev=300)
        p = res.x
        pred = geom(p)[-1]
        r = np.abs(pred - target)
        c = HUBER_K * 1.4826 * max(float(np.median(r)), 1e-4)
        w = bw * np.sqrt(np.minimum(1.0, c / np.maximum(r, 1e-12)))
    _, vals, g = unpack(p)
    return ToneCurve(values=vals), g


def neutral_support(pairs: list[PairData]) -> list:
    """Chromagraph a*, b* of near-neutral stock pixels (C* < NEUTRAL_C_MAX), by stock L* bin.
    Each pair's bin median counts once, weighted by the pair's neutral weight."""
    rng = np.random.default_rng(SEED + 1)
    nb = len(NEUTRAL_L_EDGES) - 1
    entries: list[list] = [[] for _ in range(nb)]
    for pr in pairs:
        if pr.neutral_weight <= 0:
            continue
        x, y = sample_pixels(pr, pr.valid, rng, NEUTRAL_SAMPLE)
        li, lc = enc_to_lab(enc(np.clip(x, 0, 1))), enc_to_lab(enc(np.clip(y, 0, 1)))
        m = np.hypot(li[:, 1], li[:, 2]) < NEUTRAL_C_MAX
        li, lc = li[m], lc[m]
        idx = np.digitize(li[:, 0], NEUTRAL_L_EDGES) - 1
        for k in range(nb):
            sel = idx == k
            if sel.sum() >= NEUTRAL_MIN_PX:
                entries[k].append({"pair": pr.name, "w": pr.neutral_weight, "n": int(sel.sum()),
                                   "a": float(np.median(lc[sel, 1])), "b": float(np.median(lc[sel, 2])),
                                   "a_in": float(np.median(li[sel, 1])), "b_in": float(np.median(li[sel, 2]))})
    out = []
    for k in range(nb):
        e = entries[k]
        row = {"lo": float(NEUTRAL_L_EDGES[k]), "hi": float(NEUTRAL_L_EDGES[k + 1]), "n": 0, "pairs": [],
               "a": float("nan"), "b": float("nan"), "a_in": float("nan"), "b_in": float("nan"), "per_pair": e}
        if e:
            w = [x["w"] for x in e]
            row.update(n=sum(x["n"] for x in e), pairs=[x["pair"] for x in e],
                       a=wmedian([x["a"] for x in e], w), b=wmedian([x["b"] for x in e], w),
                       a_in=wmedian([x["a_in"] for x in e], w), b_in=wmedian([x["b_in"] for x in e], w))
        out.append(row)
    return out


def white_level(pairs: list[PairData]) -> dict:
    """Highlight level the real Chromagraph images reach where the stock image is at white:
    per pair the median Chromagraph L* over overlap pixels with stock L* >= WHITE_STOCK_L
    (clipped pixels included: that is exactly what the LUT's white node receives), then the
    weighted median over pairs with enough such pixels that were not deliberately shot at a
    different exposure. Also reports each pair's WHITE_PCT percentile of Chromagraph L*."""
    levels, pct, n_px, voters = {}, {}, {}, {}
    for pr in pairs:
        Ls = Y_to_L(pr.x0[pr.bright] @ LUMA)
        Lc = Y_to_L(pr.y0[pr.bright] @ LUMA)
        if Lc.size:
            pct[pr.name] = float(np.percentile(Lc, WHITE_PCT))
        m = Ls >= WHITE_STOCK_L
        n_px[pr.name] = int(m.sum())
        if m.sum() >= WHITE_MIN_PX:
            levels[pr.name] = float(np.median(Lc[m]))
            if not pr.deliberate:
                voters[pr.name] = pr.weight
    if not voters:
        voters = {n: 1.0 for n in levels} or {n: 1.0 for n in pct}
        levels = levels or pct
    target = wmedian([levels[n] for n in voters], list(voters.values()))
    return {"levels": levels, "percentile": pct, "n_px": n_px, "voters": list(voters), "target_L": target}


def fit_look(pairs: list[PairData], toe: bool) -> LookModel:
    rng = np.random.default_rng(SEED)
    # Stage 1: neutral tone curve + exposure gains, fitted on luminance (so hue-specific
    # channel shifts such as blue skies turning cream do not bend the neutral curve),
    # using RGB-bin-balanced samples weighted by pair weight (and, for v2, toe weight).
    lx, tg, pid, gamma0, bw = [], [], [], [], []
    for i, pr in enumerate(pairs):
        xv, yv = pr.x[pr.valid], pr.y[pr.valid]
        sel = balanced_indices(enc(np.clip(xv, 0, 1)), rng, BAL_BINS, BAL_CAP, STAGE1_MAX)
        lum = xv[sel] @ LUMA
        lx.append(np.log2(np.maximum(lum, TINY))[:, None])
        tg.append(enc(np.clip(yv[sel] @ LUMA, 0, 1))[:, None])
        pid.append(np.full(len(sel), i))
        w = np.full(len(sel), pr.weight)
        if toe and pr.toe_weight != 1.0:
            ramp = np.clip((enc(np.clip(lum, 0, 1)) - TOE_LO) / (TOE_HI - TOE_LO), 0.0, 1.0)
            w = w * (pr.toe_weight + (1.0 - pr.toe_weight) * ramp)
        bw.append(w[:, None])
        ys, xs = yv @ LUMA, xv @ LUMA
        mid = (xs > 0.02) & (xs < 0.6) & (ys > 1e-4)
        gamma0.append(float(np.median(np.log2(ys[mid] / xs[mid]))) if mid.sum() > 100 else 0.0)
    base_w = np.concatenate(bw)
    curve, g = fit_tone_curve(np.concatenate(lx), np.concatenate(tg), np.concatenate(pid), np.asarray(gamma0),
                              None if np.all(base_w == 1.0) else base_w)
    gains = {pr.name: float(gi) for pr, gi in zip(pairs, g)}

    # Stage 2: smooth 3D residual on bin aggregates (balances large flat regions).
    U, D = [], []
    for pr in pairs:
        xv, yv = pr.x[pr.valid], pr.y[pr.valid]
        sel = rng.choice(len(xv), min(len(xv), int(round(STAGE2_MAX * pr.weight))), replace=False)
        u = curve(np.log2(np.maximum(xv[sel], TINY)) + gains[pr.name])
        U.append(u)
        D.append(enc(np.clip(yv[sel], 0, 1)) - u)
    U, D = np.clip(np.concatenate(U), 0.0, 1.0), np.concatenate(D)
    key = bin_key(U, RBF_BINS)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    stop = np.r_[start[1:], len(ks)]
    cu, cd, cn = [], [], []
    for a, b in zip(start, stop):
        if b - a < RBF_MIN_COUNT:
            continue
        sl = order[a:b]
        cu.append(U[sl].mean(axis=0))
        cd.append(np.median(D[sl], axis=0))
        cn.append(b - a)
    cu, cd, cn = np.asarray(cu), np.asarray(cd), np.asarray(cn, dtype=np.float64)
    smoothing = RBF_SMOOTH * RBF_REF_COUNT / np.minimum(cn, RBF_REF_COUNT)
    rbf = RBFInterpolator(cu, cd, kernel="gaussian", epsilon=1.0 / RBF_WIDTH, degree=-1, smoothing=smoothing)
    return LookModel(curve=curve, log2_gains=gains, rbf=rbf, n_rbf_centres=len(cu),
                     support=neutral_support(pairs), white=white_level(pairs))


def lattice_coverage(model: LookModel, pairs: list[PairData]) -> dict:
    """Share of the 33^3 LUT nodes with >= COVER_MIN valid samples within one lattice step."""
    rng = np.random.default_rng(SEED + 4)
    gm = model.median_gain()
    pts = []
    for pr in pairs:
        xv = pr.x[pr.valid]
        sel = rng.choice(len(xv), min(len(xv), STAGE2_MAX), replace=False)
        xe = xv[sel] * 2.0 ** (model.log2_gains[pr.name] - gm)  # where this pixel lands in the baked LUT
        xe = xe[np.all(xe <= 1.0, axis=1)]
        pts.append(enc(xe))
    tree = cKDTree(np.concatenate(pts))
    nodes = lattice(LUT_SIZE).reshape(-1, 3)
    step = 1.0 / (LUT_SIZE - 1)
    n1 = tree.query_ball_point(nodes, r=step * 1.0001, p=np.inf, return_length=True)
    n2 = tree.query_ball_point(nodes, r=2 * step * 1.0001, p=np.inf, return_length=True)
    return {"within_1_step": float((n1 >= COVER_MIN).mean()), "within_2_steps": float((n2 >= COVER_MIN).mean())}


def lut_smoothness(lut: np.ndarray) -> dict:
    """How often output lightness drops when one input channel increases by one lattice step."""
    L = enc_to_lab(lut.reshape(-1, 3)).reshape(lut.shape)[..., 0]
    out = {}
    for ax, name in enumerate("RGB"):
        d = np.diff(L, axis=ax)
        out[name] = {"drop_gt1": float(np.mean(d < -1)), "drop_gt3": float(np.mean(d < -3)), "worst": float(d.min())}
    return out


# ----------------------------------------------------------------------------- evaluation
def sample_pixels(pr: PairData, mask: np.ndarray, rng, n: int):
    xv, yv = pr.x[mask], pr.y[mask]
    sel = rng.choice(len(xv), min(len(xv), n), replace=False)
    return xv[sel].astype(np.float64), yv[sel].astype(np.float64)


def predict(lut: np.ndarray | None, x: np.ndarray, log2_k: float = 0.0) -> np.ndarray:
    s = enc(np.clip(x * 2.0**log2_k, 0.0, 1.0))
    return s if lut is None else apply_lut(lut, s)


def stats(de: np.ndarray) -> dict:
    return {"median": float(np.median(de)), "p95": float(np.percentile(de, 95))}


def best_gain(lut, x, target_s) -> float:
    """Exposure (log2) minimising the mean CIEDE2000. The mean, unlike the median, cannot be
    won by pushing a mostly dark frame to black at the expense of everything else."""
    best, best_k = np.inf, 0.0
    for k in GAIN_SEARCH:
        m = float(np.mean(delta_e(predict(lut, x, k), target_s)))
        if m < best:
            best, best_k = m, float(k)
    return best_k


def evaluate(luts: dict, pr: PairData, mask: np.ndarray, rng, match_exposure: bool = False) -> dict:
    x, y = sample_pixels(pr, mask, rng, EVAL_MAX)
    ts = enc(np.clip(y, 0, 1))
    out = {"n": int(len(x))}
    sub = rng.choice(len(x), min(len(x), GAIN_SEARCH_PX), replace=False)
    for name, lut in luts.items():
        k = best_gain(lut, x[sub], ts[sub]) if match_exposure else 0.0
        out[name] = stats(delta_e(predict(lut, x, k), ts)) | {"log2_gain": k}
    return out


# ----------------------------------------------------------------------------- previews and probes
def label_font(size: int = 16):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow without FreeType default font
        return ImageFont.load_default()


def to_u8(s: np.ndarray) -> np.ndarray:
    return (np.clip(s, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def resize_lin(img8: np.ndarray, w: int) -> np.ndarray:
    h0, w0 = img8.shape[:2]
    h = max(1, round(h0 * w / w0))
    return cv2.resize(LIN_TABLE[img8], (w, h), interpolation=cv2.INTER_AREA)


def compose(panels: list[np.ndarray], labels: list[str], pw: int, path: Path,
            boxes: list | None = None) -> str:
    ph = max(p.shape[0] for p in panels)
    W = len(panels) * pw + (len(panels) - 1) * PREVIEW_GAP
    canvas = Image.new("RGB", (W, ph + LABEL_H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = label_font()
    for i, (p, lab) in enumerate(zip(panels, labels)):
        x = i * (pw + PREVIEW_GAP)
        canvas.paste(Image.fromarray(p), (x, LABEL_H))
        draw.text((x + 5, 7), lab, fill=(235, 235, 235), font=font)
        for bx in boxes or []:
            h, w = p.shape[:2]
            draw.rectangle([x + bx[0] * w, LABEL_H + bx[1] * h, x + bx[2] * w, LABEL_H + bx[3] * h],
                           outline=(255, 0, 255), width=2)
    canvas.save(path, quality=90)
    return f"{W}x{ph + LABEL_H}"


def make_preview(stock: Decoded, chroma: Decoded, reg: Registration | None, luts: dict, path: Path) -> str:
    """Stock | each LUT | real Chromagraph (registered and cropped to the overlap when possible)."""
    n = len(luts) + 2
    pw = (PREVIEW_W - (n - 1) * PREVIEW_GAP) // n
    S, C = stock.srgb8, chroma.srgb8
    hs, ws = S.shape[:2]
    if reg is not None and reg.accepted:
        hc, wc = C.shape[:2]
        q = cv2.perspectiveTransform(np.float64([[0, 0], [wc, 0], [wc, hc], [0, hc]]).reshape(-1, 1, 2), reg.H_full).reshape(-1, 2)
        x0 = int(math.ceil(max(q[0, 0], q[3, 0], 0)))
        x1 = int(math.floor(min(q[1, 0], q[2, 0], ws)))
        y0 = int(math.ceil(max(q[0, 1], q[1, 1], 0)))
        y1 = int(math.floor(min(q[2, 1], q[3, 1], hs)))
        lin_s = resize_lin(S[y0:y1, x0:x1], pw)
        ph = lin_s.shape[0]
        sp = pw / (x1 - x0)
        cpre = min(1.0, reg.scale * sp)
        Wc, Hc = max(1, round(wc * cpre)), max(1, round(hc * cpre))
        lin_c = cv2.resize(LIN_TABLE[C], (Wc, Hc), interpolation=cv2.INTER_AREA)
        T = np.array([[sp, 0, -x0 * sp], [0, sp, -y0 * sp], [0, 0, 1.0]])
        Hp = T @ reg.H_full @ np.linalg.inv(np.diag([Wc / wc, Hc / hc, 1.0]))
        lin_c = cv2.warpPerspective(lin_c, Hp, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        c_label = "Chromagraph (registered)"
    else:
        lin_s = resize_lin(S, pw)
        lin_c = resize_lin(C, pw)
        c_label = "Chromagraph (not registered)"
    s_enc = enc(np.clip(lin_s, 0, 1))
    panels = [to_u8(s_enc)] + [to_u8(apply_lut(l, s_enc)) for l in luts.values()] + [to_u8(enc(np.clip(lin_c, 0, 1)))]
    labels = ["Stock (sRGB)"] + [f"Stock + {k}" for k in luts] + [c_label]
    return compose(panels, labels, pw, path)


def box_pixels(img: np.ndarray, boxes: list) -> np.ndarray:
    h, w = img.shape[:2]
    out = []
    for b in boxes:
        x0, x1 = int(round(b[0] * w)), max(int(round(b[0] * w)) + 1, int(round(b[2] * w)))
        y0, y1 = int(round(b[1] * h)), max(int(round(b[1] * h)) + 1, int(round(b[3] * h)))
        out.append(img[y0:y1, x0:x1].reshape(-1, img.shape[2]))
    return np.concatenate(out)


def lab_median(s: np.ndarray) -> dict:
    lab = enc_to_lab(s)
    return {"L": float(np.median(lab[:, 0])), "a": float(np.median(lab[:, 1])), "b": float(np.median(lab[:, 2]))}


def measure_pair_probes(spec: PairSpec, stock: Decoded, chroma: Decoded, reg: Registration | None,
                        pdata: PairData | None, luts: dict, diag: Path) -> list:
    results = []
    if not spec.probes:
        return results
    for pr in spec.probes:
        S = box_pixels(stock.srgb8, pr["boxes"]) / 255.0
        row = {"pair": spec.name, "probe": pr["name"], "stock": lab_median(S), "chroma": None, "how": ""}
        if pr.get("chroma_boxes"):
            row["chroma"] = lab_median(box_pixels(chroma.srgb8, pr["chroma_boxes"]) / 255.0)
            row["how"] = "boxes placed in each frame"
        elif reg is not None and reg.accepted and pdata is not None:
            row["chroma"] = lab_median(enc(np.clip(box_pixels(pdata.y0, pr["boxes"]), 0, 1)))
            row["how"] = "registered"
        row["luts"] = {k: lab_median(apply_lut(l, S)) for k, l in luts.items()}
        results.append(row)
    # Diagnostic: probe boxes on the stock (and Chromagraph, when boxed separately) images.
    pw = 700
    panels = [to_u8(enc(np.clip(resize_lin(stock.srgb8, pw), 0, 1)))]
    boxes = [b for p in spec.probes for b in p["boxes"]]
    compose(panels, [f"{spec.name}: stock probes"], pw, diag / f"probes_{spec.name}_stock.jpg", boxes)
    cb = [b for p in spec.probes for b in p.get("chroma_boxes", [])]
    if cb:
        compose([to_u8(enc(np.clip(resize_lin(chroma.srgb8, pw), 0, 1)))], [f"{spec.name}: Chromagraph probes"], pw,
                diag / f"probes_{spec.name}_chroma.jpg", cb)
    return results


def run_user_tests(cfg_path: Path, photo_dir: Path, luts: dict, out_dir: Path) -> list:
    """Apply the LUTs to the user's own photos (visual check only; never used for fitting)."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        sys.exit("pillow-heif is needed to read HEIC user photos; add `--with pillow-heif` to the uv command.")
    register_heif_opener()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    results = []
    for ph in cfg["photos"]:
        d = load_image(photo_dir / ph["file"])
        h, w = d.srgb8.shape[:2]
        sc = min(1.0, USER_LONG / max(h, w))
        lin = cv2.resize(LIN_TABLE[d.srgb8], (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        s = enc(np.clip(lin, 0, 1))
        outs = {"stock": s} | {k: apply_lut(l, s) for k, l in luts.items()}
        probes = []
        for pr in ph.get("probes", []):
            probes.append({"probe": pr["name"], "values": {k: lab_median(box_pixels(o, pr["boxes"])) for k, o in outs.items()}})
        boxes = [b for pr in ph.get("probes", []) for b in pr["boxes"]]
        pw = USER_PANEL_W
        panels = [to_u8(cv2.resize(o.astype(np.float32), (pw, round(o.shape[0] * pw / o.shape[1])), interpolation=cv2.INTER_AREA))
                  for o in outs.values()]
        labels = ["Stock (P3 -> sRGB)"] + [USER_LABELS.get(k, k) for k in luts]
        path = out_dir / f"user_test_{ph['name']}.jpg"
        size = compose(panels, labels, pw, path, boxes)
        results.append({"name": ph["name"], "file": ph["file"], "profile": d.profile, "size": f"{w}x{h}",
                        "preview": path.name, "preview_size": size, "probes": probes})
        log(f"[user] {path} ({size})")
    return results


USER_LABELS = {"v1": "v1", "v2": "v2 (fitted exposure)", "v2_bright": "v2 bright (+0.75 stop)",
               "v2_brighter": "v2 brighter (+1.5 stop)"}


# ----------------------------------------------------------------------------- look description
HUE_SECTORS = [("red", 345, 25), ("orange", 25, 60), ("yellow", 60, 100), ("green", 100, 170),
               ("cyan", 170, 230), ("blue", 230, 300), ("magenta", 300, 345)]
HUE_ORDER = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]  # increasing CIELab hue angle


def hue_mask(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (h >= lo) & (h < hi) if lo < hi else (h >= lo) | (h < hi)


def wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def toward(sector: str, dh: float) -> str:
    if abs(dh) < 3:
        return "no clear hue shift"
    i = HUE_ORDER.index(sector)
    return "toward " + HUE_ORDER[(i + (1 if dh > 0 else -1)) % len(HUE_ORDER)]


def describe_look(lut: np.ndarray, pairs: list[PairData]) -> dict:
    rng = np.random.default_rng(SEED + 3)
    S, Y = [], []
    for pr in pairs:
        x, y = sample_pixels(pr, pr.valid, rng, 40000)
        S.append(enc(np.clip(x, 0, 1)))
        Y.append(enc(np.clip(y, 0, 1)))
    S, Y = np.concatenate(S), np.concatenate(Y)
    li, lo_, lc = enc_to_lab(S), enc_to_lab(apply_lut(lut, S)), enc_to_lab(Y)
    Ci, Co, Cc = np.hypot(li[:, 1], li[:, 2]), np.hypot(lo_[:, 1], lo_[:, 2]), np.hypot(lc[:, 1], lc[:, 2])
    hi_ = np.degrees(np.arctan2(li[:, 2], li[:, 1])) % 360
    ho = np.degrees(np.arctan2(lo_[:, 2], lo_[:, 1])) % 360
    hc = np.degrees(np.arctan2(lc[:, 2], lc[:, 1])) % 360
    chrom = Ci > 20
    sat_in = Ci[chrom] / np.maximum(li[chrom, 0], 1.0)
    info = {"chroma": {"lut": float(np.median(Co[chrom] / Ci[chrom])), "data": float(np.median(Cc[chrom] / Ci[chrom])),
                       "lut_sat": float(np.median(Co[chrom] / np.maximum(lo_[chrom, 0], 1.0) / sat_in)),
                       "data_sat": float(np.median(Cc[chrom] / np.maximum(lc[chrom, 0], 1.0) / sat_in))}}
    sectors = []
    for name, a, b in HUE_SECTORS:
        m = chrom & hue_mask(hi_, a, b)
        row = {"sector": name, "n": int(m.sum())}
        if m.sum() >= 300:
            row.update({"lut_dL": float(np.median(lo_[m, 0] - li[m, 0])), "lut_C": float(np.median(Co[m] / Ci[m])),
                        "lut_dh": float(np.median(wrap(ho[m] - hi_[m]))),
                        "data_dL": float(np.median(lc[m, 0] - li[m, 0])), "data_C": float(np.median(Cc[m] / Ci[m])),
                        "data_dh": float(np.median(wrap(hc[m] - hi_[m])))})
        sectors.append(row)
    info["sectors"] = sectors
    sky = hue_mask(hi_, 200, 290) & (li[:, 0] > 55) & (Ci > 8)
    info["sky"] = {"n": int(sky.sum())}
    if sky.sum() >= 300:
        info["sky"].update({"lut_dL": float(np.median(lo_[sky, 0] - li[sky, 0])), "lut_db": float(np.median(lo_[sky, 2] - li[sky, 2])),
                            "data_db": float(np.median(lc[sky, 2] - li[sky, 2]))})
    return info


# ----------------------------------------------------------------------------- report helpers
def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def mp(d: dict) -> str:
    return f"{d['median']:.2f} / {d['p95']:.2f}"


def lab_str(v: dict | None, with_L: bool = True) -> str:
    if v is None:
        return "n/a"
    return (f"L* {v['L']:.0f}, " if with_L else "") + f"a* {v['a']:+.1f}, b* {v['b']:+.1f}"


def icc_label(d: Decoded) -> str:
    if d.profile.startswith("no embedded"):
        return "none → sRGB"
    return d.profile.split("'")[1] if "'" in d.profile else d.profile


def ramp_L(lut: np.ndarray, levels) -> list:
    g = np.asarray(levels, float)
    return [float(v) for v in enc_to_lab(apply_lut(lut, np.repeat(g[:, None], 3, axis=1)))[:, 0]]


def plot_tone_curves(models: dict, pairs: list[PairData], path: Path) -> None:
    rng = np.random.default_rng(SEED + 5)
    ref = models["v2"]
    gm = ref.median_gain()
    s = np.linspace(0, 1, 256)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)
    ax.plot(s, s, color="0.6", lw=1, ls="--", label="identity")
    for pr in pairs:
        x, y = sample_pixels(pr, pr.valid, rng, 30000)
        se = enc(np.clip((x @ LUMA64) * 2.0 ** (ref.log2_gains[pr.name] - gm), 0, 1))
        ye = enc(np.clip(y @ LUMA64, 0, 1))
        edges = np.linspace(0, 1, 41)
        idx = np.digitize(se, edges) - 1
        cx, cy = [], []
        for k in range(40):
            m = idx == k
            if m.sum() >= 30:
                cx.append(se[m].mean())
                cy.append(np.median(ye[m]))
        ax.plot(cx, cy, "o", ms=2.5, alpha=0.7, label=pr.name)
    for key, style in (("v1", ":"), ("data", "--"), ("v2", "-")):
        m = models[key]
        ax.plot(s, np.clip(m.curve(np.log2(np.maximum(dec(s), TINY)) + m.median_gain()), 0, 1), color="k", lw=2,
                ls=style, label=f"tone curve: {SHORT.get(key, key)}")
    ax.set_xlabel("stock sRGB-encoded value (data shifted to the v2 median gain)")
    ax.set_ylabel("Chromagraph sRGB-encoded value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True, help="folder with the pair images named in pairs.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="folder for the .cube files, previews and report.md")
    ap.add_argument("--pairs", type=Path, default=here / "pairs.json")
    ap.add_argument("--user-photos-dir", type=Path, default=None,
                    help="folder with the user's own photos (visual check only; never used for fitting)")
    ap.add_argument("--user-photos", type=Path, default=here / "user_photos.json")
    return ap.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    args = parse_args()
    t_start = time.time()
    data_dir, out_dir = args.data_dir.resolve(), args.out_dir.resolve()
    if out_dir == data_dir or data_dir in out_dir.parents:
        sys.exit("--out-dir must not be inside --data-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = out_dir / "diagnostics"
    diag.mkdir(exist_ok=True)
    specs = load_specs(args.pairs)
    spec_by = {s.name: s for s in specs}
    images, regs, pdata, fit_names = prepare_pairs(specs, data_dir, diag)
    sets = {"round1": [n for n in fit_names if spec_by[n].round <= 1], "all": fit_names}
    if len(sets["all"]) < 2 or len(sets["round1"]) < 2:
        sys.exit(f"need at least two registered fit pairs per data set; got {sets}")
    anchor = fit_names[0]
    cache: dict = {}

    def get_model(names: list[str], toe: bool) -> LookModel:
        key = (tuple(names), toe)
        if key not in cache:
            t = time.time()
            cache[key] = fit_look([pdata[n] for n in names], toe)
            log(f"[fit] {len(names)} pairs, toe={toe}: {time.time() - t:.1f}s")
        return cache[key]

    # ---- final models and LUTs
    models = {v.key: get_model(sets[v.data], v.toe) for v in VARIANTS}
    luts = {v.key: models[v.key].bake_variant(v) for v in VARIANTS}
    luts.update({k: brighten(luts["v2"], st) for k, st in BRIGHT.items()})
    for key, (fname, desc) in CUBES.items():
        m = models["v1" if key == "v1" else "v2"]
        gm_rel = m.median_gain() - m.log2_gains[next(iter(m.log2_gains))]
        comments = [
            f"Approximate Chromagraph film-print look, {desc}.",
            "Fitted from public stock-iPhone vs Chromagraph photo pairs by fit/fit_look.py. Approximation only; see report.md.",
            "INPUT: sRGB-encoded RGB (IEC 61966-2-1 transfer curve), Rec.709/sRGB primaries, D65. Convert Display P3",
            "photos to sRGB before applying. Domain 0-1.",
            "OUTPUT: sRGB-encoded RGB, Rec.709/sRGB primaries, D65, clamped to 0-1.",
            "The input is Apple's processed photo, not RAW or scene-linear data. No grain or halation.",
            f"Baked exposure: median per-pair gain ({gm_rel:+.2f} stops relative to pair '{next(iter(m.log2_gains))}').",
            "Order: red index varies fastest, then green, then blue.",
        ]
        write_cube(out_dir / fname, luts[key], f"Chromagraph approx {key}", comments)
        back = read_cube(out_dir / fname)
        if float(np.abs(back - luts[key]).max()) > 1e-5:
            sys.exit(f"cube round-trip mismatch for {fname}")
        luts[key] = back
        log(f"[lut] wrote {out_dir / fname}")

    # ---- leave one pair out
    loo = {}
    for held in fit_names:
        t = time.time()
        rng = np.random.default_rng(SEED + 2)
        fold_models = {v.key: get_model([n for n in sets[v.data] if n != held], v.toe) for v in VARIANTS}
        fl = {v.key: fold_models[v.key].bake_variant(v) for v in VARIANTS}
        fl.update({k: brighten(fl["v2"], st) for k, st in BRIGHT.items()})
        m2 = fold_models["v2"]
        tone = m2.bake(m2.median_gain(), residual=False)
        pr = pdata[held]
        loo[held] = {
            "valid": evaluate({"identity": None, "tone": tone} | fl, pr, pr.valid, rng),
            "exposure_matched": evaluate({"identity": None, "v1": fl["v1"], "v2": fl["v2"]}, pr, pr.valid, rng, True),
            "loose": evaluate({"identity": None, "v1": fl["v1"], "v2": fl["v2"]}, pr, pr.loose, rng),
            "white_target": m2.white["target_L"],
        }
        v = loo[held]["valid"]
        log(f"[loo] {held}: identity {mp(v['identity'])}, v1 {mp(v['v1'])}, data {mp(v['data'])}, toe {mp(v['toe'])}, "
            f"neutral {mp(v['neutral'])}, white {mp(v['white'])}, v2 {mp(v['v2'])} ({time.time() - t:.1f}s)")

    # ---- in-sample (v2)
    m2 = models["v2"]
    insample = {}
    rng = np.random.default_rng(SEED + 6)
    for n in fit_names:
        g = m2.log2_gains[n]
        insample[n] = evaluate({"identity": None, "tone_own": m2.bake(g, residual=False),
                                "v2_own": m2.bake(g, neutral=True, white=True), "v2": luts["v2"], "v1": luts["v1"]},
                               pdata[n], pdata[n].valid, rng)

    # ---- previews, probes, user photos
    prev_luts = {"v1": luts["v1"], "v2": luts["v2"]}
    probes = []
    for spec in specs:
        stock, chroma = images[spec.name]
        size = make_preview(stock, chroma, regs.get(spec.name), prev_luts, out_dir / f"preview_{spec.name}.jpg")
        log(f"[preview] preview_{spec.name}.jpg ({size})")
        probes += measure_pair_probes(spec, stock, chroma, regs.get(spec.name), pdata.get(spec.name),
                                      {"v1": luts["v1"], "v2": luts["v2"]}, diag)
    user = []
    if args.user_photos_dir is not None:
        user = run_user_tests(args.user_photos, args.user_photos_dir.resolve(),
                              {k: luts[k] for k in ("v1", "v2", "v2_bright", "v2_brighter")}, out_dir)

    plot_tone_curves(models, [pdata[n] for n in fit_names], diag / "tone_curve.png")
    ctx = build_context(args, specs, spec_by, images, regs, pdata, fit_names, sets, anchor, models, luts, loo,
                        insample, probes, user, t_start)
    write_report(out_dir / "report.md", ctx)
    (diag / "metrics.json").write_text(json.dumps(ctx["metrics"], indent=2, default=float), encoding="utf-8")
    log(f"[done] report {out_dir / 'report.md'} in {time.time() - t_start:.0f}s")
    return 0


# ----------------------------------------------------------------------------- report
RAMP_LEVELS = [0.05, 0.10, 0.20, 0.4612, 0.75, 1.0]


def build_context(args, specs, spec_by, images, regs, pdata, fit_names, sets, anchor, models, luts, loo,
                  insample, probes, user, t_start) -> dict:
    ctx: dict = {"command": f"{COMMAND_PREFIX} {shlex.join(sys.argv[1:])}", "anchor": anchor}
    held = list(loo)
    methods = ["identity", "v1", "data", "toe", "neutral", "white", "v2", "w93", "v2_bright", "v2_brighter", "tone"]
    mean = {m: {s: float(np.mean([loo[h]["valid"][m][s] for h in held])) for s in ("median", "p95")} for m in methods}
    ctx["mean"] = mean
    base = mean["data"]["median"]
    ctx["effects"] = []
    for m in ["v1", "data", "toe", "neutral", "white", "v2", "w93", "v2_bright", "v2_brighter"]:
        ref = "data" if m in ("toe", "neutral", "white") else ("v2" if m == "w93" else "v1")
        worse = [h for h in held if loo[h]["valid"][m]["median"] - loo[h]["valid"][ref]["median"] > 0.5]
        ctx["effects"].append({"key": m, "median": mean[m]["median"], "p95": mean[m]["p95"],
                               "delta": mean[m]["median"] - mean[ref]["median"], "ref": ref, "worse": worse})
    ctx["effects_by"] = {e["key"]: e for e in ctx["effects"]}
    ctx["flags"] = [e for e in ctx["effects"] if e["key"] in ("neutral", "white", "v2") and (e["delta"] > 0.5 or e["worse"])]
    ctx["wins"] = sum(loo[h]["valid"]["v2"]["median"] < loo[h]["valid"]["v1"]["median"] for h in held)
    ctx["held"] = held
    ctx["base"] = base

    # Neutral evidence and its effect.
    m2 = models["v2"]
    ctx["support"] = m2.support

    def band(lo, hi):
        rows = [b for b in m2.support if b["n"] and b["lo"] >= lo and b["hi"] <= hi]
        if not rows:
            return None
        return wmedian([b["b"] for b in rows], [b["n"] for b in rows]), sorted({p for b in rows for p in b["pairs"]})

    hl_pairs = {}
    for b in m2.support:
        if b["lo"] >= 75:
            for e in b["per_pair"]:
                hl_pairs.setdefault(e["pair"], []).append((e["n"], e["b_in"], e["b"]))
    per = "; ".join(f"{p} {wmedian([x[1] for x in v], [x[0] for x in v]):+.1f} → {wmedian([x[2] for x in v], [x[0] for x in v]):+.1f}"
                    for p, v in hl_pairs.items())
    sh, md, hi = band(0, 25), band(25, 55), band(75, 100)
    ctx["neutral_summary"] = (
        "**What the evidence shows.** Near-neutral stock pixels come out at b* "
        + (f"{sh[0]:+.1f} in shadows (L* < 25), " if sh else "")
        + (f"{md[0]:+.1f} in mid-tones (L* 25–55) " if md else "")
        + (f"and {hi[0]:+.1f} in highlights (L* > 75)" if hi else "")
        + f". Per pair, stock b* → Chromagraph b* for near-neutral highlights (L* > 75): {per}. "
        + ("Every pair with such pixels, including the indoor ones, turns neutral highlights cream, so the data does not support "
           "keeping already-neutral whites neutral: the constraint removes the v1 cast in shadows and mid-tones but keeps cream "
           "highlights at the level the data shows." if hi and hi[0] > 3 else
           "The data keeps neutral highlights close to neutral, so the constraint removes the cream cast from neutral inputs.")
    )
    grid = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    ctx["neutral_ramp"] = {k: neutral_ramp(luts[k], grid).tolist() for k in ("v1", "data", "v2", "v2_bright", "v2_brighter")}
    ctx["neutral_grid"] = grid.tolist()
    ctx["white"] = {"v1_L": float(enc_to_lab(luts["v1"][-1, -1, -1])[0]), "v2_L": float(enc_to_lab(luts["v2"][-1, -1, -1])[0]),
                    "data_L": float(enc_to_lab(luts["data"][-1, -1, -1])[0])} | m2.white
    ctx["ramp"] = {k: ramp_L(luts[k], RAMP_LEVELS) for k in ("v1", "data", "toe", "v2", "v2_bright", "v2_brighter")}
    ctx["gains"] = {k: {n: models[k].log2_gains[n] - models[k].log2_gains[anchor] for n in models[k].log2_gains}
                    for k in ("v1", "data", "toe")}
    ctx["median_gain"] = {k: models[k].median_gain() - models[k].log2_gains[anchor] for k in ("v1", "data", "toe")}
    ctx["probes"], ctx["user"] = probes, user
    ctx["coverage"] = lattice_coverage(m2, [pdata[n] for n in fit_names])
    ctx["smoothness"] = lut_smoothness(luts["v2"])
    ctx["look"] = describe_look(luts["v2"], [pdata[n] for n in fit_names])
    ctx["insample"] = insample
    ctx["loo"] = loo
    ctx["specs"], ctx["regs"], ctx["images"], ctx["fit_names"], ctx["sets"] = specs, regs, images, fit_names, sets
    ctx["spec_by"] = spec_by
    ctx["runtime_s"] = time.time() - t_start
    ctx["metrics"] = {
        "fit_pairs": fit_names, "round1_pairs": sets["round1"], "anchor": anchor, "mean_held_out": mean,
        "effects": ctx["effects"], "leave_one_out": loo, "in_sample": insample, "neutral_support": m2.support,
        "white": ctx["white"], "neutral_ramp": ctx["neutral_ramp"], "ramp_L": ctx["ramp"], "gains": ctx["gains"],
        "probes": probes, "user_tests": user, "coverage": ctx["coverage"], "smoothness": ctx["smoothness"],
        "look": ctx["look"],
        "registration": {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.__dict__.items()}
                         for n, r in regs.items()},
        "settings": {"NEUTRAL_C_MAX": NEUTRAL_C_MAX, "NEUTRAL_W_C": NEUTRAL_W_C, "FALLBACK_B": FALLBACK_B,
                     "WHITE_PCT": WHITE_PCT, "WHITE_KNEE_L": WHITE_KNEE_L, "TOE": [TOE_LO, TOE_HI], "BRIGHT": BRIGHT,
                     "RBF_WIDTH": RBF_WIDTH, "RBF_SMOOTH": RBF_SMOOTH, "SEED": SEED},
        "runtime_s": ctx["runtime_s"],
    }
    return ctx


def write_report(path: Path, c: dict) -> None:
    L: list[str] = []
    held, loo, mean = c["held"], c["loo"], c["mean"]
    L.append("# Chromagraph look approximation: fit report (v2)\n")
    L.append("Command (run from the repository root):\n")
    L.append("```\n" + c["command"] + "\n```\n")
    L.append(f"Runtime {c['runtime_s']:.0f} s, seed {SEED}. LUTs: " + ", ".join(f"`{f}` ({d})" for f, d in CUBES.values())
             + ". Previews: `preview_<pair>.jpg` (stock | v1 | v2 | Chromagraph), `user_test_<photo>.jpg`. "
               "Diagnostics: `diagnostics/` (registration overlays, probe boxes, tone curves, `metrics.json`).\n")

    # ---- verdict
    L.append("## Verdict\n")
    v1m, v2m = mean["v1"], mean["v2"]
    L.append(
        f"Averaged over all {len(held)} held-out pairs, v2's CIEDE2000 is **{v2m['median']:.2f} median / {v2m['p95']:.2f} p95** "
        f"against **{v1m['median']:.2f} / {v1m['p95']:.2f}** for v1 (identity: {mean['identity']['median']:.2f} / "
        f"{mean['identity']['p95']:.2f}). v2 has the lower median on {c['wins']} of {len(held)} pairs. "
        + ("**v2 is closer to the real Chromagraph images than v1 on held-out pairs.**" if v2m["median"] < v1m["median"]
           else "**v2 is not closer than v1 on held-out pairs.**")
        + " The brighter versions are deliberately further from the data: "
        f"{mean['v2_bright']['median']:.2f} (bright) and {mean['v2_brighter']['median']:.2f} (brighter).\n")
    worse_than_id = [h for h in held if loo[h]["valid"]["v2"]["median"] >= loo[h]["valid"]["identity"]["median"]]
    if worse_than_id:
        L.append("- **Fails identity:** on " + ", ".join(
            f"`{h}` ({loo[h]['valid']['v2']['median']:.2f} against {loo[h]['valid']['identity']['median']:.2f} unchanged)"
            for h in worse_than_id) + " v2 is further from the real Chromagraph image than the unchanged stock image.")
    if c["flags"]:
        for e in c["flags"]:
            L.append(f"- **Flag:** `{SHORT[e['key']]}` changes the mean held-out median by {e['delta']:+.2f} against "
                     f"`{SHORT[e['ref']]}`" + (f"; worse by more than 0.5 on {', '.join(e['worse'])}" if e["worse"] else "") + ".")
        L.append("")
    if c["user"]:
        L.append("User photos (median over each probe box; stock → v1 → v2 → v2 bright → v2 brighter):\n")
        rows = []
        for u in c["user"]:
            for p in u["probes"]:
                vals = p["values"]
                rows.append([u["name"], p["probe"]] + [lab_str(vals[k]) for k in ("stock", "v1", "v2", "v2_bright", "v2_brighter")])
        L.append(table(["Photo", "Probe", "Stock", "v1", "v2", "v2 bright", "v2 brighter"], rows))
        L.append("")

    # ---- v2 changes
    L.append("## v2 changes\n")
    new = [s for s in c["specs"] if s.round > 1]
    status = []
    for s in new:
        r = c["regs"].get(s.name)
        st = "preview only" if r is None else ("fitted" if r.accepted else f"dropped ({r.reason})")
        wt = f", weight {s.weight:g}" if s.weight != 1 else ""
        status.append(f"`{s.name}` {st}{wt}")
    L.append("**New data.** " + "; ".join(status) + ".\n")
    L.append("**Effect of each change on held-out scores** (mean over all held-out pairs, CIEDE2000 median / p95). "
             "`+data` is the v1 method refitted with the new pairs; `+toe`, `+neutral` and `+white` each add one change to "
             "`+data`; v2 has all three. Δ is the change in mean median against the row named in brackets.\n")
    rows = []
    for e in c["effects"]:
        rows.append([VARIANT[e["key"]].label if e["key"] in VARIANT else SHORT[e["key"]], f"{e['median']:.2f} / {e['p95']:.2f}",
                     f"{e['delta']:+.2f} ({SHORT[e['ref']]})" if e["key"] != "v1" else "—",
                     ", ".join(e["worse"]) if e["worse"] and e["key"] != "v1" else "—"])
    L.append(table(["Variant", "Held-out median / p95", "Δ median", "Pairs worse by > 0.5"], rows))
    L.append("")
    L.append("**v2 against v1 on every held-out pair** (median / p95):\n")
    rows = []
    for h in held:
        v = loo[h]["valid"]
        tag = " (same RAW)" if c["spec_by"][h].same_exposure else ""
        rows.append([h + tag, mp(v["identity"]), mp(v["v1"]), mp(v["data"]), mp(v["toe"]), mp(v["neutral"]), mp(v["white"]),
                     mp(v["v2"]), f"{v['v2']['median'] - v['v1']['median']:+.2f}"])
    L.append(table(["Held out", "Identity", "v1", "+data", "+toe", "+neutral", "+white", "v2", "v2 − v1 median"], rows))
    L.append("")

    # neutral axis
    L.append("### 1. Neutral axis\n")
    L.append(f"Evidence: real Chromagraph a*, b* of stock pixels with C* < {NEUTRAL_C_MAX:g}, by stock L* (each pair's bin median "
             "weighted by its neutral weight; `sliwex` gets 0 because of its night white balance). Where a bin has evidence, neutral "
             f"inputs are pulled to it; elsewhere the LUT's own neutral tint is clamped to a* {FALLBACK_A}, b* {FALLBACK_B}. "
             f"The pull fades with input chroma (half at C* {NEUTRAL_W_C:g}), so bluish whites keep their own shift: in the data "
             "it is skies that start blue and end cream.\n")
    L.append(c["neutral_summary"] + "\n")
    rows = []
    for b in c["support"]:
        if b["n"]:
            rows.append([f"{b['lo']:.0f}–{b['hi']:.0f}", f"{b['n']:,}", ", ".join(b["pairs"]),
                         f"{b['a_in']:+.1f} / {b['b_in']:+.1f}", f"{b['a']:+.1f} / {b['b']:+.1f}"])
    L.append(table(["Stock L*", "Pixels", "Pairs", "Stock a* / b*", "Chromagraph a* / b*"], rows))
    L.append("")
    L.append("Named neutral-white probes (median over the boxes in `diagnostics/probes_*.jpg`):\n")
    rows = []
    for p in c["probes"]:
        rows.append([p["pair"], p["probe"], lab_str(p["stock"]), lab_str(p["chroma"]) + (f" ({p['how']})" if p["how"] else ""),
                     lab_str(p["luts"]["v1"]), lab_str(p["luts"]["v2"])])
    L.append(table(["Pair", "Probe", "Stock", "Real Chromagraph", "v1", "v2"], rows))
    L.append("")
    g = c["neutral_grid"]
    rows = []
    for k in ("v1", "data", "v2"):
        rows.append([SHORT[k]] + [f"{v[1]:+.1f} / {v[2]:+.1f}" for v in c["neutral_ramp"][k]])
    L.append("Neutral inputs through each LUT (output a* / b* at input L*):\n")
    L.append(table(["LUT"] + [f"L* {x:.0f}" for x in g], rows))
    L.append("")

    # white
    w = c["white"]
    L.append("### 2. White point\n")
    rows = []
    for n, npx in w["n_px"].items():
        lvl = w["levels"].get(n)
        note = "votes" if n in w["voters"] else ("too few stock-white pixels" if lvl is None else "not voting: deliberate exposure")
        rows.append([n, f"{npx:,}", "—" if lvl is None else f"{lvl:.1f}", f"{w['percentile'].get(n, float('nan')):.1f}", note])
    L.append(f"Measured as the median real-Chromagraph L* where the stock image is at white (stock L* ≥ {WHITE_STOCK_L:g}, clipped "
             f"pixels included, since that is exactly what the LUT's white node receives); the {WHITE_PCT:g}th percentile of each "
             "whole Chromagraph frame is shown for comparison.\n")
    L.append(table(["Pair", "Stock-white px", "Chromagraph L* there", f"Chromagraph {WHITE_PCT:g}th pct L*", "Use"], rows))
    w93 = c["effects_by"]["w93"]
    L.append(f"\nTarget = weighted median of the voting pairs = **L* {w['target_L']:.1f}**. The white node moves from L* "
             f"{w['v1_L']:.1f} (v1) / {w['data_L']:.1f} (+data) to {w['v2_L']:.1f} (v2) through a smooth highlight remap above "
             f"L* {WHITE_KNEE_L:g} that keeps chromaticity; the monotone-lightness projection runs first and still holds. "
             f"The L* {WHITE_ASKED_L:g} figure came from single highlights (the `vsgfield` sun, the `bencsinC` window); across "
             f"pairs the measured level is lower. Anchoring at L* {WHITE_ASKED_L:g} instead scores {w93['median']:.2f} / "
             f"{w93['p95']:.2f} held out ({w93['delta']:+.2f} median against v2).\n")

    # toe
    L.append("### 3. Exposure and toe\n")
    gtxt = "; ".join(f"{n} {c['gains']['data'][n]:+.2f} → {c['gains']['toe'][n]:+.2f}" for n in c["gains"]["data"])
    L.append(f"`sliwex` and `fahim` get toe weight 0.25 below stock sRGB luma {TOE_LO} (full weight from {TOE_HI}) when fitting the "
             f"tone curve. Gains in stops relative to `{c['anchor']}`, +data → +toe: {gtxt}. Baked median gain "
             f"{c['median_gain']['data']:+.2f} → {c['median_gain']['toe']:+.2f}.\n")
    rd, rt, r1 = c["ramp"]["data"], c["ramp"]["toe"], c["ramp"]["v1"]
    shift = max(abs(a - b) for a, b in zip(rd, rt))
    L.append(f"Effect: the toe weighting moves the neutral ramp by at most {shift:.1f} L* (mid-grey {rd[3]:.1f} → {rt[3]:.1f}; "
             f"sRGB 0.20 {rd[2]:.1f} → {rt[2]:.1f}) and the held-out mean median by {c['effects_by']['toe']['delta']:+.2f}. "
             f"With {len(c['fit_names'])} fitted pairs, these two no longer drive the toe. The rise in mid-grey from v1 "
             f"(L* {r1[3]:.0f}) to v2 (L* {c['ramp']['v2'][3]:.0f}) comes from the new pairs, not from the toe weighting.\n")
    rows = [[SHORT[k]] + [f"{v:.0f}" for v in vals] for k, vals in c["ramp"].items()]
    L.append("Neutral ramp, output L* (input sRGB value in the header; stock L* is 4, 9, 21, 49, 77, 100):\n")
    L.append(table(["LUT"] + [f"{x:.2f}" for x in RAMP_LEVELS], rows))
    L.append(f"\n`v2 bright` and `v2 brighter` apply lin·g/(1+(g−1)·lin) per channel to the v2 output in linear light "
             "with g = 2^0.75 and 2^1.5, normalised to the v2 white node so the anchored white stays fixed (the literal curve "
             "would also lift white).\n")

    # ---- remaining sections
    L.append("## Pairs and registration\n")
    rows = []
    for s in c["specs"]:
        st, ch = c["images"][s.name]
        r = c["regs"].get(s.name)
        if r is None:
            rows.append([s.name, str(s.round), s.use, f"{s.weight:g}", icc_label(st), icc_label(ch), "—", "—", "—", "—", "preview only"])
            continue
        status = "fit" if r.accepted else f"dropped: {r.reason}"
        rows.append([s.name, str(s.round), s.use, f"{s.weight:g}", icc_label(st), icc_label(ch), f"{r.n_inliers}",
                     f"{r.reproj_median:.2f}", f"{r.struct_agree:.0%}", f"{r.n_valid:,}", status])
    L.append(table(["Pair", "Round", "Use", "Weight", "Stock ICC", "Chromagraph ICC", "Inliers", "Reproj. median (px)",
                    "Textured agree", "Valid px", "Status"], rows))
    L.append("")
    notes = [f"`{s.name}`: {s.deliberate_exposure}." for s in c["specs"] if s.deliberate_exposure]
    notes += [f"`{s.name}`: stock side is Apple's rendering of a RAW (same exposure as the Chromagraph side)."
              for s in c["specs"] if s.stock_source == "apple_raw_render"]
    notes += [f"`{s.name}`: top {s.mask_top_fraction:.0%} masked (labels)." for s in c["specs"] if s.mask_top_fraction > 0]
    L += [f"- {n}" for n in notes]
    L.append("")

    L.append("## Held-out details (v2)\n")
    rows = []
    for h in held:
        v, e, lo = loo[h]["valid"], loo[h]["exposure_matched"], loo[h]["loose"]
        rows.append([h, f"{v['n']:,}", mp(v["identity"]), mp(v["tone"]), mp(v["v2"]), mp(v["v2_bright"]), mp(v["v2_brighter"]),
                     f"{mp(e['identity'])} ({e['identity']['log2_gain']:+.1f})", f"{mp(e['v2'])} ({e['v2']['log2_gain']:+.1f})",
                     mp(lo["v2"])])
    L.append(table(["Held out", "Pixels", "Identity", "Tone curve only", "v2", "v2 bright", "v2 brighter",
                    "Identity, own exposure", "v2, own exposure", "v2, no residual masks"], rows))
    L.append("\n\"Own exposure\" gives each method one exposure gain (stops in brackets) chosen on the held-out Chromagraph image to "
             "minimise mean ΔE00, so it is optimistic. \"No residual masks\" scores every overlapping unclipped pixel.\n")
    rows = [[n, mp(r["identity"]), mp(r["tone_own"]), mp(r["v2_own"]), mp(r["v2"]), mp(r["v1"])] for n, r in c["insample"].items()]
    L.append("In-sample (all fit pairs; \"own\" uses the pair's fitted gain):\n")
    L.append(table(["Pair", "Identity", "Tone curve, own gain", "v2, own gain", "v2 as written", "v1 as written"], rows))
    L.append("")

    cov, sm = c["coverage"], c["smoothness"]
    L.append("## Coverage and smoothness (v2)\n")
    L.append(f"{cov['within_1_step']:.1%} of the 33³ nodes have at least {COVER_MIN} valid samples within one lattice step "
             f"({cov['within_2_steps']:.1%} within two); elsewhere the LUT is the tone curve plus a residual that fades to zero. "
             "Output L* drops when one input channel rises by one step: "
             + ", ".join(f"{k} {v['drop_gt1']:.2%} of steps by > 1 (worst {v['worst']:.1f})" for k, v in sm.items()) + ".\n")

    look = c["look"]
    L.append("## Observations about v2\n")
    ramp = c["ramp"]["v2"]
    L.append(f"- **Tone.** Neutral sRGB 0.05 / 0.20 / 0.46 / 0.75 / 1.00 come out at L* "
             + " / ".join(f"{v:.0f}" for v in [ramp[0], ramp[2], ramp[3], ramp[4], ramp[5]])
             + " (stock 4 / 21 / 49 / 77 / 100): deep shadows and mid-tones are pushed down hard, which is the main reason the "
               "fitted-exposure LUT looks dark; the bright and brighter versions trade fidelity for a lighter rendering.")
    ch = look["chroma"]
    parts = [f"{s['sector']} ΔL* {s['lut_dL']:+.0f}, chroma ×{s['lut_C']:.2f}, hue {s['lut_dh']:+.0f}° ({toward(s['sector'], s['lut_dh'])})"
             for s in look["sectors"] if "lut_C" in s]
    L.append(f"- **Colour.** Chroma ×{ch['lut']:.2f} (real ×{ch['data']:.2f}); saturation relative to lightness ×{ch['lut_sat']:.2f} "
             f"(real ×{ch['data_sat']:.2f}). By input hue: " + "; ".join(parts) + ".")
    if "lut_db" in look["sky"]:
        L.append(f"- **Skies.** Bright blue pixels: ΔL* {look['sky']['lut_dL']:+.0f}, Δb* {look['sky']['lut_db']:+.1f} "
                 f"(real Δb* {look['sky']['data_db']:+.1f}): skies still roll toward cream while neutral whites stay near the evidence.")
    L.append("")

    L.append("## Limits\n")
    L.append(
        f"- {len(c['fit_names'])} fitted pairs, all recompressed social-media JPEGs; colours none of them contain fall back to the tone curve.\n"
        "- The stock images are Apple's finished renders with local tone mapping, segmentation and night processing; a global "
        "LUT cannot undo processing that varies across the frame.\n"
        "- No grain or halation: the images are too downscaled and recompressed to measure them, and a LUT cannot add spatial effects.\n"
        "- Exposure differences between pairs are absorbed by per-pair gains; the baked median gain is a choice, and the "
        "fitted-exposure LUT is dark because the real Chromagraph renders are dark relative to Apple's lifted shadows.\n"
        "- Files without an ICC profile are decoded as sRGB; Display P3 colours outside sRGB clip on conversion and are masked.\n"
        "- The user photos have no Chromagraph version, so they show only how the LUTs render, not how close they are.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
