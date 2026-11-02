# Chromagraph look approximation: fit report (v2)

Command (run from the repository root):

```
uv run --no-project --python 3.12 --with numpy --with scipy --with pillow --with pillow-heif --with opencv-python-headless --with colour-science --with matplotlib python -I fit/fit_look.py --data-dir pairs-data --out-dir fit-out --user-photos-dir <your-photos-dir> --neutralize-script tools/neutralize_lut.py
```

Runtime 364 s, seed 20261102. LUTs: `chromagraph_approx_33.cube` (v1 (as shipped; kept for comparison)), `chromagraph_v2_33.cube` (v2, fitted exposure (closest to the data)), `chromagraph_v2_bright_33.cube` (v2 with shadows lifted 0.75 stop, white fixed), `chromagraph_v2_brighter_33.cube` (v2 with shadows lifted 1.5 stops, white fixed), `chromagraph_v3_33.cube` (v3: v2 plus hue-dependent chroma gain and rotation, fitted exposure), `chromagraph_v3_bright_33.cube` (v3 with shadows lifted 0.75 stop, white fixed). Previews: `preview_<pair>.jpg` (stock | v2 | v3 | Chromagraph), `user_test_<photo>.jpg`. Diagnostics: `diagnostics/` (registration overlays, probe boxes, tone curves, `metrics.json`).

## v3 verdict (round 3: saturated colours)

Averaged over all 9 held-out pairs, v3 scores **6.43 median / 14.85 p95** CIEDE2000 against **6.68 / 14.65** for v2, with the lower median on 4 of 9 pairs. **v3 beats v2 on the mean held-out median.**

`fahim` red-orange leaves: stock L 30, C 33, h 45 → v2 L 14, C 16, h 40 → v3 L 14, C 15, h 40 → real Chromagraph L 22, C 36, h 41.

Why `fahim` looks richer than v2 or v3: it is one of the few pairs where Chromagraph keeps warm chroma (reds/oranges ×1.27 against stock), while the weighted median over 8 pairs is ×0.66 and v2 already reproduces that held out. A LUT fitted to all pairs therefore cannot give `fahim` its own saturation. v2 also bakes a darker exposure than `fahim`'s Chromagraph render (leaves L 14 against 22), and CIELab chroma falls with lightness.

User photos (median over each probe box, L*, a*, b*):

| Photo | Probe | stock | v2 | v3 | v2 bright | v3 bright | v3 bright neutral |
|---|---|---|---|---|---|---|---|
| IMG_8693 | grey desk mat | L* 35, a* -0.2, b* +6.9 | L* 18, a* -1.2, b* +4.9 | L* 18, a* -1.6, b* +5.8 | L* 24, a* -1.4, b* +5.6 | L* 24, a* -1.9, b* +6.7 | L* 24, a* -1.4, b* +3.9 |
| IMG_8693 | silver laptop | L* 99, a* -0.8, b* +0.2 | L* 88, a* -0.5, b* +12.0 | L* 88, a* -0.7, b* +11.8 | L* 88, a* -0.6, b* +7.6 | L* 88, a* -0.7, b* +7.5 | L* 88, a* -0.1, b* +1.1 |
| IMG_8697 | grey desk mat | L* 28, a* +0.1, b* +6.9 | L* 13, a* -1.2, b* +4.6 | L* 13, a* -1.3, b* +5.0 | L* 18, a* -1.4, b* +5.3 | L* 18, a* -1.6, b* +5.8 | L* 18, a* -1.2, b* +3.5 |
| IMG_8697 | silver laptop | L* 94, a* -1.4, b* +1.5 | L* 84, a* -0.1, b* +12.6 | L* 84, a* -0.5, b* +12.7 | L* 86, a* -0.4, b* +8.5 | L* 86, a* -0.6, b* +8.6 | L* 86, a* -0.2, b* +1.5 |

## v3 changes

### 1. Is chroma retention consistent across pairs?

Real Chromagraph against stock, per fit pair, for stock pixels with C* > 20 and L* 15–85, excluding stock pixels with any channel ≥ 250. Each cell: chroma ratio (Chromagraph C* / stock C*), hue shift, ΔL*, pixels. Pooled = median over pairs weighted by pair weight (pairs with ≥ 300 pixels).

| Pair | reds/oranges (15–60°) | yellows (70–95°) | greens (95–150°) |
|---|---|---|---|
| vsgfield | — (0) | ×0.32, -4°, ΔL -31 (1,058) | ×0.28, +6°, ΔL -26 (121) |
| sliwex | ×0.31, -14°, ΔL -23 (55,080) | ×0.19, -23°, ΔL -27 (1,513) | ×0.22, -7°, ΔL -30 (945) |
| fahim | ×1.27, -2°, ΔL -14 (13,989) | ×1.14, -14°, ΔL -20 (3,202) | ×0.97, -8°, ΔL -18 (13,609) |
| aitree | ×0.62, +1°, ΔL -12 (77,153) | — (0) | — (0) |
| robby | ×0.73, +7°, ΔL -17 (4,388) | ×0.94, -5°, ΔL -12 (297) | ×0.93, -8°, ΔL -12 (694) |
| satoru | ×0.39, -3°, ΔL -15 (5,636) | ×0.62, -1°, ΔL -16 (1,495) | — (0) |
| pallav | ×1.03, +6°, ΔL -13 (652) | ×1.06, -3°, ΔL -14 (1,115) | ×1.08, -14°, ΔL -13 (292) |
| bencsinB | ×0.66, +3°, ΔL -3 (7,323) | ×0.76, +10°, ΔL -3 (5,477) | ×0.77, +10°, ΔL -4 (2,131) |
| bencsinC | ×0.10, +14°, ΔL -14 (1,364) | — (4) | — (0) |
| **pooled** | **×0.66, +3°, ΔL -14** (8 pairs) | **×0.62, -3°, ΔL -16** (6 pairs) | **×0.77, -8°, ΔL -12** (4 pairs) |

**Where pairs disagree** (chroma ratio more than 0.25 from the pooled value, or hue shift more than 8° off): reds/oranges: `sliwex` ×0.31, -14° (night scene; Chromagraph much darker (author: both straight out of camera)); `fahim` ×1.27, -2° (Chromagraph shot deliberately underexposed by the author); `satoru` ×0.39, -3°; `pallav` ×1.03, +6°; `bencsinC` ×0.10, +14° | yellows: `vsgfield` ×0.32, -4° (same RAW, so not an exposure difference); `sliwex` ×0.19, -23° (night scene; Chromagraph much darker (author: both straight out of camera)); `fahim` ×1.14, -14° (Chromagraph shot deliberately underexposed by the author); `pallav` ×1.06, -3°; `bencsinB` ×0.76, +10° | greens: `sliwex` ×0.22, -7° (night scene; Chromagraph much darker (author: both straight out of camera)); `bencsinB` ×0.77, +10°. Likely reasons: the stock image is Apple's per-scene rendering, whose saturation boost and local tone mapping vary from scene to scene and form the denominator of the ratio; deliberate exposure differences change CIELab chroma along with lightness; and night or mixed light changes white balance between the two renders. The lowest warm ratios come from `sliwex` (Night street, parcel locker under LED and orange sodium light); `satoru` (Lake through a window; white railing posts, pale blue sky, dry grass); `bencsinC` (Dark hallway with warm cove lighting and a bright window at the end): there the reds/oranges are largely surfaces lit by warm light, whose orange cast Chromagraph neutralizes. That is a white-balance effect rather than object saturation, but it still pulls the pooled warm ratio down. Warm chroma is kept or raised only in `fahim`, `pallav`, so "darker but just as chromatic" is not consistent across these pairs.

### 2. What v3 adds

After the tone curve and before the 3D residual, v3 multiplies CIELab a*+ib* by a complex gain G(h): |G| is a chroma gain and arg G a hue rotation, both functions of hue. G is periodic and piecewise linear over 24 knots (15° apart), fitted by weighted linear least squares over all fit pairs with their weights (51,895 bin-balanced samples, near-clipped stock pixels excluded, Huber reweighting). It has a penalty on second differences around the hue circle (λ 2) and a weak pull toward G = 1 where a hue has little data (λ 0.05). It fades in between input C* 4 and 12, so neutrals are untouched and the neutral-axis constraint still applies. I chose this over refitting the residual in L*C*h: the observed effect is a per-hue chroma ratio and rotation, which a 24-parameter complex gain captures directly. The RBF residual, smoothed and averaged across pairs, had pulled warm chroma down; it is refitted on top of the gained colours. Chroma above 85% of the sRGB limit at the same L* and hue is rolled off with tanh, after the gain and again after the residual, so nothing is hard-clipped. The monotone-lightness projection, white anchor and neutral constraint follow as in v2.

| Hue (after tone curve) | Chroma gain | Hue rotation | Data share within ±22.5° |
|---|---|---|---|
| reds/oranges (37°) | 0.68 | +7° | 57% |
| yellows (82°) | 0.76 | +0° | 42% |
| greens (122°) | 0.74 | -9° | 10% |
| blue sky (220°) | 0.61 | -107° | 1% |
| blues (270°) | 0.31 | -65° | 9% |
| magentas (330°) | 0.41 | +53° | 3% |

Plot: `diagnostics/hue_gain.png`. Share of lattice nodes with a channel at 0 or 1: v2 7.2%, v3 0.1%. Output L* still never drops when an input channel rises R 0.00% > 1, G 0.00% > 1, B 0.00% > 1.

### 3. Held-out comparison

| Held out | Identity | v2 | v3 | v3 − v2 median | v2 bright | v3 bright |
|---|---|---|---|---|---|---|
| vsgfield | 16.19 / 32.84 | 5.92 / 16.54 | 6.22 / 16.76 | +0.30 | 9.97 / 23.67 | 10.79 / 23.87 |
| sliwex | 21.09 / 29.44 | 10.37 / 16.14 | 9.57 / 15.85 | -0.79 | 14.61 / 21.45 | 13.94 / 21.50 |
| fahim | 14.65 / 19.28 | 6.12 / 10.24 | 6.15 / 10.96 | +0.03 | 7.64 / 14.54 | 7.70 / 14.55 |
| aitree | 10.96 / 17.52 | 5.37 / 13.73 | 5.43 / 13.70 | +0.05 | 4.31 / 11.65 | 3.85 / 11.22 |
| robby | 16.51 / 22.69 | 6.57 / 12.18 | 5.48 / 9.30 | -1.09 | 7.66 / 16.40 | 7.93 / 14.85 |
| satoru | 20.62 / 26.81 | 6.61 / 15.03 | 5.84 / 15.43 | -0.78 | 9.59 / 21.24 | 8.95 / 21.35 |
| pallav | 11.91 / 19.69 | 3.73 / 8.75 | 4.14 / 11.88 | +0.41 | 4.26 / 8.10 | 4.94 / 8.47 |
| bencsinB | 6.80 / 10.83 | 10.96 / 19.63 | 10.52 / 19.51 | -0.44 | 6.46 / 13.79 | 6.19 / 13.87 |
| bencsinC | 12.17 / 32.04 | 4.43 / 19.62 | 4.54 / 20.30 | +0.10 | 6.19 / 26.93 | 6.47 / 27.26 |
| **mean** |  | 6.68 / 14.65 | 6.43 / 14.85 | -0.25 | 7.85 / 17.53 | 7.86 / 17.44 |

Held-out chroma in the saturated sectors (median over held-out pairs, weighted; chroma ratio / hue shift against stock):

| Sector | Pairs | Real Chromagraph | v2 | v3 |
|---|---|---|---|---|
| reds/oranges | 8 | ×0.66 / +3° | ×0.64 / -1° | ×0.64 / +1° |
| yellows | 6 | ×0.61 / -3° | ×0.74 / -0° | ×0.74 / +2° |
| greens | 4 | ×0.77 / -7° | ×0.77 / +4° | ×0.74 / -3° |

Regions (in-sample: these pairs are in the fit; L*, C*, h):

| Pair | Region | Stock | v2 | v3 | Real Chromagraph |
|---|---|---|---|---|---|
| fahim | red-orange leaves | L 30, C 33, h 45 | L 14, C 16, h 40 | L 14, C 15, h 40 | L 22, C 36, h 41 |
| fahim | yellow chair | L 42, C 49, h 76 | L 23, C 30, h 79 | L 23, C 26, h 79 | L 27, C 43, h 55 |
| fahim | green grass | L 44, C 38, h 107 | L 27, C 28, h 108 | L 27, C 26, h 107 | L 34, C 36, h 100 |
| fahim | blue sky | L 76, C 8, h 255 | L 66, C 9, h 107 | L 67, C 10, h 106 | L 67, C 9, h 103 |
| aitree | bright red leaves | L 36, C 56, h 38 | L 18, C 33, h 36 | L 18, C 31, h 34 | L 24, C 38, h 38 |
| aitree | darker red leaves | L 22, C 33, h 30 | L 10, C 16, h 28 | L 9, C 13, h 26 | L 12, C 17, h 29 |
| aitree | blue sky | L 81, C 23, h 275 | L 70, C 7, h 141 | L 70, C 9, h 128 | L 67, C 8, h 120 |
| pallav | blue sky | L 67, C 26, h 274 | L 56, C 7, h 207 | L 56, C 9, h 215 | L 60, C 6, h 210 |
| pallav | sunlit beige building | L 32, C 8, h 65 | L 15, C 5, h 74 | L 16, C 5, h 74 | L 16, C 6, h 56 |
| pallav | red-brick building | L 41, C 8, h 48 | L 23, C 6, h 58 | L 24, C 7, h 59 | L 24, C 8, h 58 |
| pallav | bridge | L 39, C 5, h 37 | L 21, C 4, h 63 | L 22, C 4, h 55 | L 31, C 7, h 57 |

### 4. Files

`chromagraph_v3_33.cube` (fitted exposure) and `chromagraph_v3_bright_33.cube` (the same 0.75-stop lift as v2 bright), plus `chromagraph_v3_bright_neutral_33.cube`, written by running `neutralize_lut.py` on the v3 bright LUT (it pulls a*/b* of near-neutral inputs to zero at fixed L*). Previews now show stock | v2 | v3 | real Chromagraph; the user photos show stock | v2 | v3 | v2 bright | v3 bright | v3 bright neutral.

## v2 verdict (round 2)

Averaged over all 9 held-out pairs, v2's CIEDE2000 is **6.68 median / 14.65 p95** against **7.42 / 15.63** for v1 (identity: 14.54 / 23.46). v2 has the lower median on 7 of 9 pairs. **v2 is closer to the real Chromagraph images than v1 on held-out pairs.** The brighter versions are deliberately further from the data: 7.85 (bright) and 11.35 (brighter).

- **Fails identity:** on `bencsinB` (10.96 against 6.80 unchanged) v2 is further from the real Chromagraph image than the unchanged stock image.
- **Flag:** `v2` changes the mean held-out median by -0.75 against `v1`; worse by more than 0.5 on sliwex.

## v2 changes

**New data.** `robby` fitted, weight 2; `satoru` fitted, weight 2; `alli` dropped (only 44% of textured pixels agree after warping (< 50%)); `batman` preview only; `kiriya` preview only; `pallav` fitted; `bencsinB` fitted, weight 2; `bencsinC` fitted; `bencsinA` dropped (only 50% of textured pixels agree after warping (< 50%)), weight 0.5.

**Effect of each change on held-out scores** (mean over all held-out pairs, CIEDE2000 median / p95). `+data` is the v1 method refitted with the new pairs; `+toe`, `+neutral` and `+white` each add one change to `+data`; v2 has all three. Δ is the change in mean median against the row named in brackets.

| Variant | Held-out median / p95 | Δ median | Pairs worse by > 0.5 |
|---|---|---|---|
| v1 as shipped (round-1 pairs, v1 method) | 7.42 / 15.63 | — | — |
| v1 method + new pairs | 6.88 / 14.97 | -0.55 (v1) | sliwex |
| + toe down-weighting only | 6.82 / 14.61 | -0.06 (+data) | — |
| + neutral-axis constraint only | 6.73 / 14.99 | -0.15 (+data) | — |
| + white anchor only | 6.89 / 14.97 | +0.01 (+data) | — |
| v2 (all three changes) | 6.68 / 14.65 | -0.75 (v1) | sliwex |
| v2 with white at L* 93 instead of the measured level (comparison only) | 6.69 / 14.65 | +0.02 (v2) | satoru |
| v2 bright | 7.85 / 17.53 | +0.43 (v1) | vsgfield, sliwex, satoru, bencsinC |
| v2 brighter | 11.35 / 22.77 | +3.93 (v1) | vsgfield, sliwex, fahim, robby, satoru, pallav, bencsinC |

**v2 against v1 on every held-out pair** (median / p95):

| Held out | Identity | v1 | +data | +toe | +neutral | +white | v2 | v2 − v1 median |
|---|---|---|---|---|---|---|---|---|
| vsgfield (same RAW) | 16.19 / 32.84 | 7.02 / 18.41 | 5.64 / 16.82 | 6.06 / 16.37 | 5.52 / 16.95 | 5.64 / 16.82 | 5.92 / 16.54 | -1.10 |
| sliwex | 21.09 / 29.44 | 9.49 / 16.89 | 10.46 / 16.23 | 10.45 / 16.22 | 10.38 / 16.15 | 10.46 / 16.23 | 10.37 / 16.14 | +0.88 |
| fahim | 14.65 / 19.28 | 7.17 / 13.10 | 6.33 / 10.23 | 6.21 / 10.21 | 6.28 / 10.26 | 6.33 / 10.23 | 6.12 / 10.24 | -1.05 |
| aitree | 10.96 / 17.52 | 6.72 / 13.61 | 5.41 / 13.97 | 5.40 / 13.48 | 5.39 / 14.20 | 5.42 / 13.97 | 5.37 / 13.73 | -1.35 |
| robby (same RAW) | 16.51 / 22.69 | 7.48 / 13.06 | 6.73 / 12.47 | 6.77 / 12.19 | 6.66 / 12.46 | 6.70 / 12.47 | 6.57 / 12.18 | -0.91 |
| satoru | 20.62 / 26.81 | 7.26 / 15.37 | 6.54 / 15.28 | 6.46 / 14.98 | 6.54 / 15.23 | 6.68 / 15.30 | 6.61 / 15.03 | -0.64 |
| pallav | 11.91 / 19.69 | 5.25 / 9.26 | 4.64 / 9.61 | 4.30 / 8.79 | 4.09 / 9.58 | 4.64 / 9.61 | 3.73 / 8.75 | -1.52 |
| bencsinB | 6.80 / 10.83 | 12.02 / 21.73 | 11.40 / 20.11 | 10.97 / 19.62 | 11.39 / 20.13 | 11.39 / 20.11 | 10.96 / 19.63 | -1.06 |
| bencsinC | 12.17 / 32.04 | 4.39 / 19.22 | 4.73 / 19.96 | 4.71 / 19.57 | 4.30 / 19.97 | 4.73 / 19.96 | 4.43 / 19.62 | +0.04 |

### 1. Neutral axis

Evidence: real Chromagraph a*, b* of stock pixels with C* < 6, by stock L* (each pair's bin median weighted by its neutral weight; `sliwex` gets 0 because of its night white balance). Where a bin has evidence, neutral inputs are pulled to it; elsewhere the LUT's own neutral tint is clamped to a* (-1.0, 1.0), b* (-1.0, 3.0). The pull fades with input chroma (half at C* 6), so bluish whites keep their own shift: in the data it is skies that start blue and end cream.

**What the evidence shows.** Near-neutral stock pixels come out at b* -0.3 in shadows (L* < 25), +0.4 in mid-tones (L* 25–55) and +12.1 in highlights (L* > 75). Per pair, stock b* → Chromagraph b* for near-neutral highlights (L* > 75): vsgfield +3.1 → +15.0; fahim -3.4 → +11.9; aitree -2.4 → +7.1; robby -0.6 → +13.0; satoru -0.1 → +13.1; pallav +0.9 → +15.4; bencsinB +2.7 → +12.3; bencsinC +1.3 → +9.8. Every pair with such pixels, including the indoor ones, turns neutral highlights cream, so the data does not support keeping already-neutral whites neutral: the constraint removes the v1 cast in shadows and mid-tones but keeps cream highlights at the level the data shows.

| Stock L* | Pixels | Pairs | Stock a* / b* | Chromagraph a* / b* |
|---|---|---|---|---|
| 0–5 | 23,869 | fahim, robby, pallav, bencsinB, bencsinC | +0.9 / +1.3 | -0.5 / -0.0 |
| 5–10 | 32,033 | vsgfield, fahim, aitree, robby, pallav, bencsinB, bencsinC | +1.4 / +1.7 | -0.2 / -0.4 |
| 10–15 | 19,635 | vsgfield, fahim, aitree, robby, pallav, bencsinB, bencsinC | +1.8 / +1.6 | -0.2 / -0.3 |
| 15–20 | 22,885 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | +1.1 / +2.9 | -0.4 / -1.2 |
| 20–25 | 21,966 | vsgfield, fahim, aitree, robby, pallav, bencsinB, bencsinC | +0.4 / +2.4 | -0.3 / -0.2 |
| 25–30 | 18,713 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinC | +0.3 / +2.2 | +0.0 / +0.9 |
| 30–35 | 18,498 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinC | -0.1 / +2.7 | -0.7 / -0.1 |
| 35–40 | 16,603 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinC | -0.7 / +2.3 | -0.7 / +0.4 |
| 40–45 | 14,382 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -0.0 / +1.8 | -0.9 / +0.3 |
| 45–50 | 12,903 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | +0.3 / +0.9 | -1.2 / +1.4 |
| 50–55 | 17,518 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -0.3 / +0.4 | -1.4 / +1.6 |
| 55–60 | 21,939 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -0.5 / +0.2 | -1.5 / +3.6 |
| 60–65 | 21,339 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -1.1 / -0.7 | -2.3 / +4.7 |
| 65–70 | 22,275 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | +0.1 / -0.8 | -2.8 / +8.1 |
| 70–75 | 25,347 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | +0.8 / +1.3 | -2.8 / +9.9 |
| 75–80 | 19,597 | vsgfield, fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -0.2 / +1.1 | -1.9 / +12.2 |
| 80–85 | 14,103 | fahim, aitree, robby, satoru, pallav, bencsinB, bencsinC | -0.4 / +0.9 | -0.8 / +11.5 |
| 85–90 | 11,042 | fahim, aitree, satoru, pallav, bencsinB, bencsinC | -0.5 / +0.9 | +0.1 / +11.9 |
| 90–95 | 12,480 | fahim, satoru, pallav, bencsinB, bencsinC | -0.4 / -0.6 | +0.4 / +12.8 |
| 95–100 | 10,302 | satoru, bencsinB, bencsinC | -0.4 / +0.5 | -0.8 / +12.1 |

Named neutral-white probes (median over the boxes in `diagnostics/probes_*.jpg`):

| Pair | Probe | Stock | Real Chromagraph | v1 | v2 | v3 |
|---|---|---|---|---|---|---|
| robby | white structure (base container, right leg) | L* 31, a* +3.1, b* +8.2 | L* 24, a* +3.3, b* +7.8 (registered) | L* 11, a* +1.3, b* +6.0 | L* 15, a* +1.1, b* +5.6 | L* 15, a* +0.9, b* +5.8 |
| satoru | white railing posts | L* 84, a* +13.1, b* +24.4 | L* 52, a* +5.2, b* +14.7 (registered) | L* 72, a* +8.0, b* +24.6 | L* 69, a* +4.7, b* +19.9 | L* 71, a* +4.1, b* +19.8 |
| alli | cream wall | L* 84, a* +4.6, b* +13.8 | L* 72, a* +0.4, b* +20.6 (boxes placed in each frame) | L* 77, a* +3.8, b* +16.6 | L* 74, a* +1.5, b* +17.2 | L* 75, a* +1.4, b* +16.3 |
| alli | skin (forehead) | L* 65, a* +22.9, b* +21.7 | L* 48, a* +15.7, b* +22.4 (boxes placed in each frame) | L* 49, a* +21.6, b* +25.1 | L* 51, a* +13.7, b* +18.4 | L* 49, a* +14.1, b* +18.3 |
| bencsinB | white ceiling | L* 60, a* +3.4, b* +10.9 | L* 58, a* -1.6, b* +11.2 (registered) | L* 42, a* +7.1, b* +13.9 | L* 44, a* +2.6, b* +10.8 | L* 43, a* +2.7, b* +11.1 |
| bencsinB | white window shades | L* 73, a* +3.1, b* +13.9 | L* 75, a* +0.2, b* +15.9 (registered) | L* 59, a* +4.2, b* +18.1 | L* 62, a* +1.2, b* +16.1 | L* 63, a* +1.0, b* +15.0 |
| bencsinB | cream cushions | L* 87, a* +4.9, b* +19.7 | L* 83, a* +1.4, b* +16.4 (registered) | L* 79, a* +3.6, b* +19.6 | L* 76, a* +1.3, b* +18.3 | L* 77, a* +1.7, b* +18.0 |
| bencsinC | window | L* 98, a* -0.3, b* +1.0 | L* 94, a* -1.5, b* +9.8 (registered) | L* 87, a* +1.3, b* +12.6 | L* 87, a* -0.4, b* +12.0 | L* 87, a* -0.5, b* +12.1 |
| bencsinC | pale wall by window | L* 54, a* +2.2, b* +13.5 | L* 24, a* -0.3, b* +8.0 (registered) | L* 33, a* +6.5, b* +16.5 | L* 36, a* +2.2, b* +11.9 | L* 36, a* +2.0, b* +12.4 |
| bencsinC | lit ceiling (orange in stock) | L* 9, a* +8.2, b* +12.1 | L* 2, a* +0.6, b* +2.0 (registered) | L* 4, a* +0.6, b* +4.2 | L* 4, a* +0.2, b* +4.2 | L* 4, a* +0.9, b* +3.8 |
| bencsinA | white wall | L* 78, a* -0.3, b* +8.7 | n/a | L* 68, a* +1.5, b* +14.7 | L* 67, a* -0.5, b* +15.1 | L* 68, a* -0.9, b* +13.5 |

Neutral inputs through each LUT (output a* / b* at input L*):

| LUT | L* 10 | L* 30 | L* 50 | L* 70 | L* 90 | L* 100 |
|---|---|---|---|---|---|---|
| v1 | -0.5 / +0.9 | -1.4 / +3.4 | +0.9 / +3.3 | +0.1 / +8.4 | +2.0 / +13.2 | +1.2 / +11.6 |
| +data | -0.8 / -0.3 | -1.9 / +1.7 | -2.0 / -0.0 | -1.6 / +8.1 | +0.3 / +13.0 | +2.0 / +12.1 |
| v2 | -0.4 / -0.4 | -0.6 / +0.6 | -1.4 / +1.3 | -2.6 / +8.7 | +0.3 / +12.2 | -0.7 / +11.8 |

### 2. White point

Measured as the median real-Chromagraph L* where the stock image is at white (stock L* ≥ 97, clipped pixels included, since that is exactly what the LUT's white node receives); the 99.9th percentile of each whole Chromagraph frame is shown for comparison.

| Pair | Stock-white px | Chromagraph L* there | Chromagraph 99.9th pct L* | Use |
|---|---|---|---|---|
| vsgfield | 0 | — | 86.9 | too few stock-white pixels |
| sliwex | 2,817 | 93.6 | 94.0 | not voting: deliberate exposure |
| fahim | 2,190 | 81.2 | 82.2 | not voting: deliberate exposure |
| aitree | 10,678 | 86.8 | 91.4 | votes |
| robby | 1,057 | 88.5 | 88.5 | votes |
| satoru | 17,014 | 85.6 | 88.3 | votes |
| pallav | 632 | 85.3 | 86.5 | votes |
| bencsinB | 4,163 | 91.3 | 91.5 | votes |
| bencsinC | 6,806 | 94.0 | 94.2 | votes |

Target = weighted median of the voting pairs = **L* 88.5**. The white node moves from L* 87.4 (v1) / 86.6 (+data) to 88.5 (v2) through a smooth highlight remap above L* 65 that keeps chromaticity; the monotone-lightness projection runs first and still holds. The L* 93 figure came from single highlights (the `vsgfield` sun, the `bencsinC` window); across pairs the measured level is lower. Anchoring at L* 93 instead scores 6.69 / 14.65 held out (+0.02 median against v2).

### 3. Exposure and toe

`sliwex` and `fahim` get toe weight 0.25 below stock sRGB luma 0.3 (full weight from 0.46) when fitting the tone curve. Gains in stops relative to `vsgfield`, +data → +toe: vsgfield +0.00 → +0.00; sliwex +0.03 → +0.13; fahim +0.22 → +0.20; aitree +0.37 → +0.36; robby +0.31 → +0.30; satoru -0.12 → -0.13; pallav +0.33 → +0.32; bencsinB +0.78 → +0.77; bencsinC -0.29 → -0.28. Baked median gain +0.22 → +0.20.

Effect: the toe weighting moves the neutral ramp by at most 0.6 L* (mid-grey 31.4 → 31.0; sRGB 0.20 8.5 → 8.7) and the held-out mean median by -0.06. With 9 fitted pairs, these two no longer drive the toe. The rise in mid-grey from v1 (L* 26) to v2 (L* 31) comes from the new pairs, not from the toe weighting.

Neutral ramp, output L* (input sRGB value in the header; stock L* is 4, 9, 21, 49, 77, 100):

| LUT | 0.05 | 0.10 | 0.20 | 0.46 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|
| v1 | 2 | 3 | 6 | 26 | 68 | 87 |
| +data | 2 | 4 | 9 | 31 | 67 | 87 |
| +toe | 2 | 4 | 9 | 31 | 67 | 86 |
| v2 | 2 | 4 | 9 | 31 | 67 | 88 |
| v2 bright | 3 | 6 | 13 | 39 | 73 | 88 |
| v2 brighter | 5 | 10 | 19 | 47 | 78 | 88 |

`v2 bright` and `v2 brighter` apply lin·g/(1+(g−1)·lin) per channel to the v2 output in linear light with g = 2^0.75 and 2^1.5, normalised to the v2 white node so the anchored white stays fixed (the literal curve would also lift white).

## Pairs and registration

| Pair | Round | Use | Weight | Stock ICC | Chromagraph ICC | Inliers | Reproj. median (px) | Textured agree | Valid px | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| sliwex | 1 | fit | 1 | Display P3 | none → sRGB | 416 | 1.01 | 66% | 170,725 | fit |
| drift | 1 | fit | 1 | sRGB | none → sRGB | 435 | 2.44 | 17% | 127,801 | dropped: only 17% of textured pixels agree after warping (< 50%) |
| fahim | 1 | fit | 1 | Display P3 | none → sRGB | 5381 | 0.41 | 100% | 422,072 | fit |
| vsgfield | 1 | fit | 1 | none → sRGB | none → sRGB | 4964 | 0.42 | 100% | 341,510 | fit |
| vsgsun | 1 | fit | 1 | none → sRGB | none → sRGB | 10 | 1.79 | 3% | 244,177 | dropped: only 10 RANSAC inliers (< 40); only 3% of textured pixels agree after warping (< 50%) |
| aitree | 1 | fit | 1 | none → sRGB | none → sRGB | 1543 | 0.73 | 75% | 249,346 | fit |
| breaks | 1 | preview | 1 | Display P3 | none → sRGB | — | — | — | — | preview only |
| robby | 2 | fit | 2 | Display P3 | none → sRGB | 3981 | 0.66 | 100% | 719,377 | fit |
| satoru | 2 | fit | 2 | Display P3 | Display P3 | 1111 | 0.83 | 55% | 337,040 | fit |
| alli | 2 | fit | 1 | Display P3 | none → sRGB | 116 | 1.95 | 44% | 451,509 | dropped: only 44% of textured pixels agree after warping (< 50%) |
| batman | 2 | preview | 1 | none → sRGB | none → sRGB | — | — | — | — | preview only |
| kiriya | 2 | preview | 1 | Display P3 | none → sRGB | — | — | — | — | preview only |
| pallav | 2 | fit | 1 | Display P3 | none → sRGB | 4354 | 0.36 | 100% | 737,095 | fit |
| bencsinB | 2 | fit | 2 | none → sRGB | none → sRGB | 2704 | 1.36 | 91% | 515,540 | fit |
| bencsinC | 2 | fit | 1 | none → sRGB | none → sRGB | 129 | 1.39 | 96% | 315,707 | fit |
| bencsinA | 2 | fit | 0.5 | none → sRGB | none → sRGB | 304 | 1.31 | 50% | 391,092 | dropped: only 50% of textured pixels agree after warping (< 50%) |

- `sliwex`: night scene; Chromagraph much darker (author: both straight out of camera).
- `fahim`: Chromagraph shot deliberately underexposed by the author.
- `vsgfield`: stock side is Apple's rendering of a RAW (same exposure as the Chromagraph side).
- `robby`: stock side is Apple's rendering of a RAW (same exposure as the Chromagraph side).
- `vsgfield`: top 8% masked (labels).

## Held-out details (v2)

| Held out | Pixels | Identity | Tone curve only | v2 | v2 bright | v2 brighter | Identity, own exposure | v2, own exposure | v2, no residual masks |
|---|---|---|---|---|---|---|---|---|---|
| vsgfield | 100,000 | 16.19 / 32.84 | 8.17 / 18.14 | 5.92 / 16.54 | 9.97 / 23.67 | 14.87 / 31.86 | 11.50 / 17.92 (-1.1) | 6.64 / 15.01 (-0.1) | 6.35 / 17.61 |
| sliwex | 100,000 | 21.09 / 29.44 | 11.54 / 23.12 | 10.37 / 16.14 | 14.61 / 21.45 | 20.11 / 28.39 | 4.31 / 28.70 (-3.1) | 6.06 / 26.42 (-1.1) | 10.79 / 17.32 |
| fahim | 100,000 | 14.65 / 19.28 | 5.54 / 16.13 | 6.12 / 10.24 | 7.64 / 14.54 | 11.07 / 21.39 | 6.99 / 17.32 (-1.0) | 6.11 / 10.26 (+0.0) | 6.12 / 10.25 |
| aitree | 100,000 | 10.96 / 17.52 | 5.77 / 15.22 | 5.37 / 13.73 | 4.31 / 11.65 | 7.16 / 15.41 | 5.50 / 14.23 (-0.9) | 4.47 / 12.69 (+0.3) | 4.87 / 13.61 |
| robby | 100,000 | 16.51 / 22.69 | 9.21 / 14.11 | 6.57 / 12.18 | 7.66 / 16.40 | 13.71 / 23.15 | 7.69 / 16.37 (-1.1) | 6.59 / 12.17 (+0.0) | 6.65 / 12.21 |
| satoru | 100,000 | 20.62 / 26.81 | 14.80 / 17.85 | 6.61 / 15.03 | 9.59 / 21.24 | 12.85 / 27.70 | 13.83 / 19.33 (-1.1) | 4.49 / 11.23 (-0.3) | 7.79 / 19.38 |
| pallav | 100,000 | 11.91 / 19.69 | 4.82 / 19.44 | 3.73 / 8.75 | 4.26 / 8.10 | 8.23 / 14.02 | 5.67 / 17.57 (-0.8) | 3.31 / 6.83 (+0.2) | 3.72 / 8.72 |
| bencsinB | 100,000 | 6.80 / 10.83 | 10.81 / 19.34 | 10.96 / 19.63 | 6.46 / 13.79 | 4.90 / 8.85 | 6.56 / 10.17 (-0.2) | 6.23 / 12.04 (+0.7) | 10.80 / 19.91 |
| bencsinC | 100,000 | 12.17 / 32.04 | 6.15 / 18.74 | 4.43 / 19.62 | 6.19 / 26.93 | 9.25 / 34.16 | 3.81 / 30.11 (-2.1) | 2.22 / 15.47 (-0.7) | 4.67 / 21.18 |

"Own exposure" gives each method one exposure gain (stops in brackets) chosen on the held-out Chromagraph image to minimise mean ΔE00, so it is optimistic. "No residual masks" scores every overlapping unclipped pixel.

In-sample (all fit pairs; "own" uses the pair's fitted gain):

| Pair | Identity | Tone curve, own gain | v2, own gain | v2 as written | v1 as written |
|---|---|---|---|---|---|
| vsgfield | 16.22 / 32.83 | 8.03 / 14.16 | 6.44 / 13.46 | 5.55 / 16.25 | 5.12 / 15.25 |
| sliwex | 21.06 / 29.49 | 9.97 / 23.05 | 6.43 / 13.51 | 6.92 / 13.85 | 4.80 / 11.44 |
| fahim | 14.66 / 19.26 | 5.55 / 16.41 | 4.09 / 8.76 | 4.09 / 8.76 | 4.10 / 8.76 |
| aitree | 10.98 / 17.56 | 6.31 / 13.64 | 3.42 / 11.07 | 3.93 / 12.76 | 4.46 / 12.97 |
| robby | 16.55 / 22.70 | 9.86 / 14.08 | 4.35 / 8.98 | 4.09 / 7.46 | 7.50 / 13.04 |
| satoru | 20.63 / 26.86 | 12.95 / 19.56 | 3.64 / 9.98 | 5.36 / 14.09 | 7.26 / 15.39 |
| pallav | 11.97 / 19.69 | 4.50 / 17.65 | 2.68 / 6.33 | 3.17 / 7.42 | 5.25 / 9.22 |
| bencsinB | 6.78 / 10.79 | 6.93 / 12.98 | 3.81 / 11.16 | 9.24 / 17.60 | 12.02 / 21.78 |
| bencsinC | 12.23 / 32.00 | 4.11 / 15.08 | 2.77 / 10.31 | 4.09 / 18.23 | 4.41 / 19.16 |

## Coverage and smoothness (v2)

18.1% of the 33³ nodes have at least 5 valid samples within one lattice step (31.2% within two); elsewhere the LUT is the tone curve plus a residual that fades to zero. Output L* drops when one input channel rises by one step: R 0.00% of steps by > 1 (worst -0.0), G 0.00% of steps by > 1 (worst -0.0), B 0.00% of steps by > 1 (worst -0.0).

## Observations about v2

- **Tone.** Neutral sRGB 0.05 / 0.20 / 0.46 / 0.75 / 1.00 come out at L* 2 / 9 / 31 / 67 / 88 (stock 4 / 21 / 49 / 77 / 100): deep shadows and mid-tones are pushed down hard, which is the main reason the fitted-exposure LUT looks dark; the bright and brighter versions trade fidelity for a lighter rendering.
- **Colour.** Chroma ×0.54 (real ×0.54); saturation relative to lightness ×1.02 (real ×0.90). By input hue: red ΔL* -14, chroma ×0.70, hue +6° (toward orange); orange ΔL* -15, chroma ×0.55, hue -2° (no clear hue shift); yellow ΔL* -16, chroma ×0.67, hue +1° (no clear hue shift); green ΔL* -17, chroma ×0.92, hue -6° (toward yellow); blue ΔL* -10, chroma ×0.27, hue -85° (toward cyan); magenta ΔL* -18, chroma ×0.41, hue +42° (toward red).
- **Skies.** Bright blue pixels: ΔL* -11, Δb* +21.4 (real Δb* +20.3): skies still roll toward cream while neutral whites stay near the evidence.

## Limits

- 9 fitted pairs, all recompressed social-media JPEGs; colours none of them contain fall back to the tone curve.
- The stock images are Apple's finished renders with local tone mapping, segmentation and night processing; a global LUT cannot undo processing that varies across the frame.
- No grain or halation: the images are too downscaled and recompressed to measure them, and a LUT cannot add spatial effects.
- Exposure differences between pairs are absorbed by per-pair gains; the baked median gain is a choice, and the fitted-exposure LUT is dark because the real Chromagraph renders are dark relative to Apple's lifted shadows.
- Files without an ICC profile are decoded as sRGB; Display P3 colours outside sRGB clip on conversion and are masked.
- The user photos have no Chromagraph version, so they show only how the LUTs render, not how close they are.
