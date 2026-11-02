# Chromagraph Camera: clean-room reverse engineering

Sources checked 2 November 2026. Everything here comes from public pages, interviews, and documentation. Nobody downloaded, decrypted, or disassembled the app binary. Each claim below is marked **stated** (a source says so) or **inferred** (our conclusion).

## Bottom line

- Chromagraph is Steve Yedlin's film-print rendering, the one used on the *Knives Out* films, rebuilt for the iPhone. It captures RAW, renders it to a scene-linear image, then applies an empirically fitted color and tone transform. Grain and halation are added in a separate step, and the result is exported as JPEG or HEIC. **Stated:** [developer page](https://www.chromaticapplications.com/chromagraph/chromagraph), [Newsshooter](https://www.newsshooter.com/2026/10/28/chromagraph-camera-ios-app-by-steve-yedlin-asc/), [WIRED](https://www.wired.com/story/the-hollywood-color-nerd-who-shot-knives-out-explains-his-viral-iphone-camera-app/).
- No public source gives the coefficients, the LUT, the working color space, or the texture parameters. Those parts have to be measured from the app's own output.
- If you own the app, you can recreate it closely. Fit a tone curve plus a 3D LUT from pairs of RAW originals and the app's exports, and measure grain and halation separately. Then rebuild the pipeline with public iOS APIs.

## Pipeline, stage by stage

| Stage | What is known | Status |
|---|---|---|
| Capture | Starts from raw sensor data and replaces Apple's look processing. The viewfinder does not preview the look; it is applied after capture. Options include a default-lens setting, front-camera mirroring, and a macro toggle. | Stated ([developer](https://www.chromaticapplications.com/chromagraph/chromagraph), [App Store](https://apps.apple.com/us/app/chromagraph-camera/id6803329435)) |
| Bayer RAW or ProRAW | Not disclosed. A secondary source reports that App Store screenshots advertise merging multiple exposures for cleaner shadows and retained highlights. That would point to the app's own Bayer-frame merge rather than ProRAW. | Inferred, low to medium ([Bottle Rocket](https://www.bottlerocketcontent.com/rian-johnson-steve-yedlin-chromagraph-iphone-camera-app/)) |
| Working space | Originals are kept as "scene-linear" images. Yedlin's film pipelines convert each camera into a neutral space before rendering. The app probably does the same with an Apple-specific calibration. | Stated (scene-linear originals), inferred (calibration) |
| Color transform | Yedlin's published method fits a transform to measured data. He measures film negative scans, ALEXA log footage, and contact-printed film, then runs his own Python scattered-data interpolation and outputs a 3D LUT. He shapes the parts of the color volume between data points with "Cone Coordinates", which bend saturated colors more gently than HSV-based adjustments. | Stated for his films ([Display Prep FAQ](https://yedlin.net/NerdyFilmTechStuff/DispPrepDemoFAQ/)); inferred, medium, that the app uses the same family of methods |
| Tone | Film-derived shoulder and toe with channel crossover (the color channels' curves diverge in shadows and highlights). Saturation behaves differently near neutral, which matters most for faces. Saturated reds get darker and more yellow instead of washing out, so a plain saturation slider will not reproduce it. | Stated ([British Cinematographer](https://britishcinematographer.co.uk/steve-yedlin-asc-knives-out/), [Two Kinds of Saturation](https://yedlin.net/NerdyFilmTechStuff/TwoKindsOfSatuartion)) |
| Texture | Grain is synthesized from statistics measured on scanned film, using fresh random numbers each time rather than overlaying stored scans. Halation is a separate spatial step that adds a reddish glow around bright edges. The app offers grain and halation controls. | Stated ([On Film Grain Emulation](https://www.yedlin.net/NerdyFilmTechStuff/OnFilmGrainEmulation), [On Color Science](https://yedlin.net/NerdyFilmTechStuff/OnColorScience/)) |
| Sharpening | Users report soft, sometimes "muddy" detail at 100% and ask for a sharpening control. That suggests restrained or no sharpening, which is consistent with Yedlin's taste. | Inferred, medium ([App Store reviews](https://apps.apple.com/us/app/chromagraph-camera/id6803329435?platform=iphone&see-all=reviews), [MacRumors forum](https://forums.macrumors.com/threads/rian-johnsons-iphone-camera-app-goes-viral.2493228/page-3)) |
| Output | JPEG or HEIC with the look baked in; the original stays editable. Version 1.2.0 (about 31 October) added "Archive export/import", but the archive's contents are not documented. The output color space and HDR behavior are not disclosed. | Stated ([developer](https://www.chromaticapplications.com/chromagraph/chromagraph), [changelog](https://indieappcatalog.com/app/5032/changelog)) |

What users have observed, with no controlled measurements yet:
- Muted, film-like color with consistent treatment across scenes, and sky color preserved in backlit sunsets.
- Some users call the output dark, warm, brownish, or oversaturated.
- Shadows can be crushed and highlights clipped in hard scenes. Some reviewers attribute this to the absence of Apple's HDR merge.
- Exports are about 12 MP from a camera capture; a 48 MP RAW imported from the library exports at 48 MP.

## Recreation plan

### 1. Measure the look (needs your copy of the app)

1. **Get matched pairs.** Check whether "Archive export" in 1.2.0 includes the original RAW or DNG. If it does, each archived original plus its JPEG export is one training pair. If not, shoot a DNG with another app and use Chromagraph's "import from library" (users report it renders imported 48 MP RAWs).
2. **Shoot targets** on a tripod under fixed light, at fixed exposure and white balance:
   - ColorChecker Classic and Digital SG charts.
   - A gray ramp.
   - Exposure brackets of about ±4 EV to map the shoulder and toe.
   - Saturated LEDs or neon to probe the edges of the gamut.
   - Real skin and foliage.
3. **Isolate each control.** Re-export the same original while changing one control at a time. Record the app version, phone, lens, ISO, shutter, and every setting.
4. **Develop the RAWs the same way every time.** Use one fixed decode with no automatic tone mapping or enhancement. The fitted LUT maps *your* decode to Chromagraph's output, so changing the decode invalidates the fit.
5. **Fit the color and tone transform.** First fit a monotonic neutral tone curve from the brackets. Then fit a 3D residual with a regularized radial-basis-function (RBF) fit, which interpolates between scattered samples. Bake the result into a 33³ or 65³ `.cube` LUT, and compare it against a single direct 3D fit. Tools:
   - [colour-science](https://github.com/colour-science/colour) and [colour-checker-detection](https://github.com/colour-science/colour-checker-detection) for color math and chart detection.
   - [SciPy RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html) for the RBF fit.
   - [camera-match](https://github.com/ethan-ou/camera-match) and [sLUTy](https://github.com/rmanalan/sluty) for fitting a LUT from paired images.
   - [OpenColorIO](https://opencolorio.readthedocs.io/en/latest/tutorials/baking_luts.html) for shaper curves and LUT baking.
6. **Test for local processing.** Photograph identical patches against dark and bright surroundings and at different positions in the frame. A global LUT always gives the same output for the same input color. Residuals that depend on the surroundings mean adaptive tone mapping, which a LUT cannot capture. Halos at slanted edges mean sharpening.
7. **Measure texture separately.**
   - Grain: export the same RAW repeatedly with grain on and off. On flat fields, measure the residual's variance, RGB correlation, and power spectrum against brightness. Synthesize matching grain, using [Newson et al. 2017](https://www.ipol.im/pub/art/2017/192/) as the physical model or density-weighted correlated noise.
   - Halation: photograph point lights on black and fit each channel's halo radius, strength, and brightness threshold.
8. **Validate on held-out scenes.** Report the median, 95th-percentile, and worst CIEDE2000 color difference against the app's export (not against chart reference values), and do blind side-by-side comparisons.

### 2. Rebuild on iOS (public APIs)

1. **Capture:** `AVCapturePhotoOutput` with a Bayer RAW format (`isBayerRAWPixelFormat`). Bayer RAW avoids Deep Fusion, Smart HDR, and Apple's tone rendering; ProRAW does not, because its pixels are already merged and denoised. Lens and resolution support must be checked at runtime on each device. ([Apple guide](https://developer.apple.com/documentation/avfoundation/capturing-photos-in-raw-and-apple-proraw-formats))
2. **Decode:** `CIRAWFilter` with `baselineExposure`, `shadowBias`, `boostAmount`, and `localToneMapAmount` set to 0 and gamut mapping off. Set white balance explicitly. Pin the decoder version: the RAW 9 decoder introduced at WWDC26 demosaics and denoises with Core ML and ignores some of the detail controls. ([WWDC21 linear-RAW sample](https://developer.apple.com/videos/play/wwdc2021/10160/), [WWDC26](https://developer.apple.com/videos/play/wwdc2026/305/))
3. **Working space:** render to half-float (`RGBAh`) in extended-linear Display P3. Keep values above 1 and below 0.
4. **Look:** a log shaper followed by the fitted cube, either with `CIColorCubeWithColorSpace` or a Metal kernel with tetrahedral interpolation. `CIColorCube` does not document which interpolation it uses.
5. **Texture:**
   - Halation: extract highlights in linear light, blur them with a red-biased multi-scale blur, and add the result back.
   - Grain: one GPU pass of seeded, density-weighted grain.
   - Optionally, a slight blur for softness.
6. **Output:** convert to Display P3 exactly once and write with `CIContext.heifRepresentation` or the JPEG equivalent. Keep the DNG and store a versioned edit recipe (decode settings, LUT hash, texture parameters, random seed) in `PHAdjustmentData` or a sidecar file.

Comparable apps that publish parts of their pipelines: [Halide Process Zero](https://www.lux.camera/process-zero-manual/), [mood.camera](https://www.mood.camera/v2), and [Rayon](https://rayon.cam/features/computational-raw/).

### Shortcuts worth knowing
- Chromatic Applications lists Yedlin-derived OFX plug-ins as "Coming Soon" ([site](https://www.chromaticapplications.com/)). When they ship, they are the licensed way to get the real transforms.
- FilmMatch's [YedPrint LUT](https://www.film-match.com/yedprint) is a third-party approximation of the LUT from Yedlin's public Display Prep Demo, also built with scattered-data interpolation. It can serve as a starting point before you have your own measurements.

## Legal boundary (summary of sources, not legal advice)

- **Out:** decrypting the FairPlay-protected binary. [17 USC §1201](https://www.govinfo.gov/content/pkg/USCODE-2024-title17/html/USCODE-2024-title17-chap12-sec1201.htm) prohibits circumventing access controls. Its interoperability exception in §1201(f) covers making independent programs work with other programs, not recreating a look. [Apple's standard EULA](https://www.apple.com/legal/macapps/stdeula/) also forbids decompiling and disassembling.
- **Generally fine:** measuring the outputs of a lawfully owned app using your own photos, together with public information. Methods and processes are not protected by copyright ([17 USC §102(b)](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A102%28b%29+edition%3Aprelim%29)). Reverse engineering and independent derivation do not count as "improper means" under trade-secret law ([18 USC §1839(6)](https://www.govinfo.gov/content/pkg/USCODE-2024-title18/html/USCODE-2024-title18-partI-chap90-sec1839.htm)).
- **Still matters if you publish:** do not copy their code, assets, sample photos, or branding. App Store guidelines [§4.1 and §5.2](https://developer.apple.com/app-store/review/guidelines/) restrict copycat apps.

## Open questions only measurement can settle

1. Do Archive exports contain the original DNG, and in what container?
2. Can grain and halation be turned fully off for export?
3. Are exports SDR or HDR (with a gain map), and what color profile is embedded?
4. Is the output global per pixel, or is there local or adaptive tone mapping?
5. Does the rendering differ between lenses or iPhone models?
