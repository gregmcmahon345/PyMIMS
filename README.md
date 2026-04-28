# PyMIMS

A Python library for reading and processing Cameca NanoSIMS `.im` files — a modern, version-stable alternative to the OpenMIMS ImageJ plugin, with no dependency on Java, ImageJ, or Fiji.

**Status:** v0.2 prototype — early development, API not stable.

---

## What it does

- Reads Cameca NanoSIMS 50/50L `.im` binary files directly
- FFT cross-correlation drift correction across the image stack
- Publication-quality plotting with scale bars, colour bars, and contrast control
- **Isotope ratio images with full Poisson error propagation** — ratio, delta (‰) vs reference, σ(R), and σ(R)/R, with configurable low-count and high-error masking
- Works in both local Python scripts and Google Colab / Jupyter notebooks

The `.im` binary format was reverse-engineered from first principles. No Cameca documentation or OpenMIMS source code was used.

---

## Installation

PyMIMS is currently a single-file module. There's no PyPI package yet.

**In Google Colab:**

```python
!wget -q https://raw.githubusercontent.com/gregmcmahon345/PyMIMS/main/pymims.py
```

**Locally:**

Clone or download the repo, then make sure the dependencies are installed:

```bash
pip install numpy matplotlib scipy
```

(On Chromebook/Crostini add `--break-system-packages`.)

---

## Quick start

```python
from pymims import MimsImage

img = MimsImage('myfile.im')
print(img)                                   # metadata summary

img.drift_correct(reference='SE')            # FFT cross-correlation
fig = img.plot()                             # all mass channels

# Isotope ratio with Poisson error propagation
fig, result = img.plot_ratio(
    '13C', '12C',
    delta_ref=0.0112372,                     # V-PDB reference for ¹³C/¹²C
    min_counts=20,                           # mask pixels where B < 20
    max_rel_err=0.5,                         # mask pixels where σ/R > 0.5
)

# result is a dict with keys: ratio, sigma, rel_err, A, B, mask, ...
```

The `plot_ratio()` method produces a four-panel figure: raw ratio, delta (‰), absolute Poisson uncertainty σ(R), and relative uncertainty σ(R)/R.

---

## Validated against

- Cameca NanoSIMS 50/50L files from multiple instruments
- OpenMIMS reference outputs for drift correction and metadata
- Synthetic ground-truth `.im` files for ratio and error propagation accuracy

---

## Planned features

- HSI (Hue-Saturation-Intensity) composite images
- Brightness / contrast / gamma adjustment with histogram
- ROI drawing and depth profiling, reporting both `total_A/total_B` (unbiased bulk ratio with propagated Poisson σ) and `mean(A/B)` (with SEM across pixels). The ratio of the two metrics serves as a homogeneity diagnostic.
- Segmentation tools

---

## Status & contributions

This is an early-stage personal research project, shared publicly for transparency and to support reproducibility of work that depends on it. The API is expected to change.

If you find it useful or run into problems, feel free to open an issue.

---

## Author

G. McMahon — principal scientist, materials science / analytical research background.
Developed with AI-assisted coding.
