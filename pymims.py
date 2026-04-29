"""
pymims.py  v0.5
===============================================================================
DEVELOPMENT STATUS: Early prototype — not a public package.
This library is an original work in development and is NOT available on PyPI
or any public repository. Do not distribute without the author's permission.
===============================================================================

Authors   : G. McMahon (principal scientist) with AI-assisted development
Created   : March 2026
Updated   : April 2026 (v0.5 — publication-quality export, multi-panel layout)
Status    : v0.5 prototype

Description
-----------
A Python library for reading and processing Cameca NanoSIMS .im files.
The .im binary format was reverse-engineered from first principles — no
Cameca documentation or OpenMIMS source code was used. The library is
intended as a modern, version-stable alternative to the OpenMIMS ImageJ
plugin, with no dependency on Java, ImageJ, or Fiji.

Validated against:
  - Cameca NanoSIMS 50/50L files from multiple instruments
  - OpenMIMS reference outputs for drift correction and metadata

v0.3 changes
------------
  - drift_correct() supports plane binning (bin_planes), needed for
    high-spatial-resolution acquisitions where per-plane counts are too
    low for reliable cross-correlation. Three apply modes: 'same' (default),
    'interp' (linear between super-plane centres), 'super' (degrade stack
    to super-plane resolution).

v0.2 changes
------------
  - Works in both local scripts and Google Colab / Jupyter notebooks
  - Ratio images with Poisson error propagation, delta notation, and masking
  - Robust Poly_list search (handles headers with multiple Poly_list strings)
  - plot() and plot_ratio() return the Figure for in-notebook display

Description
-----------
A Python library for reading and processing Cameca NanoSIMS .im files.
The .im binary format was reverse-engineered from first principles — no
Cameca documentation or OpenMIMS source code was used. The library is
intended as a modern, version-stable alternative to the OpenMIMS ImageJ
plugin, with no dependency on Java, ImageJ, or Fiji.

Validated against:
  - Cameca NanoSIMS 50/50L files from multiple instruments
  - OpenMIMS reference outputs for drift correction and metadata

v0.2 changes
------------
  - Works in both local scripts and Google Colab / Jupyter notebooks
  - Ratio images with Poisson error propagation, delta notation, and masking
  - plot() and plot_ratio() return the Figure for in-notebook display

Planned features (not yet implemented):
  - HSI (Hue-Saturation-Intensity) composite images
  - Brightness / contrast / gamma adjustment with histogram
  - ROI drawing and depth profiling, reporting BOTH:
      * total_A / total_B   — unbiased bulk ratio (with propagated Poisson σ)
      * mean(A/B)           — pixel-wise mean ratio (with SEM across pixels)
    plus median(A/B) and pixel count. The ratio of the two metrics is a
    homogeneity diagnostic: ≈1.0 means a well-behaved ROI; notably >1.0
    flags either low-count bias (Jensen's inequality on 1/B) or genuine
    sub-ROI heterogeneity worth examining.
  - Segmentation tools

Requirements
------------
    pip install numpy matplotlib scipy
  (on Chromebook/Crostini add --break-system-packages)

Basic usage
-----------
    from pymims import MimsImage
    img = MimsImage('myfile.im')
    print(img)
    img.drift_correct(reference='SE')
    img.plot()                                       # all channels
    img.plot_ratio('13C', '12C', delta_ref=0.0112)   # ratio + delta + errors
"""

import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from scipy.ndimage import shift as nd_shift


# ── Environment detection ────────────────────────────────────────────────────

def _in_notebook():
    """Return True if running inside a Jupyter / Colab / IPython notebook.

    Detection strategy: any IPython kernel that is NOT the terminal CLI is
    treated as a notebook. Covers Jupyter (ZMQInteractiveShell), Colab
    (which uses its own subclass), VS Code notebooks, etc.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Anything except plain terminal IPython is a notebook-like environment
        # where inline display works.
        return shell.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


_IS_NOTEBOOK = _in_notebook()


# ── Utility functions ────────────────────────────────────────────────────────

def _u32(data, off): return struct.unpack_from('<I', data, off)[0]
def _i32(data, off): return struct.unpack_from('<i', data, off)[0]
def _f64(data, off): return struct.unpack_from('<d', data, off)[0]
def _str(data, off, length=16):
    return data[off:off+length].split(b'\x00')[0].decode('ascii', errors='replace').strip()


def _finalize_figure(fig, outpath, show):
    """
    Common save/show logic for plotting methods.

    - If outpath is given, save and (in scripts) close.
    - If show is True (notebook), leave the figure open for inline display.
    - If show is False (script, no outpath), close to free memory.
    Returns the figure.
    """
    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved: {outpath}")

    if show:
        # In notebook context, returning the figure causes inline display.
        # Do not close.
        pass
    else:
        plt.close(fig)
    return fig


# ── MimsImage class ──────────────────────────────────────────────────────────

class MimsImage:
    """
    Reads and processes a Cameca NanoSIMS .im file.

    Attributes
    ----------
    path        : str        full file path
    metadata    : dict       all header fields
    masses      : list[str]  mass channel labels
    nom_masses  : list[float] nominal mass values from header
    data        : np.ndarray raw image stack (planes, masses, height, width) uint32
    corrected   : np.ndarray drift-corrected stack, or None if not yet corrected
    shifts      : np.ndarray (n_planes, 2) drift shifts in pixels, or None
    """

    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.path = filepath
        self._raw_header = None
        self.metadata    = {}
        self.masses      = []
        self.nom_masses  = []
        self.data        = None
        self.corrected   = None
        self.shifts      = None
        self._load()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self):
        """Read header and image data from .im file."""
        filesize = os.path.getsize(self.path)

        with open(self.path, 'rb') as f:
            header = f.read(min(filesize, 65536))

        self._raw_header = header
        self._parse_header(header, filesize)
        self._load_data()

    def _parse_header(self, data, filesize):
        """Parse all metadata from binary header."""
        m = {}

        # Fixed offsets
        m['data_offset']  = _u32(data, 0x008)
        m['n_planes_hdr'] = _u32(data, 0x090)   # planned planes
        m['n_masses']     = _u32(data, 0x198)
        m['duration_s']   = _u32(data, 0x08c)
        m['stage_x_nm']   = _i32(data, 0x014)
        m['stage_y_nm']   = _i32(data, 0x018)
        m['z_position']   = _u32(data, 0x04c)
        m['instrument']   = _str(data, 0x3d, 16)
        m['date']         = _str(data, 0x5c, 12)
        m['time']         = _str(data, 0x6c, 8)

        # Image dimensions from Poly_list block.
        # The string 'Poly_list' can appear multiple times in the header
        # (e.g. as part of longer label strings). Walk through all occurrences
        # and pick the first one where +0x34 and +0x38 yield plausible image
        # dimensions. NanoSIMS images are typically 64–1024 px square; we
        # accept 16–8192 to be safe.
        m['width']  = None
        m['height'] = None
        search_from = 0
        while True:
            poly_off = data.find(b'Poly_list', search_from)
            if poly_off < 0:
                break
            if poly_off + 0x3c > len(data):
                break
            w_try = _u32(data, poly_off + 0x34)
            h_try = _u32(data, poly_off + 0x38)
            if 16 <= w_try <= 8192 and 16 <= h_try <= 8192:
                m['width']      = w_try
                m['height']     = h_try
                m['poly_offset'] = poly_off
                break
            search_from = poly_off + 1

        if m['width'] is None:
            raise ValueError(
                "Could not find a Poly_list block with valid image dimensions "
                "— file may be corrupt or use an unsupported header layout"
            )

        # Raster (nm) at fixed offset -68 before data block
        m['raster_nm']  = _u32(data, m['data_offset'] - 68)
        m['pixel_nm']   = m['raster_nm'] / m['width'] if m['width'] > 0 else 0
        m['field_um']   = m['raster_nm'] / 1000.0

        # Derive actual plane count from file size
        data_bytes      = filesize - m['data_offset']
        bytes_per_plane = m['n_masses'] * m['width'] * m['height'] * 4
        if bytes_per_plane > 0 and data_bytes % bytes_per_plane == 0:
            m['n_planes'] = data_bytes // bytes_per_plane
        else:
            raise ValueError(
                f"Data size {data_bytes} not divisible by "
                f"n_masses({m['n_masses']}) * w({m['width']}) * h({m['height']}) * 4"
            )

        self.metadata = m

        # Mass labels and nominal masses
        # Labels: block at 0x2c0 + n*0xC0 + 0x0d
        # Nominal mass: block at 0x200 + n*0xC0 + 0x94 (one block before label)
        block_start = 0x2c0
        block_size  = 0xC0
        mass_pre    = 0x200

        labels     = []
        nom_masses = []
        for n in range(m['n_masses']):
            label_off = block_start + n * block_size + 0x0d
            mass_off  = mass_pre   + n * block_size + 0x94
            label = _str(data, label_off, 16)
            mass  = _f64(data, mass_off) if mass_off + 8 <= len(data) else 0.0
            labels.append(label if label else 'SE')
            nom_masses.append(mass if 0.0 < mass < 200.0 else 0.0)

        self.masses     = labels
        self.nom_masses = nom_masses

    def _load_data(self):
        """Load image data into numpy array."""
        m = self.metadata
        offset = m['data_offset']
        shape  = (m['n_planes'], m['n_masses'], m['height'], m['width'])
        n_vals = m['n_planes'] * m['n_masses'] * m['height'] * m['width']

        with open(self.path, 'rb') as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(n_vals * 4), dtype='<u4')

        self.data = raw.reshape(shape)

    # ── Drift correction ─────────────────────────────────────────────────────

    def drift_correct(self, reference='SE', ref_plane=0,
                      bin_planes=1, bin_apply='same'):
        """
        Drift correct the stack using FFT cross-correlation.

        For high-spatial-resolution acquisitions where per-plane counts are
        too low for reliable cross-correlation, use ``bin_planes`` to sum
        adjacent planes into super-planes before computing shifts.

        Parameters
        ----------
        reference : str or int
            Mass channel label (e.g. 'SE', '12C 14N') or channel index to use
            as the reference signal for cross-correlation. Partial-match
            (case-insensitive) on labels.
        ref_plane : int
            Plane index (or super-plane index, when binning) to use as
            reference. Default 0 = first plane / first super-plane.
        bin_planes : int
            Group size for plane binning. 1 = no binning (default).
            E.g. bin_planes=5 sums planes (0-4), (5-9), ... into super-planes
            before cross-correlation. Must divide n_planes evenly, or any
            trailing partial group is dropped from binning (a warning is
            printed). Useful when individual planes have too few counts for
            cross-correlation to be reliable.
        bin_apply : {'same', 'interp', 'super'}
            How shifts derived between super-planes are applied:
              'same'   — every original plane in a group gets that group's
                         shift (default; no false sub-group precision).
              'interp' — shifts are linearly interpolated between super-plane
                         centres and applied per original plane. Better when
                         drift is monotonic.
              'super'  — the corrected stack stays at the super-plane level.
                         n_planes is reduced to the number of super-planes
                         and counts are summed within groups. Useful as a
                         degrade-to-fewer-planes preprocessing step.
        """
        # Resolve reference channel index
        ref_ch = self._resolve_channel(reference)
        n_planes = self.metadata['n_planes']
        n_masses = self.metadata['n_masses']
        w = self.metadata['width']

        if bin_apply not in ('same', 'interp', 'super'):
            raise ValueError(f"bin_apply must be 'same', 'interp', or 'super'; got {bin_apply!r}")
        if bin_planes < 1:
            raise ValueError(f"bin_planes must be >= 1; got {bin_planes}")

        arr = self.data.astype(float)

        # Build super-planes (or use the original stack if bin_planes == 1)
        if bin_planes == 1:
            super_arr = arr
            n_super   = n_planes
            print(f"Drift correcting using channel {ref_ch} ({self.masses[ref_ch]})...")
        else:
            n_full_groups = n_planes // bin_planes
            if n_full_groups < 2:
                raise ValueError(
                    f"bin_planes={bin_planes} gives only {n_full_groups} super-plane(s) "
                    f"from {n_planes} planes — need at least 2 to drift correct."
                )
            kept = n_full_groups * bin_planes
            if kept < n_planes:
                print(f"  Warning: dropping {n_planes - kept} trailing plane(s) "
                      f"that don't fill a complete bin of {bin_planes}.")
            # Sum within groups: shape (n_super, masses, h, w)
            super_arr = arr[:kept].reshape(
                n_full_groups, bin_planes, n_masses,
                self.metadata['height'], w
            ).sum(axis=1)
            n_super = n_full_groups
            print(f"Drift correcting using channel {ref_ch} ({self.masses[ref_ch]}); "
                  f"binning {bin_planes} planes -> {n_super} super-planes; "
                  f"apply mode '{bin_apply}'.")

        # Compute shifts on super-planes
        ref_img = super_arr[ref_plane, ref_ch]
        super_shifts = np.zeros((n_super, 2), dtype=int)
        for p in range(n_super):
            super_shifts[p] = self._xcorr_shift(ref_img, super_arr[p, ref_ch], w)

        # Branch on apply mode
        if bin_apply == 'super':
            # Operate on super-planes directly; reduce n_planes
            corrected = np.zeros_like(super_arr)
            for p in range(n_super):
                dy, dx = super_shifts[p]
                for ch in range(n_masses):
                    corrected[p, ch] = nd_shift(
                        super_arr[p, ch], (dy, dx), mode='constant', cval=0)
            self.corrected = corrected.astype(np.float32)
            self.shifts    = super_shifts
            self.metadata['n_planes_original'] = n_planes
            self.metadata['n_planes']          = n_super
            self.metadata['bin_planes']        = bin_planes
            self.metadata['bin_apply']         = bin_apply

        else:
            # Build a per-plane shift array for the original stack
            plane_shifts = np.zeros((n_planes, 2), dtype=int)

            if bin_planes == 1 or bin_apply == 'same':
                # Each original plane gets its group's shift
                for p in range(n_planes):
                    g = min(p // bin_planes, n_super - 1)
                    plane_shifts[p] = super_shifts[g]

            else:  # 'interp'
                # Super-plane g has centre at original plane g*bin_planes + (bin_planes-1)/2
                centres = np.arange(n_super) * bin_planes + (bin_planes - 1) / 2.0
                for p in range(n_planes):
                    dy = float(np.interp(p, centres, super_shifts[:, 0]))
                    dx = float(np.interp(p, centres, super_shifts[:, 1]))
                    plane_shifts[p] = (int(round(dy)), int(round(dx)))

            corrected = np.zeros_like(arr)
            for p in range(n_planes):
                dy, dx = plane_shifts[p]
                for ch in range(n_masses):
                    corrected[p, ch] = nd_shift(
                        arr[p, ch], (dy, dx), mode='constant', cval=0)

            self.corrected = corrected.astype(np.float32)
            self.shifts    = plane_shifts
            self.metadata['bin_planes'] = bin_planes
            self.metadata['bin_apply']  = bin_apply

        dy_range = self.shifts[:, 0].min(), self.shifts[:, 0].max()
        dx_range = self.shifts[:, 1].min(), self.shifts[:, 1].max()
        print(f"  Done. Y shifts: {dy_range[0]:+d} to {dy_range[1]:+d} px  |  "
              f"X shifts: {dx_range[0]:+d} to {dx_range[1]:+d} px")

    @staticmethod
    def _xcorr_shift(ref, img, w):
        """Return (dy, dx) integer shift of img relative to ref."""
        ref_n = ref - ref.mean()
        img_n = img - img.mean()
        corr  = np.fft.ifft2(
            np.fft.fft2(ref_n) * np.conj(np.fft.fft2(img_n))
        ).real
        peak  = np.unravel_index(np.argmax(corr), corr.shape)
        dy, dx = int(peak[0]), int(peak[1])
        if dy > w // 2: dy -= w
        if dx > w // 2: dx -= w
        return dy, dx

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot(self, plane=None, corrected=True, outpath=None,
             cmap='gray', scalebar_color='red', percentile=99.5,
             log_scale=False, show=None):
        """
        Plot all mass channels as a figure with colorbars and scale bar.

        Parameters
        ----------
        plane     : int or None
            If None, plots the sum of all planes.
            If int, plots that single plane (0-indexed).
        corrected : bool
            Use drift-corrected data if available (default True).
        outpath   : str or None
            If given, save to this path.
            If None and running as a script, auto-generate a path next to the .im file.
            If None and running in a notebook, do not save (just display inline).
        cmap      : str
            Matplotlib colormap (default 'gray').
        scalebar_color : str
            Scale bar colour (default 'red').
        percentile : float
            Upper display percentile for contrast (default 99.5).
        log_scale : bool
            If True, use a symmetric-log normalisation that handles zero counts
            naturally and emphasises low-intensity features. Colour bar shows
            log-spaced ticks. Default False (linear scale).
        show      : bool or None
            If True, return the figure for inline display (notebook).
            If False, close after saving (script).
            If None (default), auto-detect.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.colors import SymLogNorm, Normalize

        # Auto-detect notebook context
        if show is None:
            show = _IS_NOTEBOOK
        # In script mode with no outpath given, auto-generate one
        if outpath is None and not _IS_NOTEBOOK:
            base   = os.path.splitext(os.path.basename(self.path))[0]
            outdir = os.path.dirname(self.path) if self.path and os.path.dirname(self.path) and os.access(os.path.dirname(self.path), os.W_OK) else os.getcwd()
            suffix = f'_plane{plane+1}' if plane is not None else '_sum'
            scale_suffix = '_log' if log_scale else ''
            outpath = os.path.join(outdir, base + suffix + scale_suffix + '.png')
        # Choose data source
        if corrected and self.corrected is not None:
            arr = self.corrected
            corr_label = 'drift-corrected'
        else:
            arr = self.data.astype(float)
            corr_label = 'raw'

        # Sum or single plane
        if plane is None:
            display = arr.sum(axis=0)  # (n_masses, h, w)
            plane_label = f'all {self.metadata["n_planes"]} planes summed'
        else:
            display = arr[plane]
            plane_label = f'plane {plane + 1}'

        n_masses  = self.metadata['n_masses']
        pixel_um  = self.metadata['pixel_nm'] / 1000.0
        field_um  = self.metadata['field_um']
        bg        = '#1a1a1a'

        # Choose a round scale bar length
        scalebar_um = self._nice_scalebar(field_um)

        fig, axes = plt.subplots(1, n_masses, figsize=(4 * n_masses, 5),
                                 facecolor=bg)
        if n_masses == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            img  = display[i]
            p99  = np.percentile(img, percentile) if img.max() > 0 else 1

            if log_scale:
                # SymLogNorm: linear near zero, log for large values.
                # linthresh sets where the linear region ends; 1 is appropriate
                # for count data (sub-1 values are rare/impossible).
                norm = SymLogNorm(linthresh=1.0, linscale=1.0,
                                  vmin=0, vmax=max(p99, 2))
                im = ax.imshow(img, cmap=cmap, norm=norm,
                               extent=[0, field_um, field_um, 0])
            else:
                im = ax.imshow(img, cmap=cmap, vmin=0, vmax=p99,
                               extent=[0, field_um, field_um, 0])

            ax.set_title(self.masses[i], color='white', fontsize=11,
                         fontweight='bold', pad=4)
            ax.set_facecolor(bg)
            ax.axis('off')

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label('Counts (log)' if log_scale else 'Counts',
                           color='white', fontsize=8)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=7)
            cbar.outline.set_edgecolor('white')
            if not log_scale:
                # Linear: clean integer ticks
                cbar.locator = ticker.MaxNLocator(nbins=5, integer=True)
                cbar.update_ticks()
            # For log scale, let matplotlib's SymLogNorm pick its own
            # log-spaced ticks — they're correct out of the box.

            # Scale bar (bottom left, inset)
            margin  = field_um * 0.05
            bar_y   = field_um - margin
            bar_x0  = margin
            bar_x1  = margin + scalebar_um
            tick_h  = field_um * 0.02

            ax.plot([bar_x0, bar_x1], [bar_y, bar_y], '-',
                    color=scalebar_color, linewidth=2.5, solid_capstyle='butt')
            for xp in [bar_x0, bar_x1]:
                ax.plot([xp, xp], [bar_y - tick_h, bar_y + tick_h],
                        '-', color=scalebar_color, linewidth=2)
            ax.text((bar_x0 + bar_x1) / 2, bar_y - tick_h * 2,
                    f'{scalebar_um:g} μm', color=scalebar_color,
                    fontsize=8, ha='center', va='bottom')

            ax.set_xlim(0, field_um)
            ax.set_ylim(field_um, 0)

        title = (f"{os.path.basename(self.path)}  |  "
                 f"{self.metadata['width']}×{self.metadata['height']}  |  "
                 f"{plane_label}  |  {corr_label}  |  "
                 f"{field_um:.1f} μm field")
        fig.suptitle(title, color='white', fontsize=9, y=1.01)
        plt.tight_layout()

        return _finalize_figure(fig, outpath, show)

    def plot_drift(self, outpath=None, show=None):
        """Plot drift trajectory if drift correction has been run."""
        if self.shifts is None:
            print("No drift correction data — run drift_correct() first.")
            return None

        if show is None:
            show = _IS_NOTEBOOK
        if outpath is None and not _IS_NOTEBOOK:
            base    = os.path.splitext(os.path.basename(self.path))[0]
            outpath = os.path.join(os.getcwd(), base + '_drift.png')

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.shifts[:, 1], self.shifts[:, 0],
                'o-', color='steelblue', linewidth=1.5, markersize=5)
        for p, (dy, dx) in enumerate(self.shifts):
            ax.annotate(str(p + 1), (dx, dy), fontsize=7,
                        ha='center', va='bottom')
        ax.set_xlabel('X shift (pixels)')
        ax.set_ylabel('Y shift (pixels)')
        ax.set_title('Drift trajectory (relative to plane 1)')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.invert_yaxis()
        plt.tight_layout()

        return _finalize_figure(fig, outpath, show)

    # ── Ratio images with Poisson errors ─────────────────────────────────────

    def _resolve_channel(self, channel):
        """Resolve a channel name or index to an integer index."""
        if isinstance(channel, str):
            matches = [i for i, l in enumerate(self.masses)
                       if channel.upper() in l.upper()]
            if not matches:
                raise ValueError(
                    f"Channel '{channel}' not found. Available: {self.masses}"
                )
            return matches[0]
        return int(channel)

    def ratio(self, numerator, denominator, corrected=True,
              min_counts=None, max_rel_err=None, sum_planes=True):
        """
        Compute an isotope ratio image with Poisson error propagation.

        For two count images A (numerator) and B (denominator), both assumed
        to be Poisson-distributed:

            R       = A / B
            σ(R)    = R * sqrt(1/A + 1/B)
            σ(R)/R  = sqrt(1/A + 1/B)

        Pixels are summed across planes BEFORE division (this is the maximum
        likelihood estimator and standard practice in NanoSIMS imaging).

        Parameters
        ----------
        numerator, denominator : str or int
            Channel name (e.g. '13C', '12C') or index.
        corrected : bool
            Use drift-corrected data if available (default True).
        min_counts : float or None
            If set, mask pixels where the denominator total counts are below
            this threshold (sets to NaN).
        max_rel_err : float or None
            If set, mask pixels where σ(R)/R exceeds this threshold (sets to NaN).
        sum_planes : bool
            If True (default), sum across planes before computing the ratio.
            If False, return per-plane ratios as a 3D array.

        Returns
        -------
        dict with keys:
            ratio        : ratio image (np.ndarray, NaNs where masked)
            sigma        : absolute uncertainty σ(R)
            rel_err      : relative uncertainty σ(R)/R
            A            : numerator counts (summed)
            B            : denominator counts (summed)
            num_label    : numerator channel label
            den_label    : denominator channel label
            mask         : boolean mask of valid pixels (True = valid)
            n_planes     : number of planes contributing
        """
        num_ch = self._resolve_channel(numerator)
        den_ch = self._resolve_channel(denominator)

        # Source data
        if corrected and self.corrected is not None:
            arr = self.corrected
        else:
            arr = self.data.astype(float)

        if sum_planes:
            A = arr[:, num_ch].sum(axis=0).astype(float)
            B = arr[:, den_ch].sum(axis=0).astype(float)
        else:
            A = arr[:, num_ch].astype(float)
            B = arr[:, den_ch].astype(float)

        # Build mask of valid pixels (start: B > 0 to avoid div-by-zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            valid = B > 0

            if min_counts is not None:
                valid &= (B >= min_counts)

            R     = np.where(valid, A / np.where(B == 0, 1, B), np.nan)

            # Poisson error propagation
            # σ(R)/R = sqrt(1/A + 1/B); guard against A=0 (gives inf rel_err)
            inv_A   = np.where(A > 0, 1.0 / np.where(A == 0, 1, A), np.inf)
            inv_B   = np.where(B > 0, 1.0 / np.where(B == 0, 1, B), np.inf)
            rel_err = np.sqrt(inv_A + inv_B)
            rel_err = np.where(valid, rel_err, np.nan)
            sigma   = np.where(valid, R * rel_err, np.nan)

            if max_rel_err is not None:
                bad     = rel_err > max_rel_err
                R       = np.where(bad, np.nan, R)
                sigma   = np.where(bad, np.nan, sigma)
                rel_err = np.where(bad, np.nan, rel_err)
                valid   = valid & ~bad

        return {
            'ratio'    : R,
            'sigma'    : sigma,
            'rel_err'  : rel_err,
            'A'        : A,
            'B'        : B,
            'num_label': self.masses[num_ch],
            'den_label': self.masses[den_ch],
            'mask'     : valid,
            'n_planes' : self.metadata['n_planes'] if sum_planes else 1,
        }

    def plot_ratio(self, numerator, denominator, corrected=True,
                   delta_ref=None, min_counts=None, max_rel_err=None,
                   ratio_cmap='viridis', delta_cmap='RdBu_r',
                   delta_range=None, scalebar_color='white',
                   percentile=(1, 99), outpath=None, show=None):
        """
        Four-panel ratio figure: ratio | delta (‰) | σ(R) | σ(R)/R.

        Parameters
        ----------
        numerator, denominator : str or int
            Channel name or index.
        corrected : bool
            Use drift-corrected stack if available.
        delta_ref : float or None
            Reference ratio for delta calculation. If None, the panel uses
            the image median as the reference (deviation from local mean).
            Common references:
                ¹³C/¹²C : 0.0112372  (V-PDB)
                ¹⁵N/¹⁴N : 0.0036765  (AIR)
        min_counts : float or None
            Mask pixels where denominator counts < threshold.
        max_rel_err : float or None
            Mask pixels where σ(R)/R > threshold.
        ratio_cmap : str
            Colormap for the raw ratio panel.
        delta_cmap : str
            Diverging colormap for the delta panel.
        delta_range : float or None
            Half-range for delta panel symmetric scaling, in ‰.
            If None, uses ±2× the robust std of finite delta values.
        scalebar_color : str
            Scale bar colour (default 'white').
        percentile : tuple (lo, hi)
            Display percentiles for the ratio, σ, and rel_err panels.
        outpath, show : see plot()

        Returns
        -------
        (fig, result)
            fig    : matplotlib Figure
            result : dict from ratio() (so caller can reuse arrays)
        """
        if show is None:
            show = _IS_NOTEBOOK
        if outpath is None and not _IS_NOTEBOOK:
            base    = os.path.splitext(os.path.basename(self.path))[0]
            num     = self._resolve_channel(numerator)
            den     = self._resolve_channel(denominator)
            # Sanitize channel labels for filename use: replace spaces, slashes
            num_safe = self.masses[num].replace(' ', '').replace('/', '_')
            den_safe = self.masses[den].replace(' ', '').replace('/', '_')
            tag      = f"{num_safe}_over_{den_safe}"
            outpath  = os.path.join(os.getcwd(), f"{base}_ratio_{tag}.png")

        result = self.ratio(numerator, denominator, corrected=corrected,
                            min_counts=min_counts, max_rel_err=max_rel_err)
        R       = result['ratio']
        sigma   = result['sigma']
        rel_err = result['rel_err']
        num_lab = result['num_label']
        den_lab = result['den_label']

        # Compute delta in per mil
        finite = np.isfinite(R)
        if not finite.any():
            raise ValueError("No valid pixels after masking — relax thresholds.")

        if delta_ref is None:
            ref = float(np.nanmedian(R))
            ref_label = f'image median ({ref:.4g})'
        else:
            ref = float(delta_ref)
            ref_label = f'{ref:.4g}'
        delta = (R / ref - 1.0) * 1000.0  # per mil

        # Robust scaling for delta panel
        if delta_range is None:
            d_finite = delta[np.isfinite(delta)]
            mad      = np.median(np.abs(d_finite - np.median(d_finite)))
            sd_robust = 1.4826 * mad if mad > 0 else np.nanstd(d_finite)
            delta_range = max(2 * sd_robust, 1.0)

        # Robust ranges for ratio, sigma, rel_err
        def _prange(x):
            xf = x[np.isfinite(x)]
            if xf.size == 0:
                return 0, 1
            return np.percentile(xf, percentile[0]), np.percentile(xf, percentile[1])

        r_lo, r_hi   = _prange(R)
        s_lo, s_hi   = _prange(sigma)
        re_lo, re_hi = _prange(rel_err)
        # Cap rel_err display at 1.0 (100% relative error) — anything above
        # is noise-dominated and would otherwise drown out real signal.
        # The underlying result['rel_err'] is unchanged.
        re_hi = min(re_hi, 1.0)
        if re_lo >= re_hi:
            re_lo = 0.0

        # Figure
        bg       = '#1a1a1a'
        field_um = self.metadata['field_um']
        sb_um    = self._nice_scalebar(field_um)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor=bg)
        panels = [
            (R,       f'Ratio  {num_lab}/{den_lab}',          ratio_cmap, r_lo, r_hi, ''),
            (delta,   f'δ vs {ref_label} (‰)',                delta_cmap, -delta_range, delta_range, '‰'),
            (sigma,   f'σ(R)  Poisson',                       'magma',    s_lo, s_hi, ''),
            (rel_err, f'σ(R)/R',                              'magma',    re_lo, re_hi, ''),
        ]

        for ax, (img, title, cm, vmin, vmax, units) in zip(axes, panels):
            im = ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax,
                           extent=[0, field_um, field_um, 0],
                           interpolation='nearest')
            ax.set_title(title, color='white', fontsize=10,
                         fontweight='bold', pad=4)
            ax.set_facecolor(bg)
            ax.axis('off')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if units:
                cbar.set_label(units, color='white', fontsize=8)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=7)
            cbar.outline.set_edgecolor('white')

            # Scale bar (top-left)
            margin = field_um * 0.05
            bar_y  = field_um - margin
            bar_x0 = margin
            bar_x1 = margin + sb_um
            tick_h = field_um * 0.02
            ax.plot([bar_x0, bar_x1], [bar_y, bar_y], '-',
                    color=scalebar_color, linewidth=2.5, solid_capstyle='butt')
            for xp in [bar_x0, bar_x1]:
                ax.plot([xp, xp], [bar_y - tick_h, bar_y + tick_h],
                        '-', color=scalebar_color, linewidth=2)
            ax.text((bar_x0 + bar_x1) / 2, bar_y - tick_h * 2,
                    f'{sb_um:g} μm', color=scalebar_color,
                    fontsize=8, ha='center', va='bottom')

            ax.set_xlim(0, field_um)
            ax.set_ylim(field_um, 0)

        # Title with masking info
        n_total  = R.size
        n_finite = int(np.isfinite(R).sum())
        # "Useful" pixels: finite ratio with rel_err under 100% (not noise-dominated)
        n_useful = int((np.isfinite(R) & np.isfinite(rel_err) & (rel_err < 1.0)).sum())
        pct_finite = 100.0 * n_finite / n_total if n_total else 0
        pct_useful = 100.0 * n_useful / n_total if n_total else 0
        mask_bits = []
        if min_counts is not None:  mask_bits.append(f'B≥{min_counts:g}')
        if max_rel_err is not None: mask_bits.append(f'σ/R≤{max_rel_err:g}')
        mask_str = '; '.join(mask_bits) if mask_bits else 'no masking'

        title = (f"{os.path.basename(self.path)}  |  "
                 f"ratio {num_lab}/{den_lab}  |  "
                 f"{result['n_planes']} planes summed  |  "
                 f"finite: {pct_finite:.1f}%, σ/R<1: {pct_useful:.1f}%  |  "
                 f"{mask_str}")
        fig.suptitle(title, color='white', fontsize=10, y=1.02)
        plt.tight_layout()

        _finalize_figure(fig, outpath, show)
        return fig, result

    # ── HSI (Hue-Saturation-Intensity) composite ─────────────────────────────

    def plot_hsi(self, numerator, denominator, corrected=True,
                 intensity='denominator',
                 cmap='viridis', cmap_reverse=False,
                 ratio_min=None, ratio_max=None,
                 intensity_percentile=(1, 99.5),
                 min_counts=None,
                 scalebar_color='white',
                 outpath=None, show=None):
        """
        HSI (Hue-Saturation-Intensity) composite image.

        Hue encodes the ratio numerator/denominator; intensity (brightness)
        encodes counts in a chosen channel — typically the denominator,
        which marks "where the sample actually is". This solves the problem
        in plain ratio images that bright-but-noisy edge pixels look identical
        to bright-and-meaningful sample pixels.

        Parameters
        ----------
        numerator, denominator : str or int
            Channel name or index.
        corrected : bool
            Use drift-corrected stack if available.
        intensity : str
            Source for the brightness channel:
              'denominator' (default) — counts in the denominator channel
              'numerator'             — counts in the numerator channel
              'sum'                   — A + B
        cmap : str
            Colourmap for the hue (ratio) axis. Useful options:
              'viridis'  — perceptually uniform, recommended scientific default
              'plasma', 'inferno', 'magma' — perceptually uniform alternatives
              'twilight' — isoluminant, suited to HSI where hue alone matters
              'rainbow'  — alias for 'hsv'
              'hsv'      — same as 'rainbow', explicit name
              'classic_openmims', 'classic openmims', 'jet'
                         — the classic OpenMIMS rainbow LUT (matplotlib jet)
        cmap_reverse : bool
            Reverse the colourmap (equivalent to using a `_r` suffix).
        ratio_min, ratio_max : float or None
            Hue scaling range. If None, uses (1st, 99th) percentile of the
            ratio for valid pixels. Pixels outside this range are clipped.
        intensity_percentile : (lo, hi)
            Percentile clip for the intensity normalisation (default 1, 99.5).
        min_counts : float or None
            Mask pixels (set to black) where the denominator counts are
            below this threshold.
        scalebar_color : str
            Scale bar colour (default white).
        outpath, show : see plot()

        Returns
        -------
        (fig, info)
            fig  : matplotlib Figure
            info : dict with ratio, intensity, mask, ratio_range, cmap_name
        """
        # Map our cmap aliases
        cmap_alias = {
            'rainbow'         : 'hsv',
            'classic_openmims': 'jet',
            'classic openmims': 'jet',
            'Classic OpenMIMS': 'jet',
            'Classic OpenMIMS LUT': 'jet',
        }
        cmap_name = cmap_alias.get(cmap, cmap)
        if cmap_reverse and not cmap_name.endswith('_r'):
            cmap_name = cmap_name + '_r'

        # Auto outpath in script mode
        if show is None:
            show = _IS_NOTEBOOK
        if outpath is None and not _IS_NOTEBOOK:
            base    = os.path.splitext(os.path.basename(self.path))[0]
            num_i   = self._resolve_channel(numerator)
            den_i   = self._resolve_channel(denominator)
            num_safe = self.masses[num_i].replace(' ', '').replace('/', '_')
            den_safe = self.masses[den_i].replace(' ', '').replace('/', '_')
            outpath = os.path.join(os.getcwd(),
                                   f"{base}_hsi_{num_safe}_over_{den_safe}.png")

        # Build the ratio (sum across planes, with masking)
        result = self.ratio(numerator, denominator, corrected=corrected,
                            min_counts=min_counts)
        R       = result['ratio']     # may contain NaN
        A       = result['A']
        B       = result['B']
        num_lab = result['num_label']
        den_lab = result['den_label']

        # Choose intensity source
        if intensity == 'denominator':
            I = B
            intensity_label = den_lab
        elif intensity == 'numerator':
            I = A
            intensity_label = num_lab
        elif intensity == 'sum':
            I = A + B
            intensity_label = f'{num_lab} + {den_lab}'
        else:
            raise ValueError(f"intensity must be 'denominator', 'numerator', "
                             f"or 'sum'; got {intensity!r}")

        # Determine ratio range for hue mapping
        finite = np.isfinite(R)
        if not finite.any():
            raise ValueError("No valid pixels — relax masking thresholds.")
        if ratio_min is None:
            ratio_min = float(np.percentile(R[finite], 1))
        if ratio_max is None:
            ratio_max = float(np.percentile(R[finite], 99))
        if ratio_max <= ratio_min:
            ratio_max = ratio_min + 1e-12

        # Normalise ratio to [0, 1] for colourmap lookup
        R_norm = np.clip((R - ratio_min) / (ratio_max - ratio_min), 0, 1)
        # NaN pixels become 0 (will be black after intensity multiplication)
        R_norm = np.where(np.isnan(R_norm), 0, R_norm)

        # Apply colourmap — get RGB for each pixel
        colormap = plt.get_cmap(cmap_name)
        rgb = colormap(R_norm)[:, :, :3]  # drop alpha

        # Normalise intensity
        I_finite = I[np.isfinite(I) & (I > 0)]
        if I_finite.size > 0:
            i_lo, i_hi = np.percentile(I_finite,
                                       [intensity_percentile[0],
                                        intensity_percentile[1]])
        else:
            i_lo, i_hi = 0, 1
        if i_hi <= i_lo:
            i_hi = i_lo + 1
        I_norm = np.clip((I - i_lo) / (i_hi - i_lo), 0, 1)

        # Apply mask: pixels where ratio is invalid OR intensity-source is too low
        mask = np.isfinite(R) & np.isfinite(I)
        if min_counts is not None:
            mask &= (B >= min_counts)
        I_norm = np.where(mask, I_norm, 0)

        # Multiply colour by intensity to get final RGB
        hsi_rgb = rgb * I_norm[:, :, None]
        hsi_rgb = np.clip(hsi_rgb, 0, 1)

        # ── Build figure: HSI image + a colourbar showing the hue scale ─────
        bg       = '#1a1a1a'
        field_um = self.metadata['field_um']
        sb_um    = self._nice_scalebar(field_um)

        fig = plt.figure(figsize=(5, 5), facecolor=bg)
        # Main image axis + a thin colourbar axis on the right
        ax       = fig.add_axes([0.05, 0.08, 0.74, 0.84])
        cbar_ax  = fig.add_axes([0.82, 0.08, 0.04, 0.84])

        ax.imshow(hsi_rgb, extent=[0, field_um, field_um, 0],
                  interpolation='nearest')
        ax.set_facecolor(bg)
        ax.axis('off')
        ax.set_xlim(0, field_um)
        ax.set_ylim(field_um, 0)

        # Scale bar
        margin = field_um * 0.05
        bar_y  = field_um - margin
        bar_x0 = margin
        bar_x1 = margin + sb_um
        tick_h = field_um * 0.02
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], '-',
                color=scalebar_color, linewidth=2.5, solid_capstyle='butt')
        for xp in [bar_x0, bar_x1]:
            ax.plot([xp, xp], [bar_y - tick_h, bar_y + tick_h],
                    '-', color=scalebar_color, linewidth=2)
        ax.text((bar_x0 + bar_x1) / 2, bar_y - tick_h * 2,
                f'{sb_um:g} μm', color=scalebar_color,
                fontsize=9, ha='center', va='bottom')

        # Hue colourbar — shows the ratio scale only (no intensity component)
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(norm=Normalize(vmin=ratio_min, vmax=ratio_max),
                            cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'Ratio  {num_lab}/{den_lab}',
                       color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                      labelsize=8)
        cbar.outline.set_edgecolor('white')

        # Title (kept compact for the smaller figure)
        title = (f"HSI {num_lab}/{den_lab}  |  "
                 f"intensity: {intensity_label}  |  "
                 f"cmap: {cmap}{'_r' if cmap_reverse else ''}\n"
                 f"hue range: {ratio_min:.4g}–{ratio_max:.4g}  |  "
                 f"{os.path.basename(self.path)}")
        fig.suptitle(title, color='white', fontsize=8, y=0.98)

        info = {
            'ratio'      : R,
            'intensity'  : I,
            'mask'       : mask,
            'ratio_range': (ratio_min, ratio_max),
            'cmap_name'  : cmap_name,
        }
        _finalize_figure(fig, outpath, show)
        return fig, info

    # ── Publication-quality export ───────────────────────────────────────────

    def _draw_scalebar(self, ax, field_um, sb_um, color='white',
                       linewidth=2.5, fontsize=9):
        """Draw a scale bar with end ticks and label on the given axes."""
        margin = field_um * 0.05
        bar_y  = field_um - margin
        bar_x0 = margin
        bar_x1 = margin + sb_um
        tick_h = field_um * 0.02
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], '-',
                color=color, linewidth=linewidth, solid_capstyle='butt')
        for xp in [bar_x0, bar_x1]:
            ax.plot([xp, xp], [bar_y - tick_h, bar_y + tick_h],
                    '-', color=color, linewidth=max(1.5, linewidth - 0.5))
        ax.text((bar_x0 + bar_x1) / 2, bar_y - tick_h * 2,
                f'{sb_um:g} μm', color=color,
                fontsize=fontsize, ha='center', va='bottom')

    def _render_channel_panel(self, ax, channel, *, fig=None,
                              corrected=True, plane=None, cmap='gray',
                              log_scale=False, scalebar_color='white',
                              percentile=99.5, fontsize=9, title=None,
                              cbar_label=True):
        """Render a single channel image onto a given matplotlib Axes."""
        from matplotlib.colors import SymLogNorm
        ch = self._resolve_channel(channel)
        if corrected and self.corrected is not None:
            arr = self.corrected
        else:
            arr = self.data.astype(float)

        if plane is None:
            data = arr[:, ch].sum(axis=0)
        else:
            data = arr[plane, ch]

        field_um = self.metadata['field_um']
        sb_um    = self._nice_scalebar(field_um)
        p99      = np.percentile(data, percentile) if data.max() > 0 else 1

        if log_scale:
            norm = SymLogNorm(linthresh=1.0, linscale=1.0,
                              vmin=0, vmax=max(p99, 2))
            im = ax.imshow(data, cmap=cmap, norm=norm,
                           extent=[0, field_um, field_um, 0],
                           interpolation='nearest')
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=p99,
                           extent=[0, field_um, field_um, 0],
                           interpolation='nearest')

        ax.set_title(title if title is not None else self.masses[ch],
                     color='white', fontsize=fontsize + 1,
                     fontweight='bold', pad=4)
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        ax.set_xlim(0, field_um); ax.set_ylim(field_um, 0)

        # Colorbar
        if fig is not None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if cbar_label:
                cbar.set_label('Counts (log)' if log_scale else 'Counts',
                               color='white', fontsize=fontsize)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=fontsize - 1)
            cbar.outline.set_edgecolor('white')
            if not log_scale:
                cbar.locator = ticker.MaxNLocator(nbins=5, integer=True)
                cbar.update_ticks()

        self._draw_scalebar(ax, field_um, sb_um, color=scalebar_color,
                            linewidth=2.5, fontsize=fontsize)
        return im

    def _render_ratio_panel(self, ax, *, fig=None, kind='ratio',
                            numerator=None, denominator=None,
                            delta_ref=None, min_counts=None, max_rel_err=None,
                            cmap=None, scalebar_color='white',
                            percentile=(1, 99), delta_range=None,
                            fontsize=9, title=None,
                            precomputed=None):
        """Render one of {ratio, delta, sigma, rel_err} onto a given Axes.

        If `precomputed` is a dict from a previous ratio() call, reuse it
        rather than recomputing. Saves work when rendering several panels
        from the same A/B pair.
        """
        if precomputed is None:
            precomputed = self.ratio(numerator, denominator,
                                     min_counts=min_counts,
                                     max_rel_err=max_rel_err)
        R       = precomputed['ratio']
        sigma   = precomputed['sigma']
        rel_err = precomputed['rel_err']
        num_lab = precomputed['num_label']
        den_lab = precomputed['den_label']
        field_um = self.metadata['field_um']
        sb_um    = self._nice_scalebar(field_um)

        # Prepare image and colour scale by panel kind
        if kind == 'ratio':
            img_data = R
            cmap_use = cmap or 'viridis'
            xf       = img_data[np.isfinite(img_data)]
            vmin     = float(np.percentile(xf, percentile[0])) if xf.size else 0
            vmax     = float(np.percentile(xf, percentile[1])) if xf.size else 1
            cbar_label = ''
            panel_title = title or f'Ratio  {num_lab}/{den_lab}'

        elif kind == 'delta':
            if delta_ref is None:
                ref = float(np.nanmedian(R))
                ref_label = f'image median ({ref:.4g})'
            else:
                ref = float(delta_ref)
                ref_label = f'{ref:.4g}'
            img_data = (R / ref - 1.0) * 1000.0
            cmap_use = cmap or 'RdBu_r'
            if delta_range is None:
                d_finite = img_data[np.isfinite(img_data)]
                mad = np.median(np.abs(d_finite - np.median(d_finite)))
                sd_robust = 1.4826 * mad if mad > 0 else np.nanstd(d_finite)
                delta_range = max(2 * sd_robust, 1.0)
            vmin, vmax = -delta_range, delta_range
            cbar_label = '‰'
            panel_title = title or f'δ vs {ref_label} (‰)'

        elif kind == 'sigma':
            img_data = sigma
            cmap_use = cmap or 'magma'
            xf = img_data[np.isfinite(img_data)]
            vmin = float(np.percentile(xf, percentile[0])) if xf.size else 0
            vmax = float(np.percentile(xf, percentile[1])) if xf.size else 1
            cbar_label = ''
            panel_title = title or f'σ(R)  {num_lab}/{den_lab}'

        elif kind == 'rel_err':
            img_data = rel_err
            cmap_use = cmap or 'magma'
            xf = img_data[np.isfinite(img_data)]
            vmin = float(np.percentile(xf, percentile[0])) if xf.size else 0
            vmax = float(np.percentile(xf, percentile[1])) if xf.size else 1
            vmax = min(vmax, 1.0)
            if vmin >= vmax: vmin = 0.0
            cbar_label = ''
            panel_title = title or f'σ(R)/R  {num_lab}/{den_lab}'

        else:
            raise ValueError(f"Unknown ratio panel kind: {kind!r}")

        im = ax.imshow(img_data, cmap=cmap_use, vmin=vmin, vmax=vmax,
                       extent=[0, field_um, field_um, 0],
                       interpolation='nearest')
        ax.set_title(panel_title, color='white', fontsize=fontsize + 1,
                     fontweight='bold', pad=4)
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        ax.set_xlim(0, field_um); ax.set_ylim(field_um, 0)

        if fig is not None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if cbar_label:
                cbar.set_label(cbar_label, color='white', fontsize=fontsize)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=fontsize - 1)
            cbar.outline.set_edgecolor('white')

        self._draw_scalebar(ax, field_um, sb_um, color=scalebar_color,
                            linewidth=2.5, fontsize=fontsize)
        return precomputed

    def _render_hsi_panel(self, ax, *, fig=None, numerator=None,
                          denominator=None, intensity='denominator',
                          cmap='viridis', cmap_reverse=False,
                          ratio_min=None, ratio_max=None,
                          intensity_percentile=(1, 99.5),
                          min_counts=None, scalebar_color='white',
                          fontsize=9, title=None):
        """Render an HSI composite onto a given Axes."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        cmap_alias = {
            'rainbow': 'hsv',
            'classic_openmims': 'jet', 'classic openmims': 'jet',
            'Classic OpenMIMS': 'jet', 'Classic OpenMIMS LUT': 'jet',
        }
        cmap_name = cmap_alias.get(cmap, cmap)
        if cmap_reverse and not cmap_name.endswith('_r'):
            cmap_name = cmap_name + '_r'

        result  = self.ratio(numerator, denominator, min_counts=min_counts)
        R       = result['ratio']
        A       = result['A']; B = result['B']
        num_lab = result['num_label']; den_lab = result['den_label']

        if intensity == 'denominator':
            I = B
        elif intensity == 'numerator':
            I = A
        elif intensity == 'sum':
            I = A + B
        else:
            raise ValueError(f"intensity must be 'denominator', 'numerator', "
                             f"or 'sum'; got {intensity!r}")

        finite = np.isfinite(R)
        if not finite.any():
            raise ValueError("No valid pixels in HSI panel.")
        if ratio_min is None: ratio_min = float(np.percentile(R[finite], 1))
        if ratio_max is None: ratio_max = float(np.percentile(R[finite], 99))
        if ratio_max <= ratio_min: ratio_max = ratio_min + 1e-12

        R_norm = np.clip((R - ratio_min) / (ratio_max - ratio_min), 0, 1)
        R_norm = np.where(np.isnan(R_norm), 0, R_norm)
        cmap_obj = plt.get_cmap(cmap_name)
        rgb = cmap_obj(R_norm)[:, :, :3]

        I_finite = I[np.isfinite(I) & (I > 0)]
        if I_finite.size > 0:
            i_lo, i_hi = np.percentile(I_finite,
                                       [intensity_percentile[0],
                                        intensity_percentile[1]])
        else:
            i_lo, i_hi = 0, 1
        if i_hi <= i_lo: i_hi = i_lo + 1
        I_norm = np.clip((I - i_lo) / (i_hi - i_lo), 0, 1)

        mask = np.isfinite(R) & np.isfinite(I)
        if min_counts is not None: mask &= (B >= min_counts)
        I_norm = np.where(mask, I_norm, 0)

        hsi_rgb = np.clip(rgb * I_norm[:, :, None], 0, 1)

        field_um = self.metadata['field_um']
        sb_um    = self._nice_scalebar(field_um)

        ax.imshow(hsi_rgb, extent=[0, field_um, field_um, 0],
                  interpolation='nearest')
        ax.set_title(title or f'HSI  {num_lab}/{den_lab}',
                     color='white', fontsize=fontsize + 1,
                     fontweight='bold', pad=4)
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        ax.set_xlim(0, field_um); ax.set_ylim(field_um, 0)

        if fig is not None:
            sm = ScalarMappable(norm=Normalize(vmin=ratio_min, vmax=ratio_max),
                                cmap=cmap_obj)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(f'{num_lab}/{den_lab}', color='white',
                           fontsize=fontsize)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=fontsize - 1)
            cbar.outline.set_edgecolor('white')

        self._draw_scalebar(ax, field_um, sb_um, color=scalebar_color,
                            linewidth=2.5, fontsize=fontsize)

    def save_publication(self, panels, outpath, *,
                         grid=None, panel_size=(4, 4),
                         dpi=600, format=None, suptitle=None,
                         fontsize=9):
        """
        Save a publication-quality figure as PNG or TIFF.

        Parameters
        ----------
        panels : list of dict, or single dict
            Each dict specifies one panel:
              {'kind': 'channel',  'channel': '12C 14N', ...}
              {'kind': 'ratio',    'numerator': '12C 15N',
                                   'denominator': '12C 14N', ...}
              {'kind': 'delta',    'numerator': ..., 'denominator': ...,
                                   'delta_ref': ...}
              {'kind': 'sigma',    'numerator': ..., 'denominator': ...}
              {'kind': 'rel_err',  'numerator': ..., 'denominator': ...}
              {'kind': 'hsi',      'numerator': ..., 'denominator': ...,
                                   'cmap': 'Classic OpenMIMS LUT', ...}
            Each dict can include any kwargs the corresponding renderer
            accepts (cmap, min_counts, log_scale, scalebar_color, title, etc.).
        outpath : str
            Output filename. Extension determines format if `format` is None
            (.png → PNG, .tif/.tiff → TIFF).
        grid : (rows, cols) or None
            Grid layout. If None, auto-picks a near-square arrangement.
        panel_size : (w, h)
            Inches per panel. Total figure size is grid * panel_size.
        dpi : int
            Output resolution (default 600 — publication quality).
        format : 'png' or 'tif' or None
            Override format, otherwise inferred from outpath extension.
        suptitle : str or None
            Optional figure-wide title.
        fontsize : int
            Base font size in points (default 9). Titles use fontsize+1,
            tick labels use fontsize-1.

        Examples
        --------
        # Single HSI panel:
        img.save_publication(
            {'kind': 'hsi', 'numerator': '12C 15N', 'denominator': '12C 14N',
             'cmap': 'Classic OpenMIMS LUT'},
            'figure_hsi.png', dpi=600)

        # Multi-panel grid: ¹²C¹⁴N channel, HSI, δ panel:
        img.save_publication(
            panels=[
                {'kind': 'channel', 'channel': '12C 14N', 'log_scale': True},
                {'kind': 'hsi',     'numerator': '12C 15N',
                                    'denominator': '12C 14N',
                                    'cmap': 'Classic OpenMIMS LUT'},
                {'kind': 'delta',   'numerator': '12C 15N',
                                    'denominator': '12C 14N',
                                    'delta_ref': 0.0036765},
            ],
            outpath='figure_3panel.tif',
            grid=(1, 3),
            panel_size=(3.5, 3.5),
            dpi=600,
        )
        """
        # Normalise inputs
        if isinstance(panels, dict):
            panels = [panels]
        n = len(panels)
        if n == 0:
            raise ValueError("panels must contain at least one panel.")

        # Auto-grid: prefer wider than tall, near-square
        if grid is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            grid = (rows, cols)
        rows, cols = grid
        if rows * cols < n:
            raise ValueError(f"grid {grid} too small for {n} panels.")

        # Format inference
        if format is None:
            ext = os.path.splitext(outpath)[1].lower().lstrip('.')
            format = 'tiff' if ext in ('tif', 'tiff') else 'png'
        format = format.lower()
        if format in ('tif',): format = 'tiff'
        if format not in ('png', 'tiff'):
            raise ValueError("format must be 'png' or 'tiff'.")

        bg = '#1a1a1a'
        fig_w = panel_size[0] * cols
        fig_h = panel_size[1] * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                                 facecolor=bg, squeeze=False)

        # Cache ratio() outputs so multi-panel ratio figures reuse the work
        ratio_cache = {}

        for idx, spec in enumerate(panels):
            r, c = idx // cols, idx % cols
            ax = axes[r][c]
            kind = spec.get('kind')
            kw = {k: v for k, v in spec.items() if k != 'kind'}

            if kind == 'channel':
                self._render_channel_panel(ax, fig=fig, fontsize=fontsize, **kw)

            elif kind in ('ratio', 'delta', 'sigma', 'rel_err'):
                cache_key = (kw.get('numerator'), kw.get('denominator'),
                             kw.get('min_counts'), kw.get('max_rel_err'))
                kw.setdefault('precomputed', ratio_cache.get(cache_key))
                result = self._render_ratio_panel(ax, fig=fig, kind=kind,
                                                  fontsize=fontsize, **kw)
                ratio_cache[cache_key] = result

            elif kind == 'hsi':
                self._render_hsi_panel(ax, fig=fig, fontsize=fontsize, **kw)

            else:
                raise ValueError(f"Unknown panel kind: {kind!r}")

        # Hide any unused axes
        for idx in range(n, rows * cols):
            r, c = idx // cols, idx % cols
            axes[r][c].axis('off')
            axes[r][c].set_facecolor(bg)

        if suptitle:
            fig.suptitle(suptitle, color='white', fontsize=fontsize + 2,
                         y=0.995)

        plt.tight_layout()

        # Save with publication-grade options
        save_kw = {'dpi': dpi, 'bbox_inches': 'tight',
                   'facecolor': fig.get_facecolor()}
        if format == 'tiff':
            save_kw['format'] = 'tiff'
            save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        else:
            save_kw['format'] = 'png'

        fig.savefig(outpath, **save_kw)
        plt.close(fig)
        print(f"Saved publication figure: {outpath}  "
              f"({rows}×{cols} panels, {fig_w:.1f}×{fig_h:.1f}\" @ {dpi} dpi)")
        return outpath

    # ── Data access ──────────────────────────────────────────────────────────

    def get_channel(self, label):
        """Return the stack for a named channel as (planes, height, width)."""
        ch  = self._resolve_channel(label)
        arr = self.corrected if self.corrected is not None else self.data.astype(float)
        return arr[:, ch, :, :]

    def sum_stack(self, corrected=True):
        """Return summed stack (n_masses, height, width)."""
        if corrected and self.corrected is not None:
            return self.corrected.sum(axis=0)
        return self.data.astype(float).sum(axis=0)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _nice_scalebar(field_um):
        """Choose a round scale bar length appropriate for the field size."""
        candidates = [0.5, 1, 2, 5, 10, 20, 25, 50]
        for c in candidates:
            if c / field_um < 0.4:
                best = c
        return best

    def __repr__(self):
        m = self.metadata
        lines = [
            f"MimsImage: {os.path.basename(self.path)}",
            f"  Instrument : {m['instrument']}  |  {m['date']}  {m['time']}",
            f"  Image      : {m['width']}×{m['height']} px  |  "
            f"{m['n_planes']} planes  |  {m['field_um']:.3f} μm field",
            f"  Pixel size : {m['pixel_nm']:.4f} nm",
            f"  Masses     : {m['n_masses']}",
        ]
        for i, (label, mass) in enumerate(zip(self.masses, self.nom_masses)):
            lines.append(f"    [{i}] {label:20s}  {mass:.5f} amu")
        if self.corrected is not None:
            dy = self.shifts[:, 0]
            dx = self.shifts[:, 1]
            lines.append(f"  Drift corr : Y {dy.min():+d}..{dy.max():+d} px  |  "
                         f"X {dx.min():+d}..{dx.max():+d} px")
        else:
            lines.append(f"  Drift corr : not applied")
        return '\n'.join(lines)


# ── Command-line usage ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 pymims.py yourfile.im [output.png]")
        sys.exit(1)

    filepath = sys.argv[1]
    outpath  = sys.argv[2] if len(sys.argv) > 2 else None

    img = MimsImage(filepath)
    print(img)
    print()
    # Use SE as drift reference if present, otherwise the first channel
    ref = 'SE' if any('SE' in m.upper() for m in img.masses) else img.masses[0]
    img.drift_correct(reference=ref)
    print()
    saved = img.plot(outpath=outpath)
    img.plot_drift()
    print("\nDone.")
