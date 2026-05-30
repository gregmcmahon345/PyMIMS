"""
pymims_app.py  —  PyMIMS v0.3 desktop GUI (Panel)
=================================================================

    Imaging   — single channel / ratio / HSI; display filters, a GLOBAL
                binning (sum-pool) control applied uniformly to every image so
                overlays/side-by-sides co-register, HSI ratio-median, Plane QC
                (per-plane scan + auto/manual drop), and save-at-DPI
    HMR       — high mass resolution deconvolution            (placeholder)
    Analysis  — clustering + intensity histograms             (placeholder)
    Metadata  — full reverse-engineered Cameca header         (working)

Run:  panel serve pymims_app.py --show
(--autoreload triggers a Panel warm-up bug on some versions; omit it.)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import panel as pn
import param

from pymims import MimsImage, save_figure
from pymims import metadata as pymims_metadata
from pymims.core import _bin_sum            # same crop+sum-pool helper as the library

# Treat the GUI as an inline-display host: plot_* methods return their Figure
# without writing PNGs to disk or closing them.
import pymims.core as _pymims_core
_pymims_core._IS_NOTEBOOK = True

try:
    from pymims import __version__ as PYMIMS_VERSION
except Exception:
    PYMIMS_VERSION = "0.3.0"

pn.extension(sizing_mode="stretch_width", notifications=True)

_BIN_OPTIONS = {"1× (full res)": 1, "2×2": 2, "4×4": 4, "8×8": 8}


# ─────────────────────────────────────────────────────────────────────────────
# Shared application state
# ─────────────────────────────────────────────────────────────────────────────

class State(param.Parameterized):
    img = param.Parameter(default=None)
    drift_version = param.Integer(default=0)   # bump to re-render after drift
    plane_version = param.Integer(default=0)   # bump to re-render after a plane drop
    bin_n = param.Integer(default=1)           # GLOBAL binning (sum-pool) factor


state = State()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: file picker + global binning (sum-pool) control
# ─────────────────────────────────────────────────────────────────────────────

dir_input = pn.widgets.TextInput(name="Directory", value=os.path.expanduser("~"))
up_btn = pn.widgets.Button(name="⬆ Up one level", button_type="default", width=140)
subdir_select = pn.widgets.Select(name="Enter subfolder", options=["—"], value="—")
file_select = pn.widgets.Select(name=".im files here", options=[])
load_btn = pn.widgets.Button(name="Load file", button_type="primary", icon="upload")
load_status = pn.pane.Markdown("*No file loaded.*")

bin_select = pn.widgets.Select(name="Binning (sum-pool)",
                               options=list(_BIN_OPTIONS), value="1× (full res)")
bin_note = pn.pane.Markdown(
    "<span style='font-size:11px;opacity:0.75'>Sum-pools counts into n×n blocks "
    "and applies to **all** images (single channel, ratio, HSI) so they "
    "co-register. Non-destructive — the full-resolution data is never "
    "altered.</span>")


def _on_bin(event):
    state.bin_n = _BIN_OPTIONS.get(event.new, 1)
bin_select.param.watch(_on_bin, "value")


# Session save / load (.npz) — resume analysis without the original .im
session_dir = pn.widgets.TextInput(name="Session folder",
                                    value=os.path.expanduser("~"))
session_name = pn.widgets.TextInput(name="Save as (name)", value="session")
save_session_btn = pn.widgets.Button(name="💾 Save session",
                                      button_type="default")
session_select = pn.widgets.Select(name="Open session", options=[])
load_session_btn = pn.widgets.Button(name="📂 Load session",
                                      button_type="default")
session_status = pn.pane.Markdown("")
session_note = pn.pane.Markdown(
    "<span style='font-size:11px;opacity:0.75'>Saves the cleaned / drift-"
    "corrected stack, dropped-plane record, metadata and raw header to a "
    "<code>.npz</code> — reload it later without the original <code>.im</code>. "
    "“Open session” lists the <code>.npz</code> files in the session folder; "
    "pick one and Load. Binning is a view setting and is not saved.</span>")


def _session_folder():
    return os.path.expanduser(session_dir.value or "~")


def _refresh_sessions(*events):
    folder = _session_folder()
    try:
        npzs = sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(".npz"))
    except Exception:
        npzs = []
    session_select.options = npzs
    if session_select.value not in npzs:
        session_select.value = npzs[0] if npzs else None


def _save_session(event):
    if state.img is None:
        session_status.object = "⚠️ Load a file before saving a session."
        return
    try:
        folder = _session_folder()
        os.makedirs(folder, exist_ok=True)
        name = (session_name.value or "session").strip()
        if not name.lower().endswith(".npz"):
            name += ".npz"
        path = os.path.join(folder, name)
        state.img.save_session(path, verbose=False)
        mb = os.path.getsize(path) / 1e6
        _refresh_sessions()
        session_select.value = name                # surface the just-saved file
        session_status.object = f"✅ Saved `{name}` ({mb:.1f} MB)"
    except Exception as exc:
        session_status.object = f"❌ Save failed: `{exc}`"


def _load_session(event):
    name = session_select.value
    if not name:
        session_status.object = ("⚠️ No session selected — pick one from "
                                 "“Open session”.")
        return
    try:
        path = os.path.join(_session_folder(), name)
        if not os.path.exists(path):
            session_status.object = f"⚠️ Not found: `{path}`"
            _refresh_sessions()
            return
        loaded = MimsImage.load_session(path, verbose=False)
        state.img = loaded                       # rebuilds all tabs
        m = loaded.metadata
        drift = "drift applied" if loaded.corrected is not None else "no drift"
        session_status.object = (
            f"✅ Loaded `{name}`  \n"
            f"{m.get('width', '?')}×{m.get('height', '?')} px · "
            f"{m.get('n_planes', '?')} planes · {drift}")
        load_status.object = (
            f"📂 **Session: {name}**  \n"
            f"(from {os.path.basename(loaded.path)})")
    except Exception as exc:
        session_status.object = f"❌ Load failed: `{exc}`"


session_dir.param.watch(_refresh_sessions, "value")
save_session_btn.on_click(_save_session)
load_session_btn.on_click(_load_session)
_refresh_sessions()


def _refresh_listing(*events):
    base = os.path.expanduser(dir_input.value or "~")
    try:
        entries = os.listdir(base)
    except Exception as exc:
        subdir_select.options = ["—"]
        subdir_select.value = "—"
        file_select.options = []
        load_status.object = f"⚠️ Can't read `{base}`: `{exc}`"
        return
    dirs = sorted(f for f in entries
                  if os.path.isdir(os.path.join(base, f)) and not f.startswith("."))
    ims = sorted(f for f in entries if f.lower().endswith(".im"))
    subdir_select.options = ["—"] + dirs
    if subdir_select.value not in subdir_select.options:
        subdir_select.value = "—"
    file_select.options = ims


def _enter_subdir(event):
    if event.new and event.new != "—":
        dir_input.value = os.path.join(os.path.expanduser(dir_input.value), event.new)


def _go_up(event):
    cur = os.path.expanduser(dir_input.value or "~").rstrip("/")
    dir_input.value = os.path.dirname(cur) or "/"


def _load(event):
    if not file_select.value:
        load_status.object = "⚠️ Choose a `.im` file from the list."
        return
    path = os.path.join(os.path.expanduser(dir_input.value), file_select.value)
    load_status.object = f"Loading `{file_select.value}` …"
    try:
        img = MimsImage(path)
        state.img = img
        m = img.metadata
        load_status.object = (
            f"✅ **{file_select.value}**  \n"
            f"{m['width']}×{m['height']} px · {m['n_planes']} planes · "
            f"{m['n_masses']} channels · {m['field_um']:.2f} µm field"
        )
    except Exception as exc:
        state.img = None
        load_status.object = f"❌ Failed to load: `{exc}`"


dir_input.param.watch(_refresh_listing, "value")
subdir_select.param.watch(_enter_subdir, "value")
up_btn.on_click(_go_up)
load_btn.on_click(_load)
_refresh_listing()


# ─────────────────────────────────────────────────────────────────────────────
# Imaging tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_channel(img, channel, cmap, upper_pct, sum_all, plane,
                    filt="None", strength=3.0, bin_n=1):
    """Single channel as a matplotlib figure. Binning (sum-pool, cropped) and
    filtering are applied to the DISPLAY array only — img.data is never touched."""
    stack = img.get_channel(channel)
    if sum_all:
        data = stack.sum(axis=0).astype(float)
        plane_lbl = "summed"
    else:
        p = int(min(max(plane, 0), stack.shape[0] - 1))
        data = stack[p].astype(float)
        plane_lbl = f"plane {p}"

    extra = []
    if bin_n > 1:
        data = _bin_sum(data, bin_n)
        extra.append(f"{bin_n}×{bin_n} bin")

    is_edge = filt in ("Sobel", "Canny")
    if filt == "Median":
        from scipy.ndimage import median_filter
        data = median_filter(data, size=max(1, int(strength)))
        extra.append(f"median {int(strength)}")
    elif filt == "Gaussian":
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=float(strength))
        extra.append(f"gaussian σ{strength:g}")
    elif filt == "Sobel":
        from skimage.filters import sobel
        data = sobel(data / (data.max() or 1.0))
        extra.append("sobel")
    elif filt == "Canny":
        from skimage.feature import canny
        data = canny(data / (data.max() or 1.0), sigma=float(strength)).astype(float)
        extra.append(f"canny σ{strength:g}")

    field = img.metadata["field_um"]
    if is_edge:
        vmin, vmax, cbar_label = 0.0, float(data.max() or 1.0), "edge"
    elif data.max() > 0:
        vmin, vmax, cbar_label = 0.0, float(np.percentile(data, upper_pct)), "counts"
    else:
        vmin, vmax, cbar_label = 0.0, 1.0, "counts"
    vmax = max(vmax, 1e-9)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, field, field, 0], interpolation="nearest")
    source = "drift-corrected" if img.corrected is not None else "raw"
    suffix = (" — " + ", ".join(extra)) if extra else ""
    ax.set_title(f"{channel} — {plane_lbl}, {source}{suffix}", fontsize=10)
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _build_fig(img, mode, ch, cm, pct, sa, pl, filt, strength, bin_n,
               nu, de, mc, mr, dr, hn, hd, inten, hcm, sc, rn, rx, hmed):
    if mode == "Single channel":
        return _render_channel(img, ch, cm, pct, sa, pl, filt, strength, bin_n)
    if mode == "Ratio":
        fig, _ = img.plot_ratio(nu, de, min_counts=(mc or None),
                                max_rel_err=(mr or None), delta_ref=(dr or None),
                                bin=bin_n, show=True)
        return fig
    fig, _ = img.plot_hsi(hn, hd, intensity=inten, cmap=hcm, scale_factor=sc,
                          ratio_min=(rn or None), ratio_max=(rx or None),
                          bin=bin_n, median_smooth=hmed, show=True)
    return fig


def imaging_view(img):
    if img is None:
        return pn.pane.Markdown("### Imaging\nLoad a `.im` file from the sidebar to begin.")

    masses = img.masses
    n_planes = int(img.metadata["n_planes"])
    non_se = [m for m in masses if "SE" not in m.upper()] or masses
    den_default = non_se[1] if len(non_se) > 1 else non_se[0]

    mode = pn.widgets.RadioButtonGroup(
        name="Mode", options=["Single channel", "Ratio", "HSI"],
        value="Single channel", button_type="primary")

    # Single-channel controls (bin is global, in the sidebar)
    channel = pn.widgets.Select(name="Channel", options=masses, value=masses[0], width=160)
    cmap = pn.widgets.Select(name="Colormap",
                             options=["gray", "viridis", "magma", "inferno", "cividis"],
                             value="gray", width=140)
    upper = pn.widgets.FloatSlider(name="Contrast (upper %ile)", start=90, end=100,
                                   step=0.1, value=99.5, width=230)
    sum_toggle = pn.widgets.Checkbox(name="Sum all planes", value=True)
    plane = pn.widgets.IntSlider(name="Plane", start=0,
                                 end=(n_planes - 1 if n_planes > 1 else 1),
                                 value=0, disabled=True, width=240)
    filt = pn.widgets.Select(name="Filter (display only)",
                             options=["None", "Median", "Gaussian", "Sobel", "Canny"],
                             value="None", width=170)
    strength = pn.widgets.FloatSlider(name="Filter size / σ", start=1, end=9,
                                      step=0.5, value=3, width=190)
    filt_note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>Filters are **display only** — "
        "they do not touch or change the underlying data. Ratios, HSI and the "
        "Analysis tab always use the raw counts.</span>")

    def _toggle_plane(event):
        cur_n = int(img.metadata["n_planes"])
        plane.disabled = event.new or cur_n <= 1
    sum_toggle.param.watch(_toggle_plane, "value")

    # Ratio controls
    num = pn.widgets.Select(name="Numerator", options=masses, value=non_se[0], width=160)
    den = pn.widgets.Select(name="Denominator", options=masses, value=den_default, width=160)
    min_counts = pn.widgets.IntInput(name="Min counts (denom; 0 = off)", value=0, start=0, width=190)
    max_rel = pn.widgets.FloatInput(name="Max σ/R (0 = off)", value=0.0, start=0.0, step=0.05, width=150)
    dref = pn.widgets.FloatInput(name="δ reference (0 = image median)", value=0.0, start=0.0, width=210)

    # HSI controls
    hnum = pn.widgets.Select(name="Numerator", options=masses, value=non_se[0], width=150)
    hden = pn.widgets.Select(name="Denominator", options=masses, value=den_default, width=150)
    intensity = pn.widgets.Select(name="Intensity from",
                                  options=["denominator", "numerator", "sum"],
                                  value="denominator", width=150)
    hcmap = pn.widgets.Select(name="Colormap",
                              options=["viridis", "magma", "inferno", "hsv", "Classic OpenMIMS LUT"],
                              value="viridis", width=180)
    scale = pn.widgets.FloatInput(name="Scale factor", value=10000.0, start=1.0, width=120)
    rmin = pn.widgets.FloatInput(name="Hue min (0 = auto)", value=0.0, width=140)
    rmax = pn.widgets.FloatInput(name="Hue max (0 = auto)", value=0.0, width=140)
    hmed = pn.widgets.IntInput(name="Ratio median (px; 0/1 = off)", value=0, start=0, width=190)
    hsi_note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>Ratio-median smooths the "
        "ratio **after** it is formed (display only). For genuine noise reduction "
        "use the global bin, which sum-pools the counts and enlarges the analytical "
        "volume.</span>")

    # ── Plane QC (run BEFORE drift correction) ────────────────────────────────
    qc_channel = pn.widgets.Select(name="Scan channel",
                                   options=["all channels"] + list(masses),
                                   value="all channels", width=170)
    qc_thresh = pn.widgets.FloatInput(name="Suspect threshold (% from median)",
                                      value=30.0, start=1.0, step=5.0, width=240)
    qc_scan_btn = pn.widgets.Button(name="Scan planes", button_type="default",
                                    icon="activity", width=130)
    qc_preview_btn = pn.widgets.Button(name="Auto-drop (preview)",
                                       button_type="default", width=170)
    qc_apply_btn = pn.widgets.Button(name="Apply auto-drop",
                                     button_type="warning", width=150)
    qc_manual = pn.widgets.TextInput(name="Drop planes (indices, e.g. 3,7,12)",
                                     value="", width=260)
    qc_add_current = pn.widgets.Button(name="↳ add scrubber plane",
                                       button_type="default", width=170)
    qc_drop_btn = pn.widgets.Button(name="Drop listed", button_type="warning",
                                    width=120)
    qc_reset_btn = pn.widgets.Button(name="↺ Reset to raw (reload)",
                                     button_type="default", width=200)
    qc_status = pn.pane.Markdown("")
    qc_plot_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No scan yet — click **Scan planes**.</span>"))
    qc_note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>Scans per-plane total counts "
        "and flags outliers (charging events, glitches, blank frames). "
        "<b>Run before drift correction</b> — bad planes are the main cause of "
        "silent drift-correction failures. Dropping is destructive in memory; "
        "<b>Reset to raw</b> reloads the file from disk (this also clears drift). "
        "To eyeball a suspect plane first, switch to Single channel, untick "
        "“Sum all planes”, scrub to it, then “↳ add scrubber plane”. "
        "(plane_movie remains available in Colab for animated inspection.)</span>")

    def _qc_channel_arg():
        v = qc_channel.value
        return None if v == "all channels" else v

    def _run_scan():
        cur_n = int(img.metadata["n_planes"])
        if cur_n < 2:
            qc_plot_box.objects = [pn.pane.Markdown(
                "<span style='opacity:0.6'>Need ≥2 planes to scan.</span>")]
            return None
        try:
            res = img.plot_plane_diagnostics(
                channel=_qc_channel_arg(),
                threshold_pct=float(qc_thresh.value),
                show=False)
        except Exception as exc:
            qc_plot_box.objects = [pn.pane.Alert(f"Scan failed: `{exc}`",
                                                 alert_type="danger")]
            return None
        fig = res["figure"]
        pane = pn.pane.Matplotlib(fig, dpi=100, tight=True, height=300)
        plt.close(fig)
        qc_plot_box.objects = [pane]
        return res

    def _after_inplace_change():
        """Refresh slider bounds, drift status, the main figure and the scan
        after an in-place plane drop (img.data was mutated under us)."""
        cur_n = int(img.metadata["n_planes"])
        new_end = cur_n - 1 if cur_n > 1 else 1
        plane.end = new_end
        if plane.value > new_end:
            plane.value = new_end
        plane.disabled = sum_toggle.value or cur_n <= 1
        drift_status.object = ("Drift: **applied**" if img.corrected is not None
                               else "Drift: not applied")
        state.plane_version += 1     # force the main image to re-render
        _run_scan()                  # keep the diagnostics plot current

    def _on_scan(event):
        res = _run_scan()
        if res is None:
            return
        sus = res["suspect_indices"]
        t = float(qc_thresh.value)
        if sus:
            qc_status.object = (f"Scan: **{len(sus)}** suspect plane(s) at "
                                f"±{t:.0f}%: `{sus}` · "
                                f"{int(img.metadata['n_planes'])} planes total.")
        else:
            qc_status.object = (f"Scan: no suspects at ±{t:.0f}% · "
                                f"{int(img.metadata['n_planes'])} planes.")
    qc_scan_btn.on_click(_on_scan)

    def _on_preview(event):
        cur_n = int(img.metadata["n_planes"])
        t = float(qc_thresh.value)
        if cur_n < 2:
            qc_status.object = "Need ≥2 planes to scan."
            return
        try:
            bad = img.auto_drop_bad_planes(threshold_pct=t,
                                           channel=_qc_channel_arg(),
                                           dry_run=True, verbose=False)
        except Exception as exc:
            qc_status.object = f"Preview failed: `{exc}`"
            return
        if bad:
            qc_status.object = (f"Auto-drop **preview**: would drop {len(bad)} "
                                f"plane(s) `{bad}` → {cur_n - len(bad)} remain. "
                                f"Click **Apply auto-drop** to commit.")
        else:
            qc_status.object = f"Auto-drop preview: nothing flagged at ±{t:.0f}%."
    qc_preview_btn.on_click(_on_preview)

    def _on_apply(event):
        cur_n = int(img.metadata["n_planes"])
        t = float(qc_thresh.value)
        if cur_n < 2:
            qc_status.object = "Need ≥2 planes to scan."
            return
        try:
            dropped = img.auto_drop_bad_planes(threshold_pct=t,
                                               channel=_qc_channel_arg(),
                                               dry_run=False, verbose=False)
        except Exception as exc:
            qc_status.object = f"Auto-drop failed: `{exc}`"
            return
        if dropped:
            _after_inplace_change()
            qc_status.object = (f"✅ Dropped {len(dropped)} plane(s) `{dropped}` "
                                f"→ {int(img.metadata['n_planes'])} planes remain. "
                                f"(Reset to raw to undo.)")
        else:
            qc_status.object = f"Nothing flagged at ±{t:.0f}% — no change."
    qc_apply_btn.on_click(_on_apply)

    def _parse_indices(text):
        out = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))      # ValueError propagates to caller
        return out

    def _on_add_current(event):
        cur = qc_manual.value.strip()
        add = str(int(plane.value))
        qc_manual.value = (cur + ", " + add) if cur else add
    qc_add_current.on_click(_on_add_current)

    def _on_drop_listed(event):
        try:
            idx = _parse_indices(qc_manual.value)
        except ValueError:
            qc_status.object = ("⚠️ Couldn't parse indices — use comma-separated "
                                "integers, e.g. `3,7,12`.")
            return
        if not idx:
            qc_status.object = "⚠️ No plane indices entered."
            return
        try:
            dropped = img.drop_planes(idx, verbose=False)
        except Exception as exc:
            qc_status.object = f"❌ Drop failed: `{exc}`"
            return
        if dropped:
            _after_inplace_change()
            qc_manual.value = ""
            qc_status.object = (f"✅ Dropped {len(dropped)} plane(s) `{dropped}` "
                                f"→ {int(img.metadata['n_planes'])} planes remain. "
                                f"(Reset to raw to undo.)")
        else:
            qc_status.object = "No valid planes dropped (out of range?)."
    qc_drop_btn.on_click(_on_drop_listed)

    def _on_reset(event):
        try:
            fresh = MimsImage(img.path)
        except Exception as exc:
            qc_status.object = f"❌ Reset failed: `{exc}`"
            return
        # New object → the tab rebuilds from disk-fresh data (drift + drops cleared).
        state.img = fresh
    qc_reset_btn.on_click(_on_reset)

    qc_section = pn.Column(
        pn.pane.Markdown("**Plane QC**  <span style='font-size:11px;opacity:0.7'>"
                         "(run before drift correction)</span>"),
        pn.Row(qc_channel, qc_thresh, qc_scan_btn),
        pn.Row(qc_preview_btn, qc_apply_btn),
        pn.Row(qc_manual, qc_add_current, qc_drop_btn),
        pn.Row(qc_reset_btn),
        qc_status,
        qc_plot_box,
        qc_note,
    )

    # Drift correction (common)
    se_default = "SE" if any("SE" in m.upper() for m in masses) else masses[0]
    drift_ref = pn.widgets.Select(name="Drift reference", options=masses,
                                  value=se_default, width=170)
    drift_btn = pn.widgets.Button(name="Run drift correction",
                                  button_type="default", icon="arrows-move")
    drift_status = pn.pane.Markdown(
        "Drift: **applied**" if img.corrected is not None else "Drift: not applied")

    def _do_drift(event):
        try:
            img.drift_correct(reference=drift_ref.value)
            drift_status.object = "Drift: **applied**"
            state.drift_version += 1
        except Exception as exc:
            drift_status.object = f"Drift failed: `{exc}`"
    drift_btn.on_click(_do_drift)

    # Save current view at chosen DPI
    out_dir = pn.widgets.TextInput(name="Save to (folder)",
                                   value=os.path.expanduser("~"), width=240)
    save_name = pn.widgets.TextInput(name="Filename", value="figure", width=160)
    fmt = pn.widgets.Select(name="Format", options=["png", "pdf", "svg", "tiff"],
                            value="png", width=90)
    dpi_in = pn.widgets.IntInput(name="DPI", value=600, start=72, end=1200, width=90)
    save_btn = pn.widgets.Button(name="💾 Save view", button_type="default", width=120)
    save_status = pn.pane.Markdown("")

    def _save(event):
        try:
            fig = _build_fig(img, mode.value, channel.value, cmap.value, upper.value,
                             sum_toggle.value, plane.value, filt.value, strength.value,
                             state.bin_n, num.value, den.value, min_counts.value,
                             max_rel.value, dref.value, hnum.value, hden.value,
                             intensity.value, hcmap.value, scale.value, rmin.value,
                             rmax.value, hmed.value)
            folder = os.path.expanduser(out_dir.value or "~")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{save_name.value or 'figure'}.{fmt.value}")
            save_figure(fig, path, dpi=int(dpi_in.value))
            plt.close(fig)
            save_status.object = f"✅ Saved `{path}` at {int(dpi_in.value)} DPI"
        except Exception as exc:
            save_status.object = f"❌ Save failed: `{exc}`"
    save_btn.on_click(_save)

    def _controls(m):
        if m == "Single channel":
            return pn.Column(pn.Row(channel, cmap, upper),
                             pn.Row(sum_toggle, plane),
                             pn.Row(filt, strength),
                             filt_note)
        if m == "Ratio":
            return pn.Column(pn.Row(num, den), pn.Row(min_counts, max_rel, dref))
        return pn.Column(pn.Row(hnum, hden, intensity),
                         pn.Row(hcmap, scale, rmin, rmax),
                         pn.Row(hmed), hsi_note)

    def _view(m, ch, cm, pct, sa, pl, fl, st, nu, de, mc, mr, dr,
              hn, hd, inten, hcm, sc, rn, rx, hm, bn, _ver, _pver):
        try:
            fig = _build_fig(img, m, ch, cm, pct, sa, pl, fl, st, bn,
                             nu, de, mc, mr, dr, hn, hd, inten, hcm, sc, rn, rx, hm)
        except Exception as exc:
            return pn.pane.Alert(f"Render failed: `{exc}`", alert_type="danger")
        pane = pn.pane.Matplotlib(fig, dpi=110, tight=True, height=480)
        plt.close(fig)
        return pane

    view = pn.bind(_view, mode.param.value, channel.param.value, cmap.param.value,
                   upper.param.value, sum_toggle.param.value, plane.param.value,
                   filt.param.value, strength.param.value,
                   num.param.value, den.param.value, min_counts.param.value,
                   max_rel.param.value, dref.param.value, hnum.param.value,
                   hden.param.value, intensity.param.value, hcmap.param.value,
                   scale.param.value, rmin.param.value, rmax.param.value,
                   hmed.param.value, state.param.bin_n, state.param.drift_version,
                   state.param.plane_version)

    return pn.Column(
        mode,
        pn.panel(pn.bind(_controls, mode.param.value)),
        pn.layout.Divider(),
        qc_section,
        pn.layout.Divider(),
        pn.Row(drift_ref, drift_btn, drift_status),
        pn.Row(out_dir, save_name, fmt, dpi_in, save_btn),
        save_status,
        pn.panel(view),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metadata tab  (working)
# ─────────────────────────────────────────────────────────────────────────────

def metadata_view(img):
    if img is None:
        return pn.pane.Markdown("### Metadata\nLoad a `.im` file to view the full Cameca header.")
    try:
        meta = img.read_full_metadata()
    except Exception as exc:
        return pn.pane.Alert(f"Could not read metadata: `{exc}`", alert_type="danger")

    panes = []
    layout = meta.get("layout", {})
    if not layout.get("layout_supported", True):
        reasons = "\n".join(f"- {r}" for r in layout.get("reasons", []))
        panes.append(pn.pane.Alert(
            "**Header layout not recognised** — the analytical values below may be "
            "unreliable for this file.\n\n" + reasons, alert_type="warning"))
    else:
        panes.append(pn.pane.Alert(
            "Header layout recognised — values validated for this acquisition type.",
            alert_type="success"))

    text = pymims_metadata.format_metadata(meta, masses=img.masses)
    panes.append(pn.pane.Str(text, styles={
        "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
        "white-space": "pre-wrap",
        "font-size": "12px",
        "line-height": "1.35",
    }))
    return pn.Column(*panes, sizing_mode="stretch_width")


# ─────────────────────────────────────────────────────────────────────────────
# HMR + Analysis tabs  (placeholders)
# ─────────────────────────────────────────────────────────────────────────────

def hmr_view():
    return pn.pane.Markdown(
        "### HMR — high mass resolution deconvolution\n\n"
        "*Coming next.* Load `.hmr` / `.hmr_txt` / `.xlsx`, run the v1 deconvolution, "
        "and show the spectrum with species-ID overlay and residuals."
    )


def analysis_view(img):
    if img is None:
        return pn.pane.Markdown("### Analysis\nLoad a `.im` file to run clustering and distributions.")
    return pn.pane.Markdown(
        "### Analysis\n\n"
        "*Coming next.* k-means / hierarchical clustering (`cluster_pixels`) and "
        "pixel-intensity histograms with GMM thresholds (`plot_histograms`)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assemble the app
# ─────────────────────────────────────────────────────────────────────────────

tabs = pn.Tabs(
    ("Imaging",  pn.panel(pn.bind(imaging_view, state.param.img))),
    ("HMR",      hmr_view()),
    ("Analysis", pn.panel(pn.bind(analysis_view, state.param.img))),
    ("Metadata", pn.panel(pn.bind(metadata_view, state.param.img))),
    dynamic=True,
    sizing_mode="stretch_width",
)

sidebar = pn.Column(
    pn.pane.Markdown("### Load a file"),
    dir_input,
    up_btn,
    subdir_select,
    file_select,
    load_btn,
    load_status,
    pn.layout.Divider(),
    bin_select,
    bin_note,
    pn.layout.Divider(),
    pn.pane.Markdown("### Session"),
    session_dir,
    session_name,
    save_session_btn,
    session_select,
    load_session_btn,
    session_status,
    session_note,
    pn.layout.Divider(),
    pn.pane.Markdown(f"*PyMIMS v{PYMIMS_VERSION} — NanoSIMS .im toolkit*"),
)

template = pn.template.FastListTemplate(
    title="PyMIMS",
    theme="dark",
    header_background="#121212",
    sidebar=[sidebar],
    main=[tabs],
    sidebar_width=340,
)

template.servable()


if __name__ == "__main__":
    pn.serve(template, show=True, port=5006)
