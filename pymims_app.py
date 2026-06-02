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
import io
import contextlib
import datetime

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

# Analysis-tab backends. Canonical layout is the package submodule form
# (matching each module's own `from .histograms import ...` relative import);
# the flat fallback covers a non-package checkout.
try:
    from pymims.clustering import (
        cluster_pixels, plot_cluster_labels, plot_cluster_grid,
        plot_overlay, plot_metric_sweep, plot_dendrogram,
        extract_cluster_masks,
    )
    from pymims.histograms import plot_histograms
except ImportError:                                   # flat-layout fallback
    from pymims_clustering import (
        cluster_pixels, plot_cluster_labels, plot_cluster_grid,
        plot_overlay, plot_metric_sweep, plot_dendrogram,
        extract_cluster_masks,
    )
    from pymims_histograms import plot_histograms

try:
    from pymims import __version__ as PYMIMS_VERSION
except Exception:
    PYMIMS_VERSION = "0.3.0"

pn.extension("plotly", sizing_mode="stretch_width", notifications=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# Journal: collect rendered views in memory, export to Word / PowerPoint
# ─────────────────────────────────────────────────────────────────────────────

JOURNAL = []   # list of entries, each {'caption': str, 'added': str} plus EITHER
               #   'png' : bytes   (a figure entry), OR
               #   'text': str     (a text/summary entry)


def _journal_add_text(text, caption):
    """Append a text/summary entry to the journal (no figure)."""
    JOURNAL.append({
        "text": text,
        "caption": caption,
        "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _update_journal_status()

journal_status = pn.pane.Markdown("Journal is empty.")
journal_dir = pn.widgets.TextInput(name="Export folder",
                                   value=os.path.expanduser("~"))
journal_name = pn.widgets.TextInput(name="Export name", value="pymims_journal")
journal_fmt = pn.widgets.Select(
    name="Format", value="PowerPoint (.pptx)",
    options=["PowerPoint (.pptx)", "Word (.docx)", "Both"])
export_journal_btn = pn.widgets.Button(name="📄 Export journal",
                                       button_type="primary")
remove_last_btn = pn.widgets.Button(name="Remove last", button_type="default")
clear_journal_btn = pn.widgets.Button(name="Clear journal", button_type="default")
export_status = pn.pane.Markdown("")
journal_note = pn.pane.Markdown(
    "<span style='font-size:11px;opacity:0.75'>Use the “➕ Add … to journal” "
    "buttons on the Imaging and Analysis tabs to collect figures (with auto "
    "captions) and text summaries, then export them here. PowerPoint lays "
    "figures out 6 per slide (2×3) and gives each text summary its own slide; "
    "Word stacks one item per block. Each export is stamped with the "
    "date.</span>")


def _update_journal_status():
    if not JOURNAL:
        journal_status.object = "Journal is empty."
        return
    lines = [f"**Journal — {len(JOURNAL)} item(s):**"]
    for i, e in enumerate(JOURNAL, 1):
        cap = e['caption']
        cap = cap if len(cap) <= 64 else cap[:61] + "…"
        tag = "📝 " if "text" in e else ""
        lines.append(f"{i}. {tag}{cap}")
    journal_status.object = "  \n".join(lines)


def _build_docx(path):
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    doc.add_heading("PyMIMS Journal", level=0)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sub = doc.add_paragraph(f"Generated {stamp}  ·  {len(JOURNAL)} items")
    if sub.runs:
        sub.runs[0].italic = True

    for i, e in enumerate(JOURNAL, 1):
        if "text" in e:
            head = doc.add_paragraph(f"Note {i}.  {e['caption']}")
            for run in head.runs:
                run.bold = True
                run.font.size = Pt(10)
            body = doc.add_paragraph(e["text"])
            for run in body.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
        else:
            doc.add_picture(io.BytesIO(e['png']), width=Inches(6.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(f"Figure {i}.  {e['caption']}")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cap.runs:
                run.italic = True
                run.font.size = Pt(9)
        doc.add_paragraph("")     # spacer between items
    doc.save(path)
    return path


def _build_pptx(path):
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.333)      # 16:9
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    SW, SH = int(prs.slide_width), int(prs.slide_height)
    margin = int(Inches(0.3))
    title_h = int(Inches(0.55))
    gap = int(Inches(0.12))
    cap_h = int(Inches(0.38))
    cols, rows = 3, 2
    per = cols * rows

    grid_w = SW - 2 * margin
    grid_h = SH - title_h - 2 * margin
    cell_w = (grid_w - (cols - 1) * gap) // cols
    cell_h = (grid_h - (rows - 1) * gap) // rows
    img_h = cell_h - cap_h

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build an ordered page list: consecutive figure entries are batched into
    # 6-up grid pages; each text entry becomes its own full-width text page.
    # This preserves the order in which items were added to the journal.
    pages = []   # ('figs', [(idx, entry), ...]) | ('text', (idx, entry))
    fig_batch = []
    for idx, e in enumerate(JOURNAL):
        if "text" in e:
            if fig_batch:
                pages.append(("figs", fig_batch)); fig_batch = []
            pages.append(("text", (idx, e)))
        else:
            fig_batch.append((idx, e))
            if len(fig_batch) == per:
                pages.append(("figs", fig_batch)); fig_batch = []
    if fig_batch:
        pages.append(("figs", fig_batch))

    n_pages = max(len(pages), 1)

    def _add_title(slide, page_no):
        tb = slide.shapes.add_textbox(margin, int(Inches(0.08)), grid_w, title_h)
        tf = tb.text_frame
        tf.text = f"PyMIMS Journal — {stamp}   (slide {page_no} of {n_pages})"
        tf.paragraphs[0].font.size = Pt(14)
        tf.paragraphs[0].font.bold = True

    for page_no, (kind, payload) in enumerate(pages, 1):
        slide = prs.slides.add_slide(blank)
        _add_title(slide, page_no)

        if kind == "text":
            idx, e = payload
            box = slide.shapes.add_textbox(
                margin, title_h + margin, grid_w, grid_h)
            tf = box.text_frame
            tf.word_wrap = True
            head = tf.paragraphs[0]
            head.text = f"{idx + 1}. {e['caption']}"
            head.font.size = Pt(13)
            head.font.bold = True
            body = tf.add_paragraph()
            body.text = e["text"]
            # Monospace so threshold tables and centroid columns line up.
            for run in body.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(10)
            continue

        # Figure grid page
        for k, (idx, e) in enumerate(payload):
            r, c = divmod(k, cols)
            left = margin + c * (cell_w + gap)
            top = title_h + margin + r * (cell_h + gap)

            pic = slide.shapes.add_picture(io.BytesIO(e['png']), left, top,
                                           width=cell_w)
            if pic.height > img_h:
                pic.width = int(pic.width * img_h / pic.height)
                pic.height = int(img_h)
            pic.left = int(left + (cell_w - pic.width) / 2)   # centre in cell

            cb = slide.shapes.add_textbox(left, top + img_h, cell_w, cap_h)
            ctf = cb.text_frame
            ctf.word_wrap = True
            ctf.text = f"{idx + 1}. {e['caption']}"
            p = ctf.paragraphs[0]
            p.font.size = Pt(7)
            p.alignment = PP_ALIGN.CENTER

    prs.save(path)
    return path


def _export_journal(event):
    if not JOURNAL:
        export_status.object = "⚠️ Journal is empty — add a view first."
        return
    folder = os.path.expanduser(journal_dir.value or "~")
    base = (journal_name.value or "pymims_journal").strip()
    want_docx = journal_fmt.value in ("Word (.docx)", "Both")
    want_pptx = journal_fmt.value in ("PowerPoint (.pptx)", "Both")
    try:
        os.makedirs(folder, exist_ok=True)
        written = []
        if want_pptx:
            written.append(os.path.basename(
                _build_pptx(os.path.join(folder, base + ".pptx"))))
        if want_docx:
            written.append(os.path.basename(
                _build_docx(os.path.join(folder, base + ".docx"))))
    except ImportError as exc:
        export_status.object = (f"❌ Missing library: `{exc}`. Install with "
                                "`pip install --break-system-packages "
                                "python-docx python-pptx`.")
        return
    except Exception as exc:
        export_status.object = f"❌ Export failed: `{exc}`"
        return
    export_status.object = (f"✅ Exported {', '.join(written)} "
                            f"({len(JOURNAL)} items) to `{folder}`")


def _remove_last(event):
    if JOURNAL:
        JOURNAL.pop()
        _update_journal_status()
        export_status.object = f"Removed last — {len(JOURNAL)} remain."
    else:
        export_status.object = "Journal already empty."


def _clear_journal(event):
    JOURNAL.clear()
    _update_journal_status()
    export_status.object = "Journal cleared."


export_journal_btn.on_click(_export_journal)
remove_last_btn.on_click(_remove_last)
clear_journal_btn.on_click(_clear_journal)


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
    qc_add_journal_btn = pn.widgets.Button(name="➕ Add scan to journal",
                                           button_type="default", width=190)
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

    def _add_scan_to_journal(event):
        cur_n = int(img.metadata["n_planes"])
        if cur_n < 2:
            qc_status.object = "Need ≥2 planes to scan."
            return
        try:
            res = img.plot_plane_diagnostics(
                channel=_qc_channel_arg(),
                threshold_pct=float(qc_thresh.value),
                show=False)
        except Exception as exc:
            qc_status.object = f"❌ Couldn't render scan: `{exc}`"
            return
        fig = res["figure"]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        sus = res["suspect_indices"]
        chan = qc_channel.value
        cap = (f"Plane QC scan — {chan}, ±{float(qc_thresh.value):.0f}% threshold, "
               f"{len(sus)} of {cur_n} flagged  ·  {os.path.basename(img.path)}")
        JOURNAL.append({
            "png": buf.getvalue(),
            "caption": cap,
            "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        _update_journal_status()
        qc_status.object = f"✅ Scan added to journal — {len(JOURNAL)} view(s)."
    qc_add_journal_btn.on_click(_add_scan_to_journal)

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
        pn.Row(qc_channel, qc_thresh, qc_scan_btn, qc_add_journal_btn),
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

    # Add the current view to the journal (captured as a PNG with an auto caption)
    add_journal_btn = pn.widgets.Button(name="➕ Add current view to journal",
                                        button_type="default", width=240)
    journal_local = pn.pane.Markdown("")

    def _current_caption():
        src = "drift-corrected" if img.corrected is not None else "raw"
        fname = os.path.basename(img.path)
        binlbl = (f", {state.bin_n}×{state.bin_n} bin" if state.bin_n > 1 else "")
        m = mode.value
        if m == "Single channel":
            plane_lbl = "summed" if sum_toggle.value else f"plane {plane.value}"
            flt = ("" if filt.value == "None"
                   else f", {filt.value.lower()} {strength.value:g}")
            return (f"{channel.value} — {plane_lbl}, {src}{binlbl}{flt}"
                    f"  ·  {fname}")
        if m == "Ratio":
            return f"{num.value} / {den.value} ratio, {src}{binlbl}  ·  {fname}"
        return (f"{hnum.value} / {hden.value} HSI "
                f"(intensity={intensity.value}), {src}{binlbl}  ·  {fname}")

    def _render_current_png():
        fig = _build_fig(img, mode.value, channel.value, cmap.value, upper.value,
                         sum_toggle.value, plane.value, filt.value, strength.value,
                         state.bin_n, num.value, den.value, min_counts.value,
                         max_rel.value, dref.value, hnum.value, hden.value,
                         intensity.value, hcmap.value, scale.value, rmin.value,
                         rmax.value, hmed.value)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def _add_to_journal(event):
        try:
            png = _render_current_png()
        except Exception as exc:
            journal_local.object = f"❌ Couldn't render view: `{exc}`"
            return
        JOURNAL.append({
            "png": png,
            "caption": _current_caption(),
            "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        _update_journal_status()
        journal_local.object = f"✅ Added — {len(JOURNAL)} view(s) in journal."
    add_journal_btn.on_click(_add_to_journal)

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
        pn.Row(add_journal_btn, journal_local),
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


def _cluster_summary_text(result, k):
    """Compact text summary of a ClusterResult at a chosen k, suitable for
    the journal. Mirrors the on-screen cluster table in plain text."""
    import numpy as _np
    lines = []
    lines.append(f"Clustering — {result['method']}, "
                 f"feature space: {result['feature_space']}")
    lines.append(f"channels: {', '.join(result['feature_labels'])}")
    lines.append(f"sensible k = {result['sensible_k']}   (showing k = {k})")
    if result['method'] == 'hierarchical' and 'cophenetic_corr' in result:
        lines.append(f"cophenetic correlation = {result['cophenetic_corr']:.3f}")
    recs = ", ".join(
        f"k={r['k']} [{', '.join(m.replace('_', ' ') for m in r['methods'])}]"
        for r in result['unique_k_recommendations'])
    lines.append(f"recommendations: {recs}")
    lines.append("")

    sizes = result['cluster_sizes_by_k'][k]
    cents = result['centroids_by_k_counts'][k]
    feats = result['feature_labels']
    total = int(sum(sizes))
    header = f"{'#':>2}  {'pixels':>9}  {'%':>6}  " + \
             "  ".join(f"{f:>12}" for f in feats)
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(k):
        size = int(sizes[i]) if i < len(sizes) else 0
        pct = 100.0 * size / total if total else 0.0
        row_c = cents[i] if i < cents.shape[0] else _np.zeros(len(feats))
        cols = "  ".join(f"{v:>12.1f}" for v in row_c)
        lines.append(f"{i + 1:>2}  {size:>9,}  {pct:>5.1f}%  {cols}")
    return "\n".join(lines)


def _enlarge_for_display(fig, per_w=3.1, per_h=3.1, cap_w=34, cap_h=44):
    """Scale a matplotlib figure up so multi-panel grids render legibly.

    Infers the subplot grid from the first axes' gridspec geometry and sizes the
    figure at roughly `per_w`×`per_h` inches per panel, capped. Returns the
    figure's pixel dimensions at `dpi` so the caller can size the display pane.
    """
    try:
        nrows, ncols = fig.axes[0].get_gridspec().get_geometry()
    except Exception:
        nrows, ncols = 1, max(1, len(fig.axes))
    w_in = min(cap_w, max(9.0, ncols * per_w))
    h_in = min(cap_h, max(3.2, nrows * per_h))
    fig.set_size_inches(w_in, h_in)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return w_in, h_in


def analysis_view(img):
    if img is None:
        return pn.pane.Markdown(
            "### Analysis\nLoad a `.im` file to run clustering and distributions.")

    masses = img.masses
    non_se = [m for m in masses if "SE" not in m.upper()] or masses

    drift_hint = ("" if img.corrected is not None else
                  "<span style='color:#e0a030'>Tip: run drift correction on "
                  "the Imaging tab first — clustering and GMM fits use the "
                  "drift-corrected stack.</span>")

    def _fig_to_png(fig, dpi=150):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

    # ── Clustering subsection ────────────────────────────────────────────────
    cl_method = pn.widgets.RadioButtonGroup(
        name="Method", options=["kmeans", "hierarchical"],
        value="kmeans", button_type="primary")
    cl_kmax = pn.widgets.IntInput(name="k max", value=10, start=2, end=20, width=90)
    cl_feature = pn.widgets.Select(
        name="Feature space",
        options=["log_zscored", "log_robustz", "log", "raw"],
        value="log_zscored", width=150)
    cl_channels = pn.widgets.MultiSelect(
        name="Channels (none = auto, SE excluded)",
        options=list(masses), value=list(non_se), size=4, width=220)
    cl_include_se = pn.widgets.Checkbox(name="Include SE (auto mode only)",
                                        value=False)
    cl_mincounts = pn.widgets.IntInput(name="Min counts (0 = off)",
                                        value=0, start=0, width=150)
    cl_maskchan = pn.widgets.Select(name="Mask channel",
                                    options=["auto"] + list(masses),
                                    value="auto", width=150)
    cl_subsample = pn.widgets.IntInput(name="Subsample (hier.)",
                                       value=5000, start=200, width=140)
    cl_linkage = pn.widgets.Select(
        name="Linkage (hier.)",
        options=["ward", "complete", "average", "single", "centroid"],
        value="ward", width=140)

    cl_run_btn = pn.widgets.Button(name="Run clustering", button_type="primary",
                                   icon="player-play", width=150)
    cl_kpick = pn.widgets.Select(name="Show k", options=[], width=90,
                                 disabled=True)
    cl_display = pn.widgets.Select(
        name="Display",
        options=["Cluster labels", "Cluster grid", "Overlay on channel",
                 "Metric sweep", "Dendrogram (hier.)"],
        value="Cluster labels", width=180)
    cl_overlay_chan = pn.widgets.Select(name="Overlay channel",
                                        options=list(masses),
                                        value=non_se[0], width=150)
    cl_add_fig_btn = pn.widgets.Button(name="➕ Add figure to journal",
                                       button_type="default", width=200)
    cl_add_txt_btn = pn.widgets.Button(name="➕ Add summary to journal",
                                       button_type="default", width=210)
    cl_status = pn.pane.Markdown("")
    cl_plot_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No clustering yet — set options and click "
        "**Run clustering**.</span>"))

    _cl = {"result": None, "png": None}

    def _cl_render():
        result = _cl["result"]
        if result is None:
            return
        try:
            k = int(cl_kpick.value)
        except (TypeError, ValueError):
            k = result["sensible_k"]
        disp = cl_display.value
        try:
            if disp == "Cluster labels":
                fig = plot_cluster_labels(img, result, k=k, show=False)
            elif disp == "Cluster grid":
                fig = plot_cluster_grid(img, result, show=False)
            elif disp == "Overlay on channel":
                fig = plot_overlay(img, result, k=k, base="channel",
                                   channel=cl_overlay_chan.value, show=False)
            elif disp == "Metric sweep":
                fig = plot_metric_sweep(result, show=False)
            else:   # Dendrogram
                if result["method"] != "hierarchical":
                    cl_plot_box.objects = [pn.pane.Alert(
                        "Dendrogram is only available for hierarchical "
                        "clustering.", alert_type="warning")]
                    return
                fig = plot_dendrogram(result, k_marks=[k], show=False)
        except Exception as exc:
            cl_plot_box.objects = [pn.pane.Alert(
                f"Render failed: `{exc}`", alert_type="danger")]
            return
        _cl["png"] = _fig_to_png(fig)
        cl_plot_box.objects = [pn.pane.PNG(_cl["png"], width=720)]

    def _cl_run(event):
        ch_list = list(cl_channels.value)
        channels_arg = ch_list if ch_list else None
        try:
            result = cluster_pixels(
                img, method=cl_method.value, k_max=int(cl_kmax.value),
                channels=channels_arg, include_se=bool(cl_include_se.value),
                feature_space=cl_feature.value,
                min_counts=(int(cl_mincounts.value) or None),
                mask_channel=(None if cl_maskchan.value == "auto"
                              else cl_maskchan.value),
                subsample_size=int(cl_subsample.value),
                linkage_method=cl_linkage.value, verbose=False)
        except Exception as exc:
            cl_status.object = f"❌ Clustering failed: `{exc}`"
            return
        _cl["result"] = result
        ks = sorted(int(x) for x in result["labels_by_k"])
        cl_kpick.options = [str(x) for x in ks]
        cl_kpick.value = str(result["sensible_k"])
        cl_kpick.disabled = False
        recs = ", ".join(
            f"k={r['k']} ({', '.join(m.replace('_', ' ') for m in r['methods'])})"
            for r in result["unique_k_recommendations"])
        cl_status.object = (f"✅ Done — sensible k = **{result['sensible_k']}**. "
                            f"Recommendations: {recs}.")
        _cl_render()
    cl_run_btn.on_click(_cl_run)

    cl_kpick.param.watch(lambda e: _cl_render(), "value")
    cl_display.param.watch(lambda e: _cl_render(), "value")
    cl_overlay_chan.param.watch(lambda e: _cl_render(), "value")

    def _cl_add_fig(event):
        if _cl["png"] is None:
            cl_status.object = "Nothing to add yet — run clustering first."
            return
        k = cl_kpick.value
        res = _cl["result"]
        cap = (f"{res['method']} clustering, {cl_display.value.lower()}, k={k}, "
               f"{res['feature_space']}  ·  {os.path.basename(img.path)}")
        JOURNAL.append({"png": _cl["png"], "caption": cap,
                        "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
        _update_journal_status()
        cl_status.object = f"✅ Figure added — {len(JOURNAL)} item(s) in journal."
    cl_add_fig_btn.on_click(_cl_add_fig)

    def _cl_add_txt(event):
        res = _cl["result"]
        if res is None:
            cl_status.object = "Nothing to add yet — run clustering first."
            return
        try:
            k = int(cl_kpick.value)
        except (TypeError, ValueError):
            k = res["sensible_k"]
        text = _cluster_summary_text(res, k)
        cap = (f"{res['method']} cluster summary, k={k}  ·  "
               f"{os.path.basename(img.path)}")
        _journal_add_text(text, cap)
        cl_status.object = f"✅ Summary added — {len(JOURNAL)} item(s) in journal."
    cl_add_txt_btn.on_click(_cl_add_txt)

    clustering_section = pn.Column(
        pn.pane.Markdown("**Clustering**  <span style='font-size:11px;"
                         "opacity:0.7'>(k-means / hierarchical, metric sweep)</span>"),
        pn.Row(cl_method, cl_kmax, cl_feature),
        pn.Row(cl_channels, pn.Column(cl_include_se, cl_mincounts, cl_maskchan)),
        pn.Row(cl_subsample, cl_linkage),
        pn.Row(cl_run_btn, cl_kpick, cl_display, cl_overlay_chan),
        pn.Row(cl_add_fig_btn, cl_add_txt_btn),
        cl_status,
        cl_plot_box,
    )

    # ── Distributions (per-channel GMM histograms) subsection ────────────────
    h_channel = pn.widgets.Select(name="Channel",
                                  options=["all channels"] + list(masses),
                                  value=non_se[0], width=170)
    h_kmax = pn.widgets.IntInput(name="k max", value=6, start=1, end=10, width=90)
    h_nbins = pn.widgets.IntInput(name="Bins", value=80, start=20, end=300, width=90)
    h_dropzeros = pn.widgets.Checkbox(name="Drop zero-count pixels", value=True)
    h_jitter = pn.widgets.Checkbox(name="Jitter integer counts", value=True)
    h_tailw = pn.widgets.FloatInput(name="Tail-warning weight",
                                    value=0.15, start=0.0, step=0.05, width=160)
    h_run_btn = pn.widgets.Button(name="Run distributions",
                                  button_type="primary", icon="chart-histogram",
                                  width=170)
    h_add_fig_btn = pn.widgets.Button(name="➕ Add figure to journal",
                                      button_type="default", width=200)
    h_add_txt_btn = pn.widgets.Button(name="➕ Add summary to journal",
                                      button_type="default", width=210)
    h_status = pn.pane.Markdown("")
    h_plot_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No fit yet — set options and click "
        "**Run distributions**.</span>"))

    _h = {"png": None, "text": None}

    def _h_run(event):
        ch_arg = None if h_channel.value == "all channels" else h_channel.value
        plt.close("all")   # clean slate so plt.gcf() is unambiguous below
        buf_txt = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_txt):
                plot_histograms(
                    img, channel=ch_arg, k_max=int(h_kmax.value),
                    n_bins=int(h_nbins.value),
                    drop_zeros=bool(h_dropzeros.value),
                    jitter=bool(h_jitter.value),
                    tail_weight_threshold=float(h_tailw.value),
                    show=True, verbose=True)
        except Exception as exc:
            h_status.object = f"❌ Distribution fit failed: `{exc}`"
            return
        fig = plt.gcf()
        w_in, h_in = _enlarge_for_display(fig)        # bigger panels
        dpi = 150
        px_w, px_h = int(w_in * dpi), int(h_in * dpi)
        _h["png"] = _fig_to_png(fig, dpi=dpi)         # also closes the figure
        _h["text"] = buf_txt.getvalue().rstrip()
        # Show at natural (large) size inside a scroll box — horizontal scroll
        # for the wide single-channel row, vertical for the tall all-channels
        # stack. Capped height keeps the page navigable.
        h_plot_box.height = min(px_h + 24, 720)
        h_plot_box.scroll = True
        h_plot_box.objects = [pn.pane.PNG(_h["png"], width=px_w)]
        lbl = "all channels" if ch_arg is None else ch_arg
        h_status.object = (f"✅ GMM fit complete for {lbl}. "
                           f"<span style='opacity:0.65;font-size:11px'>"
                           f"(scroll the panel; fewer **k max** = larger panels)</span>")
    h_run_btn.on_click(_h_run)

    def _h_add_fig(event):
        if _h["png"] is None:
            h_status.object = "Nothing to add yet — run a fit first."
            return
        lbl = "all channels" if h_channel.value == "all channels" else h_channel.value
        cap = (f"GMM histograms — {lbl}, k≤{int(h_kmax.value)}  ·  "
               f"{os.path.basename(img.path)}")
        JOURNAL.append({"png": _h["png"], "caption": cap,
                        "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
        _update_journal_status()
        h_status.object = f"✅ Figure added — {len(JOURNAL)} item(s) in journal."
    h_add_fig_btn.on_click(_h_add_fig)

    def _h_add_txt(event):
        if not _h["text"]:
            h_status.object = "No summary captured — run a fit first."
            return
        lbl = "all channels" if h_channel.value == "all channels" else h_channel.value
        cap = f"GMM threshold summary — {lbl}  ·  {os.path.basename(img.path)}"
        _journal_add_text(_h["text"], cap)
        h_status.object = f"✅ Summary added — {len(JOURNAL)} item(s) in journal."
    h_add_txt_btn.on_click(_h_add_txt)

    distributions_section = pn.Column(
        pn.pane.Markdown("**Distributions**  <span style='font-size:11px;"
                         "opacity:0.7'>(per-channel GMM histograms + BIC/AIC + "
                         "crossing thresholds)</span>"),
        pn.Row(h_channel, h_kmax, h_nbins),
        pn.Row(h_dropzeros, h_jitter, h_tailw),
        pn.Row(h_run_btn, h_add_fig_btn, h_add_txt_btn),
        h_status,
        h_plot_box,
    )

    # ── ROI Manager subsection ───────────────────────────────────────────────
    # Two ways to make ROIs: (1) from a clustering run's per-cluster masks, or
    # (2) by edge/threshold detection on a chosen channel. ROIs then get
    # per-channel pooled-count statistics (and optional pooled isotope ratios).
    _roi = {"rois": {}, "pending": {}, "stats_png": None, "stats_text": None}

    def _summed_channel(channel, corrected=True):
        """Summed-over-planes (H, W) counts for one channel.

        corrected=True follows get_channel (the drift-corrected float stack if
        drift has been run, else the raw stack). corrected=False forces the raw
        integer-count stack regardless of drift state — use for Poisson error
        propagation, where you need true detected-ion counts."""
        if corrected:
            stack = np.asarray(img.get_channel(channel))
            return (stack.sum(axis=0) if stack.ndim == 3 else stack).astype(float)
        ch = list(img.masses).index(channel)
        return img.data[:, ch, :, :].sum(axis=0).astype(float)

    def _masks_from_clusters(result, k):
        """{cluster_id (1-based int): (H, W) bool mask} from the library helper.
        IDs and pixel counts match the clustering summary table; pixels excluded
        by min_counts / mask_channel are NaN in labels_by_k and fall out of
        every mask."""
        return extract_cluster_masks(result, k=k)

    def _refresh_roi_list():
        rois = _roi["rois"]
        # Attach the current masks to the image object so the stack carries its
        # ROI set (used by depth profiles here, and available to any downstream
        # consumer holding the same MimsImage). Not written to .npz sessions —
        # that would need a core save_session change.
        img.rois = {name: info["mask"] for name, info in rois.items()}
        if not rois:
            roi_list.object = "<span style='opacity:0.6'>No ROIs yet.</span>"
            return
        lines = [f"**ROIs — {len(rois)}:**"]
        for name, info in rois.items():
            lines.append(f"- `{name}` · {int(info['mask'].sum()):,} px "
                         f"<span style='opacity:0.6'>({info['source']})</span>")
        roi_list.object = "  \n".join(lines)

    # -- (1) From clusters ----------------------------------------------------
    roi_load_btn = pn.widgets.Button(name="Load clusters from last run",
                                     button_type="default", width=220)
    roi_cluster_pick = pn.widgets.MultiSelect(name="Clusters to add",
                                              options=[], size=5, width=240)
    roi_add_clusters_btn = pn.widgets.Button(name="➕ Add selected as ROIs",
                                             button_type="primary", width=200)

    def _roi_load_clusters(event):
        result = _cl["result"]
        if result is None:
            roi_status.object = "⚠️ Run clustering first (Clustering section above)."
            return
        try:
            k = int(cl_kpick.value)
        except (TypeError, ValueError):
            k = result["sensible_k"]
        try:
            masks = _masks_from_clusters(result, k)
        except Exception as exc:
            roi_status.object = f"❌ Couldn't build cluster masks: `{exc}`"
            return
        _roi["pending"] = {f"k{k}_c{cid}": m for cid, m in sorted(masks.items())}
        roi_cluster_pick.options = [
            f"{name}  ({int(m.sum()):,} px)"
            for name, m in _roi["pending"].items()]
        roi_cluster_pick.value = list(roi_cluster_pick.options)
        roi_status.object = (f"Loaded {len(masks)} cluster(s) at k={k}. "
                             f"Pick which to add, then **Add selected as ROIs**.")
    roi_load_btn.on_click(_roi_load_clusters)

    def _roi_add_clusters(event):
        if not _roi["pending"]:
            roi_status.object = "Nothing loaded — click **Load clusters** first."
            return
        label_to_name = {f"{name}  ({int(m.sum()):,} px)": name
                         for name, m in _roi["pending"].items()}
        added = 0
        for lbl in roi_cluster_pick.value:
            name = label_to_name.get(lbl)
            if name:
                _roi["rois"][name] = {"mask": _roi["pending"][name],
                                      "source": "cluster"}
                added += 1
        _refresh_roi_list()
        roi_status.object = f"✅ Added {added} ROI(s) — {len(_roi['rois'])} total."
    roi_add_clusters_btn.on_click(_roi_add_clusters)

    # -- (2) Edge / threshold detection --------------------------------------
    roi_edge_chan = pn.widgets.Select(name="Detect on channel",
                                      options=list(masses), value=non_se[0],
                                      width=160)
    roi_edge_method = pn.widgets.Select(
        name="Method",
        options=["Threshold (Otsu)", "Manual percentile", "Edge (Canny) + fill"],
        value="Threshold (Otsu)", width=190)
    roi_edge_sigma = pn.widgets.FloatInput(name="Smoothing σ", value=1.5,
                                           start=0.0, step=0.5, width=120)
    roi_edge_pct = pn.widgets.FloatInput(name="Percentile (manual)", value=90.0,
                                         start=0.0, end=100.0, step=1.0, width=160)
    roi_edge_minarea = pn.widgets.IntInput(name="Min area (px)", value=25,
                                           start=1, width=130)
    roi_edge_merge = pn.widgets.Checkbox(name="Merge into one ROI", value=False)
    roi_detect_btn = pn.widgets.Button(name="Detect regions",
                                       button_type="primary", icon="vector",
                                       width=160)
    roi_overlay_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No detection yet.</span>"))

    def _detect_regions():
        from scipy import ndimage as ndi
        data = _summed_channel(roi_edge_chan.value)
        norm = data / (data.max() or 1.0)
        sigma = float(roi_edge_sigma.value)
        method = roi_edge_method.value
        if method == "Edge (Canny) + fill":
            from skimage.feature import canny
            binary = ndi.binary_fill_holes(canny(norm, sigma=sigma))
        else:
            from skimage.filters import gaussian
            sm = gaussian(norm, sigma=sigma) if sigma > 0 else norm
            if method == "Threshold (Otsu)":
                from skimage.filters import threshold_otsu
                try:
                    thr = threshold_otsu(sm)
                except Exception:
                    thr = float(sm.mean())
            else:                                   # Manual percentile
                thr = float(np.percentile(sm, float(roi_edge_pct.value)))
            binary = ndi.binary_fill_holes(sm > thr)
        if bool(roi_edge_merge.value):
            return ({"edge_all": binary} if binary.any() else {}), data
        lab, n = ndi.label(binary)
        regions = [(lab == i) for i in range(1, n + 1)]
        regions = [m for m in regions if m.sum() >= int(roi_edge_minarea.value)]
        regions.sort(key=lambda m: -m.sum())
        return {f"r{j}": m for j, m in enumerate(regions[:50], 1)}, data

    def _roi_overlay_fig(base_channel, extra_masks=None):
        data = _summed_channel(base_channel)
        vmax = float(np.percentile(data, 99.5)) or 1.0
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        ax.imshow(data, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        cmap = plt.get_cmap("tab10")
        items = list((extra_masks or _roi["rois"]).items())
        for i, (name, info) in enumerate(items):
            m = info["mask"] if isinstance(info, dict) else info
            ax.contour(m.astype(float), levels=[0.5],
                       colors=[cmap(i % 10)], linewidths=1.2)
        ax.set_title(f"{len(items)} region(s) on {base_channel}  (pixels)",
                     fontsize=10)
        ax.set_xlabel("px"); ax.set_ylabel("px")
        fig.tight_layout()
        return fig

    def _roi_detect(event):
        try:
            found, _ = _detect_regions()
        except Exception as exc:
            roi_status.object = f"❌ Detection failed: `{exc}`"
            return
        if not found:
            roi_overlay_box.objects = [pn.pane.Markdown(
                "<span style='opacity:0.6'>Nothing detected — lower the "
                "threshold / min area.</span>")]
            roi_status.object = "No regions found."
            return
        # add directly to the ROI set, tagging the source
        for name, m in found.items():
            _roi["rois"][name] = {"mask": m, "source": "edge"}
        fig = _roi_overlay_fig(roi_edge_chan.value)
        png = _fig_to_png(fig)
        roi_overlay_box.objects = [pn.pane.PNG(png, width=560)]
        _refresh_roi_list()
        roi_status.object = (f"✅ Detected & added {len(found)} region(s) — "
                             f"{len(_roi['rois'])} ROI(s) total.")
    roi_detect_btn.on_click(_roi_detect)

    # -- ROI stats ------------------------------------------------------------
    roi_stat_chans = pn.widgets.MultiSelect(
        name="Stat channels", options=list(masses), value=list(non_se),
        size=4, width=200)
    roi_counts_src = pn.widgets.Select(
        name="Counts source",
        options=["Drift-corrected (if run)", "Raw counts (integer)"],
        value="Drift-corrected (if run)", width=210)
    roi_ratio_on = pn.widgets.Checkbox(name="Add pooled ratio", value=False)
    roi_ratio_num = pn.widgets.Select(name="Ratio numerator",
                                      options=list(masses), value=non_se[0],
                                      width=150)
    roi_ratio_den = pn.widgets.Select(
        name="Ratio denominator", options=list(masses),
        value=(non_se[1] if len(non_se) > 1 else non_se[0]), width=150)
    roi_compute_btn = pn.widgets.Button(name="Compute ROI stats",
                                        button_type="primary", icon="table",
                                        width=180)
    roi_remove_btn = pn.widgets.Button(name="Remove last ROI",
                                       button_type="default", width=150)
    roi_clear_btn = pn.widgets.Button(name="Clear ROIs",
                                      button_type="default", width=130)
    roi_add_fig_btn = pn.widgets.Button(name="➕ Add table to journal",
                                        button_type="default", width=200)
    roi_add_txt_btn = pn.widgets.Button(name="➕ Add summary to journal",
                                        button_type="default", width=210)
    roi_table_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No stats yet — add ROIs and click "
        "**Compute ROI stats**.</span>"))

    def _build_stats_rows(channels, ratio, corrected):
        summed = {ch: _summed_channel(ch, corrected) for ch in channels}
        if ratio:
            for ch in (roi_ratio_num.value, roi_ratio_den.value):
                summed.setdefault(ch, _summed_channel(ch, corrected))
        header = ["ROI", "px"]
        for ch in channels:
            header += [f"{ch} tot", f"{ch} mean"]
        if ratio:
            header += [f"{roi_ratio_num.value}/{roi_ratio_den.value}"]
        rows = []
        for name, info in _roi["rois"].items():
            m = info["mask"]
            npix = int(m.sum())
            row = [name, f"{npix:,}"]
            for ch in channels:
                vals = summed[ch][m] if npix else np.array([0.0])
                row += [f"{float(vals.sum()):,.0f}",
                        f"{float(vals.mean()) if npix else 0.0:,.1f}"]
            if ratio:
                num = float(summed[roi_ratio_num.value][m].sum()) if npix else 0.0
                den = float(summed[roi_ratio_den.value][m].sum()) if npix else 0.0
                row += [f"{(num / den):.4f}" if den else "—"]
            rows.append(row)
        return header, rows

    def _stats_text(header, rows, source_label):
        widths = [max(len(header[c]), *(len(r[c]) for r in rows)) if rows
                  else len(header[c]) for c in range(len(header))]
        def fmt(vals):
            return "  ".join(v.rjust(widths[c]) for c, v in enumerate(vals))
        lines = [f"ROI statistics  ·  {os.path.basename(img.path)}",
                 f"counts source: {source_label}",
                 "(totals are pooled counts over the summed stack; "
                 "ratio is pooled Σnum/Σden)", "",
                 fmt(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
        lines += [fmt(r) for r in rows]
        return "\n".join(lines)

    def _stats_fig(header, rows, source_label):
        ncol, nrow = len(header), len(rows) + 1
        w_in = min(2.4 + 1.6 * ncol, 34)
        h_in = max(1.8, 0.7 + 0.52 * nrow)
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        ax.axis("off")
        tbl = ax.table(cellText=rows or [["—"] * ncol], colLabels=header,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold")
            if c == 0:
                cell.set_text_props(ha="left")
        ax.set_title(f"ROI statistics ({source_label}) — "
                     f"{os.path.basename(img.path)}", fontsize=12, pad=14)
        fig.tight_layout()
        return fig

    def _roi_compute(event):
        if not _roi["rois"]:
            roi_status.object = "⚠️ No ROIs — add some from clusters or detection."
            return
        channels = list(roi_stat_chans.value) or [non_se[0]]
        ratio = bool(roi_ratio_on.value)
        corrected = roi_counts_src.value.startswith("Drift")
        source_label = ("drift-corrected" if (corrected and img.corrected is not None)
                        else "raw integer counts" if not corrected
                        else "raw (no drift applied)")
        try:
            header, rows = _build_stats_rows(channels, ratio, corrected)
        except Exception as exc:
            roi_status.object = f"❌ Stats failed: `{exc}`"
            return
        _roi["stats_text"] = _stats_text(header, rows, source_label)
        fig = _stats_fig(header, rows, source_label)
        w_in, h_in = fig.get_size_inches()
        dpi = 150
        px_w, px_h = int(w_in * dpi), int(h_in * dpi)
        _roi["stats_png"] = _fig_to_png(fig, dpi=dpi)   # closes the figure
        objs = []
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=header)
            objs.append(pn.pane.DataFrame(df, index=False, width=920))
        except Exception:
            objs.append(pn.pane.Str(_roi["stats_text"], styles={
                "font-family": "ui-monospace, Menlo, monospace",
                "white-space": "pre", "font-size": "12px"}))
        # Journal-preview render at natural (large) size in a scroll box —
        # horizontal scroll for wide many-channel tables.
        objs.append(pn.pane.Markdown(
            "<span style='font-size:11px;opacity:0.6'>Journal preview "
            "(scroll if wide):</span>"))
        objs.append(pn.Column(pn.pane.PNG(_roi["stats_png"], width=px_w),
                              scroll=True, height=min(px_h + 24, 460),
                              sizing_mode="stretch_width"))
        roi_table_box.objects = objs
        roi_status.object = (f"✅ Stats for {len(_roi['rois'])} ROI(s) over "
                             f"{len(channels)} channel(s) · {source_label}.")
    roi_compute_btn.on_click(_roi_compute)

    def _roi_remove(event):
        if _roi["rois"]:
            last = list(_roi["rois"])[-1]
            _roi["rois"].pop(last)
            _refresh_roi_list()
            roi_status.object = f"Removed `{last}` — {len(_roi['rois'])} remain."
        else:
            roi_status.object = "No ROIs to remove."
    roi_remove_btn.on_click(_roi_remove)

    def _roi_clear(event):
        _roi["rois"].clear()
        _refresh_roi_list()
        roi_overlay_box.objects = [pn.pane.Markdown(
            "<span style='opacity:0.6'>No detection yet.</span>")]
        roi_table_box.objects = [pn.pane.Markdown(
            "<span style='opacity:0.6'>No stats yet.</span>")]
        roi_status.object = "Cleared all ROIs."
    roi_clear_btn.on_click(_roi_clear)

    def _roi_add_fig(event):
        if _roi["stats_png"] is None:
            roi_status.object = "Nothing to add — compute stats first."
            return
        cap = (f"ROI statistics, {len(_roi['rois'])} ROI(s)  ·  "
               f"{os.path.basename(img.path)}")
        JOURNAL.append({"png": _roi["stats_png"], "caption": cap,
                        "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
        _update_journal_status()
        roi_status.object = f"✅ Table added — {len(JOURNAL)} item(s) in journal."
    roi_add_fig_btn.on_click(_roi_add_fig)

    def _roi_add_txt(event):
        if not _roi["stats_text"]:
            roi_status.object = "Nothing to add — compute stats first."
            return
        cap = f"ROI statistics summary  ·  {os.path.basename(img.path)}"
        _journal_add_text(_roi["stats_text"], cap)
        roi_status.object = f"✅ Summary added — {len(JOURNAL)} item(s) in journal."
    roi_add_txt_btn.on_click(_roi_add_txt)

    # -- Depth profiles (signal vs plane / sputter depth) --------------------
    n_planes = int(img.metadata.get("n_planes", 1))
    dp_chan = pn.widgets.Select(name="Profile channel", options=list(masses),
                                value=non_se[0], width=160)
    dp_metric = pn.widgets.Select(name="Metric",
                                  options=["mean counts / px", "total counts"],
                                  value="mean counts / px", width=170)
    dp_ratio = pn.widgets.Checkbox(name="Ratio profile (uses ratio num/den above)",
                                   value=False)
    dp_btn = pn.widgets.Button(name="Plot depth profiles", button_type="primary",
                               icon="chart-line", width=180)
    dp_add_fig_btn = pn.widgets.Button(name="➕ Add profile to journal",
                                       button_type="default", width=210)
    dp_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>No profile yet — add ROIs and click "
        "**Plot depth profiles**.</span>"))
    _dp = {"png": None}

    def _dp_series(channel, mask, corrected):
        """Per-plane pooled counts within a mask → 1-D array (length n_planes)."""
        if corrected:
            stack = np.asarray(img.get_channel(channel))      # (planes, H, W)
        else:
            ch = list(img.masses).index(channel)
            stack = img.data[:, ch, :, :].astype(float)
        return stack[:, mask].sum(axis=1).astype(float)

    def _dp_plot():
        rois = _roi["rois"]
        if not rois:
            return None
        corrected = roi_counts_src.value.startswith("Drift")
        ratio = bool(dp_ratio.value)
        metric_mean = dp_metric.value.startswith("mean")
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(n_planes)
        for i, (name, info) in enumerate(rois.items()):
            m = info["mask"]
            npix = int(m.sum()) or 1
            if ratio:
                num = _dp_series(roi_ratio_num.value, m, corrected)
                den = _dp_series(roi_ratio_den.value, m, corrected)
                y = np.divide(num, den, out=np.zeros_like(num),
                              where=den > 0)
                ylabel = f"{roi_ratio_num.value} / {roi_ratio_den.value}  (pooled)"
            else:
                tot = _dp_series(dp_chan.value, m, corrected)
                y = (tot / npix) if metric_mean else tot
                ylabel = (f"{dp_chan.value}  "
                          + ("mean counts / px" if metric_mean else "total counts"))
            xx = x if len(y) == len(x) else np.arange(len(y))
            ax.plot(xx, y, marker="o", ms=3, lw=1.3, color=cmap(i % 10),
                    label=f"{name}  ({npix:,} px)")
        src = "drift-corrected" if corrected and img.corrected is not None else \
              ("raw" if not corrected else "raw (no drift)")
        ax.set_xlabel("plane index  (sputter depth →)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ROI depth profiles ({src}) — {os.path.basename(img.path)}",
                     fontsize=10)
        ax.legend(fontsize=8, ncol=2, framealpha=0.4)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    def _dp_run(event):
        if not _roi["rois"]:
            roi_status.object = "⚠️ No ROIs — add some before plotting profiles."
            return
        try:
            fig = _dp_plot()
        except Exception as exc:
            roi_status.object = f"❌ Depth profile failed: `{exc}`"
            return
        _dp["png"] = _fig_to_png(fig)
        dp_box.objects = [pn.pane.PNG(_dp["png"], width=760)]
        roi_status.object = (f"✅ Depth profiles for {len(_roi['rois'])} ROI(s) "
                             f"over {n_planes} planes.")
    dp_btn.on_click(_dp_run)

    def _dp_add_fig(event):
        if _dp["png"] is None:
            roi_status.object = "Nothing to add — plot a profile first."
            return
        what = (f"{roi_ratio_num.value}/{roi_ratio_den.value} ratio"
                if dp_ratio.value else dp_chan.value)
        cap = (f"ROI depth profiles — {what}, {len(_roi['rois'])} ROI(s)  ·  "
               f"{os.path.basename(img.path)}")
        JOURNAL.append({"png": _dp["png"], "caption": cap,
                        "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
        _update_journal_status()
        roi_status.object = f"✅ Profile added — {len(JOURNAL)} item(s) in journal."
    dp_add_fig_btn.on_click(_dp_add_fig)

    roi_status = pn.pane.Markdown("")
    roi_list = pn.pane.Markdown(
        "<span style='opacity:0.6'>No ROIs yet.</span>")
    roi_note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>ROI counts are pooled over "
        "the summed stack. <b>Counts source</b>: <i>drift-corrected</i> uses the "
        "registered float stack (matches what clustering saw); <i>raw counts</i> "
        "pulls true integer detected-ion counts from the raw stack, for Poisson "
        "error propagation. Caveat — masks are defined in drift-corrected "
        "geometry, so raw totals over a mask are approximate when drift is large "
        "(the unregistered planes smear the boundary). Isotope ratios use pooled "
        "Σnumerator / Σdenominator over each ROI — the analytically correct way, "
        "not a mean of per-pixel ratios. <b>Depth profiles</b> apply each ROI "
        "mask plane-by-plane down the stack (x = plane index; no per-plane depth "
        "calibration is in the header). The current ROI set is attached to the "
        "loaded image as <code>img.rois</code>. Edge/threshold detection works "
        "on one channel; cluster ROIs come straight from the clustering run "
        "above.</span>")

    roi_section = pn.Column(
        pn.pane.Markdown("**ROI Manager**  <span style='font-size:11px;"
                         "opacity:0.7'>(clusters → ROIs, edge detection, pooled "
                         "counts &amp; ratios, depth profiles)</span>"),
        pn.pane.Markdown("<span style='font-size:12px;opacity:0.8'>"
                         "**From clusters**</span>"),
        pn.Row(roi_load_btn, roi_cluster_pick, roi_add_clusters_btn),
        pn.pane.Markdown("<span style='font-size:12px;opacity:0.8'>"
                         "**Edge / threshold detection**</span>"),
        pn.Row(roi_edge_chan, roi_edge_method, roi_edge_sigma),
        pn.Row(roi_edge_pct, roi_edge_minarea, roi_edge_merge, roi_detect_btn),
        roi_overlay_box,
        pn.pane.Markdown("<span style='font-size:12px;opacity:0.8'>"
                         "**Statistics**</span>"),
        pn.Row(roi_stat_chans,
               pn.Column(roi_counts_src, roi_ratio_on),
               pn.Column(roi_ratio_num, roi_ratio_den)),
        pn.Row(roi_compute_btn, roi_remove_btn, roi_clear_btn),
        pn.Row(roi_add_fig_btn, roi_add_txt_btn),
        pn.pane.Markdown("<span style='font-size:12px;opacity:0.8'>"
                         "**Depth profiles**</span>"),
        pn.Row(dp_chan, dp_metric, dp_ratio),
        pn.Row(dp_btn, dp_add_fig_btn),
        dp_box,
        roi_list,
        roi_status,
        roi_table_box,
        roi_note,
    )

    children = [pn.pane.Markdown("### Analysis")]
    if drift_hint:
        children.append(pn.pane.Markdown(drift_hint))
    children += [clustering_section, pn.layout.Divider(),
                 roi_section, pn.layout.Divider(), distributions_section]
    return pn.Column(*children)


# ─────────────────────────────────────────────────────────────────────────────
# 3D tab  (interactive point cloud / isosurface via plotly) + stack export
# ─────────────────────────────────────────────────────────────────────────────

def three_d_view(img):
    if img is None:
        return pn.pane.Markdown(
            "### 3D\nLoad a `.im` file to render the stack in 3D.")

    masses = img.masses
    non_se = [m for m in masses if "SE" not in m.upper()] or masses
    n_planes = int(img.metadata.get("n_planes", 1))
    field = float(img.metadata.get("field_um", 1.0)) or 1.0

    def _stack(corrected):
        if corrected:
            return np.asarray(img.get_channel(channel.value))        # (P, H, W)
        ch = list(masses).index(channel.value)
        return img.data[:, ch, :, :].astype(float)

    # ── controls ─────────────────────────────────────────────────────────────
    mode = pn.widgets.RadioButtonGroup(
        name="Mode", options=["Point cloud", "Isosurface"],
        value="Point cloud", button_type="primary")
    channel = pn.widgets.Select(name="Channel", options=list(masses),
                                value=non_se[0], width=150)
    src = pn.widgets.Select(name="Counts source",
                            options=["Drift-corrected (if run)",
                                     "Raw counts (integer)"],
                            value="Drift-corrected (if run)", width=200)
    cscale = pn.widgets.Select(
        name="Colour scale",
        options=["Viridis", "Magma", "Inferno", "Cividis", "Hot", "Greys"],
        value="Viridis", width=130)
    z_aspect = pn.widgets.FloatSlider(name="Z aspect (visual)", start=0.1,
                                      end=5.0, step=0.1, value=1.0, width=200)
    # point-cloud-specific
    thr_pct = pn.widgets.FloatSlider(name="Intensity threshold (percentile)",
                                     start=0.0, end=99.9, step=0.5, value=85.0,
                                     width=260)
    max_pts = pn.widgets.IntInput(name="Max points", value=60000, start=2000,
                                  end=400000, width=130)
    msize = pn.widgets.FloatSlider(name="Marker size", start=1.0, end=6.0,
                                   step=0.5, value=2.0, width=180)
    # isosurface-specific
    iso_n = pn.widgets.IntInput(name="Surfaces", value=3, start=1, end=10,
                                width=100)
    iso_op = pn.widgets.FloatSlider(name="Opacity", start=0.03, end=1.0,
                                    step=0.02, value=0.15, width=180)
    iso_cap = pn.widgets.IntInput(name="Max grid pts (decimate)", value=200000,
                                  start=20000, end=1000000, width=180)

    render_btn = pn.widgets.Button(name="Render 3D", button_type="primary",
                                   icon="cube", width=140)
    html_dir = pn.widgets.TextInput(name="Save to (folder)",
                                    value=os.path.expanduser("~"), width=240)
    html_name = pn.widgets.TextInput(name="HTML name", value="pymims_3d",
                                     width=160)
    html_btn = pn.widgets.Button(name="💾 Save interactive HTML",
                                 button_type="default", width=210)
    status = pn.pane.Markdown("")
    plot_box = pn.Column(pn.pane.Markdown(
        "<span style='opacity:0.6'>Set options and click **Render 3D**. "
        "Point cloud shows voxels above the threshold; isosurface draws "
        "nested shells through the volume.</span>"))
    _fig = {"obj": None}

    def _colorscale():
        return cscale.value

    def _build_pointcloud():
        import plotly.graph_objects as go
        corrected = src.value.startswith("Drift")
        s = _stack(corrected)
        P, Hh, Ww = s.shape
        thr = float(np.percentile(s, float(thr_pct.value)))
        sel = s > thr
        n_sel = int(sel.sum())
        if n_sel == 0:
            return None, "Nothing above threshold — lower the percentile."
        zz, yy, xx = np.where(sel)
        vals = s[sel]
        cap = int(max_pts.value)
        note = ""
        if n_sel > cap:                       # random subsample for responsiveness
            rng = np.random.default_rng(0)
            idx = rng.choice(n_sel, size=cap, replace=False)
            zz, yy, xx, vals = zz[idx], yy[idx], xx[idx], vals[idx]
            note = f" (showing {cap:,} of {n_sel:,} voxels)"
        x_um = xx / max(Ww - 1, 1) * field
        y_um = yy / max(Hh - 1, 1) * field
        fig = go.Figure(go.Scatter3d(
            x=x_um.tolist(), y=y_um.tolist(), z=zz.astype(float).tolist(),
            mode="markers",
            marker=dict(size=float(msize.value), color=vals.tolist(),
                        colorscale=_colorscale(), opacity=0.85,
                        colorbar=dict(title="counts")),
            hovertemplate="x=%{x:.2f}µm<br>y=%{y:.2f}µm<br>"
                          "plane=%{z:.0f}<br>counts=%{marker.color:.0f}"
                          "<extra></extra>"))
        src_lbl = ("drift-corrected" if corrected and img.corrected is not None
                   else "raw")
        fig.update_layout(
            scene=dict(xaxis_title="µm", yaxis_title="µm",
                       zaxis_title="plane (depth →)",
                       aspectmode="manual",
                       aspectratio=dict(x=1, y=1, z=float(z_aspect.value))),
            margin=dict(l=0, r=0, t=30, b=0), height=640,
            title=f"{channel.value} point cloud ({src_lbl}) · "
                  f"{os.path.basename(img.path)}")
        return fig, (f"✅ Point cloud: {channel.value}, threshold "
                     f"P{thr_pct.value:g} (>{thr:.0f} counts){note}.")

    def _build_isosurface():
        import plotly.graph_objects as go
        corrected = src.value.startswith("Drift")
        s = _stack(corrected)
        P, Hh, Ww = s.shape
        # Decimate in-plane only (keep all planes for depth fidelity).
        step = 1
        while (P * Hh * Ww) / (step * step) > int(iso_cap.value):
            step += 1
        s_d = s[:, ::step, ::step]
        Pd, Hd, Wd = s_d.shape
        zc, yc, xc = np.mgrid[0:Pd, 0:Hd, 0:Wd]
        x_um = xc / max(Wd - 1, 1) * field
        y_um = yc / max(Hd - 1, 1) * field
        z_co = zc.astype(float)   # plane index; in-plane decimation doesn't touch z
        vals = s_d.astype(float)
        vmax = float(np.percentile(vals, 99.5)) or 1.0
        vmin = float(np.percentile(vals[vals > 0], 50)) if (vals > 0).any() else 0.0
        fig = go.Figure(go.Volume(
            x=x_um.flatten().tolist(), y=y_um.flatten().tolist(),
            z=z_co.flatten().tolist(),
            value=vals.flatten().tolist(), isomin=vmin, isomax=vmax,
            opacity=float(iso_op.value), surface_count=int(iso_n.value),
            colorscale=_colorscale(), colorbar=dict(title="counts")))
        src_lbl = ("drift-corrected" if corrected and img.corrected is not None
                   else "raw")
        fig.update_layout(
            scene=dict(xaxis_title="µm", yaxis_title="µm",
                       zaxis_title="plane (depth →)",
                       aspectmode="manual",
                       aspectratio=dict(x=1, y=1, z=float(z_aspect.value))),
            margin=dict(l=0, r=0, t=30, b=0), height=640,
            title=f"{channel.value} isosurface ({src_lbl}) · "
                  f"{os.path.basename(img.path)}")
        decim = f" (in-plane decimation ×{step})" if step > 1 else ""
        return fig, f"✅ Isosurface: {channel.value}, {iso_n.value} shells{decim}."

    def _render(event):
        try:
            import plotly.graph_objects as go  # noqa: F401
        except Exception:
            plot_box.objects = [pn.pane.Alert(
                "plotly is not installed. Install with "
                "`pip install --break-system-packages plotly`.",
                alert_type="warning")]
            return
        status.object = "Rendering…"
        try:
            if mode.value == "Point cloud":
                fig, msg = _build_pointcloud()
            else:
                fig, msg = _build_isosurface()
        except Exception as exc:
            plot_box.objects = [pn.pane.Alert(f"Render failed: `{exc}`",
                                              alert_type="danger")]
            status.object = ""
            return
        if fig is None:
            status.object = f"⚠️ {msg}"
            return
        _fig["obj"] = fig
        plot_box.objects = [pn.pane.Plotly(
            fig, height=660, sizing_mode="stretch_width",
            config={"responsive": True})]
        status.object = msg
    render_btn.on_click(_render)

    def _save_html(event):
        if _fig["obj"] is None:
            status.object = "Nothing to save — render a view first."
            return
        try:
            folder = os.path.expanduser(html_dir.value or "~")
            os.makedirs(folder, exist_ok=True)
            name = (html_name.value or "pymims_3d").strip()
            if not name.lower().endswith(".html"):
                name += ".html"
            path = os.path.join(folder, name)
            _fig["obj"].write_html(path, include_plotlyjs="cdn")
            status.object = f"✅ Saved interactive view `{name}` to `{folder}`."
        except Exception as exc:
            status.object = f"❌ HTML save failed: `{exc}`"
    html_btn.on_click(_save_html)

    # ── Stack export (TIFF / NPZ) for external tools ─────────────────────────
    exp_what = pn.widgets.Select(name="Export data",
                                 options=["Drift-corrected (if run)",
                                          "Raw stack"],
                                 value="Drift-corrected (if run)", width=200)
    exp_fmt = pn.widgets.Select(name="Format",
                                options=["TIFF (ImageJ hyperstack)", "NPZ",
                                         "Both"],
                                value="TIFF (ImageJ hyperstack)", width=210)
    exp_dir = pn.widgets.TextInput(name="Export folder",
                                   value=os.path.expanduser("~"), width=240)
    exp_name = pn.widgets.TextInput(name="Export name", value="stack", width=160)
    exp_btn = pn.widgets.Button(name="📦 Export stack", button_type="primary",
                                width=160)
    exp_status = pn.pane.Markdown("")
    exp_note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>Writes the full 4-D stack "
        "(planes × channels × Y × X). TIFF is an ImageJ-style hyperstack "
        "(axes ZCYX, channel names embedded) that opens in any volumetric "
        "viewer; NPZ keeps the array plus masses and metadata for Python. "
        "Drift-corrected export uses the registered float stack; raw export "
        "keeps the original integer counts.</span>")

    def _export_stack(event):
        corrected = exp_what.value.startswith("Drift")
        if corrected and img.corrected is not None:
            arr = np.asarray(img.corrected)
            src_lbl = "drift-corrected"
        else:
            arr = np.asarray(img.data)
            src_lbl = "raw"
        # arr shape (planes, channels, H, W) == axes ZCYX
        folder = os.path.expanduser(exp_dir.value or "~")
        base = (exp_name.value or "stack").strip()
        want_tiff = exp_fmt.value in ("TIFF (ImageJ hyperstack)", "Both")
        want_npz = exp_fmt.value in ("NPZ", "Both")
        written = []
        try:
            os.makedirs(folder, exist_ok=True)
            if want_tiff:
                try:
                    import tifffile
                except Exception:
                    exp_status.object = ("❌ TIFF needs `tifffile` — "
                                         "`pip install --break-system-packages "
                                         "tifffile`.")
                    return
                tif_path = os.path.join(folder, base + ".tif")
                tifffile.imwrite(
                    tif_path, arr, imagej=True, metadata={
                        "axes": "ZCYX",
                        "Labels": list(masses),
                        "unit": "um",
                        "spacing": 1.0,
                    })
                written.append(os.path.basename(tif_path))
            if want_npz:
                npz_path = os.path.join(folder, base + ".npz")
                np.savez_compressed(
                    npz_path, stack=arr, masses=np.array(list(masses)),
                    axes="ZCYX", field_um=field, n_planes=int(arr.shape[0]),
                    corrected=bool(corrected and img.corrected is not None))
                written.append(os.path.basename(npz_path))
        except Exception as exc:
            exp_status.object = f"❌ Export failed: `{exc}`"
            return
        shp = "×".join(str(d) for d in arr.shape)
        exp_status.object = (f"✅ Exported {', '.join(written)} "
                             f"({src_lbl}, {shp} ZCYX) to `{folder}`.")
    exp_btn.on_click(_export_stack)

    # ── layout (mode-dependent controls) ─────────────────────────────────────
    def _controls(m):
        common = pn.Row(channel, src, cscale, z_aspect)
        if m == "Point cloud":
            return pn.Column(common, pn.Row(thr_pct, max_pts, msize))
        return pn.Column(common, pn.Row(iso_n, iso_op, iso_cap))

    note = pn.pane.Markdown(
        "<span style='font-size:11px;opacity:0.75'>Interactive in-browser 3D: "
        "drag to rotate, scroll to zoom. The Z axis is plane index, not "
        "calibrated depth — and remember sputter rate varies across the field, "
        "so treat this as a data-space view, not a reconstructed geometry. "
        "Point cloud stays smooth to ~60–100k points (raise the threshold or "
        "lower Max points if it lags); isosurface decimates in-plane to the "
        "grid-point cap. Save an interactive HTML to share, or export the stack "
        "below for a dedicated volumetric viewer.</span>")

    export_section = pn.Column(
        pn.layout.Divider(),
        pn.pane.Markdown("**Export stack (TIFF / NPZ)**  <span style='font-size:"
                         "11px;opacity:0.7'>for external volumetric tools</span>"),
        pn.Row(exp_what, exp_fmt),
        pn.Row(exp_dir, exp_name, exp_btn),
        exp_status,
        exp_note,
    )

    return pn.Column(
        pn.pane.Markdown("### 3D"),
        mode,
        pn.panel(pn.bind(_controls, mode.param.value)),
        pn.Row(render_btn, html_dir, html_name, html_btn),
        status,
        note,
        plot_box,
        export_section,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assemble the app
# ─────────────────────────────────────────────────────────────────────────────

tabs = pn.Tabs(
    ("Imaging",  pn.panel(pn.bind(imaging_view, state.param.img))),
    ("HMR",      hmr_view()),
    ("Analysis", pn.panel(pn.bind(analysis_view, state.param.img))),
    ("3D",       pn.panel(pn.bind(three_d_view, state.param.img))),
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
    pn.pane.Markdown("### Journal"),
    journal_status,
    journal_dir,
    journal_name,
    journal_fmt,
    export_journal_btn,
    pn.Row(remove_last_btn, clear_journal_btn),
    export_status,
    journal_note,
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
