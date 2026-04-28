"""
pymims_explore.py — interactive widget UI and list-config runner for pymims.

Drop this into a Colab cell (or %run it from a notebook) after pymims.py is
loaded. Provides two ways to explore an .im file:

  1. explore(img)              — ipywidgets dropdowns for one ratio at a time
  2. run_analyses(img, specs)  — list-config for reproducible batch runs

Example:
    !wget -q -O pymims.py https://raw.githubusercontent.com/gregmcmahon345/PyMIMS/main/pymims.py
    !wget -q -O pymims_explore.py https://raw.githubusercontent.com/gregmcmahon345/PyMIMS/main/pymims_explore.py
    from pymims import MimsImage
    from pymims_explore import explore, run_analyses, ISOTOPE_REFS

    from google.colab import files
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    img = MimsImage(fname)
    print(img)

    # Widget-driven:
    explore(img)

    # Or list-config:
    specs = [
        {'num': '15N', 'den': '14N',  'delta_ref': ISOTOPE_REFS['15N/14N']},
        {'num': '13C', 'den': '12C',  'delta_ref': ISOTOPE_REFS['13C/12C']},
        {'num': '31P', 'den': '12C 14N'},   # not an isotope pair — uses median
    ]
    run_analyses(img, specs)
"""

import re

# Common natural-abundance isotope ratios (IUPAC/community standards).
# Keys are written 'numerator/denominator'.
ISOTOPE_REFS = {
    '13C/12C':   0.0112372,    # V-PDB
    '15N/14N':   0.0036765,    # AIR
    '34S/32S':   0.0441626,    # V-CDT
    '33S/32S':   0.0078772,    # V-CDT
    '18O/16O':   0.0020052,    # V-SMOW
    '17O/16O':   0.0003799,    # V-SMOW
    '2H/1H':     0.00015576,   # V-SMOW
    # CN-isotopologue forms (NanoSIMS standard for ¹⁵N work)
    '12C 15N/12C 14N': 0.0036765,   # ¹⁵N/¹⁴N inherited via the CN pair
    '13C 14N/12C 14N': 0.0112372,   # ¹³C/¹²C inherited via the CN pair
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect isotope-pair from numerator/denominator labels
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_ref(num_label, den_label):
    """Return absolute reference ratio if (num, den) matches a known pair, else None."""
    key = f"{num_label}/{den_label}"
    if key in ISOTOPE_REFS:
        return ISOTOPE_REFS[key], key
    # Try canonical form: strip spaces
    key2 = key.replace(' ', '')
    for known, val in ISOTOPE_REFS.items():
        if known.replace(' ', '') == key2:
            return val, known
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Bulk ratio reporting (total_A / total_B with propagated Poisson sigma)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_ratio_report(result, delta_ref=None):
    """Return a printable string summarising the bulk ratio and δ."""
    import numpy as np
    A_tot = float(np.nansum(result['A']))
    B_tot = float(np.nansum(result['B']))
    if B_tot <= 0 or A_tot <= 0:
        return "  Bulk ratio: N/A (zero counts)"
    R_bulk    = A_tot / B_tot
    sigma_R   = R_bulk * (1.0/A_tot + 1.0/B_tot) ** 0.5
    rel_err   = sigma_R / R_bulk
    lines = [
        f"  Bulk ratio  totalA/totalB = {R_bulk:.5g} ± {sigma_R:.2g}  "
        f"(σ/R = {rel_err*100:.2f}%)",
        f"    totals: A = {A_tot:.0f},  B = {B_tot:.0f}",
    ]
    if delta_ref is not None and delta_ref > 0:
        delta_bulk     = (R_bulk / delta_ref - 1.0) * 1000.0
        sigma_delta    = (sigma_R / delta_ref) * 1000.0
        lines.append(
            f"    δ vs {delta_ref:.4g} = {delta_bulk:+.1f} ± {sigma_delta:.1f} ‰"
        )
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# List-config runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analyses(img, specs):
    """
    Run a list of ratio analyses against a MimsImage.

    Parameters
    ----------
    img   : MimsImage
        Already loaded (and ideally drift-corrected) MimsImage.
    specs : list of dict
        Each dict may contain:
          num, den           : channel name/index (required)
          delta_ref          : reference ratio (optional; auto if not given
                               and pair matches a known isotope ratio)
          min_counts         : low-count masking threshold (optional)
          max_rel_err        : relative-error mask threshold (optional)
          delta_range        : symmetric ±range for δ panel in ‰ (optional)

    Each spec produces a four-panel figure inline plus a bulk-ratio printout.
    """
    for i, s in enumerate(specs, 1):
        num = s['num']; den = s['den']
        # Resolve labels for the auto-ref lookup
        num_lab = img.masses[img._resolve_channel(num)]
        den_lab = img.masses[img._resolve_channel(den)]
        delta_ref = s.get('delta_ref')
        ref_origin = 'user-specified'
        if delta_ref is None:
            auto, key = _lookup_ref(num_lab, den_lab)
            if auto is not None:
                delta_ref = auto
                ref_origin = f'auto ({key})'
            else:
                ref_origin = 'image median (non-isotope pair)'

        print(f"\n[{i}/{len(specs)}] {num_lab} / {den_lab}   "
              f"δ ref: {ref_origin}")
        fig, result = img.plot_ratio(
            num, den,
            delta_ref=delta_ref,
            min_counts=s.get('min_counts'),
            max_rel_err=s.get('max_rel_err'),
            delta_range=s.get('delta_range'),
        )
        if delta_ref is not None:
            print(bulk_ratio_report(result, delta_ref=delta_ref))
        else:
            print(bulk_ratio_report(result))


# ─────────────────────────────────────────────────────────────────────────────
# Widget UI
# ─────────────────────────────────────────────────────────────────────────────

def explore(img):
    """
    Launch an ipywidgets UI for exploring ratios on a MimsImage.

    Provides dropdowns for numerator, denominator, drift reference channel,
    a bin_planes selector for low-count drift correction, and masking controls.
    Click 'Plot ratio' to render. Click 'Re-do drift correction' to re-run
    drift correction with the chosen settings.
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
    except ImportError:
        raise ImportError(
            "ipywidgets / IPython not available. Install with: "
            "pip install ipywidgets ipython"
        )

    masses = img.masses

    # ── Drift correction controls ───────────────────────────────────────────
    drift_ref = W.Dropdown(
        options=list(enumerate(masses)),
        value=0,
        description='Drift ref:',
        style={'description_width': 'initial'},
    )
    bin_planes = W.IntSlider(
        value=1, min=1, max=max(2, img.metadata['n_planes'] // 2),
        step=1, description='bin_planes:',
        style={'description_width': 'initial'},
        continuous_update=False,
    )
    bin_apply = W.Dropdown(
        options=['same', 'interp', 'super'],
        value='same', description='bin_apply:',
        style={'description_width': 'initial'},
    )
    drift_button = W.Button(description='Re-do drift correction',
                            button_style='warning')

    # ── Ratio controls ──────────────────────────────────────────────────────
    num_dd = W.Dropdown(options=list(enumerate(masses)), value=0,
                        description='Numerator:',
                        style={'description_width': 'initial'})
    den_dd = W.Dropdown(options=list(enumerate(masses)),
                        value=min(1, len(masses) - 1),
                        description='Denominator:',
                        style={'description_width': 'initial'})
    delta_text = W.Text(value='', placeholder='auto if known pair, else median',
                        description='δ ref:',
                        style={'description_width': 'initial'})
    min_counts = W.FloatText(value=0, description='min_counts (B≥):',
                             style={'description_width': 'initial'})
    max_rel = W.FloatText(value=0, description='max σ/R:',
                          style={'description_width': 'initial'})
    plot_button = W.Button(description='Plot ratio', button_style='success')

    out = W.Output()

    # ── Callbacks ───────────────────────────────────────────────────────────
    def do_drift(_):
        with out:
            clear_output(wait=True)
            try:
                img.drift_correct(
                    reference=drift_ref.value,
                    bin_planes=bin_planes.value,
                    bin_apply=bin_apply.value,
                )
            except Exception as e:
                print(f"Drift correction failed: {e}")

    def do_plot(_):
        with out:
            clear_output(wait=True)
            try:
                num = num_dd.value
                den = den_dd.value
                num_lab = masses[num]; den_lab = masses[den]

                # δ reference: explicit, auto, or median
                delta_ref = None
                ref_origin = 'image median'
                if delta_text.value.strip():
                    try:
                        delta_ref = float(delta_text.value)
                        ref_origin = f'user-specified ({delta_ref:.4g})'
                    except ValueError:
                        print(f"Could not parse δ ref '{delta_text.value}', "
                              "using image median.")
                else:
                    auto, key = _lookup_ref(num_lab, den_lab)
                    if auto is not None:
                        delta_ref = auto
                        ref_origin = f'auto ({key} = {auto:.4g})'

                print(f"{num_lab} / {den_lab}   δ ref: {ref_origin}")
                fig, result = img.plot_ratio(
                    num, den,
                    delta_ref=delta_ref,
                    min_counts=min_counts.value if min_counts.value > 0 else None,
                    max_rel_err=max_rel.value if max_rel.value > 0 else None,
                )
                print(bulk_ratio_report(result, delta_ref=delta_ref))
            except Exception as e:
                import traceback
                traceback.print_exc()

    drift_button.on_click(do_drift)
    plot_button.on_click(do_plot)

    drift_box = W.VBox([
        W.HTML("<b>Drift correction</b>"),
        drift_ref, bin_planes, bin_apply, drift_button,
    ])
    ratio_box = W.VBox([
        W.HTML("<b>Ratio</b>"),
        num_dd, den_dd, delta_text, min_counts, max_rel, plot_button,
    ])

    display(W.HBox([drift_box, ratio_box]))
    display(out)
