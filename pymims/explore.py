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
import numpy as np

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


def _parse_optional_float(text):
    """Parse text-widget value to float; '' or unparseable -> None."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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

def cluster_overlay_slider(img, result):
    """
    Interactive cluster-overlay viewer with sliders for min_pixels and k.

    Wraps `pymims_clustering.plot_overlay` in a widget UI. As the user
    drags the min_pixels slider, the cluster outlines redraw with the new
    speckle-filtering threshold. As the k dropdown changes, the cluster
    partition switches.

    Mode-specific controls auto-appear in the panel:
      * For base='delta', set the δ reference (natural-abundance ratio,
        auto-filled from ISOTOPE_REFS when num/den match a known pair)
        and δ max (colour-bar limit in permil).
      * For base='hsi', set the OpenMIMS-style scale factor (e.g. 10000
        for ¹⁵N/¹⁴N), and optional hue min/max in scaled units.
      * The min_counts field applies to all ratio-based modes (ratio,
        delta, hsi) and masks out pixels with low denominator counts.

    Parameters
    ----------
    img : MimsImage
    result : ClusterResult dict from cluster_pixels()

    Usage
    -----
        from pymims_clustering import cluster_pixels
        from pymims_explore import cluster_overlay_slider

        result = cluster_pixels(img, method='kmeans', k_max=10)
        cluster_overlay_slider(img, result)

    Notes
    -----
    The widget renders a fresh figure on each control change. The
    underlying clustering is not re-run; only the display is rebuilt,
    so dragging the slider is fast (~0.5 s on a 256×256 image).
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(f"Widget UI requires ipywidgets: {e}")

    from .clustering import plot_overlay

    # Available k values come from result['labels_by_k']
    k_options = sorted(result['labels_by_k'].keys())

    # Build controls
    k_dd = W.Dropdown(
        options=k_options, value=result['sensible_k'],
        description='k:',
        layout=W.Layout(width='180px'),
    )
    base_dd = W.Dropdown(
        options=['channel', 'ratio', 'delta', 'hsi'],
        value='channel',
        description='base:',
        layout=W.Layout(width='180px'),
    )
    channel_dd = W.Dropdown(
        options=img.masses, value=img.masses[0],
        description='channel:',
        layout=W.Layout(width='220px'),
    )
    num_dd = W.Dropdown(
        options=img.masses, value=img.masses[0],
        description='num:',
        layout=W.Layout(width='200px'),
    )
    den_dd = W.Dropdown(
        options=img.masses, value=img.masses[-1],
        description='den:',
        layout=W.Layout(width='200px'),
    )
    min_pix_slider = W.IntSlider(
        value=1, min=1, max=200, step=1,
        description='min pixels:',
        style={'description_width': 'initial'},
        continuous_update=False,    # only re-render on release, not while dragging
        layout=W.Layout(width='400px'),
    )
    min_counts_input = W.FloatText(
        value=0, description='min_counts:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    # Delta-mode controls (auto-fill from ISOTOPE_REFS when num/den match)
    delta_ref_input = W.FloatText(
        value=0.0,
        description='δ ref:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    delta_max_input = W.FloatText(
        value=10000.0,
        description='δ max (‰):',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    # HSI-mode controls
    scale_factor_input = W.FloatText(
        value=1.0,
        description='HSI scale ×:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    ratio_min_input = W.Text(
        value='', description='hue min:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
        placeholder='auto (1st pct)',
    )
    ratio_max_input = W.Text(
        value='', description='hue max:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
        placeholder='auto (99th pct)',
    )
    out = W.Output()

    def _autofill_delta_ref(*_):
        """When num/den change, look up the natural-abundance value
        from ISOTOPE_REFS and pre-fill δ ref."""
        ref_val, _key = _lookup_ref(num_dd.value, den_dd.value)
        if ref_val is not None:
            delta_ref_input.value = ref_val

    def render(*_):
        with out:
            clear_output(wait=True)
            base_kwargs = {}
            if min_counts_input.value > 0:
                base_kwargs['min_counts'] = min_counts_input.value
            if base_dd.value == 'delta':
                if delta_ref_input.value <= 0:
                    print("Render failed: 'δ ref' must be a positive value "
                          "(natural-abundance ratio). For 15N/14N use 0.0037; "
                          "for 13C/12C use 0.01124; etc.")
                    return
                base_kwargs['reference'] = delta_ref_input.value
                base_kwargs['delta_max'] = delta_max_input.value
            elif base_dd.value == 'hsi':
                if scale_factor_input.value > 0:
                    base_kwargs['scale_factor'] = scale_factor_input.value
                if ratio_min_input.value.strip():
                    try:
                        base_kwargs['ratio_min'] = float(ratio_min_input.value)
                    except ValueError:
                        pass
                if ratio_max_input.value.strip():
                    try:
                        base_kwargs['ratio_max'] = float(ratio_max_input.value)
                    except ValueError:
                        pass

            kwargs = dict(
                img=img, result=result,
                k=k_dd.value,
                base=base_dd.value,
                min_pixels=min_pix_slider.value,
                base_kwargs=base_kwargs,
                show=False,
            )

            if base_dd.value == 'channel':
                kwargs['channel'] = channel_dd.value
            else:
                kwargs['numerator']   = num_dd.value
                kwargs['denominator'] = den_dd.value

            try:
                fig = plot_overlay(**kwargs)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Render failed: {e}")

    # Wire up callbacks
    for ctrl in (k_dd, base_dd, channel_dd, num_dd, den_dd,
                 min_pix_slider, min_counts_input,
                 delta_ref_input, delta_max_input,
                 scale_factor_input, ratio_min_input, ratio_max_input):
        ctrl.observe(render, names='value')
    # Auto-fill δ ref when num/den change
    for ctrl in (num_dd, den_dd):
        ctrl.observe(_autofill_delta_ref, names='value')
    # Initial auto-fill
    _autofill_delta_ref()

    # Initial render
    render()

    # Layout: controls in three rows above the output
    controls = W.VBox([
        W.HBox([k_dd, base_dd, channel_dd]),
        W.HBox([num_dd, den_dd, min_counts_input]),
        W.HBox([delta_ref_input, delta_max_input]),
        W.HBox([scale_factor_input, ratio_min_input, ratio_max_input]),
        min_pix_slider,
    ])
    display(W.VBox([controls, out]))


def roi_rule_slider(img, hist_results=None):
    """
    Interactive two-rule ROI builder with sliders.

    Lets you build a rule-based ROI mask from up to two threshold rules,
    combine them with AND/OR, filter out small connected components via
    min_pixels, and visualise the result over a base image (channel,
    ratio, delta, or HSI). All controls update live.

    The widget is a thin wrapper around `pymims_rules.build_roi_masks`
    + `pymims_rules.plot_rule_masks`. Anything you can do in the widget
    you can also do in code; the widget exists to remove typing friction
    for the routine case of "tune the cutoff and see what happens".

    Parameters
    ----------
    img : MimsImage
        Drift-corrected image.
    hist_results : dict or None
        Output of pymims_histograms.plot_histograms(). Required only if
        you want to use 'gmm-component' mode for any rule. If None, the
        gmm-component mode option is hidden from the dropdowns.

    Usage
    -----
        from pymims_histograms import plot_histograms
        from pymims_explore import roi_rule_slider

        # Optional: pre-fit GMMs if you want gmm-component rules
        hists = plot_histograms(img, k_max=6, show=False, verbose=False)
        roi_rule_slider(img, hist_results=hists)

        # Or, with no GMM support:
        roi_rule_slider(img)

    Notes
    -----
    Sliders use continuous_update=False so the figure only redraws on
    release, not while dragging — keeps interaction responsive on a
    256×256 image (~0.5 s per redraw).
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(f"Widget UI requires ipywidgets: {e}")

    from .rules import build_roi_masks, plot_rule_masks

    has_gmm = hist_results is not None
    mode_options = ['counts', 'percentile']
    if has_gmm:
        mode_options.append('gmm-component')

    # ── Rule controls factory ─────────────────────────────────────────────
    # Each rule has: enable checkbox, channel, mode, cutoff (or k+component
    # for gmm-component mode), and a comparison toggle. The cutoff control
    # type changes with the mode, so we pre-build all variants and toggle
    # visibility.
    def make_rule_controls(rule_index, default_enabled):
        enable = W.Checkbox(
            value=default_enabled,
            description=f'Rule {rule_index + 1}',
            indent=False,
            layout=W.Layout(width='130px'),
        )
        channel = W.Dropdown(
            options=img.masses, value=img.masses[0],
            description='channel:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='200px'),
        )
        mode = W.Dropdown(
            options=mode_options, value='percentile',
            description='mode:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='200px'),
        )
        comparison = W.Dropdown(
            options=['>=', '>', '<=', '<'], value='>=',
            description='cmp:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='130px'),
        )
        # Cutoff variants — only one is visible at a time
        counts_cutoff = W.FloatText(
            value=100.0, description='counts:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='180px'),
        )
        percentile_cutoff = W.FloatSlider(
            value=90.0, min=0.0, max=100.0, step=0.5,
            description='percentile:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=W.Layout(width='350px'),
        )
        # GMM controls
        gmm_k = W.IntSlider(
            value=3, min=2, max=6,
            description='k:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout=W.Layout(width='200px'),
        )
        gmm_component = W.Dropdown(
            options=['highest', 'lowest', 0, 1, 2, 3, 4, 5],
            value='highest',
            description='component:',
            style={'description_width': 'initial'},
            layout=W.Layout(width='180px'),
        )

        def _toggle_visibility(*_):
            """Show only the controls relevant to the current mode."""
            m = mode.value
            counts_cutoff.layout.display = 'flex' if m == 'counts' else 'none'
            percentile_cutoff.layout.display = (
                'flex' if m == 'percentile' else 'none'
            )
            gmm_k.layout.display = 'flex' if m == 'gmm-component' else 'none'
            gmm_component.layout.display = (
                'flex' if m == 'gmm-component' else 'none'
            )
        mode.observe(_toggle_visibility, names='value')
        _toggle_visibility()   # set initial state

        return {
            'enable': enable,
            'channel': channel,
            'mode': mode,
            'comparison': comparison,
            'counts_cutoff': counts_cutoff,
            'percentile_cutoff': percentile_cutoff,
            'gmm_k': gmm_k,
            'gmm_component': gmm_component,
        }

    rule1 = make_rule_controls(0, default_enabled=True)
    rule2 = make_rule_controls(1, default_enabled=False)

    # ── Combine + min_pixels ─────────────────────────────────────────────
    combine_dd = W.Dropdown(
        options=['AND', 'OR'], value='AND',
        description='combine:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='150px'),
    )
    min_pix_slider = W.IntSlider(
        value=1, min=1, max=200, step=1,
        description='min pixels:',
        style={'description_width': 'initial'},
        continuous_update=False,
        layout=W.Layout(width='400px'),
    )

    # ── Base-image controls (same as cluster_overlay_slider) ─────────────
    base_dd = W.Dropdown(
        options=['channel', 'ratio'], value='channel',
        description='base:',
        layout=W.Layout(width='180px'),
    )
    base_channel_dd = W.Dropdown(
        options=img.masses, value=img.masses[0],
        description='base channel:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='220px'),
    )
    base_num_dd = W.Dropdown(
        options=img.masses, value=img.masses[0],
        description='ratio num:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='200px'),
    )
    base_den_dd = W.Dropdown(
        options=img.masses, value=img.masses[-1],
        description='ratio den:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='200px'),
    )

    out = W.Output()

    def _build_rule_dict(rule_ctrls, rule_index):
        """Translate widget state into a rule dict for build_roi_masks."""
        if not rule_ctrls['enable'].value:
            return None
        m = rule_ctrls['mode'].value
        rule = {
            'channel': rule_ctrls['channel'].value,
            'mode': m,
            'comparison': rule_ctrls['comparison'].value,
            'name': f"rule_{rule_index + 1}",
        }
        if m == 'counts':
            rule['cutoff'] = rule_ctrls['counts_cutoff'].value
            rule['name'] = (f"{rule_ctrls['channel'].value} "
                            f"{rule_ctrls['comparison'].value} "
                            f"{rule_ctrls['counts_cutoff'].value:.0f}")
        elif m == 'percentile':
            rule['cutoff'] = rule_ctrls['percentile_cutoff'].value
            rule['name'] = (f"{rule_ctrls['channel'].value} "
                            f"p{rule_ctrls['percentile_cutoff'].value:.1f}"
                            f" {rule_ctrls['comparison'].value}")
        elif m == 'gmm-component':
            rule['k'] = rule_ctrls['gmm_k'].value
            rule['component'] = rule_ctrls['gmm_component'].value
            rule['name'] = (f"{rule_ctrls['channel'].value} "
                            f"GMM k={rule_ctrls['gmm_k'].value} "
                            f"comp={rule_ctrls['gmm_component'].value}")
        return rule

    def render(*_):
        with out:
            clear_output(wait=True)
            r1 = _build_rule_dict(rule1, 0)
            r2 = _build_rule_dict(rule2, 1)
            rules = [r for r in (r1, r2) if r is not None]
            if not rules:
                print("Enable at least one rule to see the ROI mask.")
                return
            try:
                rois = build_roi_masks(
                    img, rules=rules,
                    combine=combine_dd.value,
                    histograms=hist_results,
                )
                # Apply min_pixels via connected-component filter
                if min_pix_slider.value > 1:
                    try:
                        from skimage import measure
                        for k_name, mask in list(rois.items()):
                            cc = measure.label(mask, connectivity=2)
                            if cc.max() == 0:
                                continue
                            sizes = np.bincount(cc.ravel())
                            keep = sizes >= min_pix_slider.value
                            keep[0] = False
                            rois[k_name] = keep[cc]
                    except ImportError:
                        pass

                # Render the overlay
                kwargs = dict(img=img, rois=rois,
                              base=base_dd.value, show=False)
                if base_dd.value == 'channel':
                    kwargs['channel'] = base_channel_dd.value
                else:
                    kwargs['numerator']   = base_num_dd.value
                    kwargs['denominator'] = base_den_dd.value
                fig = plot_rule_masks(**kwargs)
                display(fig)
                plt.close(fig)

                # Print summary
                total_pixels = rois['combined'].size
                combined_pixels = int(rois['combined'].sum())
                print(f"\n  Combined ROI: {combined_pixels:,} pixels "
                      f"({100 * combined_pixels / total_pixels:.1f}%) "
                      f"using {combine_dd.value} of {len(rules)} rule(s)")
                for k_name in (k for k in rois if k != 'combined'):
                    n = int(rois[k_name].sum())
                    pct = 100 * n / total_pixels
                    print(f"    {k_name}: {n:,} pixels ({pct:.1f}%)")
            except Exception as e:
                print(f"Render failed: {e}")

    # Wire up callbacks — every control re-renders
    all_ctrls = [combine_dd, min_pix_slider, base_dd,
                 base_channel_dd, base_num_dd, base_den_dd]
    for r in (rule1, rule2):
        all_ctrls.extend(r.values())
    for ctrl in all_ctrls:
        ctrl.observe(render, names='value')

    # Initial render
    render()

    # Layout
    rule1_box = W.VBox([
        W.HBox([rule1['enable'], rule1['channel'], rule1['mode'],
                rule1['comparison']]),
        W.HBox([rule1['counts_cutoff'], rule1['percentile_cutoff'],
                rule1['gmm_k'], rule1['gmm_component']]),
    ], layout=W.Layout(border='1px solid #ddd', padding='6px',
                       margin='2px 0'))
    rule2_box = W.VBox([
        W.HBox([rule2['enable'], rule2['channel'], rule2['mode'],
                rule2['comparison']]),
        W.HBox([rule2['counts_cutoff'], rule2['percentile_cutoff'],
                rule2['gmm_k'], rule2['gmm_component']]),
    ], layout=W.Layout(border='1px solid #ddd', padding='6px',
                       margin='2px 0'))
    combine_row = W.HBox([combine_dd, min_pix_slider])
    base_row = W.HBox([base_dd, base_channel_dd, base_num_dd, base_den_dd])

    display(W.VBox([rule1_box, rule2_box, combine_row, base_row, out]))


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
    # Build dropdown options as (display_label, value) tuples — ipywidgets
    # expects this order. Value is the channel index.
    channel_options = [(f"[{i}] {lab}", i) for i, lab in enumerate(masses)]

    # Compact widget layouts so the controls don't dominate the page.
    DROP_LAYOUT = W.Layout(width='220px')
    SLIDER_LAYOUT = W.Layout(width='220px')
    BTN_LAYOUT = W.Layout(width='150px')
    SHORT_BTN = W.Layout(width='115px')

    # ── Drift correction controls ───────────────────────────────────────────
    drift_ref = W.Dropdown(
        options=channel_options,
        value=0,
        description='Drift ref:',
        style={'description_width': 'initial'},
        layout=DROP_LAYOUT,
    )
    bin_planes = W.IntSlider(
        value=1, min=1, max=max(2, img.metadata['n_planes'] // 2),
        step=1, description='bin_planes:',
        style={'description_width': 'initial'},
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    bin_apply = W.Dropdown(
        options=['same', 'interp', 'super'],
        value='same', description='bin_apply:',
        style={'description_width': 'initial'},
        layout=DROP_LAYOUT,
    )
    drift_button = W.Button(description='Re-do drift correction',
                            button_style='warning', layout=BTN_LAYOUT)
    channels_button = W.Button(description='Show channels',
                               button_style='info', layout=SHORT_BTN)
    log_scale_cb = W.Checkbox(value=False, description='log scale',
                              indent=False)

    # ── Ratio controls ──────────────────────────────────────────────────────
    num_dd = W.Dropdown(options=channel_options, value=0,
                        description='Numerator:',
                        style={'description_width': 'initial'},
                        layout=DROP_LAYOUT)
    den_dd = W.Dropdown(options=channel_options,
                        value=min(1, len(masses) - 1),
                        description='Denominator:',
                        style={'description_width': 'initial'},
                        layout=DROP_LAYOUT)
    delta_text = W.Text(value='', placeholder='auto, else median',
                        description='δ ref:',
                        style={'description_width': 'initial'},
                        layout=W.Layout(width='220px'))
    min_counts = W.FloatText(value=0, description='min_counts (B≥):',
                             style={'description_width': 'initial'},
                             layout=W.Layout(width='200px'))
    max_rel = W.FloatText(value=0, description='max σ/R:',
                          style={'description_width': 'initial'},
                          layout=W.Layout(width='200px'))
    plot_button = W.Button(description='Plot ratio', button_style='success',
                           layout=BTN_LAYOUT)

    # ── HSI controls ────────────────────────────────────────────────────────
    # Display labels are user-friendly; the library handles the alias mapping.
    hsi_cmap = W.Dropdown(
        options=['viridis', 'plasma', 'inferno', 'magma', 'twilight',
                 'rainbow', 'Classic OpenMIMS LUT'],
        value='viridis',
        description='cmap:',
        style={'description_width': 'initial'},
        layout=DROP_LAYOUT,
    )
    hsi_cmap_reverse = W.Checkbox(value=False, description='reverse',
                                  indent=False)
    hsi_intensity = W.Dropdown(
        options=['denominator', 'numerator', 'sum'],
        value='denominator',
        description='intensity:',
        style={'description_width': 'initial'},
        layout=DROP_LAYOUT,
    )
    hsi_scale_factor = W.FloatText(
        value=1.0, description='scale ×:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='150px'),
    )
    hsi_ratio_min = W.Text(
        value='', placeholder='auto (1st pct)',
        description='hue min:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    hsi_ratio_max = W.Text(
        value='', placeholder='auto (99th pct)',
        description='hue max:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    hsi_button = W.Button(description='Plot HSI', button_style='primary',
                          layout=BTN_LAYOUT)

    # ── Export controls (publication-quality) ──────────────────────────────
    export_format = W.Dropdown(
        options=['png', 'tiff'], value='png',
        description='format:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='150px'),
    )
    export_dpi = W.Dropdown(
        options=[300, 600, 1200], value=600,
        description='dpi:',
        style={'description_width': 'initial'},
        layout=W.Layout(width='150px'),
    )
    export_size = W.FloatText(
        value=4.0, description='panel size (in):',
        style={'description_width': 'initial'},
        layout=W.Layout(width='180px'),
    )
    zip_download_cb = W.Checkbox(
        value=True, description='zip + download',
        indent=False,
        layout=W.Layout(width='180px'),
    )

    # Checkbox-driven selection. Each ticked checkbox produces ONE file.
    # Group A — full multi-panel figures
    cb_full_channels = W.Checkbox(value=False, description='Full channels figure',
                                  indent=False)
    cb_full_ratio    = W.Checkbox(value=False, description='Full ratio figure',
                                  indent=False)

    # Group B — individual channels (one checkbox per available mass)
    cb_channels = [W.Checkbox(value=False, description=lab, indent=False)
                   for lab in masses]

    # Group C — individual ratio sub-panels
    cb_ratio    = W.Checkbox(value=False, description='Ratio',     indent=False)
    cb_delta    = W.Checkbox(value=False, description='δ',         indent=False)
    cb_sigma    = W.Checkbox(value=False, description='σ(R)',      indent=False)
    cb_rel_err  = W.Checkbox(value=False, description='σ(R)/R',    indent=False)

    # Group D — composite
    cb_hsi      = W.Checkbox(value=False, description='HSI',       indent=False)

    # Helper buttons + main export trigger
    select_all_btn  = W.Button(description='Select all',
                               layout=W.Layout(width='100px'))
    clear_all_btn   = W.Button(description='Clear all',
                               layout=W.Layout(width='100px'))
    export_btn      = W.Button(description='Export selected',
                               button_style='success',
                               layout=W.Layout(width='180px'))

    # Convenience list of all checkboxes for select-all/clear-all
    _all_checkboxes = ([cb_full_channels, cb_full_ratio]
                       + cb_channels
                       + [cb_ratio, cb_delta, cb_sigma, cb_rel_err, cb_hsi])

    # Four output panels:
    # - status: drift correction messages, errors
    # - channels_out: all-channels plot (pinned, only changes on demand)
    # - hsi_out: HSI composite (pinned, only changes on demand)
    # - ratio_out: four-panel ratio figure (changes every Plot ratio)
    status_out   = W.Output()
    channels_out = W.Output()
    hsi_out      = W.Output()
    ratio_out    = W.Output()

    # ── Callbacks ───────────────────────────────────────────────────────────
    def do_drift(_):
        with status_out:
            clear_output(wait=True)
            try:
                img.drift_correct(
                    reference=drift_ref.value,
                    bin_planes=bin_planes.value,
                    bin_apply=bin_apply.value,
                )
            except Exception as e:
                print(f"Drift correction failed: {e}")

    def do_channels(_):
        with channels_out:
            clear_output(wait=True)
            try:
                fig = img.plot(outpath=None, show=True,
                               log_scale=log_scale_cb.value)
                display(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                import traceback
                traceback.print_exc()

    def do_hsi(_):
        with hsi_out:
            clear_output(wait=True)
            try:
                num = num_dd.value
                den = den_dd.value
                num_lab = masses[num]; den_lab = masses[den]
                # Natural-abundance lookup based on label pair
                nat_abund, nat_key = _lookup_ref(num_lab, den_lab)
                rmin = _parse_optional_float(hsi_ratio_min.value)
                rmax = _parse_optional_float(hsi_ratio_max.value)
                sf = hsi_scale_factor.value if hsi_scale_factor.value > 0 else 1.0

                ref_msg = (f'  |  nat. abund.: {nat_abund:.4g} ({nat_key})'
                           if nat_abund is not None else '')
                scale_msg = f'  |  ×{sf:g}' if sf != 1.0 else ''
                print(f"HSI {num_lab}/{den_lab}  |  intensity: {hsi_intensity.value}  |  "
                      f"cmap: {hsi_cmap.value}{'_r' if hsi_cmap_reverse.value else ''}"
                      f"{scale_msg}{ref_msg}")
                fig, info = img.plot_hsi(
                    num, den,
                    intensity=hsi_intensity.value,
                    cmap=hsi_cmap.value,
                    cmap_reverse=hsi_cmap_reverse.value,
                    min_counts=min_counts.value if min_counts.value > 0 else None,
                    ratio_min=rmin, ratio_max=rmax,
                    scale_factor=sf,
                    natural_abundance=nat_abund,
                    show_natural_abundance=True,
                    outpath=None,
                    show=True,
                )
                display(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                import traceback
                traceback.print_exc()

    def do_plot(_):
        with ratio_out:
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
                    outpath=None,
                    show=True,
                )
                display(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(bulk_ratio_report(result, delta_ref=delta_ref))
            except Exception as e:
                import traceback
                traceback.print_exc()

    def _export_outpath(tag):
        """Build a default filename for an export."""
        import os, re
        base = os.path.splitext(os.path.basename(img.path))[0]
        # Drop trailing " (1)" etc. from Colab re-uploads
        base = re.sub(r'\s*\(\d+\)$', '', base)
        ext  = export_format.value
        return f"{base}_{tag}_{export_dpi.value}dpi.{ext}"

    def _safe_label(s):
        return s.replace(' ', '').replace('/', '_')

    def do_select_all(_):
        for cb in _all_checkboxes:
            cb.value = True

    def do_clear_all(_):
        for cb in _all_checkboxes:
            cb.value = False

    def do_export(_):
        with status_out:
            clear_output(wait=True)
            try:
                # Resolve current numerator/denominator labels
                num = num_dd.value; den = den_dd.value
                num_lab = masses[num]; den_lab = masses[den]
                pair_tag = f"{_safe_label(num_lab)}_over_{_safe_label(den_lab)}"

                # Resolve δ reference like do_plot does
                delta_ref = None
                if delta_text.value.strip():
                    try: delta_ref = float(delta_text.value)
                    except ValueError: pass
                else:
                    auto, _ = _lookup_ref(num_lab, den_lab)
                    if auto is not None: delta_ref = auto

                common_ratio_kw = {
                    'numerator': num, 'denominator': den,
                    'min_counts': min_counts.value if min_counts.value > 0 else None,
                    'max_rel_err': max_rel.value if max_rel.value > 0 else None,
                }

                # Build a list of (label, panels-spec, grid, extra-suffix) jobs
                # from ticked checkboxes. Each job becomes one exported file.
                jobs = []

                if cb_full_channels.value:
                    panels = [{'kind': 'channel', 'channel': i,
                               'log_scale': log_scale_cb.value}
                              for i in range(len(masses))]
                    jobs.append(('channels_all',
                                 panels, (1, len(masses)),
                                 (export_size.value, export_size.value + 1)))

                if cb_full_ratio.value:
                    panels = [
                        {'kind': 'ratio',   **common_ratio_kw},
                        {'kind': 'delta',   'delta_ref': delta_ref, **common_ratio_kw},
                        {'kind': 'sigma',   **common_ratio_kw},
                        {'kind': 'rel_err', **common_ratio_kw},
                    ]
                    jobs.append((f'ratio_all_{pair_tag}',
                                 panels, (2, 2),
                                 (export_size.value, export_size.value)))

                # Individual channels
                for i, cb in enumerate(cb_channels):
                    if cb.value:
                        panels = [{'kind': 'channel', 'channel': i,
                                   'log_scale': log_scale_cb.value}]
                        jobs.append((f'channel_{_safe_label(masses[i])}',
                                     panels, (1, 1),
                                     (export_size.value, export_size.value)))

                # Individual ratio panels
                for cb, kind, label in [
                    (cb_ratio,   'ratio',   'ratio'),
                    (cb_delta,   'delta',   'delta'),
                    (cb_sigma,   'sigma',   'sigma'),
                    (cb_rel_err, 'rel_err', 'relerr'),
                ]:
                    if cb.value:
                        spec = {'kind': kind, **common_ratio_kw}
                        if kind == 'delta':
                            spec['delta_ref'] = delta_ref
                        jobs.append((f'{label}_{pair_tag}',
                                     [spec], (1, 1),
                                     (export_size.value, export_size.value)))

                # HSI
                if cb_hsi.value:
                    nat_abund, _ = _lookup_ref(num_lab, den_lab)
                    rmin = _parse_optional_float(hsi_ratio_min.value)
                    rmax = _parse_optional_float(hsi_ratio_max.value)
                    sf = (hsi_scale_factor.value
                          if hsi_scale_factor.value > 0 else 1.0)
                    spec = {
                        'kind': 'hsi',
                        'numerator': num, 'denominator': den,
                        'intensity': hsi_intensity.value,
                        'cmap': hsi_cmap.value,
                        'cmap_reverse': hsi_cmap_reverse.value,
                        'ratio_min': rmin, 'ratio_max': rmax,
                        'scale_factor': sf,
                        'natural_abundance': nat_abund,
                        'show_natural_abundance': True,
                        'min_counts': min_counts.value if min_counts.value > 0 else None,
                    }
                    jobs.append((f'hsi_{pair_tag}',
                                 [spec], (1, 1),
                                 (export_size.value, export_size.value)))

                if not jobs:
                    print("Nothing selected — tick at least one checkbox above.")
                    return

                print(f"Exporting {len(jobs)} file(s):")
                exported_paths = []
                for tag, panels, grid, panel_size in jobs:
                    outpath = _export_outpath(tag)
                    img.save_publication(
                        panels=panels,
                        outpath=outpath,
                        grid=grid,
                        panel_size=panel_size,
                        dpi=export_dpi.value,
                        format=export_format.value,
                    )
                    exported_paths.append(outpath)

                # Optional: bundle into a zip and (in Colab) trigger download
                if zip_download_cb.value and exported_paths:
                    import os, re, zipfile
                    base = os.path.splitext(os.path.basename(img.path))[0]
                    base = re.sub(r'\s*\(\d+\)$', '', base)
                    zip_path = f"{base}_export_{export_dpi.value}dpi.zip"
                    with zipfile.ZipFile(zip_path, 'w',
                                         compression=zipfile.ZIP_DEFLATED) as zf:
                        for p in exported_paths:
                            zf.write(p, arcname=os.path.basename(p))
                    print(f"\nZipped {len(exported_paths)} file(s) → {zip_path}")
                    try:
                        from google.colab import files as colab_files
                        colab_files.download(zip_path)
                        print("Download triggered.")
                    except ImportError:
                        print(f"(Not in Colab — zip saved at {zip_path})")
                    except Exception as e:
                        print(f"(Could not trigger download: {e})")

            except Exception as e:
                import traceback
                traceback.print_exc()

    drift_button.on_click(do_drift)
    channels_button.on_click(do_channels)
    plot_button.on_click(do_plot)
    hsi_button.on_click(do_hsi)
    select_all_btn.on_click(do_select_all)
    clear_all_btn.on_click(do_clear_all)
    export_btn.on_click(do_export)

    drift_box = W.VBox([
        W.HTML("<b>Drift correction</b>"),
        drift_ref, bin_planes, bin_apply,
        W.HBox([drift_button, channels_button, log_scale_cb]),
    ])
    ratio_box = W.VBox([
        W.HTML("<b>Ratio</b>"),
        num_dd, den_dd, delta_text, min_counts, max_rel, plot_button,
    ])
    hsi_box = W.VBox([
        W.HTML("<b>HSI</b> <small>(uses Numerator / Denominator from Ratio panel)</small>"),
        hsi_intensity, hsi_cmap, hsi_cmap_reverse,
        hsi_scale_factor, hsi_ratio_min, hsi_ratio_max,
        hsi_button,
    ])

    # Export panel — checkbox-driven, one ticked checkbox = one exported file
    # Each row of channels uses up to 4 columns to keep the layout compact.
    channel_rows = []
    row = []
    for i, cb in enumerate(cb_channels):
        row.append(cb)
        if len(row) == 4:
            channel_rows.append(W.HBox(row))
            row = []
    if row:
        channel_rows.append(W.HBox(row))

    export_box = W.VBox([
        W.HTML("<b>Export (publication quality)</b>"),
        W.HBox([export_format, export_dpi, export_size, zip_download_cb]),
        W.HBox([select_all_btn, clear_all_btn, export_btn]),
        W.HTML("<i>Each ticked checkbox produces one file. "
               "Multi-panel ticks produce a grouped figure; individual ticks "
               "produce single-panel figures.</i>"),
        W.HTML("<b>Full figures (multi-panel):</b>"),
        W.HBox([cb_full_channels, cb_full_ratio]),
        W.HTML("<b>Channels (individual):</b>"),
        *channel_rows,
        W.HTML("<b>Ratio panels (individual; uses Numerator / Denominator above):</b>"),
        W.HBox([cb_ratio, cb_delta, cb_sigma, cb_rel_err]),
        W.HTML("<b>Composite:</b>"),
        cb_hsi,
    ])

    display(W.HBox([drift_box, ratio_box, hsi_box]))
    display(export_box)
    display(status_out)
    display(channels_out)
    display(hsi_out)
    display(ratio_out)
