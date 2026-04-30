"""
pymims_rules.py — Rule-based ROI generation for NanoSIMS .im data.

Builds boolean pixel masks ("ROIs") from threshold rules expressed in
three different threshold modes:

  * 'counts'       — raw pixel-count cutoffs (e.g. ≥ 50 counts of ¹²C¹⁴N)
  * 'percentile'   — empirical-percentile cutoffs (e.g. top 10% of ¹²C¹⁴N)
  * 'gmm-component'— pixels assigned to a specific GMM component
                     (consumes the per-channel fits from
                     pymims_histograms.plot_histograms)

Multiple rules are combined with AND or OR logic. The output is a
boolean image per rule plus a `combined` boolean image, all sharing the
same shape as the input acquisition. These masks feed downstream ROI
statistics (per-region counts, ratios, depth profiles).

Author : G. McMahon (with AI-assisted development)
Created: April 2026 (v0.7 work-in-progress)

Usage
-----
    from pymims import MimsImage
    from pymims_histograms import plot_histograms
    from pymims_rules import build_roi_masks

    img = MimsImage('myfile.im')
    img.drift_correct(reference='SE')

    # Optional: pre-compute GMM fits if any rule uses 'gmm-component'.
    histograms = plot_histograms(img, k_max=6, show=False, verbose=False)

    rois = build_roi_masks(
        img,
        rules=[
            {'channel': '32S',     'mode': 'percentile',    'cutoff': 90},
            {'channel': '12C 14N', 'mode': 'counts',        'cutoff': 200},
            {'channel': '31P',     'mode': 'gmm-component', 'k': 3,
                                   'component': 'highest'},
        ],
        combine='AND',
        histograms=histograms,        # required if any rule uses gmm-component
    )

    # rois['rule_0']  → bool mask for the first rule
    # rois['rule_1']  → bool mask for the second
    # rois['combined'] → AND/OR combination, the final ROI

Visualisation:

    from pymims_rules import plot_rule_masks
    plot_rule_masks(img, rois, base='channel', channel='12C 14N')
"""

import numpy as np
import matplotlib.pyplot as plt


VALID_MODES = ('counts', 'percentile', 'gmm-component')
VALID_COMBINES = ('AND', 'OR')


# ── Threshold-mode mask builders ─────────────────────────────────────────────

def _mask_counts(img, channel, cutoff, comparison='>='):
    """
    Mask: pixels where the channel's summed counts satisfy a comparison
    against an absolute count threshold.

    Parameters
    ----------
    comparison : '>=', '>', '<=', '<'
        Default '>=' (top-end thresholding, the common case).
    """
    ch_idx = img._resolve_channel(channel)
    counts = img.sum_stack(corrected=True)[ch_idx]
    if comparison == '>=':
        return counts >= cutoff
    elif comparison == '>':
        return counts > cutoff
    elif comparison == '<=':
        return counts <= cutoff
    elif comparison == '<':
        return counts < cutoff
    else:
        raise ValueError(f"comparison must be one of >=, >, <=, <; "
                         f"got {comparison!r}")


def _mask_percentile(img, channel, cutoff, comparison='>='):
    """
    Mask: pixels above (or below) a percentile of the channel's
    count distribution.

    Parameters
    ----------
    cutoff : float
        Percentile in [0, 100]. cutoff=90 with comparison='>=' picks the
        top 10% of pixels.
    comparison : '>=', '>', '<=', '<'
    """
    if not 0 <= cutoff <= 100:
        raise ValueError(f"percentile cutoff must be in [0, 100]; got {cutoff}")
    ch_idx = img._resolve_channel(channel)
    counts = img.sum_stack(corrected=True)[ch_idx].astype(float)
    threshold = float(np.percentile(counts.ravel(), cutoff))
    if comparison == '>=':
        return counts >= threshold
    elif comparison == '>':
        return counts > threshold
    elif comparison == '<=':
        return counts <= threshold
    elif comparison == '<':
        return counts < threshold
    else:
        raise ValueError(f"comparison must be one of >=, >, <=, <; "
                         f"got {comparison!r}")


def _mask_gmm_component(img, channel, k, component, histograms):
    """
    Mask: pixels assigned to a specific GMM component for a given channel.

    Parameters
    ----------
    k : int
        Cluster count to use from the GMM fit (any value present in
        the histograms[channel]['components_by_k'] dict).
    component : int or 'highest' or 'lowest'
        Which component to extract. Components are sorted by mean
        ascending, so 0 is lowest-count, k-1 is highest. The 'highest'
        and 'lowest' string shortcuts are convenient for the common case.
    histograms : dict
        Output of pymims_histograms.plot_histograms(), needed for the
        component thresholds. Must contain an entry for `channel`.
    """
    if histograms is None:
        raise ValueError("'gmm-component' rules require histograms= "
                         "(output of plot_histograms).")
    ch_label = (channel if isinstance(channel, str)
                else img.masses[img._resolve_channel(channel)])
    if ch_label not in histograms:
        # Try fuzzy match against histogram keys
        match = None
        for key in histograms.keys():
            if key.replace(' ', '') == ch_label.replace(' ', ''):
                match = key; break
        if match is None:
            raise ValueError(
                f"Channel {ch_label!r} not in histograms dict; "
                f"available: {list(histograms.keys())}"
            )
        ch_label = match

    fit = histograms[ch_label]
    if k not in fit['thresholds_by_k']:
        raise ValueError(
            f"k={k} not fit for channel {ch_label!r}; "
            f"available: {sorted(fit['thresholds_by_k'])}"
        )
    thresholds = fit['thresholds_by_k'][k]   # k-1 cuts in raw counts

    # Resolve the component index
    if component == 'highest':
        comp_idx = k - 1
    elif component == 'lowest':
        comp_idx = 0
    elif isinstance(component, int):
        if not 0 <= component < k:
            raise ValueError(f"component={component} out of range [0, {k}-1]")
        comp_idx = component
    else:
        raise ValueError(f"component must be int, 'highest', or 'lowest'; "
                         f"got {component!r}")

    # The component lives between two crossings (or at the edges).
    # thresholds[i] separates component i from component i+1.
    lower = thresholds[comp_idx - 1] if comp_idx > 0 else -np.inf
    upper = thresholds[comp_idx]     if comp_idx < k - 1 else np.inf

    ch_idx = img._resolve_channel(ch_label)
    counts = img.sum_stack(corrected=True)[ch_idx].astype(float)
    return (counts >= lower) & (counts < upper)


# ── Top-level rule application ───────────────────────────────────────────────

def build_roi_masks(img, rules, combine='AND', histograms=None):
    """
    Build boolean ROI masks from a list of threshold rules.

    Parameters
    ----------
    img : MimsImage
        Drift-corrected image.
    rules : list[dict]
        Each rule is a dict with at minimum:
          'channel'    : channel label or index
          'mode'       : 'counts', 'percentile', or 'gmm-component'
        Plus mode-specific fields:
          counts        : 'cutoff' (float), optional 'comparison'
          percentile    : 'cutoff' (0-100), optional 'comparison'
          gmm-component : 'k' (int), 'component' (int|'highest'|'lowest')
        Optional 'name' field for labelling the output mask key
        (defaults to 'rule_{i}').
    combine : 'AND' or 'OR'
        How to combine multiple rules into the final mask. Default 'AND'
        (all rules must be satisfied at a pixel).
    histograms : dict or None
        Required when any rule uses mode='gmm-component'. Output of
        pymims_histograms.plot_histograms().

    Returns
    -------
    dict mapping rule names to 2-D bool arrays, with an extra
    'combined' entry that is the AND/OR combination of all rules.
    """
    if combine not in VALID_COMBINES:
        raise ValueError(f"combine must be 'AND' or 'OR'; got {combine!r}")
    if not rules:
        raise ValueError("rules list is empty")

    masks = {}
    individual_masks = []

    for i, rule in enumerate(rules):
        if 'channel' not in rule or 'mode' not in rule:
            raise ValueError(f"Rule {i} missing required 'channel' or 'mode'")
        if rule['mode'] not in VALID_MODES:
            raise ValueError(f"Rule {i}: mode must be one of {VALID_MODES}; "
                             f"got {rule['mode']!r}")

        name = rule.get('name', f'rule_{i}')
        channel = rule['channel']
        mode = rule['mode']

        if mode == 'counts':
            mask = _mask_counts(
                img, channel, rule['cutoff'],
                comparison=rule.get('comparison', '>='),
            )
        elif mode == 'percentile':
            mask = _mask_percentile(
                img, channel, rule['cutoff'],
                comparison=rule.get('comparison', '>='),
            )
        elif mode == 'gmm-component':
            mask = _mask_gmm_component(
                img, channel,
                k=rule['k'], component=rule['component'],
                histograms=histograms,
            )

        masks[name] = mask
        individual_masks.append(mask)

    # Combine
    if combine == 'AND':
        combined = np.logical_and.reduce(individual_masks)
    else:  # OR
        combined = np.logical_or.reduce(individual_masks)

    masks['combined'] = combined
    return masks


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_rule_masks(img, rois, base='channel', channel=None,
                    numerator=None, denominator=None,
                    cmap_base='viridis', show_individual=True,
                    contour_linewidth=1.4, panel_size=(5, 5),
                    outpath=None, show=True):
    """
    Display rule-based ROI masks as outlined regions on a base image.

    Parameters
    ----------
    img : MimsImage
    rois : dict
        Output of build_roi_masks(): {'rule_0': mask, ..., 'combined': mask}.
    base : str
        Base image to overlay on: 'channel', 'ratio', 'hsi'.
    channel, numerator, denominator : as for cluster overlay.
    cmap_base : str
        Colourmap for the base image (default 'viridis').
    show_individual : bool
        If True, draws each individual rule mask in a separate colour
        with its own outline. If False, only draws the 'combined' mask.
    contour_linewidth : float
        Outline thickness in points.
    panel_size : (w, h) tuple

    Returns
    -------
    matplotlib Figure.
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "scikit-image required for ROI mask outlines. "
            "Install with: pip install scikit-image."
        )

    field_um = img.metadata['field_um']
    fig, ax = plt.subplots(figsize=panel_size, facecolor='white')

    # ── Base image ────────────────────────────────────────────────────────
    if base == 'channel':
        if channel is None:
            channel = 0
        ch_idx = img._resolve_channel(channel)
        base_img = img.sum_stack(corrected=True)[ch_idx]
        finite = base_img[np.isfinite(base_img) & (base_img > 0)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1, 99])
        else:
            vmin, vmax = 0, 1
        ax.imshow(base_img, extent=[0, field_um, field_um, 0],
                  cmap=cmap_base, vmin=vmin, vmax=vmax,
                  interpolation='nearest')
        base_label = f"{img.masses[ch_idx]} (counts)"
    elif base == 'ratio':
        if numerator is None or denominator is None:
            raise ValueError("base='ratio' requires numerator= and denominator=")
        result = img.ratio(numerator, denominator)
        R = result['ratio']
        finite = R[np.isfinite(R)]
        vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))
        ax.imshow(R, extent=[0, field_um, field_um, 0],
                  cmap=cmap_base, vmin=vmin, vmax=vmax,
                  interpolation='nearest')
        base_label = f"Ratio {result['num_label']}/{result['den_label']}"
    else:
        raise ValueError(f"base must be 'channel' or 'ratio'; got {base!r}")

    # ── Overlay rule outlines ──────────────────────────────────────────────
    # Distinct colours per rule (high-separation palette like the cluster module)
    rule_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00',
                    '#984ea3', '#a65628']
    H, W = img.metadata['height'], img.metadata['width']
    # Half-pixel offset to align find_contours' pixel-centre coordinates
    # with imshow's extent (which places pixel edges at 0..field_um, so
    # pixel centres sit at +0.5*um_per_px). See pymims_clustering.plot_overlay
    # for the full discussion.
    um_per_px_x = field_um / W
    um_per_px_y = field_um / H

    rule_keys = [k for k in rois.keys() if k != 'combined']

    if show_individual:
        for i, key in enumerate(rule_keys):
            mask = rois[key]
            colour = rule_palette[i % len(rule_palette)]
            contours = measure.find_contours(mask.astype(float), 0.5)
            for ctr in contours:
                ys = (ctr[:, 0] + 0.5) * um_per_px_y
                xs = (ctr[:, 1] + 0.5) * um_per_px_x
                ax.plot(xs, ys, color=colour, linewidth=contour_linewidth,
                        alpha=0.85, label=None)
            # Inline legend entry
            ax.text(0.02, 0.97 - i * 0.05, f'━━ {key} ({100*mask.sum()/mask.size:.1f}%)',
                    transform=ax.transAxes, fontsize=8, color=colour,
                    fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.85, pad=2))

    # Combined mask in white, thicker, on top
    combined = rois['combined']
    contours = measure.find_contours(combined.astype(float), 0.5)
    for ctr in contours:
        ys = (ctr[:, 0] + 0.5) * um_per_px_y
        xs = (ctr[:, 1] + 0.5) * um_per_px_x
        ax.plot(xs, ys, color='white', linewidth=contour_linewidth + 0.6,
                alpha=0.95)
    n_individual = len(rule_keys)
    ax.text(0.02, 0.97 - n_individual * 0.05,
            f'━━ combined ({100*combined.sum()/combined.size:.1f}%)',
            transform=ax.transAxes, fontsize=8, color='white',
            fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='black', edgecolor='none', alpha=0.7, pad=2))

    ax.set_title(f"{base_label}  +  ROI rule masks", fontsize=11,
                 fontweight='bold', pad=8)
    ax.set_xlabel('μm')
    ax.set_ylabel('μm')
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {outpath}')
    if not show:
        plt.close(fig)
    return fig


# ── ROI statistics ───────────────────────────────────────────────────────────

def roi_statistics(img, rois, channels=None):
    """
    Compute summary statistics for each ROI in the given mask dict.

    Parameters
    ----------
    img : MimsImage
    rois : dict
        Output of build_roi_masks() — one bool mask per rule + 'combined'.
    channels : list or None
        Which channels to summarise. None = all channels.

    Returns
    -------
    dict of dicts: {roi_name: {channel_label: {n_pixels, mean, total,
                                                p5, p50, p95}}}
    """
    if channels is None:
        channels = list(range(len(img.masses)))
    ch_indices = [img._resolve_channel(c) for c in channels]
    ch_labels  = [img.masses[i] for i in ch_indices]
    summed = img.sum_stack(corrected=True)

    out = {}
    for roi_name, mask in rois.items():
        if not mask.any():
            out[roi_name] = {'n_pixels': 0}
            continue
        roi_data = {'n_pixels': int(mask.sum()),
                    'fraction': float(mask.sum() / mask.size)}
        for ch_idx, ch_lbl in zip(ch_indices, ch_labels):
            counts_in_roi = summed[ch_idx][mask]
            roi_data[ch_lbl] = {
                'mean':  float(counts_in_roi.mean()),
                'total': int(counts_in_roi.sum()),
                'p5':    float(np.percentile(counts_in_roi, 5)),
                'p50':   float(np.percentile(counts_in_roi, 50)),
                'p95':   float(np.percentile(counts_in_roi, 95)),
            }
        out[roi_name] = roi_data
    return out


def print_roi_summary(stats):
    """Tabular print of roi_statistics() output."""
    for roi_name, data in stats.items():
        if data.get('n_pixels', 0) == 0:
            print(f"  {roi_name:20s}  empty (no pixels)")
            continue
        n = data['n_pixels']
        frac = data.get('fraction', 0) * 100
        print(f"\n  {roi_name}: {n:,} pixels ({frac:.1f}%)")
        print(f"    {'channel':>20} {'mean':>10} {'total':>12} "
              f"{'p5':>8} {'p50':>8} {'p95':>8}")
        for key, val in data.items():
            if key in ('n_pixels', 'fraction'):
                continue
            print(f"    {key:>20} {val['mean']:>10.1f} "
                  f"{val['total']:>12,} "
                  f"{val['p5']:>8.1f} {val['p50']:>8.1f} {val['p95']:>8.1f}")
