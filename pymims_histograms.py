"""
pymims_histograms.py — pixel-intensity histograms with Gaussian Mixture
Model fitting in log-space for NanoSIMS .im data.

Companion to pymims.py / pymims_explore.py. Produces per-channel histograms
of summed-stack pixel intensities (drift-corrected if available), fits
1..k_max-component Gaussian mixtures, and reports the candidates side-by-side
so the user can judge overfitting visually as well as by BIC/AIC.

Outputs feed v0.6's rule-based ROI generator. Three threshold modes are
supported downstream:

  * counts          — raw pixel count cutoff (e.g. ≥50 counts).
                      Useful when count rates have absolute physical meaning,
                      but breaks across acquisitions of different durations.
  * percentile      — empirical quantile cutoff (e.g. top 10%).
                      Acquisition-invariant; same biological feature picks
                      out the same pixel set regardless of integration time.
  * gmm-component   — pixels belonging to a specific GMM component.
                      Both acquisition-invariant and morphology-aware: the
                      GMM identifies the labelled population, regardless of
                      where it sits on the count axis.

The fit results dict therefore reports all three sources:
  - thresholds_by_k    : crossing-point cutoffs (raw counts) for each k
  - empirical_quantiles: full quantile-table at fixed percentile breakpoints
  - components_by_k    : per-component mean / std / weight / quantile tables

Author : G. McMahon (with AI-assisted development)
Created: April 2026 (v0.6 work-in-progress)

Usage
-----
    from pymims import MimsImage
    from pymims_histograms import plot_histograms, best_thresholds

    img = MimsImage('myfile.im')
    img.drift_correct(reference='SE')

    # Side-by-side candidate fits (k=1..6) for every channel.
    result = plot_histograms(img, k_max=6)

    # Default ROI cutoffs (BIC pick, raw-count crossing points)
    thresholds = best_thresholds(result)

    # Override BIC where the side-by-side panels show overfitting:
    thresholds = best_thresholds(result, manual_k={'31P': 3, '12C 15N': 2})

    # For percentile-based rules, read from the empirical_quantiles entry:
    #   p90_31P_counts = result['31P']['empirical_quantiles'][90]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Fixed percentile grid reported for every channel. Chosen to span the
# typical ROI-selection use cases:
#   * extreme tails (1, 99) — outlier diagnostics
#   * shoulders (5, 95)     — broad-thresholding regions
#   * top-N-percent rules (90, 95, 99) — common biology cutoffs
#   * IQR (25, 75)          — distribution shape
#   * median (50)           — robust central tendency
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


# ── Core fitting routine ─────────────────────────────────────────────────────

def fit_channel_gmm(values, k_max=6, drop_zeros=True, random_state=0,
                    jitter=True):
    """
    Fit Gaussian mixtures with k = 1..k_max on log10(values).

    Parameters
    ----------
    values : 1-D array
        Pixel intensities for a single channel (summed over planes).
    k_max : int
        Maximum number of components to try (k=1..k_max).
    drop_zeros : bool
        If True, drop pixels with zero counts before log-transform.
        If False, add 0.5 offset (Anscombe-like) and keep them.
    random_state : int
        For reproducible sklearn GMM init.
    jitter : bool
        If True (default), add uniform [-0.5, +0.5) jitter to integer counts
        before log-transform. NanoSIMS data is integer-valued, and dominant
        single-count modes (e.g. lots of pixels at exactly 1) create spike
        artifacts that misspecify continuous GMM. Jitter softens these
        without biasing the distribution. Disable only if you have already
        non-integer or pre-binned data.

    Returns
    -------
    dict with keys:
      'log_values'         : 1-D array of log10(values) actually used in fit
      'frac_zeros'         : fraction of input pixels that were zero
      'frac_used'          : fraction of input pixels used in fit
      'n_pixels_used'      : count of pixels used in the fit (after zero-handling)
      'models'             : list[GaussianMixture] indexed by k-1
      'bics'               : array of BIC values (length k_max)
      'aics'               : array of AIC values (length k_max)
      'best_k_bic'         : int, k with lowest BIC
      'best_k_aic'         : int, k with lowest AIC
      'thresholds_by_k'    : dict {k -> list of crossing-point thresholds}
                              (linear units, NOT log; one entry per k)
      'empirical_quantiles': dict {percentile -> count value}
                              keys are PERCENTILES; values are raw counts.
                              Computed from the data WITHOUT jitter — these
                              are the honest quantiles a campaign user wants
                              for percentile-based ROI rules.
      'components_by_k'    : dict {k -> list[dict]}
                              For each k, a list of per-component summary
                              dicts (sorted by mean), each containing:
                                'weight'         : mixing fraction (Σ = 1)
                                'mean_counts'    : component mean (linear)
                                'std_log10'      : std-dev in log10-space
                                'mean_log10'     : mean in log10-space
                                'quantiles'      : dict {percentile -> count}
                                                   for the component-conditional
                                                   distribution. Use these for
                                                   "top 10% within the labelled
                                                   population" style rules.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for GMM fitting. Install with: "
            "pip install scikit-learn (add --break-system-packages on Crostini)"
        )

    v = np.asarray(values, dtype=float).ravel()
    n_total = v.size

    # Handle zeros
    if drop_zeros:
        mask = v > 0
        frac_zeros = float(np.sum(~mask) / n_total) if n_total else 0.0
        v = v[mask]
    else:
        frac_zeros = float(np.sum(v == 0) / n_total) if n_total else 0.0
        v = v + 0.5

    # Empirical quantiles — computed BEFORE jitter so they reflect the
    # actual integer count distribution. Reported in raw count units.
    if v.size:
        empirical_quantiles = {
            int(p): float(np.percentile(v, p)) for p in PERCENTILES
        }
    else:
        empirical_quantiles = {int(p): 0.0 for p in PERCENTILES}

    # Optional jitter for integer-valued counts. Done in linear space so
    # the average is preserved; we then go to log10.
    if jitter and v.size:
        rng = np.random.default_rng(random_state)
        v = v + rng.uniform(-0.5, 0.5, size=v.size)
        v = np.maximum(v, 1e-3)   # guard against negative values from jitter

    if v.size < 50:
        # Not enough data to fit anything meaningful
        return {
            'log_values': np.log10(v) if v.size else np.array([]),
            'frac_zeros': frac_zeros,
            'frac_used': v.size / n_total if n_total else 0.0,
            'n_pixels_used': int(v.size),
            'models': [],
            'bics': np.array([]),
            'aics': np.array([]),
            'best_k_bic': None,
            'best_k_aic': None,
            'thresholds_by_k': {},
            'empirical_quantiles': empirical_quantiles,
            'components_by_k': {},
        }

    log_v = np.log10(v).reshape(-1, 1)

    models = []
    bics = []
    aics = []
    thresholds_by_k = {}
    components_by_k = {}

    for k in range(1, k_max + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=random_state,
            max_iter=200,
            n_init=2,        # two random inits, keep best
        )
        gm.fit(log_v)
        models.append(gm)
        bics.append(gm.bic(log_v))
        aics.append(gm.aic(log_v))

        # Crossing-point thresholds (in linear units, where v lives)
        thresholds_by_k[k] = _crossing_thresholds(gm, log_v)
        # Per-component summaries (weight, mean, std, conditional quantiles)
        components_by_k[k] = _component_summaries(gm)

    bics = np.asarray(bics)
    aics = np.asarray(aics)

    return {
        'log_values': log_v.ravel(),
        'frac_zeros': frac_zeros,
        'frac_used': v.size / n_total if n_total else 0.0,
        'n_pixels_used': int(v.size),
        'models': models,
        'bics': bics,
        'aics': aics,
        'best_k_bic': int(np.argmin(bics)) + 1,
        'best_k_aic': int(np.argmin(aics)) + 1,
        'thresholds_by_k': thresholds_by_k,
        'empirical_quantiles': empirical_quantiles,
        'components_by_k': components_by_k,
    }


def _crossing_thresholds(gm, log_v):
    """
    Find the equal-posterior crossing points between adjacent components,
    ordered by component mean. Returns thresholds in LINEAR units.

    For k components there are k-1 crossings. Each crossing is the value
    at which posterior probability of belonging to component i equals
    component i+1 — the natural ROI cutoff.
    """
    if gm.n_components < 2:
        return []

    # Order components by mean
    means = gm.means_.ravel()
    order = np.argsort(means)
    sorted_means = means[order]

    # Search grid spans the data range
    grid = np.linspace(log_v.min(), log_v.max(), 2000).reshape(-1, 1)
    # Per-sample posterior probabilities (n_grid, n_components)
    post = gm.predict_proba(grid)[:, order]

    thresholds = []
    for i in range(gm.n_components - 1):
        # Find where posterior(i) crosses posterior(i+1) (i.e. their argmax flips)
        diff = post[:, i] - post[:, i + 1]
        # Sign change: from + (component i wins) to - (i+1 wins)
        sign = np.sign(diff)
        flips = np.where(np.diff(sign) < 0)[0]
        if len(flips) == 0:
            # No crossing in range — components fully overlap or grid misses it.
            # Fall back to midpoint between means.
            log_thr = 0.5 * (sorted_means[i] + sorted_means[i + 1])
        else:
            # Take the crossing nearest the midpoint between the two means
            mid = 0.5 * (sorted_means[i] + sorted_means[i + 1])
            best = flips[np.argmin(np.abs(grid[flips, 0] - mid))]
            # Linear interpolate between grid[best] and grid[best+1]
            x0, x1 = grid[best, 0], grid[best + 1, 0]
            d0, d1 = diff[best], diff[best + 1]
            log_thr = x0 - d0 * (x1 - x0) / (d1 - d0)
        thresholds.append(10.0 ** log_thr)

    return thresholds


def _component_summaries(gm):
    """
    Per-component summary table for a fitted GMM.

    Components are sorted by mean (ascending), so component 0 is always the
    leftmost (lowest-count) population — typically background — and the last
    is the rightmost (highest-count) — typically the labelled population.

    For each component the within-component quantile at percentile p is the
    p-th quantile of N(μ_log, σ_log²) in log10 space, then exponentiated:
        q_p = 10 ** (μ_log + σ_log · Φ⁻¹(p/100))
    This is exact for Gaussian-in-log-space components.

    Returns
    -------
    list[dict], one per component, each containing:
        weight       : mixing fraction (fits sum to 1)
        mean_counts  : 10**μ_log — component mean in linear count units
        mean_log10   : μ_log
        std_log10    : σ_log
        quantiles    : dict {percentile -> linear count}
                       per the closed-form lognormal quantile above.
    """
    means_log = gm.means_.ravel()
    stds_log  = np.sqrt(gm.covariances_.ravel())   # full covariance, 1-D ⇒ scalar
    weights   = gm.weights_

    order = np.argsort(means_log)
    summaries = []
    for j in order:
        mu  = float(means_log[j])
        sd  = float(stds_log[j])
        wt  = float(weights[j])
        quantiles = {
            int(p): float(10.0 ** (mu + sd * norm.ppf(p / 100.0)))
            for p in PERCENTILES
        }
        summaries.append({
            'weight'      : wt,
            'mean_counts' : float(10.0 ** mu),
            'mean_log10'  : mu,
            'std_log10'   : sd,
            'quantiles'   : quantiles,
        })
    return summaries


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_histograms(img, channel=None, k_max=6, n_bins=80,
                    drop_zeros=True, corrected=True, jitter=True,
                    figsize_per_panel=(2.8, 2.4), outpath=None,
                    show=True, verbose=True):
    """
    Per-channel histograms with side-by-side GMM candidate fits (k=1..k_max).

    Layout: one row per channel, k_max+1 columns. The first k_max columns
    show the histogram with the k=1, 2, …, k_max GMM overlaid; the final
    column plots BIC and AIC vs k for that channel. The min-BIC k is
    highlighted in the column headers.

    Parameters
    ----------
    img : MimsImage
        Loaded MimsImage instance.
    channel : str, int, or None
        Channel to plot. None → all channels.
    k_max : int
        Maximum number of GMM components to try.
    n_bins : int
        Number of histogram bins (in log-space).
    drop_zeros : bool
        Drop zero-count pixels before fitting (recommended).
    corrected : bool
        Use drift-corrected stack if available.
    figsize_per_panel : (w, h)
        Inches per panel. Total figure scales with n_channels and k_max.
    outpath : str or None
        If given, save the figure.
    show : bool
        If True, return the figure for inline display.
    verbose : bool
        Print BIC table and threshold summary.

    Returns
    -------
    dict {channel_label: fit_result_dict}
        Per fit_channel_gmm() return values, one entry per channel plotted.
    """
    # Resolve channels to plot
    if channel is None:
        channels = list(range(len(img.masses)))
    elif isinstance(channel, (list, tuple)):
        channels = [img._resolve_channel(c) for c in channel]
    else:
        channels = [img._resolve_channel(channel)]

    n_rows = len(channels)
    n_cols = k_max + 1  # k_max GMM panels + 1 BIC/AIC panel

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        facecolor='white',
    )

    summed = img.sum_stack(corrected=corrected)  # (n_masses, H, W)
    results = {}

    for row, ch_idx in enumerate(channels):
        label = img.masses[ch_idx]
        pixel_vals = summed[ch_idx].ravel()
        fit = fit_channel_gmm(pixel_vals, k_max=k_max, drop_zeros=drop_zeros,
                              jitter=jitter)
        results[label] = fit

        if len(fit['models']) == 0:
            # Insufficient data — annotate and skip
            for col in range(n_cols):
                ax = axes[row][col]
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            axes[row][0].set_ylabel(label, fontsize=10, fontweight='bold')
            continue

        log_v = fit['log_values']
        bics = fit['bics']
        aics = fit['aics']
        best_k = fit['best_k_bic']

        # Clip the visible range to avoid a few extreme outliers stretching
        # the axis and squashing the bulk distribution into a sliver. The
        # fit itself uses ALL pixels — only the plotting range is clipped.
        lo, hi = np.percentile(log_v, [0.5, 99.5])
        # Pad slightly so the tails stay visible
        span = hi - lo
        x_lo, x_hi = lo - 0.05 * span, hi + 0.05 * span

        # Histogram density (so GMM PDF overlays line up)
        bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)
        hist_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Restrict the histogram to the visible range so density is correct
        log_v_clip = log_v[(log_v >= x_lo) & (log_v <= x_hi)]
        counts, _ = np.histogram(log_v_clip, bins=bin_edges, density=True)

        # ── GMM panels (one per k) ──────────────────────────────────────────
        x_grid = np.linspace(x_lo, x_hi, 500)
        for k in range(1, k_max + 1):
            ax = axes[row][k - 1]
            ax.bar(hist_centres, counts,
                   width=(bin_edges[1] - bin_edges[0]),
                   color='#cccccc', edgecolor='none', alpha=0.85,
                   align='center')

            gm = fit['models'][k - 1]
            weights = gm.weights_
            means = gm.means_.ravel()
            stds = np.sqrt(gm.covariances_.ravel())

            # Order components by mean for a consistent colour mapping
            order = np.argsort(means)
            cmap = plt.get_cmap('tab10')

            # Total mixture density
            total_pdf = np.zeros_like(x_grid)
            for j_pos, j in enumerate(order):
                comp_pdf = weights[j] * norm.pdf(x_grid, means[j], stds[j])
                total_pdf += comp_pdf
                ax.plot(x_grid, comp_pdf, color=cmap(j_pos), linewidth=1.2,
                        alpha=0.85)
            ax.plot(x_grid, total_pdf, color='black', linewidth=1.3,
                    linestyle='--', alpha=0.9)

            # Mark crossing-point thresholds in log space (only those in view)
            for thr in fit['thresholds_by_k'][k]:
                lt = np.log10(thr)
                if x_lo <= lt <= x_hi:
                    ax.axvline(lt, color='red', linewidth=0.8,
                               linestyle=':', alpha=0.7)

            # Title with k and BIC. Highlight best_k.
            star = ' ★' if k == best_k else ''
            ax.set_title(f'k = {k}{star}\nBIC = {bics[k-1]:.0f}',
                         fontsize=9,
                         color='darkred' if k == best_k else 'black')
            ax.set_xlim(x_lo, x_hi)
            ax.set_yticks([])
            # Show useful x-tick positions on the bottom row only
            if row == n_rows - 1:
                tick_logs = np.array([0, 1, 2, 3, 4])
                tick_logs = tick_logs[(tick_logs >= x_lo) & (tick_logs <= x_hi)]
                ax.set_xticks(tick_logs)
                ax.set_xticklabels([f'$10^{{{int(t)}}}$' for t in tick_logs],
                                   fontsize=7)
            else:
                ax.set_xticks([])
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)

        # Channel label on leftmost panel
        axes[row][0].set_ylabel(label, fontsize=10, fontweight='bold')

        # ── BIC / AIC summary panel ─────────────────────────────────────────
        ax = axes[row][n_cols - 1]
        ks = np.arange(1, k_max + 1)
        # Plot ΔBIC and ΔAIC (relative to best) — absolute values uninterpretable
        d_bic = bics - bics.min()
        d_aic = aics - aics.min()
        ax.plot(ks, d_bic, 'o-', color='steelblue', label='ΔBIC', linewidth=1.3)
        ax.plot(ks, d_aic, 's--', color='darkorange', label='ΔAIC',
                linewidth=1.0, alpha=0.8)
        ax.axhline(10, color='gray', linewidth=0.6, linestyle=':')
        ax.axvline(best_k, color='darkred', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('k', fontsize=8)
        ax.set_ylabel('Δ from best', fontsize=8)
        ax.set_title(f'best k = {best_k}', fontsize=9, color='darkred')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right', frameon=False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Bottom-row x-axis label (counts on log scale)
    for col in range(k_max):
        axes[-1][col].set_xlabel('counts (log scale)', fontsize=8)

    fig.suptitle(
        f'Pixel-intensity histograms with GMM candidates  '
        f'(red dotted lines = crossing-point thresholds; ★ = min-BIC)',
        fontsize=10, y=1.0,
    )
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {outpath}')

    if not show:
        plt.close(fig)

    # Verbose summary
    if verbose:
        _print_summary(results)

    return results


def _print_summary(results):
    """Tabular print of BIC table, GMM thresholds, empirical quantiles,
    and per-component summaries for each channel."""
    print()
    for label, fit in results.items():
        if len(fit['models']) == 0:
            print(f'  {label:20s}  no fit (insufficient data)')
            continue
        bics = fit['bics']
        d_bic = bics - bics.min()
        best = fit['best_k_bic']
        zero_pct = fit['frac_zeros'] * 100.0
        n_used = fit.get('n_pixels_used', '?')
        print(f'  {label}   best k={best}  '
              f'zero pixels: {zero_pct:.1f}%   '
              f'pixels in fit: {n_used}')
        # ΔBIC table
        bic_str = '    ΔBIC: ' + '  '.join(
            f'k={k}:{d:.0f}' for k, d in zip(range(1, len(bics) + 1), d_bic)
        )
        print(bic_str)
        # Crossing-point thresholds for the best k
        thrs = fit['thresholds_by_k'][best]
        if thrs:
            thr_str = ', '.join(f'{t:.1f}' for t in thrs)
            print(f'    GMM crossings (counts) at best k: {thr_str}')
        # Empirical quantiles (acquisition-invariant percentile cutoffs)
        eq = fit['empirical_quantiles']
        eq_str = '    Empirical quantiles (counts):  ' + '  '.join(
            f'p{p}={v:.1f}' for p, v in eq.items()
        )
        print(eq_str)
        # Per-component breakdown at best k
        comps = fit['components_by_k'].get(best, [])
        if comps:
            print(f'    Components at k={best} (sorted by mean):')
            print(f'      {"#":>2} {"weight":>7} {"mean":>10} '
                  f'{"p5":>9} {"p50":>9} {"p95":>9}')
            for i, c in enumerate(comps):
                q = c['quantiles']
                print(f'      {i:>2} {c["weight"]:>7.1%} '
                      f'{c["mean_counts"]:>10.2f} '
                      f'{q[5]:>9.2f} {q[50]:>9.2f} {q[95]:>9.2f}')
        print()


# ── Convenience: a single best-fit threshold table ──────────────────────────

def best_thresholds(results, criterion='bic', manual_k=None):
    """
    Reduce a plot_histograms() result dict to a flat
    {channel_label: list_of_thresholds} mapping.

    Parameters
    ----------
    results : dict
        Output of plot_histograms().
    criterion : 'bic' or 'aic'
        Which criterion picks the default best k for each channel.
    manual_k : dict or None
        Per-channel override, e.g. {'31P': 3, '12C 15N': 2}. Channels not
        in the dict fall back to the criterion-selected best_k. Use this
        after eyeballing the side-by-side panels to override the BIC verdict
        on channels where it overfits.

    Returns
    -------
    dict {channel_label: list_of_thresholds_in_counts}
    """
    key = 'best_k_bic' if criterion == 'bic' else 'best_k_aic'
    manual_k = manual_k or {}
    out = {}
    for label, fit in results.items():
        if not fit['models']:
            out[label] = []
            continue
        k = manual_k.get(label, fit[key])
        if k not in fit['thresholds_by_k']:
            raise ValueError(
                f"manual_k for {label!r} = {k} is outside fitted range "
                f"(1..{len(fit['models'])})"
            )
        out[label] = fit['thresholds_by_k'][k]
    return out
