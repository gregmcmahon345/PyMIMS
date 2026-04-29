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
                    jitter=True, tail_weight_threshold=0.10):
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
    tail_weight_threshold : float, default 0.10
        Threshold for raising a tail-warning. If incrementing the sensible_k
        pick by 1 would introduce a new component whose mixing weight is
        below this fraction AND whose mean is higher than every component
        at the current sensible_k, a warning is raised. Catches cases where
        rare high-count populations (e.g. Fe:S clusters in mitochondria,
        hot-spot enrichment, isolated organelles) would be suppressed by
        the conservative default. Lower values catch rarer features at the
        cost of more false alarms. 0.10 is a conservative default; 0.05 is
        appropriate when you expect rare biological features.

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
      'tail_warning'       : None or dict. When non-None, indicates that
                              k=sensible_k+1 would reveal a rare high-count
                              component the conservative pick is suppressing.
                              Dict contains:
                                'suggested_k'   : int — k+1
                                'weight'        : float — fraction of pixels
                                                  in the new component
                                'mean_counts'   : float — its mean
                                'p5','p50','p95': float — its quantiles
                              If you see this on a channel where rare
                              high-count features matter (Fe:S clusters,
                              hot organelles), override with manual_k.
      'sensible_k'         : int, conservative-consensus pick from elbow
                              heuristics (largest-drop ∩ kneedle); favoured
                              over best_k_bic when ΔBIC has a long shallow
                              plateau (the "BIC overfitting" failure mode).
      'largest_drop_k'     : int, pick from the largest-ΔBIC-drop heuristic
      'kneedle_k'          : int, pick from the kneedle algorithm
      'heuristics_agree'   : bool, whether the two elbow methods agreed.
                              Disagreement is itself a useful signal —
                              eyeball the side-by-side panels.
      'unique_k_recommendations' : list[dict], one entry per unique k
                              picked by any of the three methods (BIC,
                              largest-drop, kneedle), ordered ascending
                              by k. Each entry has:
                                'k'       : int — the recommended k
                                'methods' : list[str] — which of
                                            ['bic', 'largest_drop',
                                            'kneedle'] picked this k
                              When all three agree there is one entry;
                              when they all disagree there are three.
                              Used for "show me a table per recommended k"
                              workflows where the disagreement itself is
                              what you want to inspect.
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
            'sensible_k': None,
            'largest_drop_k': None,
            'kneedle_k': None,
            'heuristics_agree': None,
            'unique_k_recommendations': [],
            'tail_warning': None,
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
    elbow = _sensible_k(bics, k_max)
    bic_k = int(np.argmin(bics)) + 1
    unique_recs = _unique_k_recommendations(
        bic_k=bic_k,
        sensible_k=elbow['sensible_k'],
        largest_drop_k=elbow['largest_drop_k'],
        kneedle_k=elbow['kneedle_k'],
    )
    tail_warning = _detect_tail_warning(
        components_by_k=components_by_k,
        sensible_k=elbow['sensible_k'],
        k_max=k_max,
        weight_threshold=tail_weight_threshold,
    )

    return {
        'log_values': log_v.ravel(),
        'frac_zeros': frac_zeros,
        'frac_used': v.size / n_total if n_total else 0.0,
        'n_pixels_used': int(v.size),
        'models': models,
        'bics': bics,
        'aics': aics,
        'best_k_bic': bic_k,
        'best_k_aic': int(np.argmin(aics)) + 1,
        'sensible_k'      : elbow['sensible_k'],
        'largest_drop_k'  : elbow['largest_drop_k'],
        'kneedle_k'       : elbow['kneedle_k'],
        'heuristics_agree': elbow['heuristics_agree'],
        'unique_k_recommendations': unique_recs,
        'tail_warning'     : tail_warning,
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


# ── Elbow / "sensible_k" heuristics ──────────────────────────────────────────

def _elbow_largest_drop(bics):
    """
    'Largest drop' elbow: pick k where the next ΔBIC drop becomes small.

    Looks at the first differences of BIC (bics[k+1] - bics[k]) — these are
    negative when adding a component improves the fit. Picks k+1 where the
    drop is largest, on the principle that the dominant elbow is where
    further components start adding marginal value.

    Returns the chosen k (1-indexed). If only one or two values exist,
    returns the index of the minimum.
    """
    bics = np.asarray(bics, dtype=float)
    if bics.size < 2:
        return int(np.argmin(bics)) + 1
    drops = np.diff(bics)               # bics[k+1] - bics[k]; negative = improvement
    most_improving = int(np.argmin(drops))   # the largest negative
    return most_improving + 2            # +1 for 0-index, +1 because diff shifts


def _elbow_kneedle(bics, k_values=None):
    """
    Kneedle elbow detection (Satopää 2011) on the BIC curve.

    Uses the `kneed` package if available; otherwise falls back to a
    minimal in-house implementation that finds the maximum perpendicular
    distance from the line connecting the first and last points.

    Returns the chosen k (1-indexed), or None if a knee cannot be found.
    """
    bics = np.asarray(bics, dtype=float)
    if bics.size < 3:
        return int(np.argmin(bics)) + 1   # too short to find a knee
    if k_values is None:
        k_values = np.arange(1, bics.size + 1)

    # Try the kneed library first (proper kneedle algorithm)
    try:
        from kneed import KneeLocator
        # BIC decreases (mostly) then plateaus → curve='convex', direction='decreasing'
        kl = KneeLocator(k_values, bics, curve='convex',
                         direction='decreasing', S=1.0)
        if kl.knee is not None:
            return int(kl.knee)
    except ImportError:
        pass   # fall through to in-house version
    except Exception:
        pass

    # Fallback: maximum perpendicular distance from the chord connecting
    # the first and last points. Robust, parameter-free, and doesn't need
    # any external library.
    x = k_values.astype(float)
    y = bics
    # Normalise both axes to [0, 1] so the distance is geometrically
    # comparable.
    if x.max() == x.min() or y.max() == y.min():
        return int(np.argmin(bics)) + 1
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # Vector from first to last
    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    line = p2 - p1
    line_norm = line / (np.linalg.norm(line) + 1e-12)
    # Perpendicular distance for each point
    points = np.column_stack([xn, yn]) - p1
    proj = points @ line_norm
    proj_vec = np.outer(proj, line_norm)
    perp = points - proj_vec
    dists = np.linalg.norm(perp, axis=1)
    return int(k_values[np.argmax(dists)])


def _sensible_k(bics, k_max):
    """
    Combine largest-drop and kneedle elbow heuristics into a single
    'sensible_k' recommendation.

    Returns
    -------
    dict with:
      'sensible_k'      : int — consensus pick (smaller of the two when they
                                disagree, on the conservative principle that
                                fewer components is safer)
      'largest_drop_k'  : int — pick from the largest-drop heuristic
      'kneedle_k'       : int — pick from the kneedle algorithm
      'heuristics_agree': bool — True iff the two methods picked the same k
    """
    k_values = np.arange(1, len(bics) + 1)
    drop_k = _elbow_largest_drop(bics)
    knee_k = _elbow_kneedle(bics, k_values)
    agree = (drop_k == knee_k)
    consensus = min(drop_k, knee_k)   # conservative consensus
    return {
        'sensible_k'      : consensus,
        'largest_drop_k'  : drop_k,
        'kneedle_k'       : knee_k,
        'heuristics_agree': agree,
    }


def _unique_k_recommendations(bic_k, sensible_k, largest_drop_k, kneedle_k):
    """
    Build a deduplicated, ordered list of k recommendations from BIC,
    largest-drop, and kneedle. The 'sensible_k' value is just the
    consensus of the two elbow heuristics, so it is implicitly covered
    by largest_drop_k and kneedle_k — we don't list it separately.

    Returns
    -------
    list[dict] sorted ascending by k, each entry containing:
        'k'       : int
        'methods' : list[str] — labels from ['bic', 'largest_drop',
                    'kneedle'] indicating which method(s) picked this k.

    When the methods all agree, the list has one entry. When they all
    disagree, the list has three.
    """
    # Each method contributes one (label, k) pair.
    picks = []
    if bic_k is not None:
        picks.append(('bic', bic_k))
    if largest_drop_k is not None:
        picks.append(('largest_drop', largest_drop_k))
    if kneedle_k is not None:
        picks.append(('kneedle', kneedle_k))

    # Group by k value
    by_k = {}
    for method, k in picks:
        by_k.setdefault(k, []).append(method)

    return [{'k': k, 'methods': by_k[k]} for k in sorted(by_k)]


def _detect_tail_warning(components_by_k, sensible_k, k_max,
                         weight_threshold=0.10):
    """
    Check whether incrementing sensible_k would reveal a low-weight,
    high-mean component the conservative pick is suppressing.

    The signature of "you're hiding a rare hot population":
      * k+1 introduces a NEW component (not present at k=sensible_k)
      * its mixing weight is below `weight_threshold`
      * its mean count is HIGHER than every component at k=sensible_k

    "New component at k+1 not present at k" is a fuzzy concept since
    sklearn refits each k from scratch — components don't have stable
    identities across k. We approximate it as: the highest-mean component
    at k+1 has a higher mean than the highest-mean component at k. That
    catches the "tail revealed by adding a component" case while not
    triggering when k+1 just splits an existing low-count component.

    Returns
    -------
    None if no warning. Otherwise a dict:
        'suggested_k'  : int — sensible_k + 1
        'weight'       : float — mixing fraction of the suppressed component
        'mean_counts'  : float — its linear-space mean
        'p5','p50','p95': float — quantiles of the suppressed component
    """
    if sensible_k is None or sensible_k >= k_max:
        return None
    comps_at_k   = components_by_k.get(sensible_k, [])
    comps_at_kp1 = components_by_k.get(sensible_k + 1, [])
    if not comps_at_k or not comps_at_kp1:
        return None

    max_mean_at_k = max(c['mean_counts'] for c in comps_at_k)
    # Top component at k+1 is the last (sorted by mean ascending)
    top_kp1 = comps_at_kp1[-1]

    # Trigger condition: top component at k+1 has a higher mean AND a low
    # mixing weight. Both must hold; otherwise k+1 is just splitting an
    # existing component rather than revealing a new tail.
    if (top_kp1['mean_counts'] > max_mean_at_k
            and top_kp1['weight'] < weight_threshold):
        q = top_kp1['quantiles']
        return {
            'suggested_k' : sensible_k + 1,
            'weight'      : top_kp1['weight'],
            'mean_counts' : top_kp1['mean_counts'],
            'p5'          : q[5],
            'p50'         : q[50],
            'p95'         : q[95],
        }
    return None


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_histograms(img, channel=None, k_max=6, n_bins=80,
                    drop_zeros=True, corrected=True, jitter=True,
                    tail_weight_threshold=0.10,
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
                              jitter=jitter,
                              tail_weight_threshold=tail_weight_threshold)
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
        sensible = fit['sensible_k']
        drop_k_local = fit['largest_drop_k']
        knee_k_local = fit['kneedle_k']

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

            # Title with k and BIC. Marker letters indicate which methods
            # picked this k:
            #   B = min-BIC,  D = largest-drop,  K = kneedle
            # The k chosen as 'sensible' (default for ROIs) gets a green title.
            picked_by = []
            if k == best_k:        picked_by.append('B')
            if k == drop_k_local:  picked_by.append('D')
            if k == knee_k_local:  picked_by.append('K')
            markers = f'  [{",".join(picked_by)}]' if picked_by else ''
            if k == sensible:
                title_color = 'darkgreen'   # default for ROI rules
            elif picked_by:
                title_color = 'darkred'     # picked by some method but not sensible
            else:
                title_color = 'black'
            ax.set_title(f'k = {k}{markers}\nBIC = {bics[k-1]:.0f}',
                         fontsize=9, color=title_color)
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

        # (Tail warnings are drawn after layout finalises so coordinates
        # are correct — see the deferred block below this loop.)

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
        # Mark each unique recommendation with a vertical line. The sensible
        # pick is drawn last/thickest in green; others are thinner red.
        unique_ks = sorted({best_k, drop_k_local, knee_k_local})
        for kv in unique_ks:
            if kv == sensible:
                ax.axvline(kv, color='darkgreen', linewidth=1.4, alpha=0.85)
            else:
                ax.axvline(kv, color='darkred', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('k', fontsize=8)
        ax.set_ylabel('Δ from best', fontsize=8)
        # Title: all-agree, two-disagree, or three-way fork
        if len(unique_ks) == 1:
            title_str = f'k = {sensible}  (all agree)'
            title_color = 'darkgreen'
        elif len(unique_ks) == 2:
            other = [kv for kv in unique_ks if kv != sensible][0]
            title_str = f'use k={sensible}   (other: {other})'
            title_color = 'darkgreen'
        else:
            title_str = (f'use k={sensible}   '
                         f'(B={best_k}  D={drop_k_local}  K={knee_k_local})')
            title_color = 'darkgreen'
        ax.set_title(title_str, fontsize=9, color=title_color)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right', frameon=False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Bottom-row x-axis label (counts on log scale)
    for col in range(k_max):
        axes[-1][col].set_xlabel('counts (log scale)', fontsize=8)

    fig.suptitle(
        f'Pixel-intensity histograms with GMM candidates  '
        f'(red dotted lines = crossing thresholds; '
        f'B = min-BIC, D = largest-drop, K = kneedle; green = ROI default)',
        fontsize=10, y=1.0,
    )
    fig.tight_layout()

    # If any channel has a tail warning, give it room above the row.
    any_warning = any(fit.get('tail_warning') is not None
                      for fit in results.values())
    if any_warning:
        fig.subplots_adjust(top=0.82, hspace=0.85)
        # Re-anchor the suptitle higher so the warning banners can sit
        # below it without overlapping panel titles
        for txt in fig.texts:
            pass  # keep simple; suptitle uses y=1.0 already

    # Now draw tail warnings in figure coordinates — done AFTER layout
    # settles so that axes bboxes are final and the warning is positioned
    # correctly above each affected row's panel titles.
    if any_warning:
        for row, ch_idx in enumerate(channels):
            label = img.masses[ch_idx]
            fit = results[label]
            tw = fit.get('tail_warning')
            if tw is None:
                continue
            warn_msg = (f'⚠ TAIL AT k={tw["suggested_k"]}: '
                        f'{tw["weight"]*100:.1f}% of pixels at mean '
                        f'{tw["mean_counts"]:.1f} cts (p95={tw["p95"]:.1f}) '
                        f'— consider manual_k={{{label!r}: {tw["suggested_k"]}}}')
            ax_lo = axes[row][0]
            bbox = ax_lo.get_position()
            # Place above the row's panel titles. Panel titles consume about
            # 0.025 in figure coords, so we offset by a bit more than that.
            y_pos = bbox.y1 + 0.055
            fig.text(
                bbox.x0, y_pos, warn_msg,
                fontsize=8.5, fontweight='bold', color='darkred',
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fff5cc',
                          edgecolor='darkred', linewidth=1.0,
                          alpha=0.95),
            )

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
    and per-component summaries for each channel.

    When BIC, largest-drop, and kneedle pick different k values, prints
    a separate component table for each unique k so you can compare them
    directly. When they all agree, prints a single block.
    """
    print()
    for label, fit in results.items():
        if len(fit['models']) == 0:
            print(f'  {label:20s}  no fit (insufficient data)')
            continue
        bics = fit['bics']
        d_bic = bics - bics.min()
        bic_k = fit['best_k_bic']
        sensible = fit['sensible_k']
        drop_k = fit['largest_drop_k']
        knee_k = fit['kneedle_k']
        zero_pct = fit['frac_zeros'] * 100.0
        n_used = fit.get('n_pixels_used', '?')
        unique_recs = fit.get('unique_k_recommendations', [])

        # Header line
        if len(unique_recs) == 1:
            k_msg = (f'all methods agree on k={unique_recs[0]["k"]}')
        elif bic_k == sensible:
            # Sensible_k matches BIC, but elbow methods themselves disagreed
            k_msg = (f'BIC and sensible_k agree on k={sensible}; '
                     f'largest-drop says {drop_k}, kneedle says {knee_k}')
        else:
            k_msg = (f'recommended k={sensible}; BIC says {bic_k}, '
                     f'largest-drop says {drop_k}, kneedle says {knee_k}')
        print(f'  {label}   {k_msg}   '
              f'zero pixels: {zero_pct:.1f}%   pixels in fit: {n_used}')

        # Tail-warning banner — printed prominently when a low-weight,
        # high-mean component is being suppressed. ANSI escape codes
        # (\033[1;31m … \033[0m) render as bold red in Colab/Jupyter.
        tw = fit.get('tail_warning')
        if tw is not None:
            print(
                f'    \033[1;31m⚠ TAIL WARNING:\033[0m '
                f'k={tw["suggested_k"]} reveals a {tw["weight"]*100:.1f}%-weight '
                f'component at mean {tw["mean_counts"]:.1f} counts '
                f'(p5={tw["p5"]:.1f}, p50={tw["p50"]:.1f}, p95={tw["p95"]:.1f}).'
            )
            print(
                f'      If rare high-count features (Fe:S clusters, hot organelles, '
                f'isolated enrichment) matter for this channel, '
                f'override with manual_k={{{label!r}: {tw["suggested_k"]}}}.'
            )

        # ΔBIC table
        bic_str = '    ΔBIC: ' + '  '.join(
            f'k={k}:{d:.0f}' for k, d in zip(range(1, len(bics) + 1), d_bic)
        )
        print(bic_str)

        # Empirical quantiles (model-free, shown once per channel)
        eq = fit['empirical_quantiles']
        eq_str = '    Empirical quantiles (counts):  ' + '  '.join(
            f'p{p}={v:.1f}' for p, v in eq.items()
        )
        print(eq_str)

        # ── One block per unique k recommendation ──────────────────────────
        for rec in unique_recs:
            k = rec['k']
            methods = rec['methods']
            method_str = ', '.join(methods)
            # Mark sensible-k pick (the default ROI rule) with an arrow
            arrow = '  ← default for ROIs' if k == sensible else ''
            header = f'    ── k={k}  (picked by: {method_str}){arrow} ──'
            print(header)

            # Crossings at this k
            thrs = fit['thresholds_by_k'].get(k, [])
            if thrs:
                thr_str = ', '.join(f'{t:.1f}' for t in thrs)
                print(f'      GMM crossings (counts): {thr_str}')

            # Per-component table
            comps = fit['components_by_k'].get(k, [])
            if comps:
                print(f'      {"#":>2} {"weight":>7} {"mean":>10} '
                      f'{"p5":>9} {"p50":>9} {"p95":>9}')
                for i, c in enumerate(comps):
                    q = c['quantiles']
                    print(f'      {i:>2} {c["weight"]:>7.1%} '
                          f'{c["mean_counts"]:>10.2f} '
                          f'{q[5]:>9.2f} {q[50]:>9.2f} {q[95]:>9.2f}')
        print()


# ── Convenience: a single best-fit threshold table ──────────────────────────

def best_thresholds(results, criterion='sensible', manual_k=None):
    """
    Reduce a plot_histograms() result dict to a flat
    {channel_label: list_of_thresholds} mapping.

    Parameters
    ----------
    results : dict
        Output of plot_histograms().
    criterion : 'sensible' (default), 'bic', or 'aic'
        Which criterion picks the default best k for each channel.
        'sensible' = elbow-based recommendation (largest_drop ∩ kneedle),
                     conservative consensus when they disagree. This is the
                     default because BIC tends to overfit on long shallow
                     plateaus where adding components only marginally
                     improves likelihood.
        'bic'      = global BIC minimum (legacy behaviour).
        'aic'      = global AIC minimum.
    manual_k : dict or None
        Per-channel override, e.g. {'31P': 3, '12C 15N': 2}. Channels not
        in the dict fall back to the criterion-selected k. Use this after
        eyeballing the side-by-side panels to override on channels where
        you disagree with the heuristic.

    Returns
    -------
    dict {channel_label: list_of_thresholds_in_counts}
    """
    if criterion == 'sensible':
        key = 'sensible_k'
    elif criterion == 'bic':
        key = 'best_k_bic'
    elif criterion == 'aic':
        key = 'best_k_aic'
    else:
        raise ValueError(
            f"criterion must be 'sensible', 'bic', or 'aic'; got {criterion!r}"
        )
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
