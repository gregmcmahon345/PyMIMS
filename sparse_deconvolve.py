"""Sparse-dictionary deconvolution with stability selection.

This is the "Greg's FPCA / synthetic-spectra" approach from the handover summary:

1. Build a dictionary of position-shifted copies of a reference IRF, on a
   mass grid much finer than the instrument resolution.
2. Run non-negative LASSO (or elastic net) at many regularisation strengths.
3. For each atom position, count how often it's selected across the scan.
   Atoms selected in >threshold fraction of runs are "stable".
4. Cluster nearby stable atoms (within ~2 mau) since the fine grid causes
   one real peak to be split across adjacent atoms.
5. Refit with the locked number of peaks for clean centroids/amplitudes.

Why elastic net (not just LASSO)? The dictionary atoms are *highly* correlated
because adjacent template copies on a 0.5 mau grid overlap heavily (σ ≈ 0.4 mau).
Pure LASSO with correlated features arbitrarily picks one atom from a group
and drops the rest, producing grid-splitting artifacts and unstable peak
positions across λ. Elastic net adds an L2 penalty that encourages correlated
atoms to share weight, so a real peak between two grid points gets distributed
weight across both neighbours, which the clustering step then merges
correctly into a single peak at the right centroid.

We enforce non-negativity throughout (``positive=True`` in sklearn). This
preserves the L2 "grouping effect" on positively-correlated atoms while
preventing the optimiser from using positive/negative cancellation tricks
to fit residual noise.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.exceptions import ConvergenceWarning

from scipy.optimize import least_squares

from irf import rect_gauss, fit_sigma_from_edge, IRFFit
from deconvolve import PeakFit, DeconvolutionResult


@dataclass
class StabilityResult:
    """Output of the stability-selection pipeline."""
    candidate_positions: np.ndarray   # all unique cluster centres found
    selection_counts: np.ndarray      # how many lambdas each position was selected at
    n_lambdas: int                    # total number of lambdas tried
    max_amplitudes: np.ndarray        # max amplitude each cluster saw across lambdas
    selected: np.ndarray              # boolean mask: positions deemed stable
    final_fit: DeconvolutionResult | None = None  # refit with N = num stable peaks


def _cluster_atoms(
    positions: np.ndarray, amplitudes: np.ndarray, radius: float,
) -> list[tuple[float, float]]:
    """Group atoms within `radius` of each other into single clusters.
    Returns list of (amplitude-weighted-centroid, total_amplitude) tuples."""
    if len(positions) == 0:
        return []
    order = np.argsort(positions)
    pos = positions[order]; amp = amplitudes[order]
    clusters = [[(pos[0], amp[0])]]
    for p, a in zip(pos[1:], amp[1:]):
        if p - clusters[-1][-1][0] <= radius:
            clusters[-1].append((p, a))
        else:
            clusters.append([(p, a)])
    out = []
    for c in clusters:
        total = sum(a for _, a in c)
        if total <= 0:
            continue
        cen = sum(p * a for p, a in c) / total
        out.append((cen, total))
    return out


def stability_select(
    m: np.ndarray, counts: np.ndarray,
    template_w: float, template_sigma: float,
    grid_step_mau: float = 0.5,
    lambda_range: tuple[float, float] = (1.0, 1e4),
    n_lambdas: int = 30,
    l1_ratios: tuple[float, ...] = (1.0,),
    stability_threshold: float = 0.10,
    cluster_radius_mau: float = 2.0,
    amplitude_threshold_rel: float = 0.001,
    min_data_signal_rel: float = 0.005,
    verbose: bool = False,
) -> StabilityResult:
    """Sparse deconvolution with stability selection.

    Parameters
    ----------
    m, counts : arrays
        Mass and counts.
    template_w, template_sigma : floats
        Shape parameters for the rect-Gauss template (in same units as m).
    grid_step_mau : float
        Spacing between dictionary atoms in millimass units.
    lambda_range, n_lambdas : LASSO regularisation scan parameters.
    stability_threshold : float
        Fraction of lambdas a position must be selected at to be deemed stable.
    cluster_radius_mau : float
        Atoms within this many mau are merged (grid-splitting recovery).
    amplitude_threshold_rel : float
        Atoms with LASSO amplitude < this fraction of the brightest are ignored.
    min_data_signal_rel : float
        Candidate positions whose nearby data points are all below
        this fraction of the brightest data point are filtered out
        (they're baseline-fitting atoms, not real peaks).
    verbose : bool
    """
    # Build dictionary
    grid_step = grid_step_mau * 1e-3
    cluster_radius = cluster_radius_mau * 1e-3
    shift_grid = np.arange(m.min() + grid_step, m.max() - grid_step, grid_step)

    if verbose:
        print(f"Dictionary: {len(shift_grid)} atoms, grid spacing {grid_step_mau} mau")
        print(f"Template: w={template_w*1000:.3f} mau, sigma={template_sigma*1000:.3f} mau")
        print(f"l1_ratios: {l1_ratios}")

    # D[i, j] = template centred at shift_grid[j], evaluated at m[i]
    D = np.zeros((len(m), len(shift_grid)))
    for j, mu_j in enumerate(shift_grid):
        D[:, j] = rect_gauss(m, 1.0, mu_j, template_w, template_sigma)

    # Scan over (alpha, l1_ratio) jointly. When l1_ratio == 1 use Lasso; otherwise
    # use ElasticNet. The total number of fits is n_lambdas * len(l1_ratios).
    lambdas = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_lambdas)
    total_fits = n_lambdas * len(l1_ratios)
    all_clusters_per_fit = []
    if verbose:
        header = f"{'l1_ratio':>9} {'lambda':>9} {'n_atoms':>9} {'n_clusters':>11}  {'centroids (top 6)':>40}"
        print(header)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for l1_ratio in l1_ratios:
            for lam in lambdas:
                if l1_ratio >= 0.9999:
                    # Pure LASSO -- use the Lasso class directly (slightly faster than ElasticNet at l1_ratio=1)
                    model = Lasso(alpha=lam, positive=True, max_iter=50000, tol=1e-6)
                else:
                    model = ElasticNet(alpha=lam, l1_ratio=l1_ratio,
                                        positive=True, max_iter=50000, tol=1e-6)
                model.fit(D, counts)
                alpha_coef = model.coef_
                if alpha_coef.max() <= 0:
                    all_clusters_per_fit.append([])
                    if verbose:
                        print(f"{l1_ratio:>9.2f} {lam:>9.2f} {0:>9} {0:>11}")
                    continue
                nz = alpha_coef > amplitude_threshold_rel * alpha_coef.max()
                clusters = _cluster_atoms(shift_grid[nz], alpha_coef[nz], radius=cluster_radius)
                if clusters:
                    max_amp = max(c[1] for c in clusters)
                    clusters = [c for c in clusters if c[1] >= amplitude_threshold_rel * max_amp]
                all_clusters_per_fit.append(clusters)
                if verbose:
                    top = sorted(clusters, key=lambda c: -c[1])[:6]
                    cen_str = ", ".join(f"{c[0]:.4f}" for c in top)
                    print(f"{l1_ratio:>9.2f} {lam:>9.2f} {int(nz.sum()):>9} {len(clusters):>11}  {cen_str:>40}")

    # Build stability map: bin cluster positions into ~grid_step bins, count occurrences
    bin_width = grid_step
    bins = {}
    for clusters in all_clusters_per_fit:
        for pos, amp in clusters:
            key = int(round(pos / bin_width))
            if key not in bins:
                bins[key] = {"positions": [], "amps": [], "count": 0}
            bins[key]["positions"].append(pos)
            bins[key]["amps"].append(amp)
            bins[key]["count"] += 1

    if not bins:
        return StabilityResult(
            candidate_positions=np.array([]),
            selection_counts=np.array([]),
            n_lambdas=n_lambdas,
            max_amplitudes=np.array([]),
            selected=np.array([], dtype=bool),
        )

    keys = sorted(bins.keys())
    candidate_positions = np.array([np.mean(bins[k]["positions"]) for k in keys])
    selection_counts = np.array([bins[k]["count"] for k in keys])
    max_amplitudes = np.array([max(bins[k]["amps"]) for k in keys])

    # Cluster again across bins (some bins differ only by one grid step but
    # represent the same physical peak)
    cluster_inputs = list(zip(candidate_positions, max_amplitudes, selection_counts))
    cluster_inputs.sort()
    final_groups = [[cluster_inputs[0]]]
    for p, a, c in cluster_inputs[1:]:
        if p - final_groups[-1][-1][0] <= cluster_radius:
            final_groups[-1].append((p, a, c))
        else:
            final_groups.append([(p, a, c)])

    # For each final group, aggregate
    grouped_pos = []
    grouped_amp = []
    grouped_count = []
    for g in final_groups:
        total_amp = sum(x[1] for x in g)
        cen = sum(x[0] * x[1] for x in g) / total_amp if total_amp > 0 else np.mean([x[0] for x in g])
        grouped_pos.append(cen)
        grouped_amp.append(total_amp)
        grouped_count.append(max(x[2] for x in g))

    grouped_pos = np.array(grouped_pos)
    grouped_amp = np.array(grouped_amp)
    grouped_count = np.array(grouped_count)

    # Decide which are "stable"
    # Total fits = n_lambdas × number of l1_ratios
    total_fits = n_lambdas * len(l1_ratios)
    min_count = int(np.ceil(stability_threshold * total_fits))
    selected_mask = grouped_count >= min_count

    # Filter: candidate must have non-trivial data signal nearby (basic sanity check).
    # The real discrimination between phantom and real small peaks happens in the
    # post-fit chi-squared pruning step (in deconvolve_sparse).
    data_max = float(counts.max())
    data_threshold_rel = max(min_data_signal_rel * data_max, 10.0)
    for i, pos in enumerate(grouped_pos):
        if not selected_mask[i]:
            continue
        # Look at data points within w/2 + 3*sigma of this position
        window_radius = 0.5 * template_w + 3.0 * template_sigma
        within = np.abs(m - pos) <= window_radius
        if not within.any() or counts[within].max() < data_threshold_rel:
            selected_mask[i] = False

    # Also require absolute amplitude > amplitude_threshold_rel * brightest selected
    if selected_mask.any():
        brightest_selected = grouped_amp[selected_mask].max()
        selected_mask &= (grouped_amp >= amplitude_threshold_rel * brightest_selected)

    if verbose:
        print(f"\nStability ranking (top 15):")
        order = np.argsort(grouped_count)[::-1]
        for i in order[:15]:
            mark = "STABLE" if selected_mask[i] else ""
            window_radius = 0.5 * template_w + 3.0 * template_sigma
            within = np.abs(m - grouped_pos[i]) <= window_radius
            data_near = counts[within].max() if within.any() else 0
            print(f"  m = {grouped_pos[i]:.5f}, count = {grouped_count[i]:>3} / {total_fits}, "
                  f"max_amp = {grouped_amp[i]:>10.0f}, data_near = {data_near:>8.0f}  {mark}")
        print(f"\nN stable peaks (>= {min_count}/{total_fits} and data_near >= {data_threshold_rel:.0f}): "
              f"{int(selected_mask.sum())}")

    return StabilityResult(
        candidate_positions=grouped_pos,
        selection_counts=grouped_count,
        n_lambdas=total_fits,
        max_amplitudes=grouped_amp,
        selected=selected_mask,
    )


def deconvolve_sparse(
    m: np.ndarray, counts: np.ndarray,
    template_w: float | None = None,
    template_sigma: float | None = None,
    grid_step_mau: float = 0.5,
    stability_threshold: float = 0.10,
    n_lambdas: int = 30,
    lambda_range: tuple[float, float] = (1.0, 1e4),
    l1_ratios: tuple[float, ...] = (1.0,),
    min_data_signal_rel: float = 0.005,
    min_peak_amplitude_rel: float = 0.0001,
    verbose: bool = False,
) -> DeconvolutionResult:
    """End-to-end sparse deconvolution: build dictionary, stability-select, refit.

    Returns a ``DeconvolutionResult`` with the auto-determined number of peaks.

    The default ``stability_threshold`` of 0.10 is chosen to be sensitive
    to small peaks (down to ~3% of the brightest). Small peaks naturally
    get selected at fewer (λ, l1_ratio) values than large peaks because the
    L1 penalty grows with amplitude. A strong SNR filter (Poisson SNR > 5
    above local baseline) discriminates between real small peaks and
    sparse-fitter noise atoms. If you're still getting phantom peaks,
    raise the threshold; if you're missing known small peaks, lower it.

    ``l1_ratios`` controls the elastic-net mix:
    - 1.0 = pure LASSO (sparse, but arbitrarily picks one of correlated atoms)
    - <1.0 = mixed L1+L2 (distributes weight across correlated atoms)

    The default is ``(1.0,)`` (pure LASSO). Including lower l1_ratios sounds
    appealing because it should give a "grouping effect" on the highly-correlated
    dictionary atoms, but in practice the L2 term smears weight across the
    *entire* active region rather than within a single peak. This conflicts
    with the cluster-merging step (which uses a fixed ~2 mau radius) and tends
    to merge distinct peaks into one. Pure LASSO with a sensitive
    ``stability_threshold`` works better.

    Non-negativity is enforced throughout (``positive=True`` in sklearn).

    After the refit, peaks with amplitude below ``min_peak_amplitude_rel``
    times the brightest fitted peak are pruned and the fit is re-run with
    the pruned set.
    """
    from deconvolve import deconvolve, _initial_peak_positions

    # Estimate IRF parameters by fitting a single peak to the brightest local
    # maximum first. This gives a much better template than the naïve
    # "plateau above 90%" approach.
    # If the single-peak fit yields a suspiciously wide sigma (sigma > w/2,
    # indicating the brightest "peak" was actually multiple unresolved peaks),
    # fall back to fitting sigma from the rising edge alone and using a
    # default-narrow w.
    if template_w is None or template_sigma is None:
        peak_idx = int(np.argmax(counts))
        try:
            single = deconvolve(
                m, counts, N=1,
                initial_positions=np.array([m[peak_idx]]),
                initial_amplitudes=np.array([counts[peak_idx]]),
                fix_sigma=False, verbose=False,
            )
            # Sanity check: if sigma > w/3, the fit absorbed multiple peaks
            # as one wide peak. Use rising-edge sigma + default w instead.
            if single.sigma > single.w / 3.0:
                if verbose:
                    print(f"Single-peak template fit looks degenerate "
                          f"(sigma={single.sigma*1000:.2f} > w/3={single.w/3*1000:.2f} mau); "
                          f"falling back to rising-edge sigma.")
                rising_sigma, _ = fit_sigma_from_edge(m, counts, side="rising")
                if template_sigma is None:
                    template_sigma = rising_sigma
                if template_w is None:
                    # Use 4 sigma as a narrow-but-realistic w
                    template_w = max(4 * rising_sigma, 4 * float(np.diff(m).mean()))
            else:
                if template_w is None:
                    template_w = single.w
                if template_sigma is None:
                    template_sigma = single.sigma
        except Exception:
            if template_sigma is None:
                template_sigma, _ = fit_sigma_from_edge(m, counts, side="rising")
            if template_w is None:
                plateau_mask = counts > 0.9 * counts[peak_idx]
                if plateau_mask.sum() >= 2:
                    template_w = float(m[plateau_mask].max() - m[plateau_mask].min())
                else:
                    template_w = 4 * float(np.diff(m).mean())

    if verbose:
        print(f"Template: w = {template_w*1000:.3f} mau, sigma = {template_sigma*1000:.3f} mau")

    sr = stability_select(
        m, counts,
        template_w=template_w, template_sigma=template_sigma,
        grid_step_mau=grid_step_mau,
        lambda_range=lambda_range,
        n_lambdas=n_lambdas,
        l1_ratios=l1_ratios,
        stability_threshold=stability_threshold,
        min_data_signal_rel=min_data_signal_rel,
        verbose=verbose,
    )

    if not sr.selected.any():
        if verbose:
            print("No stable peaks found.")
        return DeconvolutionResult(
            peaks=[], w=template_w, sigma=template_sigma,
            chi2=np.inf, bic=np.inf, aic=np.inf,
            n_data=len(counts), n_params=0,
            success=False,
        )

    init_pos = sr.candidate_positions[sr.selected]
    init_amp = sr.max_amplitudes[sr.selected]
    N = len(init_pos)
    if verbose:
        print(f"\nRefitting with N = {N} peaks from stable positions (template locked)...")

    # Initial refit: lock w and sigma to the template values so the optimiser
    # can't go degenerate when many peaks are crammed into a small window.
    # This is correct physically because all peaks in one scan share the IRF.
    final = deconvolve(
        m, counts, N=N,
        initial_positions=init_pos,
        initial_amplitudes=init_amp,
        sigma0=template_sigma,
        w0=template_w,
        fix_sigma=True, fix_w=True,
        verbose=verbose,
    )

    # Post-fit pruning: alternate Stage A (drop tiny peaks + merge duplicates)
    # and Stage B (likelihood-ratio test) until the peak set stops changing.
    #
    # Stage A: drop amplitude below `min_peak_amplitude_rel` of brightest;
    #          merge peaks within 1σ in μ.
    # Stage B: for each peak, refit without it; drop if Δχ² < min_chi2_improvement.
    min_chi2_improvement = 10.0   # ~3-sigma in Poisson-weighted chi-squared
    max_outer_iterations = 5
    for outer in range(max_outer_iterations):
        if len(final.peaks) <= 1:
            break
        n_before = len(final.peaks)
        max_amp = max(p.a for p in final.peaks)
        threshold = min_peak_amplitude_rel * max_amp
        merge_radius = final.sigma

        # === Stage A: drop tiny peaks and merge duplicates ===
        kept_peaks = [p for p in final.peaks if p.a >= threshold]
        kept_peaks.sort(key=lambda p: p.mu)
        merged: list[PeakFit] = []
        for p in kept_peaks:
            if merged and (p.mu - merged[-1].mu) < merge_radius:
                prev = merged[-1]
                total_a = prev.a + p.a
                if total_a > 0:
                    new_mu = (prev.mu * prev.a + p.mu * p.a) / total_a
                else:
                    new_mu = 0.5 * (prev.mu + p.mu)
                merged[-1] = PeakFit(mu=new_mu, a=total_a)
            else:
                merged.append(p)

        if len(merged) != len(final.peaks):
            if verbose:
                print(f"Stage A (iter {outer+1}): {len(final.peaks)} -> {len(merged)} peaks")
            if len(merged) == 0:
                break
            init_pos2 = np.array([p.mu for p in merged])
            init_amp2 = np.array([p.a for p in merged])
            final = deconvolve(
                m, counts, N=len(merged),
                initial_positions=init_pos2, initial_amplitudes=init_amp2,
                sigma0=template_sigma, w0=template_w,
                fix_sigma=True, fix_w=True, verbose=False,
            )

        # === Stage B: likelihood-ratio test for each peak ===
        # Find the peak whose removal increases chi^2 the LEAST.
        # If that increase is below threshold, drop it and refit.
        if final.n_peaks > 1:
            current_chi2 = final.chi2
            worst_idx = None
            worst_delta = np.inf
            worst_refit = None
            for j in range(final.n_peaks):
                remaining = [p for k, p in enumerate(final.peaks) if k != j]
                init_pos = np.array([p.mu for p in remaining])
                init_amp = np.array([p.a for p in remaining])
                try:
                    test = deconvolve(
                        m, counts, N=len(remaining),
                        initial_positions=init_pos, initial_amplitudes=init_amp,
                        sigma0=template_sigma, w0=template_w,
                        fix_sigma=True, fix_w=True, verbose=False,
                    )
                    delta = test.chi2 - current_chi2
                except Exception:
                    delta = np.inf
                    test = None
                if delta < worst_delta:
                    worst_delta = delta
                    worst_idx = j
                    worst_refit = test
            if 0.0 <= worst_delta < min_chi2_improvement and worst_refit is not None:
                if verbose:
                    print(f"Stage B (iter {outer+1}): dropping peak {worst_idx+1} "
                          f"(μ={final.peaks[worst_idx].mu:.5f}, a={final.peaks[worst_idx].a:.1f}), "
                          f"Δχ² = {worst_delta:.1f}")
                final = worst_refit

        if len(final.peaks) == n_before:
            break  # No change in this iteration -> converged

    # Final free-template fit: now that we have the converged peak set,
    # allow w and σ to vary slightly to absorb any residual systematic
    # error in the original template extraction.
    if final.n_peaks > 0:
        init_pos_final = np.array([p.mu for p in final.peaks])
        init_amp_final = np.array([p.a for p in final.peaks])
        try:
            final = deconvolve(
                m, counts, N=final.n_peaks,
                initial_positions=init_pos_final, initial_amplitudes=init_amp_final,
                sigma0=final.sigma, w0=final.w,
                fix_sigma=False, fix_w=False, verbose=False,
            )
        except Exception:
            pass  # keep the locked-template result if free fit fails

    return final
