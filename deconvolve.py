"""Multi-peak deconvolution for HMR spectra.

Given a spectrum (mass axis + counts) and an initial guess for the number of
peaks, fit a sum of rect⊗Gauss peaks where (w, sigma) are shared across peaks
(scan-level parameters set by the slit + beam optics) and (a_i, mu_i) are
per-peak parameters.

Strategy:
1. Find candidate peak positions via local maxima of the smoothed signal.
2. Initialise sigma from the brightest peak's rising edge.
3. Initialise w from the brightest peak's plateau extent above 90% max.
4. Joint nonlinear least squares for all (a_i, mu_i, w, sigma).

For unknown N, the function ``deconvolve_with_N_sweep`` runs the fit for
N = 1, 2, 3, ... and returns all results so the caller can pick by BIC or
by inspection. A stability-selection variant is left for a later module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from irf import rect_gauss, fit_sigma_from_edge


@dataclass
class PeakFit:
    """One fitted peak."""
    mu: float       # centre
    a: float        # amplitude (plateau counts above baseline)
    mu_se: float = 0.0   # standard error on mu (1-sigma)
    a_se: float = 0.0    # standard error on a


@dataclass
class DeconvolutionResult:
    """Result of fitting N peaks to a spectrum."""
    peaks: list[PeakFit]
    w: float
    sigma: float
    chi2: float          # weighted chi-squared
    bic: float           # Bayesian information criterion
    aic: float
    n_data: int
    n_params: int
    success: bool
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_peaks(self) -> int:
        return len(self.peaks)

    def model(self, m: np.ndarray) -> np.ndarray:
        """Evaluate the full model at any mass values."""
        y = np.zeros_like(m)
        for p in self.peaks:
            y += rect_gauss(m, p.a, p.mu, self.w, self.sigma)
        return y


def _initial_peak_positions(
    m: np.ndarray, counts: np.ndarray, N: int,
    smooth_width: int = 3,
    min_distance_units: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to N (positions, amplitudes) sorted by amplitude descending.

    The minimum allowed distance between peaks is set to roughly the IRF
    width (rising-edge-derived sigma times ~10, or one tenth of the scan
    range if the IRF can't be determined). This prevents the finder from
    returning multiple maxima on the same physical plateau.

    If fewer than N local maxima are found, additional positions are placed
    at the largest gaps in mass.
    """
    if smooth_width > 1:
        kernel = np.ones(smooth_width) / smooth_width
        y_smooth = np.convolve(counts, kernel, mode="same")
    else:
        y_smooth = counts.copy()

    # Estimate minimum peak separation from the IRF (rising edge)
    step = float(np.diff(m).mean())
    if min_distance_units is None:
        try:
            sigma_est, _ = fit_sigma_from_edge(m, counts, side="rising")
            # Two peaks closer than ~10*sigma will look like one bumpy peak
            min_distance_units = max(10.0 * sigma_est, 5.0 * step)
        except Exception:
            min_distance_units = (m[-1] - m[0]) / 10.0

    # Convert to integer index distance
    min_dist_idx = max(1, int(np.ceil(min_distance_units / step)))

    peaks_idx, _ = find_peaks(
        y_smooth,
        height=max(1.0, 0.001 * y_smooth.max()),
        distance=min_dist_idx,
    )
    if len(peaks_idx) == 0:
        peaks_idx = np.array([int(np.argmax(y_smooth))])

    heights = y_smooth[peaks_idx]
    order = np.argsort(heights)[::-1]
    peaks_idx = peaks_idx[order]

    if len(peaks_idx) >= N:
        peaks_idx = peaks_idx[:N]
        positions = m[peaks_idx]
        amplitudes = counts[peaks_idx]
        return positions, amplitudes

    # Need more positions: place them at the largest gaps among the m range
    positions = list(m[peaks_idx])
    amplitudes = list(counts[peaks_idx])
    while len(positions) < N:
        sorted_pos = sorted(positions + [m[0], m[-1]])
        gaps = [(sorted_pos[i + 1] - sorted_pos[i], (sorted_pos[i] + sorted_pos[i + 1]) / 2)
                for i in range(len(sorted_pos) - 1)]
        gaps.sort(reverse=True)
        new_pos = gaps[0][1]
        positions.append(new_pos)
        idx = int(np.argmin(np.abs(m - new_pos)))
        amplitudes.append(max(1.0, counts[idx]))
    return np.array(positions), np.array(amplitudes)


def deconvolve(
    m: np.ndarray, counts: np.ndarray, N: int,
    initial_positions: np.ndarray | None = None,
    initial_amplitudes: np.ndarray | None = None,
    sigma0: float | None = None,
    w0: float | None = None,
    fix_sigma: bool = False,
    fix_w: bool = False,
    verbose: bool = False,
) -> DeconvolutionResult:
    """Fit N rect⊗Gauss peaks with shared (w, sigma).

    Parameters
    ----------
    m, counts : arrays
        Mass and counts.
    N : int
        Number of peaks.
    initial_positions, initial_amplitudes : arrays, optional
        Starting guesses. If None, picked from local maxima.
    sigma0 : float, optional
        Initial sigma. If None, fitted from the brightest peak's rising edge.
    w0 : float, optional
        Initial slit width. If None, estimated from the brightest peak's
        90%-plateau extent.
    fix_sigma : bool
        If True, sigma is held at sigma0 during the fit (used when sigma
        was reliably determined separately, e.g. on a calibration scan).
    """
    n_data = len(counts)
    step = float(np.diff(m).mean())
    span = float(m[-1] - m[0])

    if initial_positions is None or initial_amplitudes is None:
        ip, ia = _initial_peak_positions(m, counts, N)
        if initial_positions is None: initial_positions = ip
        if initial_amplitudes is None: initial_amplitudes = ia

    initial_positions = np.array(initial_positions, dtype=float)
    initial_amplitudes = np.array(initial_amplitudes, dtype=float)

    if sigma0 is None:
        sigma0, _ = fit_sigma_from_edge(m, counts, side="rising")
    if w0 is None:
        # Plateau width above 90% of brightest peak
        peak_idx = int(np.argmax(counts))
        plateau_thresh = 0.9 * counts[peak_idx]
        plateau_mask = counts > plateau_thresh
        if plateau_mask.sum() >= 2:
            w0 = float(m[plateau_mask].max() - m[plateau_mask].min())
            w0 = max(w0, 2 * step)
        else:
            w0 = 4 * step  # default to a few steps

    if verbose:
        print(f"Initial sigma = {sigma0*1000:.4f}, w = {w0*1000:.4f} (mau if m is in amu)")
        print(f"Initial positions: {initial_positions}")
        print(f"Initial amplitudes: {initial_amplitudes}")

    # Parameter packing: [a_1...a_N, mu_1...mu_N, w?, sigma?]
    # depending on which are fixed.
    if fix_sigma and fix_w:
        def unpack(p):
            return p[:N], p[N:2*N], w0, sigma0
        n_params = 2 * N
    elif fix_sigma:
        def unpack(p):
            return p[:N], p[N:2*N], p[2*N], sigma0
        n_params = 2 * N + 1
    elif fix_w:
        def unpack(p):
            return p[:N], p[N:2*N], w0, p[2*N]
        n_params = 2 * N + 1
    else:
        def unpack(p):
            return p[:N], p[N:2*N], p[2*N], p[2*N + 1]
        n_params = 2 * N + 2

    def model_at(p, mm):
        a, mu, w, sigma = unpack(p)
        y = np.zeros_like(mm)
        for ai, mui in zip(a, mu):
            y += rect_gauss(mm, ai, mui, w, sigma)
        return y

    weights = 1.0 / np.sqrt(np.clip(counts, 1.0, None))

    def residuals(p):
        return (model_at(p, m) - counts) * weights

    a_max = max(initial_amplitudes.max() * 5.0, counts.max() * 2.0)
    p0_parts = [initial_amplitudes, initial_positions]
    lo_parts = [np.zeros(N), initial_positions - 0.01]
    hi_parts = [np.full(N, a_max), initial_positions + 0.01]
    if not fix_w:
        p0_parts.append([w0])
        lo_parts.append([step])
        hi_parts.append([span * 2.0])
    if not fix_sigma:
        p0_parts.append([sigma0])
        lo_parts.append([step / 100.0])
        hi_parts.append([span])
    p0 = np.concatenate(p0_parts)
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)

    result = least_squares(
        residuals, p0, bounds=(lo, hi),
        method="trf", xtol=1e-13, ftol=1e-13, max_nfev=20000,
    )
    a_fit, mu_fit, w_fit, sigma_fit = unpack(result.x)

    # Compute chi2 and BIC
    chi2 = float(np.sum(residuals(result.x) ** 2))
    bic = chi2 + n_params * np.log(n_data)
    aic = chi2 + 2 * n_params

    # Approximate parameter uncertainties from the Jacobian (linearised
    # covariance: cov = (J^T J)^-1 * chi2/(n_data - n_params))
    mu_se = np.zeros(N); a_se = np.zeros(N)
    try:
        J = result.jac
        # Reduced chi2
        dof = max(n_data - n_params, 1)
        cov = np.linalg.pinv(J.T @ J) * (chi2 / dof)
        sigmas = np.sqrt(np.maximum(np.diag(cov), 0))
        a_se = sigmas[:N]
        mu_se = sigmas[N:2*N]
    except Exception:
        pass

    peaks = [PeakFit(mu=float(mu_fit[i]), a=float(a_fit[i]),
                     mu_se=float(mu_se[i]), a_se=float(a_se[i]))
             for i in range(N)]
    peaks.sort(key=lambda p: p.mu)

    if verbose:
        print(f"Fit: chi2={chi2:.2f}, BIC={bic:.2f}, w={w_fit*1000:.4f} mau, sigma={sigma_fit*1000:.4f} mau")
        for i, p in enumerate(peaks):
            print(f"  Peak {i+1}: mu={p.mu:.6f} (±{p.mu_se*1e6:.1f} µamu), a={p.a:.1f} (±{p.a_se:.1f})")

    return DeconvolutionResult(
        peaks=peaks,
        w=float(w_fit), sigma=float(sigma_fit),
        chi2=chi2, bic=bic, aic=aic,
        n_data=n_data, n_params=n_params,
        success=bool(result.success),
        residuals=counts - model_at(result.x, m),
    )


def deconvolve_with_N_sweep(
    m: np.ndarray, counts: np.ndarray,
    N_range: tuple[int, int] = (1, 5),
    sigma0: float | None = None,
    w0: float | None = None,
    verbose: bool = False,
) -> list[DeconvolutionResult]:
    """Run deconvolve for each N in the range; return all results."""
    if sigma0 is None:
        sigma0, _ = fit_sigma_from_edge(m, counts, side="rising")
    results = []
    for N in range(N_range[0], N_range[1] + 1):
        try:
            r = deconvolve(m, counts, N, sigma0=sigma0, w0=w0,
                            fix_sigma=False, verbose=False)
            results.append(r)
            if verbose:
                print(f"N={N}: chi2={r.chi2:.1f}, BIC={r.bic:.1f}, "
                      f"AIC={r.aic:.1f}, w={r.w*1000:.3f}, sigma={r.sigma*1000:.3f} mau, "
                      f"centres={[f'{p.mu:.5f}' for p in r.peaks]}")
        except Exception as e:
            if verbose:
                print(f"N={N}: FAILED — {e}")
    return results
