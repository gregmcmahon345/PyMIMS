"""Instrument response function (IRF) for NanoSIMS HMR peak shapes.

A NanoSIMS mass peak is, to first order, the image of the entrance/exit slit
(a rectangle of width ``w`` in mass units) blurred by the ion-optical
point-spread (a Gaussian of width ``sigma``). Their convolution is a
flat-topped peak with error-function edges:

    rect_gauss(m) = a/2 * [ erf((m - (mu - w/2)) / (sqrt2 * sigma))
                          - erf((m - (mu + w/2)) / (sqrt2 * sigma)) ]

which has plateau height ``a`` (for w >> sigma), centre ``mu``, flat-top width
``w`` and edge sharpness set by ``sigma``. As ``w -> 0`` it becomes a pure
Gaussian of integrated area ``a*w``; as ``sigma -> 0`` it becomes a rectangle.

This module is the shared dependency of ``deconvolve.py`` and
``sparse_deconvolve.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit


@dataclass
class IRFFit:
    """Result of characterising the instrument response on a single peak."""
    sigma: float            # Gaussian blur (mass units)
    w: float = 0.0          # flat-top / slit width (mass units)
    mu: float = 0.0         # peak centre (mass units)
    amplitude: float = 0.0  # plateau height (counts)
    baseline: float = 0.0   # fitted baseline (counts)
    success: bool = True


def rect_gauss(m, a, mu, w, sigma):
    """Rectangle(width w) convolved with Gaussian(sigma), plateau height a.

    Vectorised over ``m``. Degrades gracefully: sigma<=0 -> hard rectangle.
    """
    m = np.asarray(m, dtype=float)
    if sigma <= 0:
        return np.where(np.abs(m - mu) <= 0.5 * w, float(a), 0.0)
    s = np.sqrt(2.0) * sigma
    z_lo = (m - (mu - 0.5 * w)) / s
    z_hi = (m - (mu + 0.5 * w)) / s
    return 0.5 * a * (erf(z_lo) - erf(z_hi))


def fit_sigma_from_edge(m, counts, side="rising"):
    """Estimate the Gaussian edge width ``sigma`` from one edge of the
    brightest peak by fitting an error-function step.

    Returns ``(sigma, info)`` where ``info`` is an :class:`IRFFit` carrying the
    fitted centre/amplitude/baseline (callers in deconvolve.py ignore the
    second element, but it's useful for diagnostics).

    The edge model on the rising side is
        f(m) = B + A/2 * (1 + erf((m - m0) / (sqrt2 * sigma)))
    and on the falling side the ``erf`` term flips sign. ``m0`` is the 50%
    crossing, so the 16%-84% rise spans 2*sigma — used both to seed the fit
    and as a fallback if the non-linear fit fails.
    """
    m = np.asarray(m, dtype=float)
    c = np.asarray(counts, dtype=float)
    n = len(c)
    if n < 4:
        return float(abs(np.diff(m).mean()) or 1.0), IRFFit(sigma=0.0, success=False)

    step = float(np.abs(np.diff(m)).mean())
    peak_idx = int(np.argmax(c))
    # plateau ~ median around the max; baseline ~ median of the far tail end
    lo = max(0, peak_idx - 2)
    plateau = float(np.median(c[lo:peak_idx + 3])) if peak_idx + 3 <= n else float(c[peak_idx])
    edge_pool = c[: max(3, n // 10)]
    baseline = float(np.median(edge_pool))
    amp = max(plateau - baseline, 1.0)

    if side == "rising":
        seg = slice(0, peak_idx + 1)
    else:
        seg = slice(peak_idx, n)
    mm = m[seg]
    cc = c[seg]
    if len(mm) < 4:
        return max(2.0 * step, 1e-6), IRFFit(sigma=2.0 * step, mu=m[peak_idx],
                                             amplitude=amp, baseline=baseline,
                                             success=False)

    # --- percentile (16/84) seed, robust to noise ---
    target_lo = baseline + 0.16 * amp
    target_hi = baseline + 0.84 * amp

    def _cross(xv, yv, level):
        # first index where yv crosses `level`, linear-interpolated
        for i in range(1, len(yv)):
            if (yv[i - 1] - level) * (yv[i] - level) <= 0 and yv[i] != yv[i - 1]:
                t = (level - yv[i - 1]) / (yv[i] - yv[i - 1])
                return xv[i - 1] + t * (xv[i] - xv[i - 1])
        return None

    if side == "rising":
        x16 = _cross(mm, cc, target_lo)
        x84 = _cross(mm, cc, target_hi)
        m0_seed = _cross(mm, cc, baseline + 0.5 * amp)
    else:  # falling: high then low
        x84 = _cross(mm, cc, target_hi)
        x16 = _cross(mm, cc, target_lo)
        m0_seed = _cross(mm, cc, baseline + 0.5 * amp)

    if x16 is not None and x84 is not None and abs(x84 - x16) > 0:
        sigma_seed = abs(x84 - x16) / 2.0
    else:
        sigma_seed = 2.0 * step
    if m0_seed is None:
        m0_seed = mm[len(mm) // 2]
    sigma_seed = max(sigma_seed, 0.25 * step)

    # --- refine with an erf-step fit ---
    sgn = 1.0 if side == "rising" else -1.0

    def edge(x, m0, sigma, A, B):
        z = sgn * (x - m0) / (np.sqrt(2.0) * max(sigma, 1e-12))
        return B + 0.5 * A * (1.0 + erf(z))

    try:
        p0 = [m0_seed, sigma_seed, amp, baseline]
        popt, _ = curve_fit(
            edge, mm, cc, p0=p0, maxfev=20000,
            bounds=([mm.min(), 0.05 * step, 0.0, -abs(amp)],
                    [mm.max(), (mm.max() - mm.min()) or (10 * step),
                     amp * 5 + 10, baseline + amp]),
        )
        sigma_fit = float(abs(popt[1]))
        info = IRFFit(sigma=sigma_fit, mu=float(popt[0]),
                      amplitude=float(popt[2]), baseline=float(popt[3]),
                      success=True)
        return sigma_fit, info
    except Exception:
        return float(sigma_seed), IRFFit(sigma=float(sigma_seed), mu=float(m0_seed),
                                         amplitude=amp, baseline=baseline,
                                         success=False)
