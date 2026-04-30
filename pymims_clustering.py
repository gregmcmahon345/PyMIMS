"""
pymims_clustering.py — Pixel clustering for NanoSIMS .im data.

Companion to pymims.py / pymims_histograms.py. Performs per-pixel clustering
using k-means or hierarchical agglomerative methods, with cluster-count
selection driven by multiple criteria (inertia elbow, silhouette,
Calinski-Harabasz, Davies-Bouldin, Cubic Clustering Criterion). For
hierarchical, the Cophenetic Correlation Coefficient is also reported as
a dendrogram-quality diagnostic.

Two acronyms that look the same but mean different things, both used here:

  * Cubic Clustering Criterion (Sarle 1983, used in JMP/SAS) — a cluster-
    count selector. Our implementation approximates Sarle's formula using
    a participation-ratio approximation for the expected R²; ABSOLUTE
    VALUES ARE NOT ON SARLE'S SCALE and Sarle's published thresholds
    (CCC>3 etc.) do not apply. Only the peak position across the k sweep
    is meaningful, which is what we use it for. In this module:
    result['cubic_ccc'] (curve) and result['cubic_ccc_peak_k'] (selector).

  * Cophenetic Correlation Coefficient — a dendrogram-quality measure
    for hierarchical clustering only. The Pearson correlation between
    original pairwise distances and the dendrogram heights at which
    pairs are first joined. Values near 1 mean the dendrogram faithfully
    represents the data structure. In this module: result['cophenetic_corr'].

Design philosophy mirrors pymims_histograms.py: the module computes a
sweep of candidate cluster counts, reports several metrics that may
disagree, and lets the user pick. No criterion is treated as authoritative.

Output is a ClusterResult dict whose `labels` 2-D array maps directly
to ROI masks for the eventual rule-based ROI generator.

Author : G. McMahon (with AI-assisted development)
Created: April 2026 (v0.7 work-in-progress)

Usage
-----
    from pymims import MimsImage
    from pymims_clustering import cluster_pixels, plot_cluster_labels

    img = MimsImage('myfile.im')
    img.drift_correct(reference='SE')

    # k-means with auto-sweep, default channels and feature space
    result = cluster_pixels(img, method='kmeans', k_max=10)

    # Hierarchical, override defaults; use outlier-robust feature space
    result = cluster_pixels(img, method='hierarchical', k_max=10,
                            subsample_size=5000,
                            feature_space='log_robustz',
                            channels=['12C 14N', '12C 15N', '31P', '32S'])

    # Display the labelled cluster image
    plot_cluster_labels(img, result, k=4)

    # The result dict carries all the metrics so you can pick a different k
    # later without re-running the clustering.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

# Import shared elbow heuristics from the histograms module
from pymims_histograms import _elbow_largest_drop, _elbow_kneedle


FEATURE_SPACES = ('log_zscored', 'log_robustz', 'log', 'raw', 'ratios')


# ── SE / topography channel detection ────────────────────────────────────────

def _is_se_channel(label):
    """
    Heuristic: does a channel label look like a secondary-electron /
    topography channel rather than a chemical mass channel?

    SE channels measure topography rather than chemistry, so they don't
    belong in chemistry-driven clustering. Different acquisitions name
    them differently — 'SE', 'Secondary Electron', 'e⁻', 'e-', 'EM',
    sometimes just a number that turns out to be the SE detector.

    This matches the common naming patterns. If your channel uses a
    non-matching label, pass channels= explicitly.
    """
    if not isinstance(label, str):
        return False
    s = label.strip().lower()
    se_patterns = ('se', 'secondary electron', 'e-', 'e⁻', 'em',
                   'topography', 'topo')
    # Match exact tokens, not substrings (so '15n' doesn't match 'em')
    return s in se_patterns or s.startswith('se ') or s.startswith('secondary')


# ── Feature-space construction ───────────────────────────────────────────────

def _build_feature_array(img, channels, feature_space, ratio_pairs=None,
                         min_counts_offset=1.0):
    """
    Build the per-pixel feature matrix for clustering.

    Parameters
    ----------
    img : MimsImage
        Drift-corrected image.
    channels : list[str or int]
        Channels to include in the feature vector.
    feature_space : str
        One of FEATURE_SPACES. See module docstring for semantics.
    ratio_pairs : list[(str, str)]
        For feature_space='ratios': list of (numerator, denominator) pairs.
        Each pair becomes one feature. Required if feature_space='ratios'.
    min_counts_offset : float
        Added to counts before log-transform to avoid log(0).

    Returns
    -------
    X : (n_pixels, n_features) array
    feature_labels : list[str], one per column of X
    image_shape : (H, W) tuple
    """
    summed = img.sum_stack(corrected=True)   # (n_masses, H, W)
    H, W = summed.shape[1], summed.shape[2]

    if feature_space == 'ratios':
        if not ratio_pairs:
            raise ValueError("feature_space='ratios' requires ratio_pairs= "
                             "argument, e.g. [('12C 15N', '12C 14N'), ...]")
        feats = []
        labels = []
        for num, den in ratio_pairs:
            res = img.ratio(num, den)
            r = res['ratio']
            r = np.where(np.isfinite(r), r, 0.0)
            feats.append(r.ravel())
            labels.append(f"{res['num_label']}/{res['den_label']}")
        X = np.column_stack(feats)
        feature_labels = labels
    else:
        ch_indices = [img._resolve_channel(c) for c in channels]
        feature_labels = [img.masses[i] for i in ch_indices]
        # Stack as (n_pixels, n_channels)
        flat = np.stack([summed[i].ravel() for i in ch_indices], axis=1)

        if feature_space == 'raw':
            X = flat.astype(float)
        elif feature_space == 'log':
            X = np.log10(flat + min_counts_offset)
        elif feature_space == 'log_zscored':
            log_x = np.log10(flat + min_counts_offset)
            mu  = log_x.mean(axis=0, keepdims=True)
            sig = log_x.std(axis=0, keepdims=True)
            sig = np.where(sig > 0, sig, 1.0)   # avoid divide-by-zero
            X = (log_x - mu) / sig
        elif feature_space == 'log_robustz':
            # Robust z-scoring: median and MAD instead of mean and std.
            # Outlier-resistant by design — a few hot pixels (cosmic rays,
            # edge artefacts) won't dominate the feature scale. The MAD
            # scaling factor 1.4826 makes MAD an unbiased estimator of σ
            # for normally-distributed data, so 'log_zscored' and
            # 'log_robustz' give identical scales when there are no outliers.
            log_x = np.log10(flat + min_counts_offset)
            med = np.median(log_x, axis=0, keepdims=True)
            mad = np.median(np.abs(log_x - med), axis=0, keepdims=True)
            mad_scaled = mad * 1.4826
            mad_scaled = np.where(mad_scaled > 0, mad_scaled, 1.0)
            X = (log_x - med) / mad_scaled
        else:
            raise ValueError(f"feature_space must be one of {FEATURE_SPACES}; "
                             f"got {feature_space!r}")

    return X, feature_labels, (H, W)


# ── Pixel masking ────────────────────────────────────────────────────────────

def _build_pixel_mask(img, image_shape, min_counts=None,
                      mask_channel=None, pixel_filter=None):
    """
    Construct a boolean mask of pixels to include in clustering.

    Parameters
    ----------
    img : MimsImage
    image_shape : (H, W)
    min_counts : float or None
        Mask out pixels where mask_channel total counts are below this.
    mask_channel : str, int, or None
        Channel used for min_counts threshold. Defaults to channel 0
        (typically a major mass like SE or 12C 14N).
    pixel_filter : 2-D bool array or None
        User-supplied mask. True = include; False = exclude. Combined
        with min_counts via logical AND.

    Returns
    -------
    mask_flat : 1-D bool array of length H*W
    """
    H, W = image_shape
    mask_2d = np.ones((H, W), dtype=bool)

    if min_counts is not None and min_counts > 0:
        mc_idx = (img._resolve_channel(mask_channel)
                  if mask_channel is not None else 0)
        denom = img.sum_stack(corrected=True)[mc_idx]
        mask_2d &= (denom >= min_counts)

    if pixel_filter is not None:
        pf = np.asarray(pixel_filter, dtype=bool)
        if pf.shape != (H, W):
            raise ValueError(f"pixel_filter shape {pf.shape} does not match "
                             f"image shape {(H, W)}")
        mask_2d &= pf

    return mask_2d.ravel()


# ── k-means with metric sweep ────────────────────────────────────────────────

def _run_kmeans_sweep(X, k_max, random_state):
    """
    k-means for k=2..k_max, recording inertia, silhouette,
    Calinski-Harabasz, Davies-Bouldin, and Cubic Clustering Criterion
    scores at each k.

    Returns a dict with arrays indexed by k=2..k_max (k=1 is meaningless
    for silhouette/CH/DB/CCC so we start at 2).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (silhouette_score,
                                     calinski_harabasz_score,
                                     davies_bouldin_score)
    except ImportError:
        raise ImportError(
            "scikit-learn is required for clustering. Install with: "
            "pip install scikit-learn (add --break-system-packages on Crostini)."
        )

    # For silhouette on large samples, use a random subset to keep
    # runtime bounded. Still uses ALL points for cluster fitting.
    n = X.shape[0]
    sil_sample = min(5000, n)
    rng = np.random.default_rng(random_state)
    sil_idx = rng.choice(n, size=sil_sample, replace=False)

    ks = np.arange(2, k_max + 1)
    inertias  = np.full(k_max + 1, np.nan)   # 1-indexed; ignore index 0
    sils      = np.full(k_max + 1, np.nan)
    chs       = np.full(k_max + 1, np.nan)
    dbs       = np.full(k_max + 1, np.nan)
    fits      = {}   # k -> KMeans object

    for k in ks:
        km = KMeans(n_clusters=k, init='k-means++', n_init=5,
                    random_state=random_state, max_iter=300)
        labels = km.fit_predict(X)
        fits[int(k)] = km
        inertias[k] = km.inertia_

        # Quality metrics
        sils[k] = silhouette_score(X[sil_idx], labels[sil_idx])
        chs[k]  = calinski_harabasz_score(X, labels)
        dbs[k]  = davies_bouldin_score(X, labels)

    # k=1 inertia for elbow detection (no clusters → total scatter)
    inertias[1] = float(np.sum((X - X.mean(axis=0))**2))

    # Cubic Clustering Criterion across the swept k range
    ccc_values = _cubic_clustering_criterion(X, ks, inertias)
    cccs = np.full(k_max + 1, np.nan)
    for i, k in enumerate(ks):
        cccs[k] = ccc_values[i]

    return {
        'k_range'              : ks,
        'inertias'             : inertias,
        'silhouettes'          : sils,
        'calinski_harabasz'    : chs,
        'davies_bouldin'       : dbs,
        'cubic_ccc'            : cccs,
        'fits'                 : fits,
    }


# ── Hierarchical with subsample-then-assign ──────────────────────────────────

def _run_hierarchical(X, k_max, subsample_size, linkage_method, random_state):
    """
    Hierarchical agglomerative clustering on a representative subsample;
    every pixel is then assigned to the nearest cluster centroid.

    Returns dict with metrics analogous to k-means and the linkage matrix
    (which drives the dendrogram).
    """
    try:
        from scipy.cluster.hierarchy import linkage, fcluster, cophenet
        from scipy.spatial.distance import pdist
        from sklearn.metrics import (silhouette_score,
                                     calinski_harabasz_score,
                                     davies_bouldin_score)
    except ImportError:
        raise ImportError(
            "scipy and scikit-learn are required for clustering. Install with: "
            "pip install scipy scikit-learn (add --break-system-packages on Crostini)."
        )

    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    sub_n = min(subsample_size, n)
    sub_idx = rng.choice(n, size=sub_n, replace=False)
    X_sub = X[sub_idx]

    # Linkage on subsample
    pairwise = pdist(X_sub, metric='euclidean')
    Z = linkage(pairwise, method=linkage_method)

    # Cophenetic correlation coefficient — quality diagnostic for the
    # dendrogram. Computed once; doesn't depend on the cut height.
    coph_corr, _ = cophenet(Z, pairwise)

    # Sweep k=2..k_max: cut the tree, assign each subsample point a label,
    # compute quality metrics on the subsample (full-image silhouette is
    # too slow at every k).
    sil_sample = min(5000, sub_n)
    sil_idx_local = rng.choice(sub_n, size=sil_sample, replace=False)

    inertias  = np.full(k_max + 1, np.nan)
    sils      = np.full(k_max + 1, np.nan)
    chs       = np.full(k_max + 1, np.nan)
    dbs       = np.full(k_max + 1, np.nan)
    centroids_by_k = {}
    labels_sub_by_k = {}

    for k in range(2, k_max + 1):
        labels_sub = fcluster(Z, t=k, criterion='maxclust')
        labels_sub_by_k[k] = labels_sub
        # Centroid of each cluster in feature space
        unique_labels = np.unique(labels_sub)
        cents = np.zeros((len(unique_labels), X.shape[1]))
        for j, lab in enumerate(unique_labels):
            cents[j] = X_sub[labels_sub == lab].mean(axis=0)
        centroids_by_k[k] = cents

        # Within-cluster sum of squares (analogous to k-means inertia)
        wcss = 0.0
        for j, lab in enumerate(unique_labels):
            diffs = X_sub[labels_sub == lab] - cents[j]
            wcss += float(np.sum(diffs ** 2))
        inertias[k] = wcss

        # Quality metrics on subsample
        try:
            sils[k] = silhouette_score(X_sub[sil_idx_local],
                                        labels_sub[sil_idx_local])
        except ValueError:
            # Silhouette undefined if a cluster has size 1 in the sample
            sils[k] = np.nan
        try:
            chs[k] = calinski_harabasz_score(X_sub, labels_sub)
        except ValueError:
            chs[k] = np.nan
        try:
            dbs[k] = davies_bouldin_score(X_sub, labels_sub)
        except ValueError:
            dbs[k] = np.nan

    # k=1 inertia for elbow
    inertias[1] = float(np.sum((X_sub - X_sub.mean(axis=0))**2))

    # Cubic Clustering Criterion (computed on the same subsample we
    # clustered, since that's what the inertias array reflects)
    ccc_values = _cubic_clustering_criterion(
        X_sub, np.arange(2, k_max + 1), inertias,
    )
    cccs = np.full(k_max + 1, np.nan)
    for i, k in enumerate(range(2, k_max + 1)):
        cccs[k] = ccc_values[i]

    return {
        'k_range'              : np.arange(2, k_max + 1),
        'inertias'             : inertias,
        'silhouettes'          : sils,
        'calinski_harabasz'    : chs,
        'davies_bouldin'       : dbs,
        'cubic_ccc'            : cccs,
        'linkage_matrix'       : Z,
        'cophenetic_corr'      : float(coph_corr),
        'subsample_indices'    : sub_idx,
        'centroids_by_k'       : centroids_by_k,
        'labels_sub_by_k'      : labels_sub_by_k,
    }


def _assign_to_centroids(X, centroids):
    """Nearest-centroid assignment in Euclidean distance. Returns labels
    in 1..k (matching scipy fcluster convention)."""
    # Vectorised: ||X - c||^2 for each centroid c
    # Shape: (n_pixels, k)
    sq_dists = np.sum(X[:, None, :] ** 2, axis=2) \
             - 2 * X @ centroids.T \
             + np.sum(centroids ** 2, axis=1)[None, :]
    return np.argmin(sq_dists, axis=1) + 1   # 1-indexed


# ── Cubic Clustering Criterion (approximate, peak-only) ──────────────────────

def _cubic_clustering_criterion(X, k_range, inertias):
    """
    Approximate Cubic Clustering Criterion as a k-selector.

    Computes a quantity inspired by Sarle's (1983) CCC: the log-ratio of
    (1 - expected R²) to (1 - observed R²), scaled by √(n · p* / 2),
    where p* is the effective dimensionality of the data (participation
    ratio of the covariance eigenvalues). The peak across k is taken as
    a recommendation, alongside silhouette / CH / DB.

    HONESTY NOTE
    ------------
    This implementation differs from Sarle's canonical SAS formula in
    the expected-R² derivation (we use a participation-ratio
    approximation rather than Sarle's full c-prime adjustment). The
    consequence is that the *absolute values* this function returns are
    NOT on Sarle's scale and should NOT be compared to his published
    thresholds (CCC > 3 = good evidence, etc.). Only the *peak position*
    across the k sweep is meaningful — which is what the cluster-count
    selector uses.

    A faithful implementation of Sarle's full formula is non-trivial
    (the SAS source includes empirical adjustments documented only in
    the program text); for k-selection purposes the peak position is
    what matters and this approximation has been validated against
    well-separated synthetic clusters.

    Parameters
    ----------
    X : (n, p) feature matrix
    k_range : iterable of ints
    inertias : array indexed by k — within-cluster SS at each k, with
        index 1 holding the total scatter.

    Returns
    -------
    1-D array of CCC-approximation values, one per k in k_range. Higher
    is better; only relative comparisons across k are meaningful.

    Reference
    ---------
    Sarle, W.S. (1983). "Cubic Clustering Criterion." SAS Technical
    Report A-108. (For the original formulation; this implementation
    approximates it.)
    """
    n, p = X.shape
    k_range = np.asarray(list(k_range))

    if n < 2:
        return np.full(len(k_range), np.nan)

    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    eigvals = (s ** 2) / max(n - 1, 1)
    eigvals = np.maximum(eigvals, 1e-12)

    total_ss = float(np.sum(Xc ** 2))
    if total_ss <= 0:
        return np.full(len(k_range), np.nan)

    # Effective dimensionality (participation ratio)
    eff_p = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    eff_p = max(eff_p, 1.0)

    out = np.full(len(k_range), np.nan)

    for idx, k in enumerate(k_range):
        if k < 2 or k >= n:
            continue
        wcss = float(inertias[k])
        if not np.isfinite(wcss) or wcss <= 0:
            continue

        r2_obs = 1.0 - wcss / total_ss
        r2_obs = np.clip(r2_obs, 1e-9, 1 - 1e-9)

        nk  = max(k, 2)
        npk = max(n / nk, 1.0)
        denom = (nk ** (2.0 / eff_p) + npk ** (2.0 / eff_p) - 2.0)
        if denom <= 0:
            continue
        r2_exp = (nk ** (2.0 / eff_p) - 1.0) / denom
        r2_exp = np.clip(r2_exp, 1e-9, 1 - 1e-9)

        try:
            log_term = np.log((1.0 - r2_exp) / (1.0 - r2_obs))
            scale = np.sqrt(n * eff_p / 2.0) / ((0.001 + r2_exp) ** 1.2)
            out[idx] = float(log_term * scale)
        except (ValueError, FloatingPointError):
            pass

    return out


# ── Cluster-count selection ──────────────────────────────────────────────────

def _select_cluster_count(metrics):
    """
    Aggregate the k-selection metrics into a recommendation set,
    same pattern as pymims_histograms.

    Returns a dict with picks from each criterion. Conservative consensus
    is the smaller of (inertia elbow) and (silhouette peak).
    """
    inertias  = metrics['inertias']
    sils      = metrics['silhouettes']
    chs       = metrics['calinski_harabasz']
    dbs       = metrics['davies_bouldin']
    cccs      = metrics.get('cubic_ccc')
    k_range   = metrics['k_range']

    # Inertia elbow heuristics — reuse our existing infrastructure
    finite_inertias = inertias[~np.isnan(inertias)]
    largest_drop_k = _elbow_largest_drop(finite_inertias)
    kneedle_k      = _elbow_kneedle(finite_inertias)

    # Quality metric peaks/valleys (only over k=2..k_max where they're defined)
    valid = ~np.isnan(sils[k_range])
    sil_peak_k = (int(k_range[valid][np.argmax(sils[k_range][valid])])
                  if valid.any() else None)
    valid = ~np.isnan(chs[k_range])
    ch_peak_k = (int(k_range[valid][np.argmax(chs[k_range][valid])])
                 if valid.any() else None)
    valid = ~np.isnan(dbs[k_range])
    db_min_k = (int(k_range[valid][np.argmin(dbs[k_range][valid])])
                if valid.any() else None)
    if cccs is not None:
        valid = ~np.isnan(cccs[k_range])
        cubic_ccc_peak_k = (int(k_range[valid][np.argmax(cccs[k_range][valid])])
                            if valid.any() else None)
        # Absolute interpretation: best CCC value across all k
        cubic_ccc_max_value = (float(np.nanmax(cccs[k_range]))
                                if valid.any() else None)
    else:
        cubic_ccc_peak_k = None
        cubic_ccc_max_value = None

    # Conservative consensus: inertia elbow + silhouette peak, take min
    candidates = [k for k in (largest_drop_k, kneedle_k, sil_peak_k) if k]
    sensible_k = min(candidates) if candidates else 2

    # Build unique recommendations list — now with cubic CCC included
    by_k = {}
    method_picks = [
        ('inertia_largest_drop', largest_drop_k),
        ('inertia_kneedle', kneedle_k),
        ('silhouette_peak', sil_peak_k),
        ('calinski_harabasz_peak', ch_peak_k),
        ('davies_bouldin_min', db_min_k),
        ('cubic_ccc_peak', cubic_ccc_peak_k),
    ]
    for label, k in method_picks:
        if k is not None:
            by_k.setdefault(k, []).append(label)
    unique_recs = [{'k': k, 'methods': by_k[k]} for k in sorted(by_k)]

    return {
        'sensible_k'              : sensible_k,
        'inertia_largest_drop_k'  : largest_drop_k,
        'inertia_kneedle_k'       : kneedle_k,
        'silhouette_peak_k'       : sil_peak_k,
        'calinski_harabasz_peak_k': ch_peak_k,
        'davies_bouldin_min_k'    : db_min_k,
        'cubic_ccc_peak_k'        : cubic_ccc_peak_k,
        'cubic_ccc_max_value'     : cubic_ccc_max_value,
        'unique_k_recommendations': unique_recs,
    }


# ── Top-level entry point ────────────────────────────────────────────────────

def cluster_pixels(img, method='kmeans', k_max=10,
                   channels=None, include_se=False,
                   feature_space='log_zscored',
                   ratio_pairs=None,
                   min_counts=None, mask_channel=None, pixel_filter=None,
                   subsample_size=5000, linkage_method='ward',
                   random_state=0):
    """
    Cluster the pixels of a NanoSIMS image.

    Parameters
    ----------
    img : MimsImage
        Drift-corrected image.
    method : 'kmeans' or 'hierarchical'
        Clustering algorithm. k-means is fast on every pixel; hierarchical
        runs on a subsample (subsample_size) and then assigns the remainder
        by nearest centroid.
    k_max : int, default 10
        Maximum cluster count to evaluate. The sweep runs k=2..k_max.
    channels : list[str or int] or None
        Channels to include in the feature vector. None (default) = all
        chemical mass channels, with any SE-like topography channels
        auto-excluded (see include_se). Pass an explicit list to override:
        channels=['12C 14N', '12C 15N', '31P', '32S'].
    include_se : bool, default False
        If True, includes any SE-like topography channels in the
        clustering features. Default behaviour excludes them because SE
        measures topography rather than chemistry, and chemistry-driven
        clustering should not be confounded by surface relief. Set True
        only if you specifically want SE in the feature space (rare).
        Has no effect if `channels` is given explicitly — the user list
        is taken at face value.
    feature_space : str
        'log_zscored' (default) — log10 then z-score per channel; equalises
                                 channel contributions and is the standard
                                 NanoSIMS default. Sensitive to a handful
                                 of extreme outlier pixels.
        'log_robustz'          — log10 then median/MAD scaling per channel.
                                 Outlier-resistant: a few hot pixels (cosmic
                                 rays, edge artefacts, glitches) won't
                                 distort the feature scale. Use this when
                                 you suspect the image has a few extreme
                                 pixels you don't want dominating clusters.
                                 Identical to 'log_zscored' for outlier-
                                 free data thanks to MAD's 1.4826 scaling
                                 factor.
        'log'                  — log10 only; preserves relative magnitudes.
        'raw'                  — raw counts; high-rate channels dominate.
        'ratios'               — use ratio images instead of channel counts;
                                 requires ratio_pairs argument.
    ratio_pairs : list[(num, den), ...] or None
        Required when feature_space='ratios'. Each pair becomes one feature.
    min_counts : float or None
        Mask out pixels where mask_channel total counts are below this.
        Default None = no count-based masking.
    mask_channel : str, int, or None
        Channel used for the min_counts test. Defaults to channel 0.
    pixel_filter : 2-D bool array or None
        Custom mask. True = include; False = exclude. Combined with
        min_counts via logical AND. Use this to apply v0.6 GMM-derived
        thresholds (e.g. cluster only pixels above the labelled-population
        crossing for a given channel).
    subsample_size : int, default 5000
        Hierarchical only: number of representative pixels for the
        subsample-then-assign workflow. Increase for more representative
        rare populations; decrease for speed.
    linkage_method : str, default 'ward'
        Hierarchical only: scipy linkage method. 'ward' minimises within-
        cluster variance and is the safe default for z-scored Euclidean
        features. Others: 'single', 'complete', 'average', 'centroid'.
    random_state : int, default 0
        Seed for k-means initialisation and hierarchical subsample.

    Returns
    -------
    ClusterResult : dict with keys:
        'method'         : 'kmeans' or 'hierarchical'
        'feature_space'  : as supplied
        'feature_labels' : list[str], one per feature dimension
        'image_shape'    : (H, W)
        'mask'           : 2-D bool array, True where pixel was clustered
        'k_range'        : array of k values evaluated
        'inertias'       : within-cluster sum of squares vs k
        'silhouettes'    : silhouette score vs k
        'calinski_harabasz' : CH score vs k
        'davies_bouldin' : DB score vs k
        'sensible_k'     : conservative consensus k recommendation
        'unique_k_recommendations' : list of {'k', 'methods'} dicts
        'labels_by_k'    : dict {k -> 2-D label image; NaN where masked}
        'centroids_by_k_features' : dict {k -> centroids in feature space}
        'centroids_by_k_counts'   : dict {k -> centroids in raw count units,
                                          for biological interpretation}
        'cluster_sizes_by_k' : dict {k -> array of pixel counts per cluster}
        # Hierarchical only:
        'linkage_matrix' : scipy linkage Z (drives dendrogram)
        'cophenetic_corr': CCC quality diagnostic
        'subsample_indices' : the pixels used in the subsample
    """
    if method not in ('kmeans', 'hierarchical'):
        raise ValueError(f"method must be 'kmeans' or 'hierarchical'; "
                         f"got {method!r}")
    if feature_space not in FEATURE_SPACES:
        raise ValueError(f"feature_space must be one of {FEATURE_SPACES}")

    # Resolve channel list
    if channels is None and feature_space != 'ratios':
        # Default: all channels EXCEPT SE-like topography channels.
        # SE measures sample topography, not chemistry, so it doesn't
        # belong in chemistry-driven clustering. Use include_se=True if
        # you genuinely want to include it.
        all_indices = list(range(len(img.masses)))
        if include_se:
            channels = all_indices
        else:
            channels = [i for i in all_indices
                        if not _is_se_channel(img.masses[i])]
            excluded = [img.masses[i] for i in all_indices
                        if _is_se_channel(img.masses[i])]
            if excluded:
                print(f"Auto-excluded SE-like channel(s) from clustering: "
                      f"{excluded}. Pass include_se=True to keep them, or "
                      f"channels=[...] for explicit control.")

    # Build full feature matrix
    X_full, feature_labels, (H, W) = _build_feature_array(
        img, channels, feature_space, ratio_pairs=ratio_pairs,
    )

    # Apply pixel mask
    mask_flat = _build_pixel_mask(img, (H, W), min_counts=min_counts,
                                  mask_channel=mask_channel,
                                  pixel_filter=pixel_filter)
    if not mask_flat.any():
        raise ValueError("Pixel mask excludes all pixels — relax masking.")

    X = X_full[mask_flat]

    # Cluster
    if method == 'kmeans':
        sweep = _run_kmeans_sweep(X, k_max, random_state)
        # Per-k labels: refit centroids would just be sweep['fits'][k].labels_
        labels_by_k_flat = {}
        centroids_by_k = {}
        cluster_sizes_by_k = {}
        for k in sweep['k_range']:
            kfit = sweep['fits'][int(k)]
            lab_clustered = kfit.labels_ + 1   # 1-indexed for consistency
            full = np.full(H * W, np.nan)
            full[mask_flat] = lab_clustered
            labels_by_k_flat[int(k)] = full.reshape(H, W)
            centroids_by_k[int(k)] = kfit.cluster_centers_
            unique, counts = np.unique(lab_clustered, return_counts=True)
            cluster_sizes_by_k[int(k)] = counts
        selection = _select_cluster_count(sweep)
        result = {
            'method': 'kmeans',
            'feature_space': feature_space,
            'feature_labels': feature_labels,
            'image_shape': (H, W),
            'mask': mask_flat.reshape(H, W),
            'k_range': sweep['k_range'],
            'inertias': sweep['inertias'],
            'silhouettes': sweep['silhouettes'],
            'calinski_harabasz': sweep['calinski_harabasz'],
            'davies_bouldin': sweep['davies_bouldin'],
            'cubic_ccc': sweep['cubic_ccc'],
            'labels_by_k': labels_by_k_flat,
            'centroids_by_k_features': centroids_by_k,
            'cluster_sizes_by_k': cluster_sizes_by_k,
            'random_state': random_state,
            **selection,
        }

    else:   # hierarchical
        sweep = _run_hierarchical(X, k_max, subsample_size,
                                  linkage_method, random_state)
        # Assign every pixel to nearest centroid for each k
        labels_by_k_flat = {}
        centroids_by_k = {}
        cluster_sizes_by_k = {}
        for k in sweep['k_range']:
            cents = sweep['centroids_by_k'][int(k)]
            lab = _assign_to_centroids(X, cents)
            full = np.full(H * W, np.nan)
            full[mask_flat] = lab
            labels_by_k_flat[int(k)] = full.reshape(H, W)
            centroids_by_k[int(k)] = cents
            unique, counts = np.unique(lab, return_counts=True)
            cluster_sizes_by_k[int(k)] = counts
        selection = _select_cluster_count(sweep)
        result = {
            'method': 'hierarchical',
            'feature_space': feature_space,
            'feature_labels': feature_labels,
            'image_shape': (H, W),
            'mask': mask_flat.reshape(H, W),
            'k_range': sweep['k_range'],
            'inertias': sweep['inertias'],
            'silhouettes': sweep['silhouettes'],
            'calinski_harabasz': sweep['calinski_harabasz'],
            'davies_bouldin': sweep['davies_bouldin'],
            'cubic_ccc': sweep['cubic_ccc'],
            'linkage_matrix': sweep['linkage_matrix'],
            'cophenetic_corr': sweep['cophenetic_corr'],
            'subsample_indices': sweep['subsample_indices'],
            'linkage_method': linkage_method,
            'subsample_size': subsample_size,
            'labels_by_k': labels_by_k_flat,
            'centroids_by_k_features': centroids_by_k,
            'cluster_sizes_by_k': cluster_sizes_by_k,
            'random_state': random_state,
            **selection,
        }

    # Translate centroids into raw-count space for biological interpretation
    result['centroids_by_k_counts'] = _centroids_to_counts(
        img, channels, feature_space, centroids_by_k,
    )

    return result


def _centroids_to_counts(img, channels, feature_space, centroids_by_k):
    """
    Translate cluster centroids in feature space back into raw-count units
    so the user can read "this cluster has ~X counts of 12C 14N, ~Y counts
    of 31P" rather than reasoning about z-scored log-space.

    For non-ratio feature spaces, this is the inverse of the feature
    transform applied to the per-pixel data. For 'ratios' feature space,
    centroids are already in physically meaningful units, so we pass through.
    """
    if feature_space == 'ratios':
        return {k: c.copy() for k, c in centroids_by_k.items()}

    # Reconstruct the global mean and std in log space
    summed = img.sum_stack(corrected=True)
    ch_indices = [img._resolve_channel(c) for c in channels]
    flat = np.stack([summed[i].ravel() for i in ch_indices], axis=1)
    log_x = np.log10(flat + 1.0)
    mu  = log_x.mean(axis=0)
    sig = log_x.std(axis=0)
    sig = np.where(sig > 0, sig, 1.0)

    out = {}
    for k, c in centroids_by_k.items():
        if feature_space == 'log_zscored':
            log_centroids = c * sig + mu
            counts = 10.0 ** log_centroids - 1.0
        elif feature_space == 'log_robustz':
            # Recompute median/MAD on the same data
            med = np.median(log_x, axis=0)
            mad = np.median(np.abs(log_x - med), axis=0) * 1.4826
            mad = np.where(mad > 0, mad, 1.0)
            log_centroids = c * mad + med
            counts = 10.0 ** log_centroids - 1.0
        elif feature_space == 'log':
            counts = 10.0 ** c - 1.0
        elif feature_space == 'raw':
            counts = c.copy()
        else:
            counts = c.copy()
        out[k] = np.maximum(counts, 0.0)   # tiny negatives from offset
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def _cophenetic_interpretation(corr):
    """
    Short interpretive label for a cophenetic correlation value.

    Returns (band_label, plain_english) — used on figure titles to
    save the user from having to remember the thresholds. Bands follow
    the standard cluster-analysis convention (Romesburg 1984).
    """
    if corr >= 0.90:
        return ('strong',
                'dendrogram faithfully represents the data')
    elif corr >= 0.80:
        return ('good',
                'dendrogram represents the data well')
    elif corr >= 0.70:
        return ('moderate',
                'some distortion; cluster cuts still usable')
    else:
        return ('weak',
                'noticeable distortion; treat cluster cuts with caution')


def _format_cophenetic_block(corr):
    """Compact two-line block for figure titles. Bold-ish via newline."""
    band, hint = _cophenetic_interpretation(corr)
    return f'cophenetic corr = {corr:.3f}  ({band})\n{hint}'


def plot_cluster_labels(img, result, k=None, cmap='tab10',
                        outpath=None, show=True):
    """
    Display the labelled cluster image at a chosen k.

    Parameters
    ----------
    img : MimsImage
    result : ClusterResult dict from cluster_pixels()
    k : int or None
        Which k to display. None = result['sensible_k'].
    cmap : str
        Categorical colormap. 'tab10' is good up to 10 clusters; 'tab20' for more.
    outpath : str or None
        If given, save to file.
    show : bool
        If False, return figure without displaying.
    """
    if k is None:
        k = result['sensible_k']
    if k not in result['labels_by_k']:
        raise ValueError(f"k={k} not in result['labels_by_k']; "
                         f"available: {sorted(result['labels_by_k'])}")

    labels = result['labels_by_k'][k]
    field_um = img.metadata['field_um']

    fig, (ax_img, ax_table) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={'width_ratios': [1.2, 1]},
        facecolor='white',
    )

    # Cluster label image
    cmap_obj = plt.get_cmap(cmap)
    n_colours = max(k, 1)
    # Build a discrete cmap with NaN handled as transparent
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = [cmap_obj(i / max(n_colours - 1, 1)) for i in range(n_colours)]
    discrete = ListedColormap(colors)
    bounds = np.arange(0.5, n_colours + 1.5)
    norm = BoundaryNorm(bounds, discrete.N)

    masked = np.ma.masked_invalid(labels)
    discrete.set_bad(color='black')
    im = ax_img.imshow(masked, extent=[0, field_um, field_um, 0],
                       cmap=discrete, norm=norm,
                       interpolation='nearest')
    ax_img.set_title(f"{result['method']}  k={k}  ({result['feature_space']})",
                     fontsize=11, fontweight='bold')
    ax_img.set_xlabel('μm'); ax_img.set_ylabel('μm')
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.04, pad=0.03,
                         ticks=np.arange(1, n_colours + 1))
    cbar.set_label('cluster')

    # Per-cluster summary table on the right axis
    ax_table.axis('off')
    counts_centroids = result['centroids_by_k_counts'][k]
    sizes = result['cluster_sizes_by_k'][k]
    feat_labels = result['feature_labels']

    # Header
    header = ['#', 'pixels', '%'] + [f'{f}\n(counts)' for f in feat_labels]
    rows = []
    total = sum(sizes)
    for i in range(n_colours):
        cluster_id = i + 1
        size = sizes[i] if i < len(sizes) else 0
        pct = 100.0 * size / total if total else 0
        # Centroid in raw-count space, rounded to integer
        centroid = counts_centroids[i] if i < counts_centroids.shape[0] else \
                   np.zeros(len(feat_labels))
        row = [str(cluster_id), f'{size:,}', f'{pct:.1f}%'] + \
              [f'{v:.1f}' for v in centroid]
        rows.append(row)
    # Render as a matplotlib table
    tbl = ax_table.table(cellText=rows, colLabels=header,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    # Colour the cluster-id column to match the image cmap
    for i in range(n_colours):
        cell = tbl[i + 1, 0]   # +1 for header row
        cell.set_facecolor(colors[i])
        cell.set_text_props(color='white' if i % 2 == 0 else 'black',
                            fontweight='bold')

    ax_table.set_title('Cluster summary (centroid means, raw counts)',
                       fontsize=10, pad=8)

    # If hierarchical, add a 2-line cophenetic-correlation footnote below
    # the table. The table itself sits in the upper portion of ax_table;
    # this places the footnote in the lower portion.
    if result['method'] == 'hierarchical':
        band, hint = _cophenetic_interpretation(result['cophenetic_corr'])
        footnote = (
            f"Cophenetic correlation = {result['cophenetic_corr']:.3f} ({band})\n"
            f"   {hint}.\n"
            f"   Measures how well the dendrogram preserves the original\n"
            f"   pairwise distances between pixels. ≥ 0.9 = strong;\n"
            f"   0.8–0.9 = good; 0.7–0.8 = moderate; < 0.7 = weak."
        )
        ax_table.text(
            0.0, -0.05, footnote,
            transform=ax_table.transAxes,
            fontsize=8, ha='left', va='top',
            family='monospace',
            color='#444',
        )

    # Top-level info
    info_bits = [f"feature space: {result['feature_space']}",
                 f"channels: {', '.join(feat_labels)}"]
    if result['method'] == 'hierarchical':
        band, _ = _cophenetic_interpretation(result['cophenetic_corr'])
        info_bits.append(
            f"cophenetic corr: {result['cophenetic_corr']:.3f} ({band})"
        )
    fig.suptitle('  |  '.join(info_bits), fontsize=9, y=1.0)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {outpath}')
    if not show:
        plt.close(fig)
    return fig


def plot_metric_sweep(result, outpath=None, show=True):
    """
    Plot the five cluster-count selection metrics on a 2×3 grid:
    inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, and Cubic CCC.
    The picks from each method are marked with vertical lines.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor='white')

    k_range = result['k_range']
    inertias = result['inertias'][k_range]
    sils     = result['silhouettes'][k_range]
    chs      = result['calinski_harabasz'][k_range]
    dbs      = result['davies_bouldin'][k_range]
    cccs     = result.get('cubic_ccc', np.full(len(inertias) + 2, np.nan))[k_range]

    sensible = result['sensible_k']

    panels = [
        (axes[0][0], 'Inertia (within-cluster SS)', inertias, 'lower=better',
         result['inertia_largest_drop_k'], 'darkred', 'largest drop',
         result['inertia_kneedle_k'], 'darkblue', 'kneedle'),
        (axes[0][1], 'Silhouette score', sils, 'higher=better',
         result['silhouette_peak_k'], 'darkgreen', 'peak',
         None, None, None),
        (axes[0][2], 'Cubic Clustering Criterion (approx)', cccs,
         'higher=better; peak position only',
         result.get('cubic_ccc_peak_k'), 'darkgreen', 'peak',
         None, None, None),
        (axes[1][0], 'Calinski-Harabasz', chs, 'higher=better',
         result['calinski_harabasz_peak_k'], 'darkgreen', 'peak',
         None, None, None),
        (axes[1][1], 'Davies-Bouldin', dbs, 'lower=better',
         result['davies_bouldin_min_k'], 'darkgreen', 'min',
         None, None, None),
    ]

    # Hide the unused 6th panel
    axes[1][2].axis('off')

    for ax, title, y, hint, pk1, c1, l1, pk2, c2, l2 in panels:
        ax.plot(k_range, y, 'o-', color='steelblue', linewidth=1.4)
        if pk1 is not None:
            ax.axvline(pk1, color=c1, linewidth=1.0, linestyle='--',
                       alpha=0.8, label=f'{l1}: k={pk1}')
        if pk2 is not None:
            ax.axvline(pk2, color=c2, linewidth=1.0, linestyle=':',
                       alpha=0.8, label=f'{l2}: k={pk2}')
        ax.axvline(sensible, color='black', linewidth=1.5, alpha=0.5,
                   label=f'sensible: k={sensible}')
        ax.set_xlabel('k')
        ax.set_title(f'{title}\n({hint})', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xticks(k_range)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # CCC panel: no reference lines — our implementation isn't on Sarle's
    # absolute scale; only the peak position is meaningful.

    title = f"Cluster-count selection  ({result['method']}"
    if result['method'] == 'hierarchical':
        band, _ = _cophenetic_interpretation(result['cophenetic_corr'])
        title += (f", cophenetic corr={result['cophenetic_corr']:.3f} "
                  f"({band})")
    title += ')'
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {outpath}')
    if not show:
        plt.close(fig)
    return fig


def plot_dendrogram(result, k_marks=None, outpath=None, show=True):
    """
    Dendrogram for hierarchical results, with optional horizontal cut
    lines at chosen k values.

    Parameters
    ----------
    result : ClusterResult dict from cluster_pixels(method='hierarchical')
    k_marks : list[int] or None
        Cut-line annotations. None = result['sensible_k'].
    """
    if result['method'] != 'hierarchical':
        raise ValueError("plot_dendrogram requires a hierarchical clustering "
                         "result.")
    from scipy.cluster.hierarchy import dendrogram, fcluster

    Z = result['linkage_matrix']
    if k_marks is None:
        k_marks = [result['sensible_k']]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    # Truncate to 30 leaves for readability — we have 5000 subsample points
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_font_size=8, leaf_rotation=90,
               above_threshold_color='gray',
               color_threshold=Z[-(max(k_marks)) + 1, 2] if k_marks else None)

    # Cut lines at chosen k values
    for k in k_marks:
        if k <= 1 or k > len(Z) + 1:
            continue
        # Height at which the (n - k + 1)th merge happens — anything below
        # that height yields k clusters.
        height = Z[-(k - 1), 2]
        ax.axhline(height, color='darkred', linewidth=1.0, linestyle='--',
                   alpha=0.7, label=f'k={k}')

    title_main = f"Dendrogram ({result['linkage_method']} linkage)"
    coph_block = _format_cophenetic_block(result['cophenetic_corr'])
    ax.set_title(f"{title_main}\n{coph_block}",
                 fontsize=11, pad=10)
    ax.set_xlabel('cluster ID (or count of merged leaves)', fontsize=9)
    ax.set_ylabel('linkage distance', fontsize=9)
    if k_marks:
        ax.legend(fontsize=8, frameon=False, loc='upper right')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved: {outpath}')
    if not show:
        plt.close(fig)
    return fig
