# `pymims_clustering.py` — Pixel clustering

This module performs per-pixel clustering on a NanoSIMS stack using k-means or hierarchical agglomerative methods, with cluster-count selection driven by multiple criteria that may disagree. It produces ROI-ready boolean masks per cluster and the overlays needed to display them on top of any image (mass channel, ratio, δ, or HSI). It is the structural counterpart to `pymims_histograms.py`: the histogram module asks "what populations exist on each channel separately?", the clustering module asks "what populations exist when all channels are considered together?"

## Contents

1. [What it does](#what-it-does)
2. [Pixel clustering vs channel clustering](#pixel-clustering-vs-channel-clustering)
3. [Quick start](#quick-start)
4. [The feature space](#the-feature-space)
5. [SE auto-exclusion](#se-auto-exclusion)
6. [Cluster-count selection: five criteria, one consensus](#cluster-count-selection-five-criteria-one-consensus)
7. [The cubic CCC and why our implementation is approximate](#the-cubic-ccc-and-why-our-implementation-is-approximate)
8. [Cophenetic correlation: dendrogram quality](#cophenetic-correlation-dendrogram-quality)
9. [Plotting functions and overlays](#plotting-functions-and-overlays)
10. [The half-pixel coordinate-frame issue](#the-half-pixel-coordinate-frame-issue)
11. [Pre-masking and feeding from `pymims_histograms`](#pre-masking-and-feeding-from-pymims_histograms)
12. [`extract_cluster_masks` and the depth-profile path](#extract_cluster_masks-and-the-depth-profile-path)
13. [Design decisions](#design-decisions)
14. [Known limitations](#known-limitations)

---

## What it does

Given a drift-corrected `MimsImage`, this module:

1. Constructs a per-pixel feature vector across selected channels (default: all chemical channels in z-scored log-counts; SE auto-excluded)
2. Runs k-means or hierarchical clustering with k = 2..k_max
3. Reports five cluster-count selection metrics that may disagree (inertia elbow methods, silhouette, Calinski-Harabasz, Davies-Bouldin, plus an approximate Cubic Clustering Criterion)
4. For hierarchical, reports the cophenetic correlation as a dendrogram-quality diagnostic
5. Renders cluster-label images, side-by-side k comparisons, metric panels, and contour overlays on any base image (channel, ratio, δ, HSI)
6. Returns ROI-ready boolean masks per cluster for downstream analysis (statistics, depth profiling)

The output is a single dict (`ClusterResult`) that carries every k worth examining. Visualisation is then a question of which k to display, not of which to compute.

## Pixel clustering vs channel clustering

This module performs **pixel** clustering: each pixel is a point in N-channel space, and we cluster the points. The output groups pixels by their multi-channel chemical signature, producing ROI-ready masks for biology like "this region of the cell" or "the labelled organelles".

**Channel clustering** — treating each channel as a point in N-pixel space and clustering the channels — is a different analysis that answers "which mass channels co-localise?". It is a multivariate-statistics question (PCA, correlation matrices, MFA) that doesn't produce ROI masks. It belongs in a separate module which this library does not yet provide.

If you find yourself wanting to know "are 31P and 32S behaving the same way across the image?", that's channel clustering and is on the v0.9 roadmap.

## Quick start

```python
from pymims import MimsImage
from pymims_clustering import (cluster_pixels, plot_cluster_labels,
                                plot_metric_sweep, plot_overlay,
                                plot_cluster_grid, extract_cluster_masks)

img = MimsImage('data.im')
img.auto_drop_bad_planes()                 # see pymims.md
img.drift_correct(reference='SE')

# k-means with auto-sweep over k=2..10
result = cluster_pixels(img, method='kmeans', k_max=10, random_state=0)

# Quality metrics — five panels showing where each criterion peaks/valleys
plot_metric_sweep(result)

# Cluster image at the recommended (sensible_k) cut
plot_cluster_labels(img, result)

# Side-by-side comparison of k=2,3,4,5 partitions
plot_cluster_grid(img, result, k_list=[2, 3, 4, 5])

# Cluster outlines overlaid on the 32S channel image
plot_overlay(img, result, k=3, base='channel', channel='32S',
             min_pixels=10)

# Boolean masks per cluster for downstream ROI analysis
masks = extract_cluster_masks(result, k=3)
```

Hierarchical clustering uses the same API:

```python
result_h = cluster_pixels(img, method='hierarchical', k_max=10,
                          subsample_size=5000, random_state=0)
```

## The feature space

What goes into the per-pixel feature vector matters. Four options, controlled by the `feature_space=` kwarg:

| Value | What it does | When to use |
|-------|--------------|-------------|
| `'log_zscored'` (default) | log10 then z-score per channel | Standard NanoSIMS default; equalises channel contributions |
| `'log_robustz'` | log10 then median/MAD per channel | Outlier-resistant; use if you suspect cosmic rays or hot pixels |
| `'log'` | log10 only | Preserves relative magnitudes; high-rate channels still dominate but less than raw |
| `'raw'` | raw counts | Almost never the right choice; high-rate channels overwhelm the clustering |
| `'ratios'` | use ratio images instead of channel counts | When biology is fundamentally about ratios (¹⁵N enrichment); requires `ratio_pairs=` |

The default `'log_zscored'` is what you want 95% of the time. Z-scoring (mean=0, std=1 per channel) makes a 200-count change in 31P count for as much as a 200-count change in 12C 14N, even though absolute counts differ by an order of magnitude. Without it, clustering collapses to "intensity bins of the brightest channel".

Empirical finding from the v0.7 testing: **`log_robustz` makes essentially no difference for the typical NanoSIMS contamination pattern** (a few cosmic-ray pixels on one channel). The standard z-scoring is more outlier-tolerant than expected because cosmic rays usually form their own cluster anyway. Where `log_robustz` does earn its keep is when outliers are on a low-count channel (say a single 50-count pixel on a channel where most pixels are 1-2 counts), or when an instrument glitch hits multiple channels simultaneously. Use it when you have specific reason to suspect either.

## SE auto-exclusion

By default, channels recognised as secondary-electron / topography channels are **auto-excluded** from clustering. Detection covers the common naming variants:

- `SE`, `se`
- `Secondary Electron`
- `e-`, `e⁻`
- `EM`, `topography`, `topo`

The auto-exclusion happens only when `channels=None` (the default channel-list shortcut). If you pass an explicit `channels=` list, that list is taken at face value — your choice overrides the heuristic.

To override:

```python
# Include SE deliberately
result = cluster_pixels(img, include_se=True)

# Or specify channels explicitly
result = cluster_pixels(img, channels=['12C 14N', '31P', '32S'])
```

When auto-exclusion fires, the module prints a one-line message naming the excluded channels. The message is **deduplicated within a session** — it appears once per (image, exclusion-list) combination, then stays quiet on subsequent calls. To silence it entirely: `verbose=False`.

The reason for this default: SE measures sample topography (surface relief, edges, charging), not chemistry. Including it in chemistry-driven clustering biases the partition toward geometric features rather than biological structure. Materials-science applications without an SE channel are unaffected because the heuristic just never matches.

## Cluster-count selection: five criteria, one consensus

The cluster count k is the central modelling choice and there is no single right answer. The module computes five quantities and lets you see all of them:

### Inertia (within-cluster sum of squares)

The standard k-means objective. Decreases monotonically with k — you cannot pick a "best" k from this curve directly. Two elbow heuristics applied:

- **Largest drop**: the k just after the biggest single decrement in inertia.
- **Kneedle** (Satopää 2011): point of maximum perpendicular distance from the line connecting first and last points after axis normalisation.

Both implementations are reused from `pymims_histograms.py` so the elbow logic is identical to what's used for the GMM ΔBIC curve.

### Silhouette score

For each pixel, the ratio of "how close it is to its own cluster's other pixels" vs "how close it is to the nearest other cluster's pixels". Range −1 to +1; higher is better. Computed on a 5000-pixel subsample for speed (full computation is O(n²) and impractical at 65,000 pixels).

### Calinski-Harabasz score

Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better, peak indicates the most-separable k.

### Davies-Bouldin score

Average ratio of within-cluster scatter to between-cluster separation. Lower is better; minimum indicates the cleanest k.

### Cubic Clustering Criterion (approximate)

See [next section](#the-cubic-ccc-and-why-our-implementation-is-approximate).

### The consensus

`sensible_k` is the conservative consensus: the smaller of (inertia-elbow consensus) and (silhouette peak). When the methods agree, that is the consensus. When they disagree, the conservative pick errs toward fewer clusters — easier to recover from (override with `k=N` later) than to detect (a too-many-clusters partition can hide real structure inside an over-fit).

`unique_k_recommendations` lists every k picked by any of the five methods, with the methods that picked it. This is the input for `plot_cluster_grid(...)` which renders one panel per unique recommendation so you can visually compare them.

The default `sensible_k` will be wrong in some cases — particularly when the conservative pick hides biological structure. The grid view exists exactly to catch that. The pattern is: run with default, look at the grid, override with `k=N` in your downstream `plot_*` calls.

## The cubic CCC and why our implementation is approximate

The Cubic Clustering Criterion (CCC, Sarle 1983) is the canonical SAS / JMP cluster-count selector. The published interpretation:

- CCC > 3 = good evidence of clustering
- 2 < CCC ≤ 3 = supports the existence of clusters
- CCC < 2 = weak or no evidence

These thresholds are decades-validated empirical guidance from SAS practice.

**Our implementation is *approximate* and these thresholds DO NOT apply to our values.** This is documented prominently in the source code, the print summary, and the metric-panel hint. The reason for the approximation: Sarle's full formula uses an "equivalent number of dimensions" p* derived from a c-prime adjustment that is only documented in SAS source code. We use a participation-ratio approximation (the effective rank of the covariance eigenvalues) that captures the curve shape but produces values orders of magnitude larger than Sarle's.

What our implementation IS good for: **the peak position across k**. When all five criteria agree, the peak position contributes a fifth vote. When they disagree, the cubic CCC peak is one piece of evidence to weigh against the others. Where it shouldn't be used: claims like "CCC=3.5 means we have good evidence of clustering". Our values aren't on that scale.

A faithful Sarle implementation would be a worthwhile future enhancement (it's a small but fiddly job — Milligan & Cooper 1985 has the formulae and validation cases). For now, treat cubic CCC as a relative-only selector.

## Cophenetic correlation: dendrogram quality

For hierarchical clustering only, `result['cophenetic_corr']` reports the cophenetic correlation coefficient — a separate quantity from the Cubic Clustering Criterion despite the same acronym in some literatures. **The acronyms collide and we deliberately distinguish them in this module.**

The cophenetic correlation is the Pearson correlation between:

- The original pairwise distances between data points in feature space
- The cophenetic distances — the dendrogram heights at which each pair of points is first joined into the same cluster

Range 0 to 1; higher means the dendrogram structure faithfully represents the original distances. Standard interpretation (Romesburg 1984):

- ≥ 0.90 — strong: dendrogram faithfully represents the data
- 0.80–0.90 — good: represents the data well
- 0.70–0.80 — moderate: some distortion; cluster cuts still usable
- < 0.70 — weak: noticeable distortion; treat cluster cuts with caution

The interpretation appears on every figure that uses the cophenetic correlation (cluster-labels suptitle, metric-sweep title, dendrogram footnote) so you don't have to remember the bands.

When cophenetic is weak, it means the dendrogram cut question becomes unreliable but **k-means on the same feature space is unaffected**. The standard cross-check: run both methods at the suspect k value and compare their partitions. If they agree, the partition is real even if the dendrogram structure is distorted; if they disagree, both methods are seeing only weak structure.

## Plotting functions and overlays

| Function | What it shows |
|----------|---------------|
| `plot_cluster_labels(img, result, k=...)` | Single labelled cluster image with summary table (sizes, raw-count centroids per cluster). Default k = `sensible_k`. |
| `plot_cluster_grid(img, result, k_list=[2,3,4,5])` | Side-by-side comparison of multiple k values. The `sensible_k` panel is highlighted in green; other panels show which method(s) picked them. |
| `plot_metric_sweep(result)` | 2×3 grid showing inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, and cubic CCC vs k, with picks from each method marked. |
| `plot_dendrogram(result, k_marks=[3,4])` | Full hierarchical dendrogram with horizontal cut lines at chosen k. Available for hierarchical results only; not part of the routine clustering workflow but useful for understanding tree structure. |
| `plot_overlay(img, result, k=..., base='channel'\|'ratio'\|'delta'\|'hsi', ...)` | Cluster contour outlines drawn on top of a base image. Each cluster gets its own colour. |

`plot_overlay` is the workhorse for "which pixels of *this* image are which clusters". The four base modes:

- `base='channel'` requires `channel='12C 14N'` (or similar)
- `base='ratio'` requires `numerator=` and `denominator=`
- `base='delta'` requires numerator/denominator plus `base_kwargs={'reference': 0.0037}` for the natural-abundance reference
- `base='hsi'` requires numerator/denominator plus optional scaling kwargs in `base_kwargs`

Cluster outlines come from `skimage.measure.find_contours` (sub-pixel marching squares) so the boundaries are smooth rather than jagged.

## The half-pixel coordinate-frame issue

A subtle bug we hit during development that's worth documenting because it's a class of error endemic to scientific image-processing code:

`matplotlib.pyplot.imshow` with `extent=[0, field_um, field_um, 0]` places the **outer edge** of the image at 0 and `field_um`. The centre of pixel (0, 0) is therefore at `(0.5*um_per_px, 0.5*um_per_px)` — half a pixel inset from the corner. Meanwhile, `skimage.measure.find_contours` returns coordinates where integer (i, j) is the **pixel centre**. Naively converting find_contours output to micrometres by dividing by `H/field_um` shifts every contour up-and-left by half a pixel.

The visible symptom: cluster outlines that look "almost right but a bit off" — they sit inside the visible cluster boundary on the underlying image rather than on it. The fix is a `+ 0.5` in both the row and column conversions:

```python
ys = (ctr[:, 0] + 0.5) * (field_um / H)
xs = (ctr[:, 1] + 0.5) * (field_um / W)
```

Both `plot_overlay` (in this module) and `plot_rule_masks` (in `pymims_rules.py`) apply this correction. Validation: a synthetic circle at known coordinates is recovered to sub-pixel precision.

If you see contour offsets in figures from any third-party plotting code, this is the first thing to check.

## `min_pixels`: filtering speckle

Cluster boundaries can be visually noisy when scattered single-pixel "islands" of one cluster appear inside another. These are usually statistical artefacts rather than biological structure — Poisson noise is enough to flip individual pixels at cluster boundaries.

`plot_overlay` and `plot_rule_masks` both accept `min_pixels=N`: connected components smaller than N pixels are filtered out before contouring. Default is 1 (no filtering). For real biological work:

- `min_pixels=5–10` — light cleanup, removes truly isolated single pixels
- `min_pixels=20–30` — moderate, suppresses speckle while keeping organelle-scale features
- `min_pixels=50–100` — aggressive, only shows large contiguous regions

The slider widget (`cluster_overlay_slider` in `pymims_explore.py`) lets you experiment with this interactively. **Be careful with aggressive filtering** — Poisson-pattern hot-spots (e.g. the Fe:S clusters from the v0.6 test data) can legitimately be single-pixel features and disappear entirely under high `min_pixels`. The right value depends on what you're trying to see.

Implementation uses `skimage.measure.label` with 8-connectivity (diagonal pixels count as connected), then thresholds the component sizes from `np.bincount`.

## Pre-masking and feeding from `pymims_histograms`

By default, clustering operates on every pixel in the image. Two ways to restrict to a subset:

```python
# Simple count threshold on a chosen channel
result = cluster_pixels(img, min_counts=200, mask_channel='12C 14N')

# Arbitrary boolean mask (e.g. derived from histogram thresholds)
from pymims_histograms import plot_histograms, best_thresholds
hists = plot_histograms(img, k_max=6, show=False, verbose=False)
# Mask = pixels where 12C 14N is above the labelled-population GMM crossing
threshold = best_thresholds(hists)['12C 14N'][0]
custom_mask = img.sum_stack(corrected=True)[1] >= threshold
result = cluster_pixels(img, pixel_filter=custom_mask)
```

This composability is the point: histogram-derived thresholds become inputs to clustering, clustering masks become inputs to ROI rules, ROI rules become inputs to depth profiles. Each layer feeds the next.

## `extract_cluster_masks` and the depth-profile path

```python
masks = extract_cluster_masks(result, k=3)
# {1: <2D bool>, 2: <2D bool>, 3: <2D bool>}
```

Cluster IDs are 1-indexed to match the rest of the module (the cluster summary tables, the colour palette, etc.). Each mask is a 2-D boolean array of the same shape as the image.

This is the **depth-profile-ready API**. Per-plane statistics inside any cluster are then a nested loop:

```python
for plane_idx in range(img.data.shape[0]):
    for cluster_id, mask in masks.items():
        for ch_idx, ch_label in enumerate(img.masses):
            counts = img.data[plane_idx, ch_idx][mask].sum()
            # store/plot/whatever
```

A proper `depth_profile` function is on the roadmap (v0.7+) but the data structure is already in place. The masks don't depend on the summed-stack representation — they are pure pixel-position information that can be applied to any 4-D `(planes, channels, H, W)` array.

## Design decisions

A few choices worth documenting because the rationale isn't obvious from the code:

**Why five cluster-count selectors instead of one?** Same reason `pymims_histograms` shows multiple k values: when methods agree the answer is robust; when they disagree the disagreement itself is information. A single criterion can be silently wrong; five criteria with documented bias directions are harder to fool simultaneously.

**Why approximate the cubic CCC instead of using a published implementation?** Because there isn't one in the Python scientific stack. SAS source is the canonical reference and is not open. A faithful reimplementation is non-trivial work that would only marginally improve on what the other four selectors already provide. For a research tool this is the right cost-benefit.

**Why subsample-then-assign for hierarchical?** Full hierarchical clustering is O(n²) memory at minimum. On a 256² = 65,536-pixel image, the pairwise distance matrix is 8 GB. Subsampling 5000 representative pixels (default) makes hierarchical fit-time-bounded; assigning each remaining pixel to its nearest centroid is then a fast O(n × k) operation. The dendrogram represents the subsample, not the full image, which is why CCC is computed on the subsample.

**Why high-separation palette by default rather than tab10?** Because tab10's blue (cluster 1) and cyan (cluster 3) are nearly indistinguishable on a dark HSI background. The custom palette (`HIGH_SEP_PALETTE` in the module) is hand-picked for maximum perceptual separation in the first six entries. Pass `cmap='tab10'` to force the legacy palette.

**Why `verbose=` plus per-session deduplication?** The auto-exclusion message is genuinely useful the first time (a new user might not know SE was excluded). It's noise when run in a loop or repeatedly during interactive work. Deduplication by `(id(img), exclusion-tuple)` solves both — visible once, quiet thereafter, until you load a different image.

**Why does `cluster_pixels` modify `img.data` shape via auto-drop downstream?** It doesn't. The clustering reads `img.sum_stack(corrected=True)` and is unaffected by whatever drift correction or plane dropping has happened. The image and the clustering are kept logically separate; you can re-cluster after dropping more planes without invalidating prior results.

## Known limitations

- **k-means assumes spherical clusters in feature space.** If your true populations are elongated (e.g. a cell type that varies smoothly in 12C 14N but tightly in 32S), k-means will partition them awkwardly. Hierarchical with Ward linkage is more flexible; consider it as a cross-check.
- **Cluster identity isn't stable across k.** k=3 cluster 1 isn't necessarily a refinement of k=2 cluster 1 — sklearn refits each k from scratch and we sort components by mean intensity for display, but biology might say a different sort. Visual inspection of `plot_cluster_grid` is the resolution.
- **The dendrogram represents a subsample, not the full image.** Cophenetic correlation reflects the subsample's structure. With well-clustered data the subsample is representative; with diffuse structure it can be misleading. Increase `subsample_size` if cophenetic comes back unexpectedly low.
- **Cubic CCC absolute values are not on Sarle's scale.** Use peak position only.
- **Performance on very large images.** A 1024² image is 1M pixels; k-means at k=10 takes ~20–30 s on Colab. Hierarchical with default subsample stays fast (the subsample size doesn't scale with image size). For very deep stacks consider downsampling spatially before clustering, or sticking to a smaller k_max.
- **Disconnected cluster regions.** A single cluster can appear as many spatially-disconnected blobs across the image. This is correct (clustering is on chemistry, not geometry) but `min_pixels` filtering treats each blob independently. If you want spatially contiguous clusters specifically, `cluster_pixels(method='hierarchical', linkage_method='ward')` with a connectivity constraint would be needed — not currently exposed.
