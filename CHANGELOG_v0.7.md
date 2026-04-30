# PyMIMS v0.7 — Change log

This document records the work done in the v0.7 development session: what was added, what was fixed, what design decisions were made, and the reasoning behind them. The "why" is the most decay-resistant content here — six months from now you'll thank past-you for writing down why `tail_weight_threshold=0.15` rather than 0.10, why SE is auto-excluded, and why the cubic CCC implementation is approximate.

## Summary

v0.7 adds **pixel clustering** (`pymims_clustering.py`), **rule-based ROI generation** (`pymims_rules.py`), and **bad-plane detection** (added to `pymims.py`). It also improves visualisation across the library (high-separation palette, sub-pixel cluster contour overlays, multi-format publication export) and restructures the example Colab notebook for clarity.

The three new analysis modes — clustering, rules, and the histogram thresholds from v0.6 — produce the same kind of output (boolean pixel masks) and feed the same downstream consumers. They are interchangeable inputs for "where do I want to compute statistics?" and a future depth-profile module will consume any of them.

## New features

### `pymims_clustering.py` (new module)

K-means and hierarchical agglomerative clustering on per-pixel feature vectors. The principal API is `cluster_pixels(img, method='kmeans'|'hierarchical', ...)` returning a `ClusterResult` dict with labels at every k from 2 to `k_max`, plus five cluster-count selection metrics. See `docs/pymims_clustering.md` for the full module tour.

Plotting functions:

- `plot_cluster_labels` — single labelled cluster image with a per-cluster summary table in raw counts
- `plot_cluster_grid` — side-by-side comparison of multiple k partitions
- `plot_metric_sweep` — 2×3 grid of inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, and cubic CCC vs k, with picks marked
- `plot_dendrogram` — hierarchical tree with cut lines (importable but not part of the routine notebook flow)
- `plot_overlay` — cluster contour outlines overlaid on any base image (channel, ratio, δ, HSI)

Helper:

- `extract_cluster_masks` — `{cluster_id: 2-D bool mask}` for downstream ROI work

### `pymims_rules.py` (new module)

Rule-based ROI generation in three threshold modes (raw counts, empirical percentiles, GMM-component assignment) with AND/OR combination. The principal API is `build_roi_masks(img, rules=[...], combine='AND', histograms=...)` returning a dict of boolean masks. See `docs/pymims_rules.md` for the full tour.

Plotting and statistics:

- `plot_rule_masks` — outline overlays (analogous to `plot_overlay`) for rule masks
- `roi_statistics` — per-region per-channel mean/total/p5/p50/p95
- `print_roi_summary` — tabular print of the above

### Bad-plane detection in `pymims.py`

Three new `MimsImage` methods, intended as pre-processing before drift correction:

- `plot_plane_diagnostics(channel=None, threshold_pct=30)` — total counts per plane with suspect planes flagged at ±N% from the median
- `auto_drop_bad_planes(threshold_pct=30, dry_run=False)` — destructive removal in place; tracks dropped indices in `img._dropped_planes`
- `plane_movie(channel=0, interval_ms=200)` — animated flipthrough of every plane for visual scanning

Bad planes (charging events, electronic dropouts, partial blank frames) cause silent drift-correction failures because the FFT cross-correlation has no real peak to lock onto. The diagnostic plot makes the problem visible and the auto-drop offers a one-line fix.

### Interactive widgets in `pymims_explore.py`

Two new widget functions wrapping the v0.7 plotting:

- `cluster_overlay_slider(img, result)` — interactive overlay viewer with `k`, `base`, `min_pixels`, and base-mode kwargs (delta reference, HSI scale factor) all exposed as widgets. Auto-fills the natural-abundance reference from `ISOTOPE_REFS` when the chosen num/den match a known pair.
- `roi_rule_slider(img, hist_results=None)` — two-rule ROI builder with mode-adaptive cutoff controls (FloatText for counts, FloatSlider for percentiles, IntSlider+Dropdown for GMM components), AND/OR combine, `min_pixels` filter, and live overlay rendering on a chosen base image.

### `save_figure` helper in `pymims.py`

```python
from pymims import save_figure
fig = plot_overlay(...)
save_figure(fig, 'figure.pdf', dpi=600)
```

General-purpose figure exporter handling PNG, PDF, SVG, EPS, JPG, TIF. Vector formats (PDF, SVG) ignore DPI by design — they're resolution-independent. Auto-detects format from the file extension when `format=` isn't specified.

### `dpi=` kwarg on every plotting function

Every `plot_*` function in the library now accepts `dpi=` to control the resolution of `outpath`-saved figures. Default is 200 DPI for screen-quality; pass 600 for journal-grade.

### Restructured Colab notebook

The example notebook (`PyMIMS.ipynb`) was reorganised from "every feature gets a cell" (which made sense during development) to a clear analytical workflow:

1. Setup (single wget block + all imports)
2. Load image
3. Pre-processing (bad-plane diagnostics + drift correction)
4. Visual exploration
5. Histograms
6. Clustering
7. Cluster overlay slider
8. Rule-based ROIs
9. Export

Each section has a markdown header explaining what it does and what to watch for. Variables computed in earlier sections (`hists`, `result_km`, `result_h`) are reused downstream rather than re-computed. Imports are consolidated into the setup cell.

## Bug fixes

### Half-pixel coordinate offset in cluster contour overlays

`matplotlib.pyplot.imshow` with `extent=[0, field_um, field_um, 0]` places the **outer edge** of the image at 0 and `field_um`, so pixel (0, 0)'s centre is at `(0.5*um_per_px, 0.5*um_per_px)`. `skimage.measure.find_contours` returns coordinates where integer (i, j) is the **pixel centre**. Naively converting find_contours output by dividing by `H/field_um` shifted every contour up-and-left by half a pixel.

The visible symptom: cluster outlines that sat *inside* the visible cluster boundary on the underlying image rather than on it. Especially noticeable on dark resin regions where the cluster-2 outline appeared 1–2 pixels inside the resin.

Fix: add `+ 0.5` to the pixel coordinates before scaling. Applied in both `pymims_clustering.plot_overlay` and `pymims_rules.plot_rule_masks`. Validated with a synthetic circle at known coordinates — the cluster contour now matches the theoretical boundary to sub-pixel precision.

This is a class of bug endemic to scientific image-processing code: every library has its own convention about whether (0,0) is the pixel corner or the pixel centre, and mixing matplotlib + skimage is exactly where these conventions collide.

### `plot_histograms` tail-warning threshold raised from 0.10 to 0.15

The original 0.10 default was too tight for biological work — Fe:S clusters in mitochondria typically occupy 10–15% of pixels (rare enough that the conservative GMM consensus suppresses them, common enough that 0.10 misses them). The 0.15 default catches the typical biological rare-population range without false-firing on plain bimodal channels. Validated on real ¹²C¹⁴N / ³²S data from the user's NS50L acquisitions.

Comparison operator also changed from `<` to `<=` so a population at exactly the threshold value is flagged, matching user expectations.

### Cluster colour palette: tab10 → high-separation custom

`tab10`'s blue (cluster 1) and cyan (cluster 3) are nearly indistinguishable on dark HSI backgrounds, where cluster outlines need to read against viridis-like base colours. New `HIGH_SEP_PALETTE` constant: red / green / blue / orange / pink / yellow as the first six entries. After 6 clusters, falls back to `tab10` for the long tail. Pass `cmap='tab10'` to force the legacy palette.

## Design decisions worth remembering

### `tail_weight_threshold=0.15` is a biology-aware default

Materials science applications might want 0.05 (catch single-percent rare features); biological applications want 0.15 (catch Fe:S-cluster-scale populations). The 0.15 default reflects the user's primary domain. Document this explicitly in `pymims_histograms.md` so users in other fields know to tune it.

### Five cluster-count selectors instead of one

Same philosophy as `pymims_histograms`: when methods agree the answer is robust; when they disagree the disagreement itself is information. A single criterion can be silently wrong; five with documented bias directions are harder to fool simultaneously.

The five: inertia largest-drop, inertia kneedle, silhouette peak, Calinski-Harabasz peak, Davies-Bouldin minimum, plus the approximate cubic CCC peak (six total — but inertia largest-drop and kneedle are usually counted as two halves of "elbow detection"). The conservative consensus `sensible_k` is the smaller of (inertia elbow consensus) and (silhouette peak); `unique_k_recommendations` lists every k picked by any method for `plot_cluster_grid` to display.

### Cubic CCC implementation is approximate, NOT on Sarle's scale

The published interpretation (CCC > 3 = good evidence, etc.) does not apply to our values. Our implementation uses a participation-ratio approximation for the expected R² rather than Sarle's full c-prime adjustment. The peak position across k is meaningful — that's what we use. Absolute values aren't.

This is documented prominently in:

- The function docstring
- The module preamble
- The metric-panel hint ("approx — peak position only")

A faithful Sarle reimplementation is a known future enhancement. For research-tool purposes, the relative selector is sufficient.

### SE auto-exclusion from clustering, with `verbose=` deduplication

SE measures topography, not chemistry. Including it in chemistry-driven clustering biases the partition toward geometric features (edges, surface relief) rather than biological structure. Auto-exclusion with a printed message gives the user discoverability of what happened. Per-session deduplication (keyed on `id(img) + exclusion-tuple`) prevents the message spamming when `cluster_pixels` is called repeatedly. `verbose=False` silences it entirely for batch processing.

The detection heuristic matches `SE`, `se`, `Secondary Electron`, `e-`, `e⁻`, `EM`, `topography`, `topo`. It explicitly does NOT match substrings (so `12C2` doesn't match `c2` against `EM`).

### Cluster IDs are 1-indexed throughout

The cluster summary tables, the colour palette, the legend entries, and `extract_cluster_masks` all use 1-indexed cluster IDs (1, 2, 3...). NumPy and sklearn use 0-indexing internally but the user-facing API hides that. Consistency matters more than convention here — mixing 0- and 1-indexing across the API would be confusing.

### Subsample-then-assign for hierarchical clustering

Full hierarchical on a 256² = 65,536-pixel image is O(n²) memory minimum (~8 GB pairwise distance matrix). Subsampling 5000 representative pixels (`subsample_size=5000` default kwarg) makes hierarchical fit-time-bounded; assigning each remaining pixel to its nearest subsample-cluster centroid is then a fast O(n × k) operation. The dendrogram represents the subsample, not the full image, which is why the cophenetic correlation reflects subsample structure.

### Cophenetic correlation interpretation bands (Romesburg 1984)

- ≥ 0.90 — strong: dendrogram faithfully represents the data
- 0.80–0.90 — good: represents the data well
- 0.70–0.80 — moderate: some distortion; cluster cuts still usable
- < 0.70 — weak: noticeable distortion; treat cluster cuts with caution

The interpretation appears on every figure that shows the cophenetic value (cluster-labels suptitle, metric-sweep title, dendrogram footnote) so users don't have to remember the bands.

### Bad-plane detection: median-relative threshold, not previous-plane diff

User originally suggested "% difference from previous plane". The weakness: two consecutive bad planes look fine relative to each other. Median-relative thresholding is robust because the median is unaffected by outliers — even with 20% bad planes, the median still represents the "typical good plane". Sum-over-all-channels per plane is the most robust default; channel-restricted detection is available via `channel=` for known per-channel artefacts.

The 30% threshold default is informed by what biological samples typically look like — modest plane-to-plane variation is normal (slow beam-current drift, sample charging slowly recovering), but anything beyond 30% deviation from the median is almost always a real problem. Tightenable to 15-20% for very stable acquisitions.

### Bad-plane drop is destructive, with provenance recording

`auto_drop_bad_planes` modifies `img.data` in place and stores the dropped plane indices in `img._dropped_planes`. The alternative — a `valid_planes` boolean mask — would be non-destructive but require every downstream operation to consult the mask. The destructive choice keeps the rest of the codebase simple; users who want to recover dropped planes re-load the file. Provenance is preserved via `_dropped_planes` for documentation purposes.

If `drift_correct` had already run when planes are dropped, the prior correction is invalidated and `img.corrected` is reset. A note tells the user to re-run drift correction.

### Pre-processing order: bad-plane detection BEFORE drift correction

Bad planes corrupt the FFT cross-correlation (no real peak → garbage shift). Removing them first means drift correction operates on a clean stack and produces accurate alignment. The notebook ordering reflects this: section 3a (bad planes) precedes section 3b (drift correction).

### Plane binning (`bin_planes=N`) lives in `drift_correct`, not as a separate method

Plane binning's only use is stabilising drift correction on low-count acquisitions. Treating it as a `drift_correct` kwarg keeps the operation tied to its purpose. The notebook explains when to use it (high-resolution fine-beam acquisitions where per-plane counts are low) but the machinery doesn't need to be visible elsewhere.

### Notebook restructuring: workflow over feature-by-feature

The original notebook was "one cell per feature" because that's how features got built. The reorganised version is "one section per analysis stage" because that's how a user actually consumes it. Markdown headers between sections make Colab's table-of-contents sidebar navigable; dependency flow goes downward (no upward references); variables computed early are reused downstream rather than re-computed.

## Things that were considered and rejected

### A `valid_planes` boolean mask instead of destructive auto-drop

Considered for non-destructiveness but rejected because every downstream operation would need to consult the mask. The complexity isn't worth it for a research tool — re-loading the file is cheap, and `_dropped_planes` provenance covers the audit trail.

### Self-organising maps as a third clustering method

Originally suggested by the user. Considered but rejected: SOMs are interesting for *visualising relationships between clusters* but for the v0.7 use case (produce ROI masks) they're doing extra work for the same output as k-means + hierarchical. They also have hyperparameters (grid size, learning rate, neighbourhood function) with no BIC-equivalent for choosing them — they'd need to be tuned by eye, which is a workflow regression. Marked as "maybe later" if cluster-relationship visualisation becomes a real need.

### A faithful Sarle Cubic Clustering Criterion implementation

Considered but rejected for v0.7. The full SAS formula uses an empirical c-prime adjustment documented only in SAS source code. The participation-ratio approximation captures the curve shape (peak position is correct) which is what we need; absolute values just aren't on Sarle's scale. A faithful reimplementation is on the roadmap but not blocking.

### A super-widget combining cluster overlay + rule slider

Both widgets produce overlay-on-base-image renderings. Tempting to merge into one. Rejected because clusters and rule-based ROIs are different analytical approaches that just happen to produce the same kind of output. Mashing them together would obscure that. Two separate widgets with consistent UX is the right answer.

### Minimum-component-size filtering at the clustering stage

Considered: filter out connected components below `min_pixels` *during* clustering rather than at visualisation time. Rejected because the right `min_pixels` depends on what you're trying to see — and a Poisson-pattern population (Fe:S clusters) might legitimately be single-pixel features that you don't want filtered. Keeping the filter at visualisation time means you can experiment without re-clustering.

## Roadmap (what's next)

The pieces in place after v0.7:

- Histogram-derived thresholds (v0.6)
- Pixel clustering with five-criterion k selection (v0.7)
- Rule-based ROI generation in three modes (v0.7)
- ROI statistics tabulation (v0.7)
- Boolean masks per region as the universal output format

What's not yet in place:

- **Depth profiling**. The architecture is ready (`extract_cluster_masks` and `rois['combined']` both produce 2-D bool masks; per-plane stats are a nested loop) but no convenience function exists. Priority for next work.
- **Channel clustering / co-localisation analysis**. Different from pixel clustering — answers "which channels behave similarly across the image?". Belongs in a multivariate-statistics module.
- **SEM / TEM correlative registration**. Affine alignment of NanoSIMS images to electron microscopy. Mooted for v0.8.
- **Faithful Sarle Cubic Clustering Criterion**. Worth doing eventually for users who want the published thresholds.
- **A `Channel` object to clean up label/index resolution**. The current `_resolve_channel` heuristic is repeated across modules; a unified type would consolidate it.
- **PyPI package release with Zenodo DOI**. v1.0 milestone.

The Colab notebook is now structured so that adding a new section (e.g. depth profiling) is a clean append at the bottom — no upstream restructuring required. Same for adding new clustering or rule modes.

## Validation summary

Every new feature was tested on synthetic data with known ground truth before being declared working. The synthetic test cases:

- 3-region image (resin / bulk / Fe:S-like hot-spots) for clustering — k-means and hierarchical both recovered the truth (ARI = 1.000 against ground truth)
- 30-plane stack with 3 deliberately-bad planes (charging spike, dropout, second spike) for bad-plane detection — all three correctly flagged at the 30% threshold
- Synthetic circle at known coordinates for the half-pixel offset fix — sub-pixel alignment confirmed
- Three-population synthetic data for rule-based ROIs — top-10% of 32S correctly identified 1641 pixels (ground truth: 1585 hot-spot pixels), GMM-component rule on 31P correctly identified 1585 pixels (ground truth: 1585) — perfect match

Real-data validation came from the user's NS50L acquisitions on biological samples. The histogram tail-warning threshold and the SE auto-exclusion default are both informed by what worked on real data.
