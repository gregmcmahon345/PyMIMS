# `pymims_rules.py` — Rule-based ROI generation

This module builds boolean pixel masks ("ROIs") from threshold rules expressed in three modes (raw counts, empirical percentiles, GMM-component assignment). Multiple rules combine with AND or OR. The output is a boolean image per rule plus a `combined` mask, all sharing the input image's shape and ready to feed downstream ROI statistics, depth profiles, or visualisation.

This module is the explicit, programmatic counterpart to clustering: clustering finds populations from the data; rules apply user-defined thresholds. Both produce the same kind of output (boolean masks per region) and both feed the same downstream analyses. Use clustering when you don't know what populations exist; use rules when you do.

## Contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Three threshold modes](#three-threshold-modes)
4. [AND vs OR combination](#and-vs-or-combination)
5. [Visualising rule masks](#visualising-rule-masks)
6. [`roi_statistics` and `print_roi_summary`](#roi_statistics-and-print_roi_summary)
7. [Interactive rule slider](#interactive-rule-slider)
8. [Composing with histograms and clustering](#composing-with-histograms-and-clustering)
9. [Design decisions](#design-decisions)
10. [Known limitations](#known-limitations)

---

## What it does

Given a `MimsImage` and a list of rule dicts, the top-level function `build_roi_masks` produces a dict of boolean masks:

```python
rois = build_roi_masks(img, rules=[
    {'channel': '32S',     'mode': 'percentile', 'cutoff': 90},
    {'channel': '12C 14N', 'mode': 'counts',     'cutoff': 200},
], combine='AND', histograms=hists)
# {'rule_0': <bool 2D>, 'rule_1': <bool 2D>, 'combined': <bool 2D>}
```

The mask shapes match the input image's spatial dimensions. The `combined` entry is the AND or OR reduction of the individual rule masks.

A separate `plot_rule_masks` function draws cluster-style outline overlays of each mask on a chosen base image, with the combined mask drawn on top in a thicker white outline. `roi_statistics` and `print_roi_summary` produce per-region per-channel summary statistics.

## Quick start

```python
from pymims import MimsImage
from pymims_histograms import plot_histograms
from pymims_rules import (build_roi_masks, plot_rule_masks,
                           roi_statistics, print_roi_summary)

img = MimsImage('data.im')
img.auto_drop_bad_planes()
img.drift_correct(reference='SE')

# Pre-fit GMMs only if any rule will use mode='gmm-component'
hists = plot_histograms(img, k_max=6, show=False, verbose=False)

rois = build_roi_masks(img, rules=[
    {'channel': '32S',     'mode': 'percentile', 'cutoff': 90,
     'name': '32S top 10%'},
    {'channel': '12C 14N', 'mode': 'counts',     'cutoff': 200,
     'name': '12C 14N >= 200'},
], combine='AND', histograms=hists)

# Visualise — outlines on a chosen base image
plot_rule_masks(img, rois, base='channel', channel='12C 14N')

# Per-ROI per-channel summary statistics
stats = roi_statistics(img, rois)
print_roi_summary(stats)
```

## Three threshold modes

Each rule dict has a `'channel'` and a `'mode'`. The mode determines the rule semantics:

### `mode: 'counts'`

> "Pick all pixels where 32S has at least 100 raw counts."

```python
{'channel': '32S', 'mode': 'counts', 'cutoff': 100}                 # ≥ 100 (default comparison)
{'channel': '32S', 'mode': 'counts', 'cutoff': 5, 'comparison': '<='}  # ≤ 5
```

Useful when count rates have absolute physical meaning — comparing to instrument detection limits, or quoting absolute counts in a paper. Breaks across acquisitions of different durations: 100 counts after 1 hour is biologically not the same as 100 counts after 12 hours. Don't use this mode if comparing pixels across acquisitions of different lengths.

### `mode: 'percentile'`

> "Pick the top 10% of 32S pixels by count."

```python
{'channel': '32S', 'mode': 'percentile', 'cutoff': 90}              # top 10% (≥ p90)
{'channel': '32S', 'mode': 'percentile', 'cutoff': 5, 'comparison': '<='}  # bottom 5% (≤ p5)
```

The `cutoff` is the percentile (0–100), not the percent. Acquisition-invariant — same biological feature picks out the same pixel set regardless of integration time. Breaks across morphologies: top 10% of a sparsely-labelled sample picks a different structural pattern from top 10% of a densely-labelled one.

### `mode: 'gmm-component'`

> "Pick pixels in the highest-mean GMM component for 31P at k=3."

```python
{'channel': '31P', 'mode': 'gmm-component', 'k': 3, 'component': 'highest'}
{'channel': '31P', 'mode': 'gmm-component', 'k': 3, 'component': 0}      # lowest-mean component
{'channel': '31P', 'mode': 'gmm-component', 'k': 3, 'component': 2}      # third (top) component
```

Both acquisition-invariant and morphology-aware. The GMM identifies populations regardless of where they sit on the count axis. Breaks when distributions aren't actually log-Gaussian-mixtures (skewed bulk distributions, integer-dominated trace channels) — see `pymims_histograms.md` for the detailed limitation.

Component selectors:

- `'highest'` — top component by mean (the labelled population in most cases)
- `'lowest'` — bottom component by mean (background/resin)
- An integer `0..k-1` — explicit component index, sorted by mean ascending

GMM-component rules require a `histograms=` argument to `build_roi_masks` — the output of `pymims_histograms.plot_histograms`. The function pulls the equal-posterior crossings for the chosen channel at the specified k and uses them as raw-count cuts.

## AND vs OR combination

```python
# Pixels that match BOTH rules (intersection)
rois = build_roi_masks(img, rules=[...], combine='AND')

# Pixels that match EITHER rule (union)
rois = build_roi_masks(img, rules=[...], combine='OR')
```

When you have multiple rules, the `combined` mask is the AND/OR reduction of all individual rule masks.

The biological intent usually maps to AND: "32S-enriched AND in the cell" — the cell mask comes from one rule, the enrichment from another. OR is useful for "any of these markers": "high 32S OR high 31P OR high ¹⁵N enrichment" picks pixels matching any single criterion.

A subtle point on OR: when rules overlap (common when biological markers are correlated), the OR combination is *less than* the sum of individual rule masks. From the v0.7 testing on synthetic data: three top-5% rules with strong correlation gave a combined OR of 8.6%, not 15%, because the same hot-spots were enriched in all three channels. That pattern of "less overlap than expected" is itself biological information.

## Visualising rule masks

```python
plot_rule_masks(img, rois, base='channel', channel='12C 14N',
                show_individual=True, min_pixels=10)
```

Each individual rule mask gets its own coloured outline (using a high-separation palette matching the rest of the library). The combined mask is drawn on top in thicker white. The legend lists each mask with its pixel count and percentage.

Base modes are the same as `pymims_clustering.plot_overlay`: `'channel'`, `'ratio'`, plus you can pass `min_pixels=` to filter speckled regions before contouring (same connected-component logic as the cluster overlays).

`show_individual=False` suppresses the per-rule outlines and only draws the combined mask — useful when you have many rules and the figure gets cluttered.

## `roi_statistics` and `print_roi_summary`

```python
stats = roi_statistics(img, rois)
print_roi_summary(stats)
```

The output is a per-ROI per-channel table with mean, total, p5, p50, p95 in raw counts. Sample output:

```
  32S top 10%: 1,641 pixels (10.0%)
                 channel       mean        total       p5      p50      p95
                 12C 14N      703.7    1,154,707    659.0    701.0    755.0
                 12C 15N      695.6    1,141,548    648.0    697.0    742.0
                     31P      195.7      321,171    169.0    199.0    224.0
                     32S      584.5      959,193    549.0    597.0    639.0
```

The `roi_statistics` return is also useful programmatically — e.g. for batch reports across many acquisitions where you want to extract `stats['combined']['32S']['mean']` for each one.

## Interactive rule slider

`pymims_explore.roi_rule_slider(img, hist_results=hists)` provides a widget UI for the same functionality. Two rules with mode dropdowns, percentile sliders / count text fields, AND/OR combine, `min_pixels` slider, and live overlay rendering on a chosen base image. See `pymims_explore.md` for the full widget tour.

The widget uses the same `build_roi_masks` and `plot_rule_masks` functions internally, so anything you can do in the widget you can also do in code. The widget exists to remove typing friction from the routine "tune this cutoff" workflow.

## Composing with histograms and clustering

The strength of the rule generator is that it composes naturally with the rest of the pipeline:

**From histogram thresholds:**

```python
hists = plot_histograms(img, k_max=6)
rois = build_roi_masks(img, rules=[
    {'channel': '12C 14N', 'mode': 'gmm-component', 'k': 3,
     'component': 'highest', 'name': 'cell body'},
    {'channel': '32S', 'mode': 'gmm-component', 'k': 3,
     'component': 'highest', 'name': 'sulphur-rich'},
], combine='AND', histograms=hists)
```

**As a pre-mask for clustering:**

```python
hists = plot_histograms(img, k_max=6)
rois = build_roi_masks(img, rules=[
    {'channel': '12C 14N', 'mode': 'gmm-component', 'k': 3,
     'component': 'highest'},
], histograms=hists)
# Cluster only pixels that pass the cell-body rule
result = cluster_pixels(img, pixel_filter=rois['combined'])
```

**As ROI definitions for downstream depth profiling:**

```python
rois = build_roi_masks(img, rules=[...])
# rois['combined'] is a 2-D bool mask
# img.data has shape (n_planes, n_channels, H, W)
depth_profile = []
for plane_idx in range(img.data.shape[0]):
    for ch_idx in range(len(img.masses)):
        counts = img.data[plane_idx, ch_idx][rois['combined']].sum()
        depth_profile.append((plane_idx, ch_idx, counts))
```

The point: rule outputs are general boolean masks. Anything that wants a "pick these pixels" specification can consume them.

## Design decisions

**Why three modes rather than one?** Because they answer different questions and have different invariance properties:

- Counts: physically meaningful, breaks across durations
- Percentile: duration-invariant, breaks across morphologies
- GMM-component: morphology-aware, breaks when distributions aren't log-Gaussian

For most multi-acquisition biological work the right combination is **GMM-component on dense channels and percentile on sparse channels**. The user picks per channel.

**Why is there both a code-based API and a slider widget?** Reproducibility. The slider is convenient for exploring; once you've found the right thresholds you write them down as a list of dicts and that becomes the analysis specification. Code is what you cite in a paper; sliders are what you use to find what to cite.

**Why does AND/OR combination not support more sophisticated boolean expressions?** Because the dual-rule case covers 95% of the workflow (one channel for the cell, another for the marker) and adding a parser for "rule_a AND (rule_b OR rule_c)" would be substantial work for marginal benefit. If you need a complex combination, build the masks individually and combine with NumPy logical ops on the boolean arrays.

**Why are GMM-component rules tied to the histograms output specifically?** Because the threshold cuts come from there. The alternative would be a parallel GMM fit inside the rules module, which would either duplicate code or make `pymims_histograms` an undeclared dependency for the convenience case where users haven't run histograms first. Explicit dependency is cleaner.

**Why does `roi_statistics` use raw counts rather than corrected ratios?** Because the rules are about *defining* regions, and statistics about those regions are most useful as raw counts (which can then be summed, ratio'd, or fed into δ calculations downstream). A separate `roi_ratios()` function for ratio-style summaries would be a reasonable future addition.

## Known limitations

- **No spatial constraint.** A rule might pick scattered single-pixel hot-spots; you can clean these up at the visualisation step with `min_pixels=` but the underlying mask still includes them. For analysis that needs spatially-contiguous regions, post-process the mask with `skimage.measure.label` and a size threshold.
- **Fuzzy matching of channel labels.** The GMM-component mode does fuzzy matching against the histograms dict keys (so `'12C 14N'` matches `'12C14N'`). Other modes don't — you must use the exact label. This inconsistency is a wart; the channel-label resolution should be unified in a future revision.
- **No lookup of GMM-component crossings without re-fitting.** If you change the histograms dict (e.g. with a `manual_k` override), you have to pass the new dict to `build_roi_masks` explicitly. There's no internal cache; the rule reads thresholds fresh every call. This is by design (no hidden state) but means batch loops re-pay the histogram lookup cost per call.
- **No vectorised batch over multiple images.** Each call to `build_roi_masks` operates on one image. For batch processing across acquisitions, loop in user code.
- **AND/OR only, no NOT.** A rule can express ">=" or "<=" via the comparison operator, but inverting the *combination* (e.g. "match rule_a but NOT rule_b") requires post-processing with `~rois['rule_b']`.
