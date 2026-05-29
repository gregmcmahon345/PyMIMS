"""
PyMIMS — Python tools for reading and analysing Cameca NanoSIMS .im files.

The public API is re-exported here so users import from the top-level
package regardless of internal layout:

    from pymims import MimsImage, explore, plot_histograms, cluster_pixels

See the submodules (pymims.core, pymims.metadata, pymims.histograms,
pymims.clustering, pymims.rules, pymims.explore) for full documentation.
"""

from .core import MimsImage, save_figure
from . import metadata

from .histograms import plot_histograms, best_thresholds, fit_channel_gmm
from .clustering import (cluster_pixels, plot_cluster_labels,
                         plot_metric_sweep, plot_cluster_grid, plot_overlay,
                         plot_dendrogram, extract_cluster_masks)
from .rules import (build_roi_masks, plot_rule_masks, roi_statistics,
                    print_roi_summary)
from .explore import (explore, run_analyses, cluster_overlay_slider,
                      roi_rule_slider, bulk_ratio_report, ISOTOPE_REFS)

__version__ = "0.3.0"

__all__ = [
    "MimsImage", "save_figure", "metadata",
    "plot_histograms", "best_thresholds", "fit_channel_gmm",
    "cluster_pixels", "plot_cluster_labels", "plot_metric_sweep",
    "plot_cluster_grid", "plot_overlay", "plot_dendrogram",
    "extract_cluster_masks",
    "build_roi_masks", "plot_rule_masks", "roi_statistics", "print_roi_summary",
    "explore", "run_analyses", "cluster_overlay_slider", "roi_rule_slider",
    "bulk_ratio_report", "ISOTOPE_REFS",
    "__version__",
]
