"""acc - Anticlustering-Clustering per la focalizzazione su grandi popolazioni."""
from .core import (ATTRIBUTION_RULES, adaptive_phase1, heterogeneity_t, outer_ratio, outer_stratum,
                   pipeline_v03, V03_CONFIG, anticluster, balance_report, coverage_radius,
                   define_region, fit_cate, knn_radius, mmd, ols_effect,
                   population_diagnostic, select_hybrid, select_kcenter,
                   select_projected, select_representative, select_tempered, sequential_phase1,
                   standardize, zoom_phase1)

__version__ = "0.2.0"
__all__ = ["ATTRIBUTION_RULES", "adaptive_phase1", "heterogeneity_t", "outer_ratio", "outer_stratum",
           "pipeline_v03", "V03_CONFIG", "anticluster", "balance_report", "coverage_radius",
           "define_region", "fit_cate", "knn_radius", "mmd", "ols_effect",
           "population_diagnostic", "select_hybrid", "select_kcenter",
           "select_projected", "select_representative", "select_tempered", "sequential_phase1",
           "standardize", "zoom_phase1"]
