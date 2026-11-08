"""acc - Anticlustering-Clustering per la focalizzazione su grandi popolazioni."""
from .core import (adaptive_phase1, anticluster, balance_report, coverage_radius,
                   define_region, fit_cate, knn_radius, mmd, ols_effect,
                   population_diagnostic, select_hybrid, select_kcenter,
                   select_representative, select_tempered, standardize)

__version__ = "0.2.0"
__all__ = ["adaptive_phase1", "anticluster", "balance_report", "coverage_radius",
           "define_region", "fit_cate", "knn_radius", "mmd", "ols_effect",
           "population_diagnostic", "select_hybrid", "select_kcenter",
           "select_representative", "select_tempered", "standardize"]
