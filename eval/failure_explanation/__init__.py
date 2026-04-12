# failure_explanation/__init__.py
# Package for failure explanation system

from .failure_explainer import explain_failures
from .failure_taxonomy import FAILURE_CATEGORIES, ALL_FAILURE_LABELS, SUPPORTED_TASK_TYPES
from .vlm_judges import query_vlm  # Allow users to override this

__all__ = [
    "explain_failures",
    "FAILURE_CATEGORIES",
    "ALL_FAILURE_LABELS",
    "SUPPORTED_TASK_TYPES",
    "query_vlm"
]