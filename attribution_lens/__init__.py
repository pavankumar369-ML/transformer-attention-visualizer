"""Attribution & Faithfulness lens: which words changed the answer, and can we trust that?

Heavy libraries (shap, captum) are imported lazily inside functions, so importing
this package is cheap. Note: ``render`` here is the app-facing *function*; the plotting
module is ``attribution_lens.render`` (import it with ``from attribution_lens.render import ...``).
"""
from . import agreement, attribute, faithfulness, predict
from .lens import analyze, compute, render

__all__ = ["agreement", "attribute", "faithfulness", "predict", "analyze", "compute", "render"]
