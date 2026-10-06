"""
What every lens hands back to the app.

Each lens exposes compute(text, model_name) -> LensResult and
render(result). compute() does the numbers and must not touch Streamlit;
render() draws them. Keeping the two apart is what lets the tests call
compute() without a running app.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LensResult:
    lens: str                       # key from shared.config.LENS_NAMES
    text: str                       # the sentence that was analysed
    model_name: str                 # Hugging Face id of the model used
    data: Dict[str, Any] = field(default_factory=dict)  # lens-specific payload
    notes: List[str] = field(default_factory=list)      # warnings to show in the UI
