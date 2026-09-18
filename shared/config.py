"""
Central configuration for the Multi-Lens Visualization Framework.

Every lens imports from here so that all four views analyse the SAME model
with the SAME visual language. Do not hardcode model names or colours
inside a lens module.
"""

# ---------------------------------------------------------------- models
# Base model used by the Attention lens and the Probing lens.
BASE_MODEL = "bert-base-uncased"

# Fine-tuned classifier used by the SHAP / token-importance lens.
CLASSIFIER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Lighter alternative for slow machines / free-tier deployment.
LIGHT_MODEL = "distilbert-base-uncased"

# Max tokens we ever feed the model. Keeps heatmaps readable.
MAX_LENGTH = 64

# ---------------------------------------------------------------- theme
# One palette for all four lenses. Low attention -> cool, high -> warm.
COLOR_SCALE = "RdBu_r"
ACCENT = "#E4572E"
ACCENT_SOFT = "#F5C7BB"
NEUTRAL = "#4A4E69"
BG_SOFT = "#F7F7F9"
GRID = "#E3E3E8"

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, sans-serif"

# ---------------------------------------------------------------- layout
BERT_LAYERS = 12
BERT_HEADS = 12

LENS_NAMES = {
    "attention": "Attention Lens",
    "shap": "Token Importance Lens",
    "probing": "Layer Probing Lens",
    "bias": "Bias Analysis Lens",
}
