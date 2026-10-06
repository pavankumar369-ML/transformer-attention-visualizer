"""
Central configuration for the Multi-Lens Visualization Framework.

Every lens imports from here so that all five views analyse the SAME model
with the SAME visual language. Do not hardcode model names or colours
inside a lens module.
"""

# ---------------------------------------------------------------- models
# Base model used by the Attention lens and the Probing lens.
BASE_MODEL = "bert-base-uncased"

# Fine-tuned sentiment classifier: default for the Attribution and Causal lenses.
CLASSIFIER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Sentiment checkpoints the Attribution and Causal lenses compare.
# Values are display names. Verify each id on huggingface.co before use.
SENTIMENT_MODELS = {
    "distilbert-base-uncased-finetuned-sst-2-english": "DistilBERT (SST-2)",
    "textattack/bert-base-uncased-SST-2": "BERT (SST-2)",
    "textattack/roberta-base-SST-2": "RoBERTa (SST-2)",
}

# Lighter alternative for slow machines / free-tier deployment.
LIGHT_MODEL = "distilbert-base-uncased"

# Max tokens we ever feed the model. Keeps heatmaps readable.
MAX_LENGTH = 64

# ---------------------------------------------------------------- theme
# One palette for all five lenses. Low attention -> cool, high -> warm.
COLOR_SCALE = "RdBu_r"
ACCENT = "#E4572E"
ACCENT_SOFT = "#F5C7BB"
NEUTRAL = "#4A4E69"
BG_SOFT = "#F7F7F9"
GRID = "#E3E3E8"

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, sans-serif"

# Probing lens: one categorical colour per probe task. The control task is
# drawn dashed in its real task's colour, so it needs no colour of its own.
TASK_COLORS = {
    "pos": ACCENT,
    "ner": "#3A7CA5",
    "sentiment": "#6A994E",
}

# ---------------------------------------------------------------- layout
BERT_LAYERS = 12
BERT_HEADS = 12

# Plain encoders the Probing lens compares. Values are display names.
PROBING_MODELS = {
    "bert-base-uncased": "BERT",
    "distilbert-base-uncased": "DistilBERT",
    "roberta-base": "RoBERTa",
}

# Masked language models the Bias lens scores (same encoders as Probing).
BIAS_MODELS = PROBING_MODELS

LENS_NAMES = {
    "attention": "Attention Lens",
    "attribution": "Attribution & Faithfulness Lens",
    "probing": "Representation & Probing Lens",
    "bias": "Bias Lens",
    "causal": "Causal Head-Ablation Lens",
}
