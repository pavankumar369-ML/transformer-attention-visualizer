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
# Palette: white paper, ink-navy text, one cobalt accent, amber as the
# second signal. Every lens imports these, so changing them here restyles
# all five views at once.
INK = "#1C2433"
NEUTRAL = "#5B6577"      # secondary text, negative bars
GRID = "#E6E9EF"         # rules and gridlines
MIST = "#F5F7FA"         # quiet surfaces
BG_SOFT = "#FFFFFF"      # chart background
ACCENT = "#2D5BE3"       # cobalt: the one accent
ACCENT_SOFT = "#C7D3F7"
AMBER = "#C27C0E"
TEAL = "#1F8A70"

# Diverging scale for signed scores (attribution). Keep a name both
# Plotly and Matplotlib know.
COLOR_SCALE = "RdBu_r"
# Sequential scale for attention weights (0 = white, high = deep cobalt).
SEQ_SCALE = [[0.0, "#FFFFFF"], [0.35, "#C7D3F7"], [0.7, "#5C80EC"], [1.0, "#1A3FB0"]]

FONT_FAMILY = "'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"

# Probing lens: one categorical colour per probe task. The control task is
# drawn dashed in its real task's colour, so it needs no colour of its own.
TASK_COLORS = {
    "pos": ACCENT,
    "ner": AMBER,
    "sentiment": TEAL,
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

# The one model the app runs live. DistilBERT keeps BERT's design at half
# the depth (6 layers, 12 heads), so every view loads about twice as fast.
# APP_CLASSIFIER is the same model fine-tuned for sentiment; the lenses
# that explain a prediction (Attribution, Causal) need that version.
# The other models stay in the offline experiments and their saved results.
APP_MODEL = "distilbert-base-uncased"
APP_CLASSIFIER = "distilbert-base-uncased-finetuned-sst-2-english"
APP_MODEL_NAME = "DistilBERT"
APP_MODEL_DETAIL = "66M parameters, 6 layers, 12 heads"

# Masked language models the Bias lens scores (same encoders as Probing).
BIAS_MODELS = PROBING_MODELS

LENS_NAMES = {
    "attention": "Attention Lens",
    "attribution": "Attribution & Faithfulness Lens",
    "probing": "Representation & Probing Lens",
    "bias": "Bias Lens",
    "causal": "Causal Head-Ablation Lens",
}
