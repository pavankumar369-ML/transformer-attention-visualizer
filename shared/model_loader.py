"""
Single place where transformer models get loaded.

Why this exists: four people loading BERT four different ways means four
different tokenisations and a demo that contradicts itself. Everyone calls
load_base_model() / load_classifier() from here instead.

Models are cached in-process, so repeated calls are free.
"""

from functools import lru_cache

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shared.config import BASE_MODEL, CLASSIFIER_MODEL, MAX_LENGTH


@lru_cache(maxsize=4)
def load_base_model(model_name: str = BASE_MODEL):
    """Return (tokenizer, model) for the plain encoder.

    The model is returned with attention outputs enabled and in eval mode,
    which is what the Attention and Probing lenses both need.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # attn_implementation="eager" is required: the default fast kernels
    # (SDPA / flash attention) never materialise the attention matrix, so
    # newer transformers versions return no attention weights without it.
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=4)
def load_classifier(model_name: str = CLASSIFIER_MODEL):
    """Return (tokenizer, model) for the sentiment classifier.

    Used by the SHAP lens, and by the Bias lens when it needs a prediction
    to attribute.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, attn_implementation="eager"
    )
    model.eval()
    return tokenizer, model


def encode(tokenizer, text: str):
    """Tokenise one sentence the way every lens should tokenise it."""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def tokens_of(tokenizer, text: str):
    """Human-readable token list, including [CLS] and [SEP].

    Keep the special tokens visible: the fact that [CLS] soaks up a large
    share of attention is one of the more interesting things to point at
    during the demo.
    """
    enc = encode(tokenizer, text)
    return tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
