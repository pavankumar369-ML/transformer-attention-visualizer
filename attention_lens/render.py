"""
Attention Lens - rendering.

Two views, deliberately different jobs:

  heatmap()  - the full token x token matrix. Good for "here is everything".
  arc()      - one query token's attention drawn as arcs over the sentence.
               Good for "here is the one thing I want you to notice".

The arc view is the one to lead the demo with. A 20x20 heatmap is honest
but nobody reads it in five seconds; three arcs over a sentence they can
read is instantly legible.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from shared.config import ACCENT, COLOR_SCALE, GRID, NEUTRAL

_ARC_CMAP = LinearSegmentedColormap.from_list("tav", ["#D8DAE5", ACCENT])


def heatmap(
    matrix: np.ndarray,
    tokens: List[str],
    title: str = "",
    ax: Optional[plt.Axes] = None,
):
    """Token x token attention heatmap. Rows = query, columns = key."""
    if ax is None:
        size = max(4.5, 0.42 * len(tokens))
        _, ax = plt.subplots(figsize=(size, size))

    im = ax.imshow(matrix, cmap=COLOR_SCALE, vmin=0, aspect="equal")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attends to", fontsize=9, color=NEUTRAL)
    ax.set_ylabel("token", fontsize=9, color=NEUTRAL)

    if title:
        ax.set_title(title, fontsize=11, color=NEUTRAL, pad=12)

    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    return ax


def arc(
    matrix: np.ndarray,
    tokens: List[str],
    query_token: str,
    top_k: int = 4,
    title: str = "",
    ax: Optional[plt.Axes] = None,
):
    """Draw the sentence in a line and arc from query_token to what it attends to.

    Arc thickness and opacity both scale with weight, so the dominant link
    is obvious from across a room.
    """
    if query_token not in tokens:
        raise ValueError(f"{query_token!r} is not in {tokens}")

    q = tokens.index(query_token)
    row = matrix[q].copy()
    row[q] = 0  # self-attention is usually large and never interesting here

    top = np.argsort(row)[::-1][:top_k]
    peak = row[top].max() if len(top) else 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 0.75 * len(tokens)), 3.2))

    xs = np.arange(len(tokens))
    ax.scatter(xs, np.zeros_like(xs), s=10, color=GRID, zorder=2)

    for i, tok in enumerate(tokens):
        weight = "bold" if i == q else "normal"
        color = ACCENT if i == q else NEUTRAL
        ax.text(
            i, -0.12, tok,
            ha="center", va="top", rotation=35,
            fontsize=9, color=color, fontweight=weight,
        )

    for j in top:
        w = row[j] / peak if peak else 0
        if w < 0.05:
            continue
        mid = (q + j) / 2
        height = 0.18 + 0.55 * abs(q - j) / max(1, len(tokens))
        curve_x = np.linspace(q, j, 80)
        curve_y = height * np.sin(
            np.pi * (curve_x - min(q, j)) / max(1e-9, abs(q - j))
        )
        ax.plot(
            curve_x, curve_y,
            linewidth=0.8 + 5.0 * w,
            alpha=0.25 + 0.7 * w,
            color=_ARC_CMAP(w),
            solid_capstyle="round",
            zorder=3,
        )
        ax.text(
            mid, height + 0.03, f"{row[j]:.2f}",
            ha="center", fontsize=8, color=NEUTRAL, alpha=0.8,
        )

    ax.set_ylim(-0.75, 1.0)
    ax.set_xlim(-0.8, len(tokens) - 0.2)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, color=NEUTRAL, pad=10)
    plt.tight_layout()
    return ax


def head_grid(attn: np.ndarray, tokens: List[str], layer: int, max_heads: int = 12):
    """Small-multiples: every head in one layer, side by side.

    This is the slide that makes the point that heads specialise - some
    track the previous token, some track punctuation, a few do something
    that looks like syntax.
    """
    n = min(max_heads, attn.shape[1])
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for h in range(n):
        ax = axes[h]
        ax.imshow(attn[layer][h], cmap=COLOR_SCALE, vmin=0)
        ax.set_title(f"head {h}", fontsize=9, color=NEUTRAL)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    for h in range(n, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Layer {layer} - all heads", fontsize=12, color=NEUTRAL)
    plt.tight_layout()
    return fig
