"""
Attention Lens - views for the app.

Plotly and plain HTML instead of Matplotlib. Matplotlib turned every view
into a large PNG on each rerun, which is what made scrolling through the
head grid sluggish. These views are small, crisp at any size, and only
built when the user opens them.

  token_strip_html   the sentence, each word tinted by how much one word attends to it
  ranked_html        the top few targets as a compact bar list
  arc_figure         arcs from the focus word to where its attention goes
  heatmap_figure     the full token x token matrix
  head_grid_figure   every head in one layer, side by side
"""

import html
from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.config import ACCENT, GRID, INK, NEUTRAL, SEQ_SCALE

SPECIAL = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"}


def content_index(tokens: Sequence[str]) -> List[int]:
    """Positions of real tokens, without [CLS]/[SEP]."""
    return [i for i, t in enumerate(tokens) if t not in SPECIAL]


def display(tok: str) -> str:
    """'##ing' -> 'ing' for display; everything else unchanged."""
    return tok[2:] if tok.startswith("##") else tok


def focus_weights(matrix: np.ndarray, tokens: Sequence[str], query: int):
    """Attention from token `query` to every content token, renormalised.

    [CLS]/[SEP] soak up a large share of attention without carrying meaning,
    so they are dropped and the rest rescaled to sum to 1.
    """
    idx = content_index(tokens)
    row = np.asarray(matrix[query], dtype=float)[idx]
    total = row.sum()
    return idx, (row / total if total > 0 else row)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def token_strip_html(tokens: Sequence[str], matrix: np.ndarray, query: int) -> str:
    """The sentence, each word shaded by the focus word's attention to it."""
    idx, w = focus_weights(matrix, tokens, query)
    peak = w.max() if len(w) and w.max() > 0 else 1.0
    parts = []
    for i, weight in zip(idx, w):
        strength = float(weight / peak)
        alpha = 0.06 + 0.84 * strength
        color = "#FFFFFF" if alpha > 0.55 else INK
        classes = "tav-tok" + (" q" if i == query else "") + (" sub" if tokens[i].startswith("##") else "")
        title = f"{display(tokens[i])}: {weight:.0%} of the attention"
        parts.append(
            f'<span class="{classes}" title="{html.escape(title)}" '
            f'style="background:{_rgba(ACCENT, alpha)};color:{color}">'
            f"{html.escape(display(tokens[i]))}</span>"
        )
    return f'<div class="tav-strip">{" ".join(parts)}</div>'


def ranked_html(tokens: Sequence[str], matrix: np.ndarray, query: int, k: int = 5) -> str:
    idx, w = focus_weights(matrix, tokens, query)
    order = [o for o in np.argsort(w)[::-1] if idx[o] != query][:k]
    peak = w[order[0]] if order else 1.0
    rows = []
    for o in order:
        rows.append(
            f'<span class="w">{html.escape(display(tokens[idx[o]]))}</span>'
            f'<span class="bar"><i style="width:{100 * w[o] / peak:.1f}%"></i></span>'
            f'<span class="pct">{w[o]:.0%}</span>'
        )
    return f'<div class="tav-rank">{"".join(rows)}</div>'


def arc_figure(tokens: Sequence[str], matrix: np.ndarray, query: int, top_k: int = 4) -> go.Figure:
    """Arcs from the focus word to its strongest targets (self excluded)."""
    idx, w = focus_weights(matrix, tokens, query)
    pos = {tok_i: x for x, tok_i in enumerate(idx)}
    qx = pos[query]
    targets = [o for o in np.argsort(w)[::-1] if idx[o] != query][:top_k]
    peak = w[targets[0]] if targets else 1.0
    n = len(idx)

    fig = go.Figure()
    for o in sorted(targets, key=lambda o: w[o]):          # strongest drawn last, on top
        jx, strength = pos[idx[o]], float(w[o] / peak)
        xs = np.linspace(qx, jx, 64)
        span = max(abs(jx - qx), 1e-9)
        height = 0.28 + 0.72 * abs(jx - qx) / max(n - 1, 1)
        ys = height * np.sin(np.pi * (xs - min(qx, jx)) / span)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="text",
            text=f"{display(tokens[query])} to {display(tokens[idx[o]])}: {w[o]:.0%}",
            line=dict(color=_rgba(ACCENT, 0.3 + 0.7 * strength), width=1.5 + 6 * strength, shape="spline"),
        ))
        fig.add_annotation(x=(qx + jx) / 2, y=height + 0.07, text=f"{w[o]:.0%}", showarrow=False,
                           font=dict(size=12, color=NEUTRAL))

    fig.add_trace(go.Scatter(x=list(range(n)), y=[0] * n, mode="markers", hoverinfo="skip",
                             marker=dict(size=6, color=[INK if i == query else GRID for i in idx])))
    labels = [f"<b>{html.escape(display(tokens[i]))}</b>" if i == query else html.escape(display(tokens[i]))
              for i in idx]
    fig.update_layout(
        height=300, showlegend=False, margin=dict(l=8, r=8, t=12, b=8),
        xaxis=dict(tickmode="array", tickvals=list(range(n)), ticktext=labels, tickangle=0,
                   showgrid=False, showline=False, tickfont=dict(size=13, color=INK),
                   range=[-0.6, n - 0.4], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.04, 1.2], fixedrange=True),
    )
    return fig


def _heatmap_trace(matrix: np.ndarray, labels: List[str], coloraxis: str = None) -> go.Heatmap:
    n = len(labels)
    hover = [[f"{labels[r]} to {labels[c]}: {matrix[r, c]:.1%}" for c in range(n)] for r in range(n)]
    kwargs = dict(coloraxis=coloraxis) if coloraxis else dict(colorscale=SEQ_SCALE, zmin=0, zmax=float(matrix.max()))
    return go.Heatmap(z=matrix, x=list(range(n)), y=list(range(n)), text=hover,
                      hoverinfo="text", xgap=1, ygap=1, **kwargs)


def heatmap_figure(tokens: Sequence[str], matrix: np.ndarray) -> go.Figure:
    """Full matrix, special tokens included: this is the honest raw view."""
    labels = [display(t) for t in tokens]
    n = len(labels)
    fig = go.Figure(_heatmap_trace(np.asarray(matrix), labels))
    side = min(640, 140 + 30 * n)
    axis = dict(tickmode="array", tickvals=list(range(n)), ticktext=labels, showgrid=False,
                fixedrange=True, tickfont=dict(size=12, color=INK))
    fig.update_layout(height=side, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(axis, side="top", tickangle=-45, title=dict(text="attends to")),
                      yaxis=dict(axis, autorange="reversed", title=dict(text="word")))
    fig.update_traces(colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%"))
    return fig


def head_grid_figure(tokens: Sequence[str], layer_attn: np.ndarray, cols: int = 6) -> go.Figure:
    """Every head in one layer, sharing one colour scale."""
    heads = layer_attn.shape[0]
    rows = int(np.ceil(heads / cols))
    labels = [display(t) for t in tokens]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Head {h + 1}" for h in range(heads)],
                        horizontal_spacing=0.02, vertical_spacing=0.08)
    for h in range(heads):
        fig.add_trace(_heatmap_trace(layer_attn[h], labels, coloraxis="coloraxis"),
                      row=h // cols + 1, col=h % cols + 1)
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, autorange="reversed", fixedrange=True)
    fig.update_annotations(font=dict(size=12, color=NEUTRAL))
    fig.update_layout(height=60 + 175 * rows, margin=dict(l=8, r=8, t=30, b=8),
                      coloraxis=dict(colorscale=SEQ_SCALE, cmin=0, cmax=float(layer_attn.max()),
                                     colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%")))
    return fig
