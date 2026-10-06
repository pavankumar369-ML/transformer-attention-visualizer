"""
Probing Lens - rendering. Every function returns a Plotly figure.

Plotly rather than matplotlib because these charts are read by hovering:
"layer 7, POS, 95.8%" is the kind of number people want exactly.
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shared.config import (
    ACCENT,
    COLOR_SCALE,
    FONT_FAMILY,
    GRID,
    NEUTRAL,
    TASK_COLORS,
)

TASK_LABELS = {"pos": "Part of speech", "ner": "Named entities", "sentiment": "Sentiment"}


def _style(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    base = dict(
        height=height,
        font=dict(family=FONT_FAMILY, color=NEUTRAL, size=13),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_layout(**{**base, **layout})
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def probe_curves(df: pd.DataFrame) -> go.Figure:
    """Panel 1: accuracy per layer, one solid line per task, dashed POS control.

    df holds one model's rows from data/probing_results.csv.
    """
    fig = go.Figure()
    for task in ("pos", "ner", "sentiment"):
        rows = df[df.task == task].sort_values("layer")
        if rows.empty:
            continue
        color = TASK_COLORS[task]
        fig.add_trace(go.Scatter(
            x=rows.layer, y=rows.accuracy, name=TASK_LABELS[task],
            mode="lines", line=dict(color=color, width=3),
            hovertemplate="%{y:.1%}",
        ))
        if task == "pos" and rows.control_accuracy.notna().any():
            fig.add_trace(go.Scatter(
                x=rows.layer, y=rows.control_accuracy, name="POS control (random labels)",
                mode="lines", line=dict(color=color, width=2, dash="dash"),
                hovertemplate="%{y:.1%}",
            ))
        best = rows.loc[rows.accuracy.idxmax()]
        fig.add_trace(go.Scatter(
            x=[best.layer], y=[best.accuracy], mode="markers+text",
            marker=dict(color=color, size=11, line=dict(color="white", width=2)),
            text=[f"best: layer {int(best.layer)}"], textposition="top center",
            textfont=dict(color=color, size=12),
            showlegend=False, hoverinfo="skip",
        ))

    n_layers = int(df.layer.max()) if not df.empty else 12
    fig.update_xaxes(title="layer (0 = embeddings)", dtick=1, range=[-0.3, n_layers + 0.3])
    fig.update_yaxes(title="probe accuracy on held-out data", tickformat=".0%")
    return _style(fig, height=420)


def selectivity_bars(df: pd.DataFrame) -> go.Figure:
    """Panel 2: POS accuracy minus control accuracy, one bar per layer."""
    rows = df[df.task == "pos"].sort_values("layer")
    sel = rows.accuracy - rows.control_accuracy
    fig = go.Figure(go.Bar(
        x=rows.layer, y=sel, marker_color=TASK_COLORS["pos"],
        customdata=np.stack([rows.accuracy, rows.control_accuracy], axis=-1),
        hovertemplate=(
            "layer %{x}<br>selectivity %{y:.1%}"
            "<br>real %{customdata[0]:.1%} · control %{customdata[1]:.1%}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="layer", dtick=1)
    fig.update_yaxes(title="real − control accuracy", tickformat=".0%")
    return _style(fig, height=320, hovermode="closest")


def cka_heatmap(
    matrix: np.ndarray,
    y_name: str,
    x_name: str,
) -> go.Figure:
    """Panel 3: layer x layer CKA. Rows are layers of y_name, columns of x_name."""
    n_y, n_x = matrix.shape
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(range(n_x)), y=list(range(n_y)),
        zmin=0, zmax=1, colorscale=COLOR_SCALE,
        colorbar=dict(title="CKA", thickness=12),
        hovertemplate=f"{y_name} layer %{{y}}<br>{x_name} layer %{{x}}"
                      "<br>CKA %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title=f"{x_name} layer", dtick=1, showgrid=False)
    fig.update_yaxes(title=f"{y_name} layer", dtick=1, showgrid=False,
                     autorange="reversed", scaleanchor="x")
    return _style(fig, height=460, hovermode="closest")


def self_similarity_line(similarity: Sequence[float], word: str) -> go.Figure:
    """Panel 4a: a word against itself in the previous layer."""
    layers = list(range(1, len(similarity) + 1))
    fig = go.Figure(go.Scatter(
        x=layers, y=similarity, mode="lines+markers",
        line=dict(color=NEUTRAL, width=3), marker=dict(size=7),
        name=f"'{word}' vs previous layer",
        hovertemplate="layer %{x} vs %{customdata}: %{y:.2f}<extra></extra>",
        customdata=[l - 1 for l in layers],
    ))
    fig.update_xaxes(title="layer", dtick=1)
    fig.update_yaxes(title="cosine with previous layer", range=[0, 1.02])
    return _style(fig, height=300, hovermode="closest", showlegend=False)


def pair_similarity_line(
    similarity: Sequence[float],
    word: str,
    senses: List[str],
    separation: Optional[int],
) -> go.Figure:
    """Panel 4b: the same word in two contexts, cosine per layer."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(similarity))), y=similarity, mode="lines+markers",
        line=dict(color=ACCENT, width=3), marker=dict(size=7),
        name=f"'{word}' ({senses[0]}) vs '{word}' ({senses[1]})",
        hovertemplate="layer %{x}: %{y:.2f}<extra></extra>",
    ))
    if separation is not None:
        fig.add_vline(x=separation, line=dict(color=NEUTRAL, dash="dot"))
        fig.add_annotation(
            x=separation, y=1.0, yref="paper", text=f"separate by layer {separation}",
            showarrow=False, xanchor="left", xshift=6, font=dict(color=NEUTRAL),
        )
    fig.update_xaxes(title="layer (0 = embeddings)", dtick=1)
    fig.update_yaxes(title="cosine between the two uses", range=[0, 1.02])
    return _style(fig, height=300, hovermode="closest", showlegend=False)
