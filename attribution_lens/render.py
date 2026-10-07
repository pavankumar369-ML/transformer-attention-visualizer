"""Streamlit UI for the Attribution & Faithfulness tab (four vertical panels)."""
from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ._compat import get_palette
from .attribute import METHOD_LABELS, METHODS

def _colors() -> dict[str, str]:
    """All colours come from shared/config.py (via _compat.get_palette)."""
    pal = get_palette()
    return {
        "pos": pal["positive"], "neg": pal["negative"], "random": pal["neutral"],
        "lines": [pal["positive"], pal["negative"], pal["accent"], pal["light_blue"]],
    }


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _clean(tok: str, family: str, first: bool) -> tuple[str, bool]:
    """Return (display text, starts_new_word)."""
    if tok.startswith("##"):
        return tok[2:], False
    if family == "roberta":
        return (tok[1:], True) if tok.startswith("\u0120") else (tok, first)
    return tok, True


def highlight_html(tokens: list[str], scores: np.ndarray, special: np.ndarray,
                   family: str, pos_hex: str, neg_hex: str) -> str:
    """Sentence as <span> tags: blue = pushes toward prediction, red = against; stronger = larger."""
    s = np.asarray(scores, dtype=float)
    keep = ~np.asarray(special, dtype=bool)
    scale = max(float(np.abs(s[keep]).max()) if keep.any() else 0.0, 1e-12)
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if not keep[i]:
            continue
        word, new_word = _clean(tok, family, not parts)
        alpha = 0.10 + 0.80 * min(1.0, abs(s[i]) / scale)
        r, g, b = _rgb(pos_hex if s[i] >= 0 else neg_hex)
        fg = "#ffffff" if alpha > 0.6 else "#111111"
        sep = " " if (new_word and parts) else ""
        parts.append(f"{sep}<span title='{html.escape(tok)}: {s[i]:+.3f}' style='background-color:"
                     f"rgba({r},{g},{b},{alpha:.2f});color:{fg};padding:2px 4px;border-radius:4px;"
                     f"font-size:1.15rem;'>{html.escape(word)}</span>")
    return f"<div style='line-height:2.3'>{''.join(parts)}</div>"


def faithfulness_figure(report: dict) -> go.Figure:
    """Line chart: share of words removed (x) vs drop in predicted-class probability (y)."""
    c = _colors()
    fig = go.Figure()
    for i, m in enumerate(METHODS):
        r = report[m]
        fig.add_trace(go.Scatter(x=r["fractions"], y=r["drops"], mode="lines+markers",
                                 name=f"{METHOD_LABELS[m]} (AOPC {r['aopc']:.3f})",
                                 line=dict(color=c["lines"][i % len(c["lines"])], width=3)))
    r = report["Random"]
    fig.add_trace(go.Scatter(x=r["fractions"], y=r["drops"], mode="lines", name=f"Random (AOPC {r['aopc']:.3f})",
                             line=dict(color=c["random"], dash="dash", width=2)))
    fig.update_layout(xaxis_title="Share of words removed", yaxis_title="Drop in prediction probability",
                      xaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def agreement_figure(spearman: np.ndarray) -> go.Figure:
    """4x4 Spearman heatmap (-1..1)."""
    c = _colors()
    fig = go.Figure(go.Heatmap(z=spearman, x=list(METHODS), y=list(METHODS), zmin=-1, zmax=1,
                               colorscale=[[0, c["neg"]], [0.5, "#ffffff"], [1, c["pos"]]],
                               text=np.round(spearman, 2), texttemplate="%{text}",
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    return fig


def render_panels(p: dict) -> None:
    """Draw the four panels from an analysis payload (see lens.analyze)."""
    c = _colors()
    tokens, special = p["tokens"], np.array(p["special"], dtype=bool)

    # Panel 1: prediction
    color = "#1C2433"
    prob = p["pred_prob"]
    shown = "over 99.9%" if prob >= 0.999 else f"{prob:.1%}"
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:0.9rem;margin:0.25rem 0 1.25rem'>"
        f"<span style='font-size:0.9rem;color:#5B6577'>Prediction</span>"
        f"<span style='font-size:1.6rem;font-weight:600;color:{color}'>{p['pred_label'].capitalize()}</span>"
        f"<span style='font-size:0.95rem;color:#5B6577'>{shown} confident</span></div>",
        unsafe_allow_html=True,
    )

    # Panel 2: highlighted sentence, SHAP and IG
    for m in ("SHAP", "IG"):
        st.markdown(f"**{METHOD_LABELS[m]}**")
        st.markdown(highlight_html(tokens, np.array(p["scores"][m]), special, p["family"],
                                   c["pos"], c["neg"]), unsafe_allow_html=True)
    st.caption("Blue = pushes toward the prediction. Red = pushes against it. Darker = stronger.")
    with st.expander("How to read this"):
        st.write("Each word is shaded by how much it changed the model's answer. If SHAP and IG "
                 "highlight the same words, we can trust the explanation more. Hover for exact scores. "
                 f"IG convergence delta: {p['ig_delta']:.4f} (steps: {p['ig_steps']}).")

    # Panel 3: faithfulness chart
    st.plotly_chart(faithfulness_figure(p["faithfulness"]), width="stretch")
    with st.expander("How to read this"):
        st.write("We delete the words each method calls important and watch the prediction fall. "
                 "The faster a line rises, the more faithful the method. A line near the dashed grey "
                 "random line is no better than guessing. AOPC (in the legend) is the average drop.")

    # Panel 4: agreement heatmap
    st.plotly_chart(agreement_figure(np.array(p["agreement"]["spearman"])), width="stretch")
    st.caption("Spearman rank correlation between methods. Low numbers mean the methods rank the "
               "words very differently, so at most one of them can be right.")
    with st.expander("How to read this"):
        st.write("1 = identical word ranking, 0 = unrelated, -1 = reversed. Low agreement between "
                 "attention and SHAP/IG supports the claim that attention is not explanation.")
