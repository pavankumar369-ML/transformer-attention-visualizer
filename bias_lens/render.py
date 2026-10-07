"""Plotly figures for the Bias tab (no Streamlit here)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.config import ACCENT, ACCENT_SOFT, BG_SOFT, FONT_FAMILY, GRID, NEUTRAL


def stereotype_fig(df) -> go.Figure:
    """df: rows of bias_results.csv for ONE model."""
    d = df.set_index("category")
    order = ["overall"] + sorted(c for c in d.index if c != "overall")
    d = d.loc[order].iloc[::-1]  # overall on top
    sig = d["ci_excludes_50"].to_numpy()
    labels = [c.replace("-", " ").replace("_", " ").capitalize() for c in d.index]
    fig = go.Figure(go.Bar(
        y=labels, x=d["score"] - 50, base=50, orientation="h",
        marker_color=[ACCENT if s else ACCENT_SOFT for s in sig],
        error_x=dict(type="data", symmetric=False, color=NEUTRAL, thickness=1.2, width=4,
                     array=(d["ci_high"] - d["score"]).to_numpy(),
                     arrayminus=(d["score"] - d["ci_low"]).to_numpy()),
        customdata=np.stack([d["n"], d["ci_low"], d["ci_high"], d["p_value"], d["score"]], axis=1),
        hovertemplate=("%{y}<br>stereotype score %{customdata[4]:.1f}%<br>95%% CI %{customdata[1]:.1f}–"
                       "%{customdata[2]:.1f}<br>n=%{customdata[0]}, p=%{customdata[3]:.3g}<extra></extra>"),
    ))
    fig.add_vline(x=50, line_width=1.5, line_color=NEUTRAL,
                  annotation_text="No preference", annotation_position="top",
                  annotation_font=dict(size=12, color=NEUTRAL))
    fig.update_layout(xaxis=dict(range=[0, 100], title="% of pairs where the stereotyped sentence scores higher",
                                 gridcolor=GRID),
                      height=120 + 36 * len(d), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor=BG_SOFT, font_family=FONT_FAMILY, showlegend=False)
    return fig


def _tok_bars(fig, row, tokens, vals, highlight, diverging=False):
    idx = list(range(1, len(tokens) - 1))              # hide [CLS]/[SEP]
    vals = np.asarray(vals, dtype=float)
    colors = []
    for i in idx:
        if diverging:
            colors.append(ACCENT if vals[i] >= 0 else NEUTRAL)
        else:
            colors.append(ACCENT if i in highlight else NEUTRAL)
    fig.add_trace(go.Bar(x=idx, y=[vals[i] for i in idx], marker_color=colors,
                         hovertext=[tokens[i] for i in idx],
                         hovertemplate="%{hovertext}: %{y:.4f}<extra></extra>"), row=row, col=1)
    fig.update_xaxes(tickmode="array", tickvals=idx, ticktext=[tokens[i] for i in idx], row=row, col=1)


def pair_figs(cmp: dict):
    """Returns (attention_fig, attribution_fig); each has rows A, B, difference."""
    figs = []
    for key, name in (("att", "Attention received"), ("attr", "Occlusion attribution (Δ P(positive))")):
        # Titles go in up front: Plotly creates no annotation for an empty
        # title, so setting them afterwards by index raised IndexError.
        titles = (f"{name}: sentence A", f"{name}: sentence B",
                  "Difference (B − A) on shared words")
        fig = make_subplots(rows=3, cols=1, subplot_titles=titles, vertical_spacing=0.14)
        _tok_bars(fig, 1, cmp["tokens_a"], cmp[f"{key}_a"], set(cmp["swapped_a"]))
        _tok_bars(fig, 2, cmp["tokens_b"], cmp[f"{key}_b"], set(cmp["swapped_b"]))
        diff = np.zeros(len(cmp["tokens_a"]))
        diff[cmp["shared_a"]] = cmp[f"{key}_diff"]
        _tok_bars(fig, 3, cmp["tokens_a"], diff, set(), diverging=True)
        fig.update_layout(height=640, showlegend=False, plot_bgcolor=BG_SOFT,
                          font_family=FONT_FAMILY, margin=dict(l=10, r=10, t=40, b=10))
        figs.append(fig)
    return figs[0], figs[1]
