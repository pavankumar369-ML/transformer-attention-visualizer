"""Plotly figures for the Causal tab (no Streamlit here)."""
import plotly.graph_objects as go

from shared.config import ACCENT, ACCENT_SOFT, AMBER, BG_SOFT, FONT_FAMILY, GRID, NEUTRAL


def importance_fig(acc_drop) -> go.Figure:
    L, H = acc_drop.shape
    # acc_drop is "baseline minus accuracy after removal"; plot the actual change,
    # so a head whose removal hurts shows a negative number.
    z = -acc_drop * 100
    lim = max(float(abs(z).max()), 0.5)
    # Centred on zero: cobalt = removing the head hurts, amber = removing it helps.
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"H{h + 1}" for h in range(H)], y=[f"L{l + 1}" for l in range(L)],
        colorscale=[[0, ACCENT], [0.5, "#FFFFFF"], [1, AMBER]], zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Accuracy change<br>when removed (pts)", side="right"),
                      thickness=10, outlinewidth=0),
        hovertemplate="Layer %{y}, head %{x}<br>removing it changes accuracy by %{z:+.1f} pts<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 30 * L, margin=dict(l=10, r=10, t=20, b=10), font_family=FONT_FAMILY)
    return fig


def pruning_fig(df) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.pct_removed, y=df.acc_importance * 100, name="Least important removed first",
                             line=dict(color=ACCENT, width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df.pct_removed, y=df.acc_random_mean * 100, name="Random order (average of 5)",
                             line=dict(color=NEUTRAL, dash="dash"), mode="lines+markers",
                             error_y=dict(type="data", array=df.acc_random_std * 100, color=ACCENT_SOFT)))
    fig.update_layout(xaxis=dict(title="% of heads removed", gridcolor=GRID),
                      yaxis=dict(title="Accuracy (%)", gridcolor=GRID), plot_bgcolor=BG_SOFT,
                      height=380, margin=dict(l=10, r=10, t=20, b=10), font_family=FONT_FAMILY)
    return fig
