"""
One visual language for the whole app.

apply_theme(st) is called once at the top of app/main.py. It
  1. injects the page CSS (type, spacing, input, header),
  2. registers a Plotly template so every lens's charts share fonts,
     colours and gridlines without each lens styling them by hand,
  3. makes st.plotly_chart default to that template and hide Plotly's
     floating toolbar, which is visual noise in a presentation.

Lens code needs no changes to pick this up.
"""

import plotly.graph_objects as go
import plotly.io as pio

from shared.config import ACCENT, AMBER, FONT_FAMILY, GRID, INK, NEUTRAL, TEAL

PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
  --ink: #1C2433; --slate: #5B6577; --rule: #E2E6EC; --mist: #F5F7FA;
  --cobalt: #2D5BE3; --amber: #C27C0E;
  --font: 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Page frame */
.stApp { background: #FFFFFF; color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], footer { display: none; }
.block-container { max-width: 1120px; padding: 2.25rem 2rem 5rem; }

/* Type. Targets text elements only, so Streamlit's icon font is untouched. */
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp button p, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"] { font-family: var(--font); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--ink); letter-spacing: -0.01em; }
.stApp h3 { font-size: 1.15rem; font-weight: 600; padding-top: 1.5rem; }
.stApp h4 { font-size: 1.02rem; font-weight: 600; padding-top: 1.25rem; }
.stApp [data-testid="stCaptionContainer"] { color: var(--slate); }
.stApp p { text-wrap: pretty; }

/* Masthead */
.tav-top {
  display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem;
  padding-bottom: 1.25rem; margin-bottom: 1.75rem; border-bottom: 1px solid var(--rule);
}
.tav-name { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.015em; line-height: 1.2; }
.tav-tag { color: var(--slate); font-size: 0.97rem; margin-top: 0.35rem; max-width: 62ch; }
.tav-model { text-align: right; font-size: 0.92rem; font-weight: 500; white-space: nowrap; }
.tav-model span { display: block; color: var(--slate); font-weight: 400; font-size: 0.8rem; margin-top: 0.15rem; }

/* The sentence is the hero: large, quiet field */
.stTextInput input { font-size: 1.4rem; line-height: 1.4; padding: 0.85rem 1rem; color: var(--ink); }
.stTextInput [data-baseweb="input"] { border-radius: 8px; border: 1px solid #D5DAE2; background: #FFFFFF; }
.stTextInput [data-baseweb="base-input"], .stTextInput input { background: #FFFFFF; }
.stTextInput [data-baseweb="input"]:focus-within {
  border-color: var(--cobalt); box-shadow: 0 0 0 3px rgba(45, 91, 227, 0.15);
}
.tav-note { color: var(--slate); font-size: 0.93rem; margin: 0.9rem 0 0; max-width: 72ch; }
.tav-note b { color: var(--ink); font-weight: 500; }

/* Lens heading */
.tav-lens-head { margin: 2rem 0 1.25rem; }
.tav-lens-head h2 { font-size: 1.65rem; font-weight: 600; margin: 0; padding: 0; letter-spacing: -0.015em; }
.tav-lens-head p { color: var(--slate); margin: 0.45rem 0 0; max-width: 68ch; }
.tav-meta { color: var(--slate); font-size: 0.82rem; margin-top: 0.6rem; }

/* Attention: the sentence tinted by where one word looks */
.tav-strip { font-size: 1.6rem; line-height: 2.5; margin: 0.25rem 0 0.25rem; }
.tav-tok { padding: 0.18rem 0.32rem; border-radius: 4px; margin: 0 0.05rem; transition: background-color 120ms; }
.tav-tok.q { box-shadow: inset 0 -2px 0 var(--ink); font-weight: 600; }
.tav-tok.sub { margin-left: -0.18rem; }

/* Ranked list next to the arc chart */
.tav-rank { display: grid; grid-template-columns: minmax(4rem, 8rem) 1fr 3rem; gap: 0.55rem 0.75rem;
            align-items: center; font-size: 0.93rem; margin-top: 0.5rem; }
.tav-rank .bar { height: 8px; background: var(--mist); border-radius: 4px; overflow: hidden; }
.tav-rank .bar i { display: block; height: 100%; background: var(--cobalt); border-radius: 4px; }
.tav-rank .pct { color: var(--slate); text-align: right; font-variant-numeric: tabular-nums; }
.tav-rank .w { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Containers */
[data-testid="stExpander"] details { border: 1px solid var(--rule); border-radius: 8px; }
[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stMetricValue"] { font-family: var(--font); font-weight: 600; }

/* Keyboard focus stays visible */
.stApp button:focus-visible, .stApp [role="tab"]:focus-visible { outline: 2px solid var(--cobalt); outline-offset: 2px; }

@media (max-width: 680px) {
  .block-container { padding: 1.25rem 1rem 3rem; }
  .tav-top { flex-direction: column; align-items: flex-start; gap: 0.75rem; }
  .tav-model { text-align: left; }
  .stTextInput input { font-size: 1.15rem; }
  .tav-strip { font-size: 1.25rem; }
}
@media (prefers-reduced-motion: reduce) { .tav-tok { transition: none; } }
</style>
"""


def _register_plotly_template() -> None:
    axis = dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="",
                title=dict(font=dict(size=12, color=NEUTRAL)), tickfont=dict(size=12, color=NEUTRAL))
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        colorway=[ACCENT, AMBER, TEAL, NEUTRAL],
        xaxis=axis, yaxis=axis,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID,
                        font=dict(family=FONT_FAMILY, color=INK, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=NEUTRAL)),
        margin=dict(l=8, r=8, t=36, b=8),
        title=dict(font=dict(size=14, color=INK)),
    )
    pio.templates["tav"] = template
    pio.templates.default = "plotly_white+tav"


def _patch_plotly_chart(st) -> None:
    """Default every st.plotly_chart call to our template and a clean config."""
    if getattr(st.plotly_chart, "_tav", False):
        return
    original = st.plotly_chart

    def plotly_chart(fig, *args, **kwargs):
        kwargs.setdefault("theme", None)          # keep our template, not Streamlit's
        kwargs.setdefault("config", PLOTLY_CONFIG)
        return original(fig, *args, **kwargs)

    plotly_chart._tav = True
    st.plotly_chart = plotly_chart


def apply_theme(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _register_plotly_template()
    _patch_plotly_chart(st)
