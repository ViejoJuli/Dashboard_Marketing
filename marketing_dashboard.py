import numpy as np
import pandas as pd
from datetime import datetime

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context


# =========================================================
# THEME / CONSTANTS
# =========================================================
FONT_STACK = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"
STAGES = ["Impression", "Click", "Lead", "MQL", "SQL", "Won"]
EMPLOYEES = ["All", "Sofía", "Mateo", "Valentina", "Juan"]

THEME = {
    "bg": "#070812",
    "text": "#F3F4FF",
    "muted": "rgba(243,244,255,0.72)",
    "muted2": "rgba(243,244,255,0.55)",
    "border": "rgba(255,255,255,0.10)",
    "border_soft": "rgba(255,255,255,0.08)",
    "panel": "rgba(255,255,255,0.055)",
    "panel2": "rgba(0,0,0,0.22)",
    "radius": "22px",
    "pill": "999px",
    "shadow": "0 10px 26px rgba(0,0,0,0.42)",
}

STAGE_COLORS = {
    "Impression": "#5B5CFF",
    "Click":      "#7B2FF7",
    "Lead":       "#B44CFF",
    "MQL":        "#FF2BD6",
    "SQL":        "#FF3D7F",
    "Won":        "#FF6A3D",
}

KPI_HELP = {
    "Impressions": "Total de veces que se mostró el anuncio/contenido. Base del funnel.",
    "CTR": "Click-Through Rate = Clicks / Impressions. Mide atractivo del creativo/copy.",
    "Click → Lead": "Leads / Clicks. Mide landing + fricción del formulario + oferta.",
    "Lead → MQL": "MQL / Leads. Mide calidad del lead y el scoring de marketing.",
    "MQL → SQL": "SQL / MQL. Mide alineación marketing–ventas y proceso de calificación.",
    "SQL → Won": "Won / SQL. Mide cierre final del pipeline comercial.",
}

OBJECTIVES = [
    "Medir eficiencia del funnel por etapa (conversión vs paso anterior).",
    "Detectar cuellos de botella para priorizar optimizaciones (landing, scoring, ventas).",
    "Comparar rendimiento por empleado para evaluar impacto y coaching.",
]


# =========================================================
# DUMMY DATA
# =========================================================
def _rng(seed=7):
    return np.random.default_rng(seed)


def make_dummy_funnel(seed=11):
    rng = _rng(seed)
    base = int(rng.integers(1_800_000, 2_500_000))

    ctr = float(np.clip(rng.normal(0.018, 0.006), 0.006, 0.06))
    click = int(base * ctr)

    click_to_lead = float(np.clip(rng.normal(0.07, 0.02), 0.02, 0.22))
    lead = int(click * click_to_lead)

    lead_to_mql = float(np.clip(rng.normal(0.45, 0.12), 0.10, 0.85))
    mql = int(lead * lead_to_mql)

    mql_to_sql = float(np.clip(rng.normal(0.35, 0.10), 0.08, 0.80))
    sql = int(mql * mql_to_sql)

    sql_to_won = float(np.clip(rng.normal(0.18, 0.07), 0.02, 0.55))
    won = int(sql * sql_to_won)

    click = min(click, base)
    lead = min(lead, click)
    mql = min(mql, lead)
    sql = min(sql, mql)
    won = min(won, sql)
    return [base, click, lead, mql, sql, won]


def split_by_employees(all_counts, seed=11):
    rng = _rng(seed)
    weights = np.array([0.28, 0.25, 0.25, 0.22])
    weights = weights / weights.sum()
    emps = ["Sofía", "Mateo", "Valentina", "Juan"]

    per_emp = {}
    for i, e in enumerate(emps):
        factor = weights[i] * float(np.clip(rng.normal(1.0, 0.08), 0.85, 1.15))
        counts = [int(c * factor) for c in all_counts]
        for k in range(1, len(counts)):
            counts[k] = min(counts[k], counts[k-1])
        per_emp[e] = counts

    summed = [0] * len(all_counts)
    for e in emps:
        for i in range(len(all_counts)):
            summed[i] += per_emp[e][i]
    per_emp["All"] = summed
    return per_emp


def make_3_months_kpis(counts_now, seed=77):
    rng = _rng(seed)
    end = datetime.now().replace(day=1)
    months = [
        (end - pd.DateOffset(months=2)).strftime("%Y-%m"),
        (end - pd.DateOffset(months=1)).strftime("%Y-%m"),
        (end - pd.DateOffset(months=0)).strftime("%Y-%m"),
    ]

    base = np.array(counts_now, dtype=float)
    rows = []
    for idx, m in enumerate(months):
        drift = 0.90 + 0.06 * idx + \
            float(np.clip(rng.normal(0, 0.03), -0.05, 0.05))
        vals = (base * drift).astype(int)
        for k in range(1, len(vals)):
            vals[k] = min(vals[k], vals[k-1])

        imp, clk, lead, mql, sql, won = vals

        def safe_pct(num, den):
            return (num / den * 100.0) if den else 0.0

        rows.append({
            "month": m,
            "impressions": int(imp),
            "clicks": int(clk),
            "leads": int(lead),
            "mql": int(mql),
            "sql": int(sql),
            "won": int(won),
            "ctr": safe_pct(clk, imp),
            "click_to_lead": safe_pct(lead, clk),
            "lead_to_mql": safe_pct(mql, lead),
            "mql_to_sql": safe_pct(sql, mql),
            "sql_to_won": safe_pct(won, sql),
        })
    return pd.DataFrame(rows)


BASE_COUNTS = make_dummy_funnel(seed=11)
DATA_BY_EMP = split_by_employees(BASE_COUNTS, seed=11)


# =========================================================
# FORMATTERS
# =========================================================
def fmt_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def pct_prev(curr: int, prev: int) -> float:
    return (curr / prev * 100.0) if prev else 0.0


def pct_prev_str(curr: int, prev: int) -> str:
    return f"{pct_prev(curr, prev):.2f}%"


# =========================================================
# FIGURES
# =========================================================
def build_funnel_bar(counts):
    colors = [STAGE_COLORS[s] for s in STAGES]
    labels = []
    for i, c in enumerate(counts):
        if i == 0:
            labels.append(f"{fmt_int(c)} · Base")
        else:
            labels.append(f"{fmt_int(c)} · {pct_prev_str(c, counts[i-1])}")

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=STAGES,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=labels,
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>",
            opacity=0.98,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["text"], "family": FONT_STACK, "size": 13},
        margin={"l": 28, "r": 24, "t": 8, "b": 8},
        barcornerradius=18,
        autosize=True,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


def build_kpi_trend(df, col, title):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["text"], "family": FONT_STACK},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        title={"text": title, "x": 0.02, "xanchor": "left"},
    )
    fig.add_trace(
        go.Scatter(
            x=df["month"].tolist(),
            y=df[col].tolist(),
            mode="lines+markers",
            line={"width": 4},
            marker={"size": 10},
            hovertemplate="%{x}<br><b>%{y}</b><extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# =========================================================
# UI COMPONENTS
# =========================================================
def help_icon(text):
    return html.Span(
        "?",
        title=text,
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "width": "18px",
            "height": "18px",
            "borderRadius": THEME["pill"],
            "border": "1px solid rgba(255,255,255,0.18)",
            "background": "rgba(0,0,0,0.20)",
            "fontSize": "12px",
            "fontWeight": "950",
            "cursor": "help",
            "userSelect": "none",
            "marginLeft": "8px",
            "color": THEME["text"],
            "opacity": 0.92,
        }
    )


def kpi_card(kpi_id, title, value, subtitle, accent):
    return html.Button(
        id={"type": "kpi", "kpi": kpi_id},
        n_clicks=0,
        style={"all": "unset", "cursor": "pointer",
               "display": "block", "borderRadius": THEME["radius"]},
        children=html.Div(
            style={
                "padding": "12px 12px 12px 14px",
                "borderRadius": THEME["radius"],
                "background": THEME["panel2"],
                "border": f"1px solid {THEME['border_soft']}",
                "minHeight": "118px",
                "position": "relative",
                "overflow": "hidden",
            },
            children=[
                html.Div(
                    style={
                        "position": "absolute",
                        "left": 0, "top": 0, "bottom": 0,
                        "width": "6px",
                        "background": f"linear-gradient(180deg, {accent} 0%, rgba(0,0,0,0) 140%)",
                    }
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "justifyContent": "space-between"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center"},
                            children=[
                                html.Div(title, style={
                                         "fontSize": "12.8px", "fontWeight": "950", "letterSpacing": "0.2px"}),
                                help_icon(KPI_HELP.get(
                                    title, "KPI del dashboard.")),
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "8px", "height": "8px",
                                "borderRadius": THEME["pill"],
                                "background": accent,
                                "boxShadow": f"0 0 10px {accent}66",
                            }
                        ),
                    ],
                ),
                html.Div(style={"height": "10px"}),
                html.Div(
                    value,
                    style={
                        "display": "inline-block",
                        "padding": "9px 12px",
                        "borderRadius": THEME["pill"],
                        "background": f"linear-gradient(90deg, {accent} 0%, rgba(255,255,255,0.06) 140%)",
                        "border": f"1px solid {accent}55",
                        "boxShadow": f"0 0 18px {accent}2f",
                        "fontSize": "21px",
                        "fontWeight": "950",
                    },
                ),
                html.Div(subtitle, style={
                         "fontSize": "11.2px", "color": THEME["muted"], "marginTop": "10px", "fontWeight": "850"}),
                html.Div("Click → evolución (3 meses)", style={
                         "fontSize": "10.4px", "color": "rgba(243,244,255,0.45)", "marginTop": "6px", "fontWeight": "900"}),
            ],
        )
    )


def panel(title, subtitle=None, right=None, children=None):
    head = [
        html.Div([
            html.Div(title, style={"fontSize": "17px", "fontWeight": "950"}),
            html.Div(subtitle, style={
                     "fontSize": "12px", "color": THEME["muted"], "marginTop": "3px", "fontWeight": "800"}) if subtitle else None,
        ]),
        right,
    ]
    head = [x for x in head if x is not None]

    return html.Div(
        style={
            "height": "100%",
            "borderRadius": THEME["radius"],
            "background": THEME["panel"],
            "border": f"1px solid {THEME['border']}",
            "boxShadow": THEME["shadow"],
            "padding": "12px",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
            "minHeight": 0,
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "gap": "10px", "flexWrap": "wrap"},
                children=head,
            ),
            html.Div(style={
                     "height": "1px", "background": "rgba(255,255,255,0.06)", "margin": "10px 0"}),
            html.Div(children=children, style={
                     "flex": "1 1 auto", "minHeight": 0}),
        ],
    )


# =========================================================
# APP
# =========================================================
app = Dash(__name__)
server = app.server
app.title = "Marketing Funnel Dashboard"

app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>
      :root {{
        --bg: {THEME["bg"]};
        --text: {THEME["text"]};
        --muted: {THEME["muted"]};
        --border: {THEME["border"]};
        --radius: {THEME["radius"]};
        --pill: {THEME["pill"]};
        --font: {FONT_STACK};
      }}

      * {{ box-sizing: border-box; }}
      html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;  /* ✅ no scroll global */
        background: var(--bg);
        color: var(--text);
        font-family: var(--font);
      }}
      #react-entry-point, #react-entry-point > div {{
        height: 100%;
        min-height: 100%;
      }}

      /* Tabs */
      .dash-tabs-parent {{
        border-bottom: 0px !important;
        display: flex !important;
        justify-content: center !important;
      }}
      .dash-tabs .tab {{
        background: rgba(0,0,0,0.14) !important;
        color: rgba(243,244,255,0.78) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        padding: 8px 16px !important;
        font-weight: 950 !important;
        font-size: 12px !important;
        border-radius: var(--pill) !important;
        margin: 0 6px !important;
      }}
      .dash-tabs .tab--selected {{
        background: rgba(255,255,255,0.08) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
      }}

      /* Dropdown */
      .dash-dropdown {{ min-width: 180px; }}
      .Select-control {{
        background: rgba(0,0,0,0.20) !important;
        border-radius: var(--pill) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: var(--text) !important;
        box-shadow: none !important;
      }}
      .Select-placeholder, .Select-value-label {{
        color: rgba(243,244,255,0.90) !important;
        font-weight: 950 !important;
      }}
      .Select-menu-outer {{
        background: rgba(10,10,20,0.98) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important;
      }}

      /* Funnel background */
      .funnel-bg {{
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 10px;
        height: 100%;
        background:
          radial-gradient(1200px 600px at 10% 10%, rgba(123,47,247,0.20), transparent 50%),
          radial-gradient(900px 500px at 90% 20%, rgba(255,43,214,0.14), transparent 55%),
          radial-gradient(900px 600px at 60% 90%, rgba(255,106,61,0.12), transparent 55%),
          linear-gradient(180deg, rgba(0,0,0,0.16) 0%, rgba(0,0,0,0.08) 100%);
      }}
      .funnel-wrap {{
        position: relative;
        border-radius: var(--radius);
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }}
      .funnel-wrap::before {{
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
        background-size: 60px 60px;
        opacity: 0.14;
      }}

      /* Force Plotly to fill */
      #funnel-chart, #funnel-chart .dash-graph, #funnel-chart .js-plotly-plot, #funnel-chart .plot-container {{
        height: 100% !important;
      }}
      #kpi-trend, #kpi-trend .dash-graph, #kpi-trend .js-plotly-plot, #kpi-trend .plot-container {{
        height: 100% !important;
      }}

      /* ✅ KPI panel distribution */
      .kpi-panel-wrap{{
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}
      .kpi-grid{{
        flex: 1 1 auto;
        min-height: 0;
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-auto-rows: minmax(118px, auto);
        gap: 10px;
        overflow: auto;              /* ✅ scroll inside */
        padding-right: 2px;
      }}
      .kpi-grid::-webkit-scrollbar {{ width: 10px; }}
      .kpi-grid::-webkit-scrollbar-thumb {{
        background: rgba(255,255,255,0.14);
        border-radius: 999px;
        border: 2px solid rgba(0,0,0,0.35);
      }}
      .kpi-grid::-webkit-scrollbar-track {{
        background: rgba(0,0,0,0.10);
        border-radius: 999px;
      }}

      @media (max-width: 1200px) {{
        .grid-main {{ grid-template-columns: 1fr !important; }}
        .grid-details {{ grid-template-columns: 1fr !important; }}
        .kpi-grid {{ grid-template-columns: 1fr !important; }}
      }}
    </style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""

controls = html.Div(
    style={"display": "flex", "alignItems": "center",
           "gap": "10px", "flexWrap": "wrap"},
    children=[
        html.Div("Empleado", style={"fontSize": "12px",
                 "color": THEME["muted"], "fontWeight": "950"}),
        dcc.Dropdown(
            id="employee",
            options=[{"label": e, "value": e} for e in EMPLOYEES],
            value="All",
            clearable=False,
            className="dash-dropdown",
        ),
    ],
)

overview = html.Div(
    id="overview-view",
    style={"height": "100%", "minHeight": 0},
    children=html.Div(
        className="grid-main",
        style={
            "height": "100%",
            "display": "grid",
            "gridTemplateColumns": "7fr 3fr",  # ✅ 70/30
            "gap": "12px",
            "alignItems": "stretch",
            "minHeight": 0,
        },
        children=[
            panel(
                "Funnel",
                "El % mostrado es vs el paso anterior.",
                right=controls,
                children=html.Div(
                    style={"height": "100%", "display": "flex",
                           "flexDirection": "column", "gap": "10px", "minHeight": 0},
                    children=[
                        html.Div(
                            style={
                                "padding": "10px 12px",
                                "borderRadius": THEME["radius"],
                                "background": "rgba(0,0,0,0.16)",
                                "border": f"1px solid {THEME['border_soft']}",
                                "flex": "0 0 auto",
                            },
                            children=[
                                html.Div("Objetivos del dashboard", style={
                                         "fontWeight": "950", "fontSize": "13px"}),
                                html.Ul(
                                    style={"margin": "8px 0 0 18px", "padding": 0,
                                           "color": THEME["muted"], "fontWeight": "850", "fontSize": "12px", "lineHeight": "1.45"},
                                    children=[html.Li(x) for x in OBJECTIVES],
                                ),
                            ],
                        ),
                        html.Div(
                            className="funnel-wrap",
                            style={"flex": "1 1 auto", "minHeight": 0,
                                   "height": "clamp(360px, 52vh, 640px)"},
                            children=[
                                html.Div(
                                    className="funnel-bg",
                                    style={"height": "100%"},
                                    children=[
                                        dcc.Graph(
                                            id="funnel-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "100%",
                                                   "minHeight": 0},
                                        )
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ),

            panel(
                "KPIs",
                "Hover en “?” · Click en tarjeta → Details (3 meses).",
                children=html.Div(
                    className="kpi-panel-wrap",
                    children=[
                        html.Div(id="kpi-panel", className="kpi-grid"),
                        html.Div(
                            id="insights",
                            style={
                                "flex": "0 0 auto",
                                "padding": "12px",
                                "borderRadius": THEME["radius"],
                                "background": "rgba(0,0,0,0.16)",
                                "border": f"1px solid {THEME['border_soft']}",
                            },
                        ),
                    ],
                ),
            ),
        ],
    ),
)

details = html.Div(
    id="details-view",
    style={"height": "100%", "minHeight": 0, "display": "none"},
    children=html.Div(
        className="grid-details",
        style={
            "height": "100%",
            "display": "grid",
            "gridTemplateColumns": "0.60fr 0.40fr",
            "gap": "12px",
            "alignItems": "stretch",
            "minHeight": 0,
        },
        children=[
            panel(
                "Detalle KPI",
                "Evolución trimestral (3 meses).",
                children=html.Div(
                    style={"height": "100%", "display": "flex",
                           "flexDirection": "column", "minHeight": 0},
                    children=[
                        dcc.Graph(id="kpi-trend", config={"displayModeBar": False}, style={
                                  "flex": "1 1 auto", "minHeight": 0}),
                        html.Div(id="kpi-detail-text", style={
                                 "marginTop": "10px", "color": THEME["muted"], "fontWeight": "850", "fontSize": "12px"}),
                    ],
                ),
            ),
            panel(
                "Tabla rápida",
                "Últimos 3 meses (mes a mes).",
                children=html.Div(
                    id="kpi-table", style={"fontSize": "12px", "color": THEME["muted"], "fontWeight": "850"}),
            ),
        ],
    ),
)

app.layout = html.Div(
    style={
        "background": THEME["bg"],
        "height": "100dvh",
        "minHeight": "100dvh",
        "overflow": "hidden",
        "padding": "10px 12px",
        "paddingBottom": "16px",
        "fontFamily": FONT_STACK,
        "color": THEME["text"],
    },
    children=[
        dcc.Store(id="selected-kpi", data="impressions"),
        html.Div(
            style={
                "width": "100%",
                "maxWidth": "1900px",
                "margin": "0 auto",
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "height": "100%",
                "minHeight": 0,
            },
            children=[
                html.Div(
                    style={
                        "borderRadius": THEME["radius"],
                        "padding": "12px 14px",
                        "background": "linear-gradient(90deg, #7B2FF7 0%, #FF2BD6 45%, #FF6A3D 100%)",
                        "boxShadow": THEME["shadow"],
                        "border": f"1px solid {THEME['border']}",
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "10px",
                        "flexWrap": "wrap",
                        "flex": "0 0 auto",
                    },
                    children=[
                        html.Div([
                            html.Div("Marketing Dashboard", style={
                                     "fontSize": "12px", "opacity": 0.92, "fontWeight": "950"}),
                            html.Div("Funnel Overview", style={
                                     "fontSize": "26px", "fontWeight": "950", "lineHeight": "1.08"}),
                            html.Div("Funcional pero divertido • Click KPIs = detalle", style={
                                     "fontSize": "12px", "opacity": 0.92, "marginTop": "4px", "fontWeight": "850"}),
                        ]),
                        html.Div(
                            style={
                                "background": "rgba(0,0,0,0.20)",
                                "padding": "8px 10px",
                                "borderRadius": THEME["pill"],
                                "border": "1px solid rgba(255,255,255,0.18)",
                                "backdropFilter": "blur(6px)",
                                "fontSize": "12px",
                                "fontWeight": "950",
                            },
                            children=[
                                html.Span("Rango base: ", style={
                                          "opacity": 0.9}),
                                html.Span("Dummy (3 meses)", style={
                                          "fontWeight": "950"}),
                            ],
                        ),
                    ],
                ),

                dcc.Tabs(
                    id="tabs",
                    value="tab-overview",
                    parent_className="dash-tabs-parent",
                    className="dash-tabs",
                    children=[
                        dcc.Tab(label="Overview", value="tab-overview",
                                className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Details", value="tab-details",
                                className="tab", selected_className="tab--selected"),
                    ],
                ),

                html.Div(
                    id="content-wrapper",
                    style={
                        "flex": "1 1 auto",
                        "minHeight": 0,
                        "height": "calc(100dvh - 170px)",
                        "overflowY": "auto",
                        "overflowX": "hidden",
                        "paddingBottom": "12px",
                    },
                    children=[overview, details],
                ),
            ],
        ),
    ],
)


# =========================================================
# CALLBACKS
# =========================================================
@app.callback(
    Output("overview-view", "style"),
    Output("details-view", "style"),
    Input("tabs", "value"),
)
def toggle_views(tab):
    if tab == "tab-details":
        return {"display": "none"}, {"display": "block", "height": "100%", "minHeight": 0}
    return {"display": "block", "height": "100%", "minHeight": 0}, {"display": "none"}


@app.callback(
    Output("funnel-chart", "figure"),
    Output("kpi-panel", "children"),
    Output("insights", "children"),
    Input("employee", "value"),
)
def update_overview(employee):
    counts = DATA_BY_EMP.get(employee, DATA_BY_EMP["All"])
    fig = build_funnel_bar(counts)

    imp, clk, lead, mql, sql, won = counts

    cards = [
        kpi_card("impressions", "Impressions",
                 f"{imp:,}", "Base del funnel", STAGE_COLORS["Impression"]),
        kpi_card("ctr", "CTR", f"{pct_prev(clk, imp):.2f}%",
                 "Click / Impression", STAGE_COLORS["Click"]),
        kpi_card("click_to_lead", "Click → Lead",
                 f"{pct_prev(lead, clk):.2f}%", "Lead / Click", STAGE_COLORS["Lead"]),
        kpi_card("lead_to_mql", "Lead → MQL",
                 f"{pct_prev(mql, lead):.2f}%", "MQL / Lead", STAGE_COLORS["MQL"]),
        kpi_card("mql_to_sql", "MQL → SQL",
                 f"{pct_prev(sql, mql):.2f}%", "SQL / MQL", STAGE_COLORS["SQL"]),
        kpi_card("sql_to_won", "SQL → Won",
                 f"{pct_prev(won, sql):.2f}%", "Won / SQL", STAGE_COLORS["Won"]),
    ]

    rates = {
        "CTR": pct_prev(clk, imp),
        "Click → Lead": pct_prev(lead, clk),
        "Lead → MQL": pct_prev(mql, lead),
        "MQL → SQL": pct_prev(sql, mql),
        "SQL → Won": pct_prev(won, sql),
    }
    worst = min(rates.keys(), key=lambda k: rates[k])
    best = max(rates.keys(), key=lambda k: rates[k])

    insights = html.Div(
        children=[
            html.Div("Insights del rango (dummy)", style={
                     "fontWeight": "950", "fontSize": "13px", "marginBottom": "10px"}),
            html.Div(f"• Mayor caída: {worst} · {rates[worst]:.2f}%", style={
                     "fontWeight": "900", "color": THEME["muted"]}),
            html.Div(f"• Mejor etapa: {best} · {rates[best]:.2f}%", style={
                     "fontWeight": "900", "color": THEME["muted"], "marginTop": "6px"}),
            html.Div(f"• Won total: {won:,}", style={
                     "fontWeight": "900", "color": THEME["muted"], "marginTop": "6px"}),
            html.Div("Tip: dale click a un KPI para ver su evolución 3 meses.", style={
                     "marginTop": "10px", "color": THEME["muted2"], "fontWeight": "900", "fontSize": "11.5px"}),
        ]
    )

    return fig, cards, insights


@app.callback(
    Output("selected-kpi", "data"),
    Output("tabs", "value"),
    Input({"type": "kpi", "kpi": "impressions"}, "n_clicks"),
    Input({"type": "kpi", "kpi": "ctr"}, "n_clicks"),
    Input({"type": "kpi", "kpi": "click_to_lead"}, "n_clicks"),
    Input({"type": "kpi", "kpi": "lead_to_mql"}, "n_clicks"),
    Input({"type": "kpi", "kpi": "mql_to_sql"}, "n_clicks"),
    Input({"type": "kpi", "kpi": "sql_to_won"}, "n_clicks"),
    prevent_initial_call=True,
)
def kpi_click(*_):
    trig = callback_context.triggered
    if not trig:
        return "impressions", "tab-overview"

    prop_id = trig[0]["prop_id"]
    kpi = "impressions"
    if '"kpi":"' in prop_id:
        kpi = prop_id.split('"kpi":"')[1].split('"')[0]
    return kpi, "tab-details"


@app.callback(
    Output("kpi-trend", "figure"),
    Output("kpi-detail-text", "children"),
    Output("kpi-table", "children"),
    Input("selected-kpi", "data"),
    Input("employee", "value"),
)
def render_details(selected_kpi, employee):
    counts = DATA_BY_EMP.get(employee, DATA_BY_EMP["All"])
    seed = 77 if employee == "All" else (abs(hash(employee)) % 10_000)
    df = make_3_months_kpis(counts, seed=seed)

    meta = {
        "impressions": ("Impressions", "impressions", "Total de impresiones por mes."),
        "ctr": ("CTR", "ctr", "Clicks / Impressions (%)."),
        "click_to_lead": ("Click → Lead", "click_to_lead", "Leads / Clicks (%)."),
        "lead_to_mql": ("Lead → MQL", "lead_to_mql", "MQL / Leads (%)."),
        "mql_to_sql": ("MQL → SQL", "mql_to_sql", "SQL / MQL (%)."),
        "sql_to_won": ("SQL → Won", "sql_to_won", "Won / SQL (%)."),
    }

    title, col, expl = meta.get(selected_kpi, meta["ctr"])
    fig = build_kpi_trend(df, col, f"{title} · Últimos 3 meses")
    txt = f"Empleado: {employee} · {expl}"

    view = df[["month", col]].copy()
    if col in ["impressions", "clicks", "leads", "mql", "sql", "won"]:
        view[col] = view[col].map(lambda v: f"{int(v):,}")
    else:
        view[col] = view[col].map(lambda v: f"{float(v):.2f}%")

    table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(
                html.Tr([
                    html.Th("Mes", style={"textAlign": "left", "padding": "10px 8px",
                            "borderBottom": "1px solid rgba(255,255,255,0.08)", "color": THEME["text"]}),
                    html.Th(title, style={"textAlign": "left", "padding": "10px 8px",
                            "borderBottom": "1px solid rgba(255,255,255,0.08)", "color": THEME["text"]}),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(r["month"], style={
                            "padding": "10px 8px", "borderBottom": "1px solid rgba(255,255,255,0.06)"}),
                    html.Td(r[col], style={
                            "padding": "10px 8px", "borderBottom": "1px solid rgba(255,255,255,0.06)", "fontWeight": "950"}),
                ])
                for _, r in view.iterrows()
            ])
        ]
    )

    return fig, txt, table


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8050")
    app.run(host="0.0.0.0", port=8050, debug=True)
