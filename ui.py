"""Tampilan bersama: warna, CSS, komponen HTML, gaya grafik Plotly, dan konteks aplikasi."""
from __future__ import annotations

import html as _html
import inspect
import re
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import energy_core as ec

NAVY = "#10233F"
SOFT = "#53647C"
VOLT = "#FFC400"
BLUE = "#1456D6"
GREEN = "#1B7A57"  # khusus untuk uang
AMBER = "#D9770A"
TEAL = "#1F9AA0"  # energi terbarukan
RED = "#C62828"
GRID = "#E3E9F0"
LINE = "#D2DBE6"
MUTED = "#A9BAD1"
FONT = "Public Sans, system-ui, sans-serif"
DISPLAY_FONT = "Barlow Condensed, Arial Narrow, sans-serif"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Public+Sans:wght@400;500;600;700&display=swap');

:root { --navy:#10233F; --soft:#53647C; --volt:#FFC400; --blue:#1456D6; --green:#1B7A57; --amber:#D9770A; --line:#D2DBE6; }

[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(rgba(16,35,63,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(16,35,63,.05) 1px, transparent 1px);
  background-size: 28px 28px;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button {
  font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
  font-weight: 700; letter-spacing: 0; color: var(--navy);
}
.stApp h2 { font-size: 2rem; line-height: 1.1; padding-top: .2rem; }
.stApp h3 { font-size: 1.5rem; line-height: 1.15; }
[data-testid="stSidebar"] h4 { color: #fff; font-size: 1.3rem; margin-top: .5rem; }

/* Kepala halaman */
.hero { display:flex; gap:18px; align-items:center; margin: 0 0 1.4rem; }
.hero-mark { width:62px; height:62px; flex:none; border-radius:12px; background:var(--navy); display:grid; place-items:center; }
.hero-bolt { width:26px; height:38px; background:var(--volt); clip-path:polygon(58% 0%, 10% 58%, 44% 58%, 32% 100%, 92% 36%, 56% 36%); }
.stApp .hero h1 { font-size: clamp(2.1rem, 4.4vw, 3.1rem); line-height:1; margin:0; padding:0; }
.hero p { margin:.4rem 0 0; color:var(--soft); font-size:1.02rem; max-width: 66ch; }

/* Tab dan formulir */
.stTabs [role="tablist"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [role="tab"] { padding: .55rem 1.15rem; }
.stTabs [role="tab"] p { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.3rem; font-weight:600; color:var(--soft); }
.stTabs [role="tab"][aria-selected="true"] p { color:var(--navy); }
.stTabs [data-baseweb="tab-highlight"], .stTabs .react-aria-SelectionIndicator { background-color: var(--volt) !important; height: 4px !important; border-color: var(--volt) !important; }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:10px; padding:1.3rem 1.4rem; }

/* Angka ringkas */
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:8px; padding:.85rem 1.05rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div { color:var(--soft); font-size:.92rem;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height:1.3; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
  font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:2.1rem; color:var(--navy); line-height:1.1;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}

/* Tombol utama berwarna kuning volt */
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"],
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--volt); border: 0; border-radius: 8px; padding: .65rem 1rem;
}
.stApp button[kind="primary"] p, .stApp button[kind="primaryFormSubmit"] p,
.stApp [data-testid="stBaseButton-primary"] p, .stApp [data-testid="stBaseButton-primaryFormSubmit"] p {
  color: var(--navy); font-weight: 700; font-size: 1.02rem;
}
.stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover,
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover { background:#FFD43D; }

/* Meteran kWh */
.meter { background: linear-gradient(160deg, #17324F 0%, #0B1A2D 100%); border:1px solid #08121F; border-radius:14px;
  padding: 1.15rem 1.4rem 1.2rem; color:#E8EEF6; box-shadow: inset 0 0 0 4px rgba(255,255,255,.04); }
.meter-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; }
.meter-title { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.4rem; font-weight:600; color:#fff; }
.meter-unit { font-size:.92rem; color:#A8BBD3; }
.odo { display:flex; gap:4px; justify-content:center; align-items:center; margin:.95rem 0 .75rem; }
.odo .d { position:relative; overflow:hidden; display:grid; place-items:center; border-radius:3px;
  width:clamp(2.1rem, 6.4vw, 3.15rem); height:clamp(3.2rem, 9.2vw, 4.7rem);
  background: linear-gradient(180deg, #000 0%, #2B3139 48%, #000 100%);
  color:#fff; font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; line-height:1;
  font-size: clamp(2.5rem, 7.2vw, 3.8rem); box-shadow: 0 1px 0 rgba(255,255,255,.08); }
.odo .d::after { content:""; position:absolute; inset:0 0 52% 0; background:linear-gradient(180deg, rgba(255,255,255,.16), rgba(255,255,255,0)); pointer-events:none; }
.odo .d.r { background: linear-gradient(180deg, #F6F6F1 0%, #D9D9D1 50%, #F6F6F1 100%); color:#C62828; }
.odo .d.r::after { background:linear-gradient(180deg, rgba(255,255,255,.55), rgba(255,255,255,0)); }
.odo .d b { font-weight:700; display:block; animation: roll .75s cubic-bezier(.2,.8,.2,1) both; }
.odo .pt { width:7px; height:7px; border-radius:50%; background:var(--volt); margin:0 2px; align-self:flex-end; margin-bottom:.45rem; }
@keyframes roll { from { transform: translateY(-115%); } to { transform: translateY(0); } }
.meter-foot { font-size:.95rem; line-height:1.5; color:#BCCBDF; text-align:center; }
.meter-foot b { color: var(--volt); font-weight:600; }

/* Kartu keterangan di bawah meteran */
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap:10px; margin-top:1rem; }
.stat { background:#fff; border:1px solid var(--line); border-radius:8px; padding:.7rem .9rem; }
.stat small { display:block; color:var(--soft); font-size:.82rem; }
.stat b { display:block; font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.6rem; font-weight:700; color:var(--navy); line-height:1.15; }
.stat span.sub { display:block; color:var(--soft); font-size:.8rem; line-height:1.35; }
.dot { display:inline-block; width:.62em; height:.62em; border-radius:50%; margin-right:.4em; vertical-align:.06em; }
.dot.low { background:#1B7A57; } .dot.mid { background:#FFC400; } .dot.high { background:#C62828; }

/* Struk biaya */
.struk { background:#fff; border:1px solid var(--line); border-left:6px solid var(--green); border-radius:6px; padding:1rem 1.2rem 1rem; margin-top:1rem; }
.struk-title { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:1.45rem; color:var(--navy); margin-bottom:.3rem; }
.struk-row { display:flex; align-items:baseline; gap:10px; padding:.3rem 0; }
.struk-row .lab { color:var(--navy); font-weight:500; }
.struk-row .lab small { display:block; color:var(--soft); font-size:.8rem; font-weight:400; }
.struk-row .lead { flex:1; min-width:14px; border-bottom:2px dotted #B7C4D4; transform:translateY(-4px); }
.struk-row .amt { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:1.6rem; color:var(--green); font-variant-numeric:tabular-nums; white-space:nowrap; }
.struk-note { margin-top:.55rem; font-size:.82rem; color:var(--soft); line-height:1.45; }

/* Judul grafik */
.chart-head { margin: 1.5rem 0 .35rem; }
.chart-head b { display:block; font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.5rem; font-weight:700; color:var(--navy); line-height:1.15; }
.chart-head span { display:block; color:var(--soft); font-size:.92rem; line-height:1.4; margin-top:.15rem; }

/* Sidebar */
.side-brand { display:flex; align-items:center; gap:12px; margin: .2rem 0 .9rem; }
.side-brand .hero-mark { width:42px; height:42px; background:#1B3558; border-radius:9px; }
.side-brand .hero-bolt { width:17px; height:25px; }
.side-brand span { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.45rem; font-weight:700; color:#fff; line-height:1.05; }
.tarif-chip { display:inline-block; margin:.2rem 0 .4rem; padding:.35rem .7rem; border-radius:6px; background:var(--volt); color:var(--navy);
  font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:1.35rem; }
.side-note { color:#A8BBD3; font-size:.82rem; line-height:1.5; }

@media (prefers-reduced-motion: reduce) { .odo .d b { animation: none; } }
"""
_CSS_MIN = re.sub(r"\s+", " ", CSS)


def inject_css() -> None:
    st.markdown("<style>" + _CSS_MIN + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Konteks yang dibagikan ke semua halaman
# --------------------------------------------------------------------------- #
@dataclass
class AppContext:
    raw: pd.DataFrame  # seluruh data (bersih)
    data: pd.DataFrame  # seluruh data + kolom turunan
    df: pd.DataFrame  # data setelah filter sidebar
    bundle: ec.ModelBundle
    tariff: float
    model_name: str
    source: str
    is_default: bool


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #
def _stretch(fn) -> dict:
    """Kompatibel dengan versi Streamlit lama (use_container_width) maupun baru (width)."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


STRETCH_BTN = _stretch(st.button)
STRETCH_PLOT = _stretch(st.plotly_chart)
STRETCH_DF = _stretch(st.dataframe)


def esc(value: object) -> str:
    return _html.escape(str(value))


def one_line(s: str) -> str:
    """Satukan HTML jadi satu baris agar tidak terbaca sebagai blok kode Markdown."""
    return "".join(line.strip() for line in s.splitlines())


def hero(title: str, sub: str) -> None:
    st.markdown(
        one_line(f'<div class="hero"><div class="hero-mark"><div class="hero-bolt"></div></div>'
                 f"<div><h1>{esc(title)}</h1><p>{esc(sub)}</p></div></div>"),
        unsafe_allow_html=True,
    )


def chart_head(title: str, sub: str | None = None) -> None:
    extra = f"<span>{sub}</span>" if sub else ""
    st.markdown(one_line(f'<div class="chart-head"><b>{title}</b>{extra}</div>'), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Grafik
# --------------------------------------------------------------------------- #
def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=13, color=NAVY),
        separators=",.",
        hoverlabel=dict(bgcolor=NAVY, font=dict(color="#FFFFFF", family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, automargin=True)
    return fig


def base_fig(height: int = 340) -> go.Figure:
    return style_fig(go.Figure(), height)


def style_subplot_titles(fig: go.Figure, size: int = 17) -> None:
    for ann in fig.layout.annotations:
        ann.font = dict(family=DISPLAY_FONT, size=size, color=NAVY)


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, theme=None, config={"displaylogo": False, "displayModeBar": False}, **STRETCH_PLOT)


# --------------------------------------------------------------------------- #
# Komponen HTML
# --------------------------------------------------------------------------- #
def meter_html(value: float, foot: str, title: str = "Prediksi konsumsi", unit: str = "kWh per jam") -> str:
    v = max(float(value), 0.0)
    int_part, dec_part = f"{v:07.2f}".split(".")
    wheels = "".join(
        f'<span class="d"><b style="animation-delay:{i * 70}ms">{ch}</b></span>' for i, ch in enumerate(int_part)
    )
    wheels += '<span class="pt"></span>'
    wheels += "".join(
        f'<span class="d r"><b style="animation-delay:{(len(int_part) + i) * 70}ms">{ch}</b></span>'
        for i, ch in enumerate(dec_part)
    )
    label = f"{ec.num_id(v, 2)} {unit}"
    return one_line(
        f'<div class="meter"><div class="meter-head"><span class="meter-title">{title}</span>'
        f'<span class="meter-unit">{unit}</span></div>'
        f'<div class="odo" role="img" aria-label="{label}">{wheels}</div>'
        f'<div class="meter-foot">{foot}</div></div>'
    )


def stat_cards_html(cards: list[tuple[str, str, str]]) -> str:
    items = "".join(
        f'<div class="stat"><small>{label}</small><b>{value}</b><span class="sub">{sub}</span></div>'
        for label, value, sub in cards
    )
    return one_line(f'<div class="stats">{items}</div>')


def receipt_html(rows: list[tuple[str, float, float]], tariff: float) -> str:
    items = "".join(
        f'<div class="struk-row"><span class="lab">{label}<small>{ec.num_id(kwh, 2)} kWh</small></span>'
        f'<span class="lead"></span><span class="amt">{ec.rupiah(cost)}</span></div>'
        for label, kwh, cost in rows
    )
    return one_line(
        f'<div class="struk"><div class="struk-title">Estimasi biaya listrik</div>{items}'
        f'<div class="struk-note">Tarif {ec.rupiah(tariff, 2)} per kWh, belum termasuk pajak dan biaya lain. '
        f"Proyeksi 24 jam dan 30 hari mengasumsikan kondisi yang sama berlangsung terus.</div></div>"
    )
