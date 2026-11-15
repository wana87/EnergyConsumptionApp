"""Meteran prediksi energi: aplikasi Streamlit untuk memprediksi konsumsi listrik per jam.

Jalankan lokal:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import energy_core as ec

st.set_page_config(
    page_title="Meteran Prediksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

# --------------------------------------------------------------------------- #
# Warna dan tipografi
# --------------------------------------------------------------------------- #
NAVY = "#10233F"
SOFT = "#53647C"
VOLT = "#FFC400"
BLUE = "#1456D6"
GREEN = "#1B7A57"
AMBER = "#D9770A"
GRID = "#E3E9F0"
LINE = "#D2DBE6"
MUTED_BAR = "#A9BAD1"
FONT = "Public Sans, system-ui, sans-serif"

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
[data-testid="stSidebar"] h4 { color: #fff; font-size: 1.35rem; margin-top: .6rem; }

/* Kepala halaman */
.hero { display:flex; gap:18px; align-items:center; margin: 0 0 1.4rem; }
.hero-mark { width:62px; height:62px; flex:none; border-radius:12px; background:var(--navy); display:grid; place-items:center; }
.hero-bolt { width:26px; height:38px; background:var(--volt); clip-path:polygon(58% 0%, 10% 58%, 44% 58%, 32% 100%, 92% 36%, 56% 36%); }
.stApp .hero h1 { font-size: clamp(2.2rem, 4.6vw, 3.3rem); line-height:1; margin:0; padding:0; }
.hero p { margin:.4rem 0 0; color:var(--soft); font-size:1.02rem; max-width: 60ch; }

/* Tab */
.stTabs [role="tablist"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [role="tab"] { padding: .55rem 1.15rem; }
.stTabs [role="tab"] p { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.3rem; font-weight:600; color:var(--soft); }
.stTabs [role="tab"][aria-selected="true"] p { color:var(--navy); }
.stTabs [data-baseweb="tab-highlight"], .stTabs .react-aria-SelectionIndicator { background-color: var(--volt) !important; height: 4px !important; border-color: var(--volt) !important; }

/* Formulir dan angka ringkas */
[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:10px; padding:1.3rem 1.4rem; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:8px; padding:.85rem 1.05rem; }
[data-testid="stMetricLabel"] p { color:var(--soft); font-size:.92rem; }
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
.side-brand { display:flex; align-items:center; gap:12px; margin: .2rem 0 1rem; }
.side-brand .hero-mark { width:42px; height:42px; background:#1B3558; border-radius:9px; }
.side-brand .hero-bolt { width:17px; height:25px; }
.side-brand span { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-size:1.45rem; font-weight:700; color:#fff; line-height:1.05; }
.tarif-chip { display:inline-block; margin:.2rem 0 .4rem; padding:.35rem .7rem; border-radius:6px; background:var(--volt); color:var(--navy);
  font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:1.35rem; }
.side-note { color:#A8BBD3; font-size:.82rem; line-height:1.5; }

@media (prefers-reduced-motion: reduce) { .odo .d b { animation: none; } }
"""
CSS_MIN = re.sub(r"\s+", " ", CSS)
st.markdown("<style>" + CSS_MIN + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Utilitas tampilan
# --------------------------------------------------------------------------- #
def _stretch(fn) -> dict:
    """Kompatibel dengan versi Streamlit lama (use_container_width) maupun baru (width)."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


STRETCH_BTN = _stretch(st.button)
STRETCH_PLOT = _stretch(st.plotly_chart)
STRETCH_DF = _stretch(st.dataframe)


def html(s: str) -> str:
    """Satukan HTML jadi satu baris agar tidak terbaca sebagai blok kode Markdown."""
    return "".join(line.strip() for line in s.splitlines())


def chart_head(title: str, sub: str | None = None) -> None:
    extra = f"<span>{sub}</span>" if sub else ""
    st.markdown(html(f'<div class="chart-head"><b>{title}</b>{extra}</div>'), unsafe_allow_html=True)


def base_fig(height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=13, color=NAVY),
        separators=",.",
        hoverlabel=dict(bgcolor=NAVY, font=dict(color="#FFFFFF", family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, automargin=True),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, automargin=True),
    )
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, theme=None, config={"displaylogo": False, "displayModeBar": False}, **STRETCH_PLOT)


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
    return html(
        f'<div class="meter"><div class="meter-head"><span class="meter-title">{title}</span>'
        f'<span class="meter-unit">{unit}</span></div>'
        f'<div class="odo" role="img" aria-label="{label}">{wheels}</div>'
        f'<div class="meter-foot">{foot}</div></div>'
    )


def receipt_html(rows: list[tuple[str, float, float]], tariff: float) -> str:
    items = "".join(
        f'<div class="struk-row"><span class="lab">{label}<small>{ec.num_id(kwh, 2)} kWh</small></span>'
        f'<span class="lead"></span><span class="amt">{ec.rupiah(cost)}</span></div>'
        for label, kwh, cost in rows
    )
    return html(
        f'<div class="struk"><div class="struk-title">Estimasi biaya listrik</div>{items}'
        f'<div class="struk-note">Tarif {ec.rupiah(tariff, 2)} per kWh, belum termasuk pajak dan biaya lain. '
        f"Proyeksi 24 jam dan 30 hari mengasumsikan kondisi yang sama berlangsung terus.</div></div>"
    )


# --------------------------------------------------------------------------- #
# Sumber data dan model (dibuat sekali, lalu disimpan di cache)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    return ec.load_raw()


@st.cache_resource(show_spinner="Menyiapkan model…")
def get_model() -> ec.EnergyModel:
    return ec.train_model(get_raw())


TARIFFS = {
    "R-1 1.300–2.200 VA": 1444.70,
    "R-1 900 VA (RTM)": 1352.00,
    "R-2/R-3 3.500 VA ke atas": 1699.53,
    "R-1 900 VA bersubsidi": 605.00,
    "R-1 450 VA bersubsidi": 415.00,
}
CUSTOM_TARIFF = "Tarif sendiri"


def render_sidebar(model: ec.EnergyModel) -> float:
    with st.sidebar:
        st.markdown(
            html('<div class="side-brand"><div class="hero-mark"><div class="hero-bolt"></div></div><span>Meteran prediksi energi</span></div>'),
            unsafe_allow_html=True,
        )
        st.markdown("#### Tarif listrik")
        choice = st.selectbox(
            "Golongan tarif PLN",
            [*TARIFFS, CUSTOM_TARIFF],
            key="tariff_choice",
            help="RTM adalah Rumah Tangga Mampu. R-1 untuk daya sampai 2.200 VA, R-2 untuk 3.500–5.500 VA, R-3 untuk 6.600 VA ke atas.",
        )
        if choice == CUSTOM_TARIFF:
            tariff = st.number_input("Tarif (Rp per kWh)", min_value=0.0, value=1444.70, step=10.0, format="%.2f")
        else:
            tariff = TARIFFS[choice]
        st.markdown(f'<div class="tarif-chip">{ec.rupiah(tariff, 2)} per kWh</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-note">Tarif non-subsidi periode Oktober sampai Desember 2026. '
            "Ganti dengan tarif di tagihan Anda jika berbeda.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Model")
        st.markdown(
            html(
                f'<div class="side-note">Linear Regression, dilatih pada {model.metrics["n_train"]} jam data. '
                f'R² {ec.num_id(model.metrics["R2"], 3)} pada {model.metrics["n_test"]} jam data uji, '
                f'rata-rata meleset sekitar {ec.num_id(model.metrics["MAE"], 2)} kWh.</div>'
            ),
            unsafe_allow_html=True,
        )
    return float(tariff)


# --------------------------------------------------------------------------- #
# Tab 1: prediksi manual
# --------------------------------------------------------------------------- #
def contribution_chart(model: ec.EnergyModel, row: pd.DataFrame) -> go.Figure:
    contrib = model.contributions(row)
    contrib = contrib.reindex(contrib.abs().sort_values().index)  # terbesar di atas
    values = contrib.to_numpy()
    span = max(abs(values).max(), 1.0)
    fig = base_fig(370)
    fig.add_trace(
        go.Bar(
            x=values,
            y=[ec.LABELS[f] for f in contrib.index],
            orientation="h",
            marker_color=[AMBER if v > 0 else BLUE for v in values],
            text=[ec.num_id(v, 2, signed=True) for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(range=[-span * 1.55, span * 1.4], zeroline=True, zerolinecolor=NAVY, zerolinewidth=1.5,
                     title="kWh dibanding rata-rata")
    fig.update_yaxes(showgrid=False)
    return fig


def sweep_chart(model: ec.EnergyModel, row: pd.DataFrame, pred: float, feature: str, title: str, unit: str) -> go.Figure:
    if feature == "Hour":
        xs = np.arange(24)
    elif feature == "Occupancy":
        xs = np.arange(0, 10)
    else:
        lo, hi = model.train_range.loc[feature, "min"], model.train_range.loc[feature, "max"]
        xs = np.linspace(lo, hi, 41)
    frame = pd.concat([row] * len(xs), ignore_index=True)
    frame[feature] = xs
    ys = model.predict(frame)

    mid, half = (ys.max() + ys.min()) / 2, max((ys.max() - ys.min()) / 2 * 1.3, 3.0)
    fig = base_fig(370)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=3),
                             hovertemplate=f"{title} %{{x:.1f}} {unit}: %{{y:.2f}} kWh<extra></extra>", name="Prediksi"))
    current = float(row[feature].iloc[0])
    fig.add_trace(go.Scatter(x=[current], y=[pred], mode="markers", name="Kondisi saat ini",
                             marker=dict(size=15, color=VOLT, line=dict(color=NAVY, width=2.5)),
                             hovertemplate=f"Saat ini: %{{y:.2f}} kWh<extra></extra>"))
    fig.update_xaxes(title=f"{title} ({unit})" if unit else title)
    fig.update_yaxes(title="kWh per jam", range=[mid - half, mid + half])
    fig.update_layout(showlegend=False)
    return fig


SWEEPS = {
    "Suhu": ("Temperature", "Suhu", "°C"),
    "Jumlah penghuni": ("Occupancy", "Jumlah penghuni", "orang"),
    "Kelembapan": ("Humidity", "Kelembapan", "%"),
    "Energi terbarukan": ("RenewableEnergy", "Energi terbarukan", "%"),
    "Luas bangunan": ("SquareFootage", "Luas bangunan", "kaki²"),
    "Jam": ("Hour", "Jam", "pukul"),
}


def render_manual(model: ec.EnergyModel, tariff: float) -> None:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader("Kondisi bangunan")
        with st.form("form_manual"):
            c1, c2 = st.columns(2)
            temperature = c1.slider("Suhu (°C)", 20.0, 30.0, 25.0, 0.1)
            humidity = c2.slider("Kelembapan (%)", 30.0, 60.0, 45.0, 0.1)
            square = c1.slider("Luas bangunan (kaki²)", 1000, 2000, 1500, 10)
            occupancy = c2.slider("Jumlah penghuni", 0, 9, 5)
            renewable = c1.slider("Energi terbarukan (%)", 0.0, 30.0, 15.0, 0.5)
            hour = c2.slider("Jam", 0, 23, 12)
            t1, t2 = st.columns(2)
            hvac = t1.toggle("HVAC menyala", value=False)
            lighting = t2.toggle("Lampu menyala", value=False)
            holiday = t1.toggle("Hari libur", value=False)
            weekend = t2.toggle("Akhir pekan", value=False)
            submitted = st.form_submit_button("Prediksi konsumsi energi", type="primary", **STRETCH_BTN)
        st.caption("Ubah nilai, lalu tekan tombol prediksi. Rentang slider mengikuti rentang data latih.")
        if submitted:
            st.toast("Prediksi diperbarui")

    row = pd.DataFrame(
        [{
            "Temperature": temperature, "Humidity": humidity, "SquareFootage": square, "Occupancy": occupancy,
            "HVACUsage": int(hvac), "LightingUsage": int(lighting), "RenewableEnergy": renewable,
            "Holiday": int(holiday), "Hour": hour, "IsWeekend": int(weekend),
        }]
    )[ec.FEATURES]
    pred = float(model.predict(row)[0])

    day_frame = pd.concat([row] * 24, ignore_index=True)
    day_frame["Hour"] = np.arange(24)
    day_kwh = float(model.predict(day_frame).sum())

    diff = pred - model.baseline
    direction = "lebih tinggi" if diff >= 0 else "lebih rendah"
    foot = (
        f"Rata-rata data historis {ec.num_id(model.baseline, 2)} kWh per jam. "
        f"Kondisi ini <b>{ec.num_id(abs(diff), 2)} kWh {direction}</b>."
    )

    with right:
        st.markdown(meter_html(pred, foot), unsafe_allow_html=True)
        st.markdown(
            receipt_html(
                [
                    ("1 jam", pred, pred * tariff),
                    ("24 jam", day_kwh, day_kwh * tariff),
                    ("30 hari", day_kwh * 30, day_kwh * 30 * tariff),
                ],
                tariff,
            ),
            unsafe_allow_html=True,
        )

    a, b = st.columns(2, gap="large")
    with a:
        chart_head(
            "Apa yang mendorong angka ini",
            f"Dibanding kondisi rata-rata data latih ({ec.num_id(model.baseline, 2)} kWh), tiap faktor menambah (oranye) "
            "atau mengurangi (biru) prediksi.",
        )
        show(contribution_chart(model, row), "chart_contrib")
    with b:
        chart_head("Jika satu faktor berubah", "Faktor lain tetap seperti isian di samping. Titik kuning adalah kondisi saat ini.")
        choice = st.selectbox("Faktor yang divariasikan", list(SWEEPS), key="sweep_choice")
        feature, title, unit = SWEEPS[choice]
        show(sweep_chart(model, row, pred, feature, title, unit), "chart_sweep")


# --------------------------------------------------------------------------- #
# Tab 2: prediksi dari file
# --------------------------------------------------------------------------- #
COLUMN_SPEC = pd.DataFrame(
    [
        ("Timestamp", "Waktu pengamatan. Jika tidak ada, isi Hour dan IsWeekend.", "2026-11-16 08:00:00"),
        ("Temperature", "Suhu (°C)", "25.4"),
        ("Humidity", "Kelembapan (%)", "45.0"),
        ("SquareFootage", "Luas bangunan (kaki²)", "1500"),
        ("Occupancy", "Jumlah penghuni (0–9)", "5"),
        ("HVACUsage", "Status HVAC: On atau Off", "On"),
        ("LightingUsage", "Status lampu: On atau Off", "Off"),
        ("RenewableEnergy", "Kontribusi energi terbarukan (%)", "15.0"),
        ("Holiday", "Hari libur: Yes atau No", "No"),
        ("EnergyConsumption", "Opsional. Konsumsi aktual, dipakai untuk mengukur akurasi.", "75.2"),
    ],
    columns=["Kolom", "Isi", "Contoh"],
)


def render_file(model: ec.EnergyModel, tariff: float, raw: pd.DataFrame) -> None:
    st.subheader("Prediksi dari file")
    st.write("Unggah tabel berisi banyak kondisi bangunan. Setiap baris diprediksi konsumsi kWh per jam-nya, lalu biayanya dihitung.")

    c_up, c_act = st.columns([3, 2], gap="large")
    with c_up:
        uploaded = st.file_uploader("Unggah file CSV atau Excel (.xlsx)", type=["csv", "xlsx"], key="uploader")
    with c_act:
        st.download_button(
            "Unduh template CSV",
            data=ec.template_frame().to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_energi.csv",
            mime="text/csv",
            **STRETCH_BTN,
        )
        if st.session_state.get("use_sample"):
            if st.button("Hapus data contoh", **STRETCH_BTN):
                st.session_state["use_sample"] = False
                st.rerun()
        elif st.button("Coba dengan data contoh", **STRETCH_BTN):
            st.session_state["use_sample"] = True
            st.rerun()
        st.caption("Data contoh adalah 200 jam terakhir dari dataset beserta konsumsi aktualnya, jadi akurasi model langsung terlihat.")

    with st.expander("Kolom yang dibutuhkan"):
        st.dataframe(COLUMN_SPEC, hide_index=True, **STRETCH_DF)
        st.caption("Nama kolom tidak peka huruf besar-kecil. Nilai On/Off dan Yes/No juga bisa ditulis Ya/Tidak atau 1/0. "
                   "File CSV dengan pemisah titik koma ikut terbaca.")

    if uploaded is not None:
        st.session_state["use_sample"] = False
        try:
            df_in = ec.read_table(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"File tidak bisa dibaca: {exc}. Pastikan formatnya CSV atau .xlsx yang valid.")
            return
    elif st.session_state.get("use_sample"):
        df_in = ec.sample_upload_frame(raw)
    else:
        st.info("Belum ada data. Unggah file atau tekan Coba dengan data contoh.")
        return

    check = ec.check_upload(df_in, model)
    for message in check.errors:
        st.error(message)
    if not check.ok:
        return
    for message in check.warnings:
        st.warning(message)

    result = ec.predict_upload(check, model)
    result["Biaya_Rp"] = result["Prediksi_kWh"] * tariff
    lead = (["Timestamp"] if check.has_timestamp else []) + ["Prediksi_kWh", "Biaya_Rp"]
    result = result[lead + [c for c in result.columns if c not in lead]]
    ok = result[check.valid].copy()
    if check.has_timestamp:
        ok = ok.sort_values("Timestamp")

    total_kwh = float(ok["Prediksi_kWh"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baris diprediksi", ec.num_id(len(ok), 0))
    k2.metric("Total konsumsi (kWh)", ec.num_id(total_kwh, 1))
    k3.metric("Rata-rata per jam (kWh)", ec.num_id(ok["Prediksi_kWh"].mean(), 2))
    k4.metric("Estimasi biaya (Rp)", ec.rupiah_short(total_kwh * tariff), help=f"Tepatnya {ec.rupiah(total_kwh * tariff)}.")

    x = ok["Timestamp"] if check.has_timestamp else pd.Series(np.arange(1, len(ok) + 1), index=ok.index)
    x_title = "Waktu" if check.has_timestamp else "Urutan baris"

    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        chart_head("Konsumsi per jam", "Garis biru adalah prediksi" + (", garis navy putus-putus adalah konsumsi aktual." if check.has_actual else "."))
        fig = base_fig(340)
        if check.has_actual:
            fig.add_trace(go.Scatter(x=x, y=ok[ec.TARGET], mode="lines", name="Aktual",
                                     line=dict(color=NAVY, width=1.6, dash="dot"),
                                     hovertemplate="Aktual: %{y:.2f} kWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=ok["Prediksi_kWh"], mode="lines", name="Prediksi", line=dict(color=BLUE, width=2.4),
                                 hovertemplate="Prediksi: %{y:.2f} kWh<extra></extra>"))
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="kWh per jam")
        show(fig, "chart_file_line")
    with g2:
        chart_head("Biaya kumulatif", f"Dihitung dengan tarif {ec.rupiah(tariff, 2)} per kWh.")
        cum = (ok["Prediksi_kWh"] * tariff).cumsum()
        fig = base_fig(340)
        fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", line=dict(color=GREEN, width=2.6), fill="tozeroy",
                                 fillcolor="rgba(27,122,87,.14)", hovertemplate="Rp %{y:,.0f}<extra></extra>"))
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="Rupiah", tickformat=",.0f")
        show(fig, "chart_file_cost")

    if check.has_actual:
        ev = ok.dropna(subset=[ec.TARGET])
        if len(ev) >= 2:
            chart_head("Seberapa akurat prediksinya", "Dibandingkan dengan konsumsi aktual pada file yang sama.")
            mae = mean_absolute_error(ev[ec.TARGET], ev["Prediksi_kWh"])
            rmse = float(np.sqrt(mean_squared_error(ev[ec.TARGET], ev["Prediksi_kWh"])))
            r2 = r2_score(ev[ec.TARGET], ev["Prediksi_kWh"])
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{ec.num_id(mae, 4)} kWh", help="Rata-rata selisih mutlak antara prediksi dan aktual.")
            m2.metric("RMSE", f"{ec.num_id(rmse, 4)} kWh", help="Seperti MAE, tetapi selisih besar dihukum lebih berat.")
            m3.metric("R²", ec.num_id(r2, 4), help="Bagian variasi konsumsi yang berhasil dijelaskan model. 1 berarti sempurna.")

            e1, e2 = st.columns(2, gap="large")
            with e1:
                lo, hi = float(min(ev[ec.TARGET].min(), ev["Prediksi_kWh"].min())), float(max(ev[ec.TARGET].max(), ev["Prediksi_kWh"].max()))
                fig = base_fig(340)
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NAVY, width=1.5, dash="dash"),
                                         hoverinfo="skip", name="Prediksi sempurna"))
                fig.add_trace(go.Scatter(x=ev[ec.TARGET], y=ev["Prediksi_kWh"], mode="markers", name="Jam pengamatan",
                                         marker=dict(color=BLUE, size=7, opacity=.6),
                                         hovertemplate="Aktual %{x:.2f}, prediksi %{y:.2f} kWh<extra></extra>"))
                fig.update_xaxes(title="Aktual (kWh)")
                fig.update_yaxes(title="Prediksi (kWh)")
                show(fig, "chart_file_avp")
            with e2:
                resid = ev[ec.TARGET] - ev["Prediksi_kWh"]
                fig = base_fig(340)
                fig.add_trace(go.Histogram(x=resid, nbinsx=24, marker_color=AMBER, opacity=.9,
                                           hovertemplate="Selisih %{x:.1f} kWh: %{y} jam<extra></extra>"))
                fig.add_vline(x=0, line=dict(color=NAVY, width=1.5, dash="dash"))
                fig.update_xaxes(title="Selisih aktual dikurangi prediksi (kWh)")
                fig.update_yaxes(title="Jumlah jam")
                show(fig, "chart_file_resid")

    chart_head("Tabel hasil", "Kolom Prediksi_kWh dan Biaya_Rp ditempatkan di depan, diikuti data asli Anda.")
    config = {
        "Prediksi_kWh": st.column_config.NumberColumn("Prediksi (kWh)", format="%.2f"),
        "Biaya_Rp": st.column_config.NumberColumn("Biaya (Rp)", format="%.0f"),
    }
    if check.has_timestamp:
        config["Timestamp"] = st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm")
    st.dataframe(result, hide_index=True, column_config=config, height=320, **STRETCH_DF)
    st.download_button(
        "Unduh hasil prediksi (CSV)",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_prediksi_energi.csv",
        mime="text/csv",
        type="primary",
    )


# --------------------------------------------------------------------------- #
# Tab 3: grafik pemakaian historis
# --------------------------------------------------------------------------- #
def render_usage(raw: pd.DataFrame, tariff: float) -> None:
    st.subheader("Grafik pemakaian listrik")
    start, end = raw["Timestamp"].min(), raw["Timestamp"].max()
    st.write(
        f"Data historis yang dipakai melatih model: {ec.num_id(len(raw), 0)} jam, "
        f"{start:%d %b %Y} sampai {end:%d %b %Y}."
    )

    target = raw[ec.TARGET]
    total = float(target.sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rata-rata per jam (kWh)", ec.num_id(target.mean(), 2))
    k2.metric("Tertinggi (kWh)", ec.num_id(target.max(), 2))
    k3.metric("Terendah (kWh)", ec.num_id(target.min(), 2))
    k4.metric("Biaya seluruh periode (Rp)", ec.rupiah_short(total * tariff),
              help=f"Tepatnya {ec.rupiah(total * tariff)} untuk {ec.num_id(total, 0)} kWh, dengan tarif di sidebar.")

    chart_head("Konsumsi sepanjang periode")
    grain = st.radio("Tampilkan", ["Per jam", "Rata-rata harian", "Rata-rata mingguan"], horizontal=True, key="grain")
    series = raw.set_index("Timestamp")[ec.TARGET]
    if grain == "Rata-rata harian":
        series = series.resample("D").mean()
    elif grain == "Rata-rata mingguan":
        series = series.resample("W").mean()
    fig = base_fig(350)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers" if grain != "Per jam" else "lines",
                             line=dict(color=BLUE, width=2), marker=dict(size=6), name=grain,
                             hovertemplate="%{x|%d %b %Y %H:%M}: %{y:.2f} kWh<extra></extra>"))
    fig.add_hline(y=float(target.mean()), line=dict(color=AMBER, width=1.5, dash="dash"),
                  annotation_text=f"Rata-rata {ec.num_id(target.mean(), 1)} kWh", annotation_position="top left",
                  annotation_font=dict(color=AMBER, size=12))
    fig.update_yaxes(title="kWh per jam")
    show(fig, "chart_usage_series")

    a, b = st.columns(2, gap="large")
    with a:
        by_hour = raw.groupby(raw["Timestamp"].dt.hour)[ec.TARGET].mean()
        chart_head(
            "Rata-rata menurut jam",
            f"Selisih antarjam hanya sekitar {ec.num_id(by_hour.max() - by_hour.min(), 1)} kWh, jadi jam bukan faktor utama.",
        )
        fig = base_fig(330)
        fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour.values, mode="lines+markers", line=dict(color=NAVY, width=2.4),
                                 marker=dict(size=7, color=VOLT, line=dict(color=NAVY, width=1.5)),
                                 hovertemplate="Pukul %{x}:00 rata-rata %{y:.2f} kWh<extra></extra>"))
        fig.update_xaxes(title="Jam", dtick=3)
        fig.update_yaxes(title="kWh per jam", range=[by_hour.min() - 4, by_hour.max() + 4])
        show(fig, "chart_usage_hour")
    with b:
        chart_head("Suhu dan konsumsi", "Suhu adalah faktor numerik terkuat. HVAC menyala menggeser titik ke atas.")
        fig = base_fig(330)
        for status, color in (("Off", BLUE), ("On", AMBER)):
            sub = raw[raw["HVACUsage"] == status]
            fig.add_trace(go.Scatter(x=sub["Temperature"], y=sub[ec.TARGET], mode="markers",
                                     name=f"HVAC {'menyala' if status == 'On' else 'mati'}",
                                     marker=dict(size=6, color=color, opacity=.55),
                                     hovertemplate="%{x:.1f} °C: %{y:.2f} kWh<extra></extra>"))
        slope, intercept = np.polyfit(raw["Temperature"], target, 1)
        xs = np.array([raw["Temperature"].min(), raw["Temperature"].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Garis tren",
                                 line=dict(color=NAVY, width=2.4), hoverinfo="skip"))
        fig.update_xaxes(title="Suhu (°C)")
        fig.update_yaxes(title="kWh per jam")
        show(fig, "chart_usage_temp")

    chart_head("Rata-rata konsumsi menurut kondisi", "Batang dimulai dari nol agar selisihnya tidak terlihat lebih besar dari aslinya.")
    weekend = raw["Timestamp"].dt.dayofweek >= 5
    groups = [
        ("HVAC", raw.groupby("HVACUsage")[ec.TARGET].mean().reindex(["Off", "On"]), ["Mati", "Menyala"]),
        ("Lampu", raw.groupby("LightingUsage")[ec.TARGET].mean().reindex(["Off", "On"]), ["Mati", "Menyala"]),
        ("Hari libur", raw.groupby("Holiday")[ec.TARGET].mean().reindex(["No", "Yes"]), ["Bukan", "Libur"]),
        ("Tipe hari", target.groupby(weekend).mean().reindex([False, True]), ["Hari kerja", "Akhir pekan"]),
    ]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, subplot_titles=[g[0] for g in groups], horizontal_spacing=0.03)
    for i, (_, values, names) in enumerate(groups, start=1):
        fig.add_trace(
            go.Bar(x=names, y=values.values, marker_color=[MUTED_BAR, AMBER], showlegend=False,
                   text=[ec.num_id(v, 1) for v in values.values], textposition="outside", cliponaxis=False,
                   hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>"),
            row=1, col=i,
        )
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(family=FONT, size=13, color=NAVY), separators=",.",
                      hoverlabel=dict(bgcolor=NAVY, font=dict(color="#FFFFFF", family=FONT)))
    fig.update_yaxes(range=[0, 100], gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_xaxes(linecolor=LINE, automargin=True)
    fig.update_yaxes(title="kWh per jam", col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(family="Barlow Condensed, Arial Narrow, sans-serif", size=18, color=NAVY)
    show(fig, "chart_usage_groups")


# --------------------------------------------------------------------------- #
# Tab 4: tentang model
# --------------------------------------------------------------------------- #
def render_about(model: ec.EnergyModel) -> None:
    st.subheader("Tentang model")
    m = model.metrics
    st.write(
        "Model akhir adalah Linear Regression. Dari empat algoritma yang dibandingkan pada notebook, "
        "model ini punya kesalahan terendah dan R² tertinggi pada data uji."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{ec.num_id(m['MAE'], 4)} kWh", help="Rata-rata selisih mutlak prediksi dan aktual pada data uji.")
    k2.metric("RMSE", f"{ec.num_id(m['RMSE'], 4)} kWh", help="Seperti MAE, tetapi selisih besar dihukum lebih berat.")
    k3.metric("R²", ec.num_id(m["R2"], 4), help="Model menjelaskan sekitar 59% variasi konsumsi pada data uji.")

    chart_head("Perbandingan empat model", "Hasil pada 200 jam data uji. Linear Regression ditandai biru.")
    comp = ec.REPORTED_COMPARISON
    colors = [BLUE if name == "Linear Regression" else MUTED_BAR for name in comp["Model"]]
    titles = ("MAE, makin kecil makin baik", "RMSE, makin kecil makin baik", "R², makin besar makin baik")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.05)
    for i, col in enumerate(["MAE", "RMSE", "R2"], start=1):
        fig.add_trace(
            go.Bar(x=comp[col], y=comp["Model"], orientation="h", marker_color=colors, showlegend=False,
                   text=[ec.num_id(v, 4) for v in comp[col]], textposition="outside", cliponaxis=False,
                   hovertemplate="%{y}: %{x:.4f}<extra></extra>"),
            row=1, col=i,
        )
        fig.update_xaxes(range=[0, comp[col].max() * 1.35], gridcolor=GRID, row=1, col=i)
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_xaxes(automargin=True)
    fig.update_layout(height=290, margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(family=FONT, size=13, color=NAVY), separators=",.",
                      hoverlabel=dict(bgcolor=NAVY, font=dict(color="#FFFFFF", family=FONT)))
    for ann in fig.layout.annotations:
        ann.font = dict(family="Barlow Condensed, Arial Narrow, sans-serif", size=17, color=NAVY)
    show(fig, "chart_about_compare")

    effects = model.effects_table()
    temp_effect = float(effects.loc[effects["Feature"] == "Temperature", "Efek"].iloc[0])
    hvac_effect = float(effects.loc[effects["Feature"] == "HVACUsage", "Efek"].iloc[0])
    chart_head(
        "Faktor mana yang paling berpengaruh",
        f"Setiap kenaikan suhu 1 °C menambah prediksi sekitar {ec.num_id(temp_effect, 2)} kWh per jam, "
        f"dan HVAC menyala menambah sekitar {ec.num_id(hvac_effect, 2)} kWh.",
    )
    a, b = st.columns([3, 2], gap="large")
    with a:
        table = pd.DataFrame(
            {
                "Faktor": effects["Faktor"],
                "Perubahan": effects["Perubahan"],
                "Efek (kWh per jam)": [ec.num_id(v, 3, signed=True) for v in effects["Efek"]],
            }
        )
        st.dataframe(table, hide_index=True, **STRETCH_DF)
    with b:
        order = effects.reindex(effects["Efek1SD"].abs().sort_values().index)
        fig = base_fig(370)
        fig.add_trace(go.Bar(x=order["Efek1SD"], y=order["Faktor"], orientation="h",
                             marker_color=[AMBER if v > 0 else BLUE for v in order["Efek1SD"]],
                             hovertemplate="%{y}: %{x:+.2f} kWh<extra></extra>"))
        fig.update_xaxes(title="kWh per 1 simpangan baku", zeroline=True, zerolinecolor=NAVY)
        fig.update_yaxes(showgrid=False)
        show(fig, "chart_about_effects")
        st.caption("Perubahan satu simpangan baku membuat faktor yang skalanya berbeda bisa dibandingkan langsung.")

    with st.expander("Cara model dibuat"):
        tr, te = model.periods["train"], model.periods["test"]
        st.markdown(
            f"""
1. **Data.** {ec.num_id(m['n_train'] + m['n_test'], 0)} observasi per jam, kolom DayOfWeek tidak dipakai.
2. **Fitur waktu.** Hour dan IsWeekend diturunkan dari Timestamp.
3. **Encoding.** HVACUsage, LightingUsage, dan Holiday diubah menjadi 0 (Off/No) dan 1 (On/Yes).
4. **Pembagian data kronologis.** Latih {m['n_train']} jam ({tr[0]:%d %b} sampai {tr[1]:%d %b %Y}),
   uji {m['n_test']} jam ({te[0]:%d %b} sampai {te[1]:%d %b %Y}).
5. **Standardisasi.** Lima fitur numerik distandardisasi dengan parameter dari data latih saja, supaya data uji tidak bocor ke pelatihan.
6. **Model.** Linear Regression dilatih pada data latih lalu dievaluasi pada data uji.

Saat Anda memasukkan data, aplikasi memakai langkah yang sama persis, termasuk standardisasi, sebelum memanggil model.
"""
        )

    with st.expander("Batasan yang perlu diingat"):
        st.markdown(
            f"""
- **Data sintetis.** Dataset meniru skenario nyata, jadi polanya belum tentu sama dengan bangunan sebenarnya.
- **Akurasi sedang.** R² {ec.num_id(m['R2'], 3)} berarti sekitar {ec.num_id((1 - m['R2']) * 100, 0)}% variasi konsumsi belum dijelaskan oleh fitur yang ada.
  Pakai hasilnya sebagai perkiraan, bukan angka pasti.
- **Rentang terbatas.** Model hanya melihat suhu 20–30 °C, kelembapan 30–60 %, luas 1.000–2.000 kaki², dan penghuni 0–9.
  Nilai di luar rentang itu diberi peringatan saat unggah file karena model linear akan meneruskan garisnya begitu saja.
- **Satuan.** Aplikasi menampilkan hasil dalam kWh per jam, mengikuti antarmuka prediksi pada laporan.
"""
        )


# --------------------------------------------------------------------------- #
# Susunan halaman
# --------------------------------------------------------------------------- #
raw_data = get_raw()
energy_model = get_model()
tariff_value = render_sidebar(energy_model)

st.markdown(
    html(
        '<div class="hero"><div class="hero-mark"><div class="hero-bolt"></div></div><div>'
        "<h1>Meteran prediksi energi</h1>"
        "<p>Isi kondisi bangunan, lihat perkiraan kWh per jam dan biayanya. Bisa satu per satu, bisa sekaligus dari file.</p>"
        "</div></div>"
    ),
    unsafe_allow_html=True,
)

tab_manual, tab_file, tab_usage, tab_about = st.tabs(
    ["Prediksi manual", "Prediksi dari file", "Grafik pemakaian", "Tentang model"]
)
with tab_manual:
    render_manual(energy_model, tariff_value)
with tab_file:
    render_file(energy_model, tariff_value, raw_data)
with tab_usage:
    render_usage(raw_data, tariff_value)
with tab_about:
    render_about(energy_model)
