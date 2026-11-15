"""Halaman eksplorasi data: Dashboard, Analisis, Data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import energy_core as ec
import ui
from ui import AMBER, BLUE, GREEN, MUTED, NAVY, TEAL, VOLT

T = ec.TARGET


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def dashboard(ctx: ui.AppContext) -> None:
    df, tariff = ctx.df, ctx.tariff
    a, b = df["Timestamp"].min(), df["Timestamp"].max()
    ui.hero(
        "Dashboard konsumsi energi",
        f"Periode {a:%d %b %Y} sampai {b:%d %b %Y}, {ec.num_id(len(df), 0)} catatan per jam. Sumber data: {ctx.source}.",
    )

    total = float(df[T].sum())
    peak = df.loc[df[T].idxmax()]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total (kWh)", ec.num_id(total, 0))
    k2.metric("Rata-rata/jam (kWh)", ec.num_id(df[T].mean(), 2))
    k3.metric("Puncak (kWh)", ec.num_id(peak[T], 2), help=f"Terjadi pada {peak['Timestamp']:%d %b %Y pukul %H:%M}.")
    k4.metric("Biaya (Rp)", ec.rupiah_short(total * tariff),
              help=f"Tepatnya {ec.rupiah(total * tariff)} dengan tarif {ec.rupiah(tariff, 2)} per kWh.")
    k5.metric("Terbarukan (%)", ec.num_id(df["RenewableEnergy"].mean(), 1),
              help="Rata-rata kolom RenewableEnergy, dibaca sebagai persen dari konsumsi (sesuai legenda notebook).")

    by_hour = df.groupby("Hour")[T].mean()
    by_day = df.groupby("Hari")[T].mean().reindex(ec.DAY_NAMES_ID).dropna()
    st.caption(
        f"Jam dengan rata-rata tertinggi: pukul {int(by_hour.idxmax()):02d}:00 ({ec.num_id(by_hour.max(), 1)} kWh). "
        f"Hari tertinggi: {by_day.idxmax()} ({ec.num_id(by_day.max(), 1)} kWh)."
    )

    # -- tren ---------------------------------------------------------------- #
    ui.chart_head("Tren konsumsi", "Pilih rincian waktu dan satuan. Hari pertama dan terakhir bisa belum penuh 24 jam.")
    c1, c2 = st.columns(2)
    grain = c1.radio("Rincian", ["Per jam", "Total per hari", "Total per minggu"], horizontal=True, key="dash_grain")
    unit = c2.radio("Satuan", ["kWh", "Rupiah"], horizontal=True, key="dash_unit")
    series = df.set_index("Timestamp")[T]
    if grain == "Total per hari":
        series = series.resample("D").sum()
    elif grain == "Total per minggu":
        series = series.resample("W").sum()
    money = unit == "Rupiah"
    if money:
        series = series * tariff
    when = "%{x|%d %b %Y %H:%M}" if grain == "Per jam" else "%{x|%d %b %Y}"
    value = "Rp %{y:,.0f}" if money else "%{y:.1f} kWh"
    fig = ui.base_fig(350)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines" if grain == "Per jam" else "lines+markers",
                             line=dict(color=GREEN if money else BLUE, width=2), marker=dict(size=6),
                             hovertemplate=f"{when}: {value}<extra></extra>", name=grain))
    if grain == "Per jam" and not money:
        fig.add_hline(y=float(df[T].mean()), line=dict(color=AMBER, width=1.5, dash="dash"),
                      annotation_text=f"Rata-rata {ec.num_id(df[T].mean(), 1)} kWh", annotation_position="top left",
                      annotation_font=dict(color=AMBER, size=12))
    fig.update_yaxes(title="Rupiah" if money else "kWh", tickformat=",.0f" if money else None)
    ui.show(fig, "dash_trend")

    ui.chart_head("Sumber energi per hari",
                  "Porsi terbarukan dihitung dari kolom RenewableEnergy (persen) dikali konsumsi. Sisanya dari jaringan.")
    daily = df.groupby("Tanggal")[["Jaringan_kWh", "Terbarukan_kWh"]].sum().reset_index()
    fig = ui.base_fig(320)
    fig.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Jaringan_kWh"], name="Dari jaringan", marker_color=BLUE,
                         hovertemplate="%{x|%d %b}: %{y:.0f} kWh<extra></extra>"))
    fig.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Terbarukan_kWh"], name="Dari energi terbarukan", marker_color=TEAL,
                         hovertemplate="%{x|%d %b}: %{y:.0f} kWh<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="kWh per hari")
    ui.show(fig, "dash_sources")

    # -- pola jam dan hari ---------------------------------------------------- #
    a1, a2 = st.columns(2, gap="large")
    with a1:
        ui.chart_head("Rata-rata menurut jam", f"Selisih antarjam sekitar {ec.num_id(by_hour.max() - by_hour.min(), 1)} kWh.")
        fig = ui.base_fig(320)
        fig.add_trace(go.Bar(x=by_hour.index, y=by_hour.values,
                             marker_color=[AMBER if h == by_hour.idxmax() else BLUE for h in by_hour.index],
                             hovertemplate="Pukul %{x}:00: %{y:.2f} kWh<extra></extra>"))
        fig.update_xaxes(title="Jam", dtick=3)
        fig.update_yaxes(title="kWh per jam")
        ui.show(fig, "dash_hour")
    with a2:
        ui.chart_head("Rata-rata menurut hari", "Hari dihitung dari Timestamp.")
        fig = ui.base_fig(320)
        fig.add_trace(go.Bar(x=by_day.index, y=by_day.values,
                             marker_color=[AMBER if d == by_day.idxmax() else NAVY for d in by_day.index],
                             text=[ec.num_id(v, 1) for v in by_day.values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>"))
        fig.update_yaxes(title="kWh per jam", range=[0, float(by_day.max()) * 1.15])
        ui.show(fig, "dash_day")

    # -- peralatan dan libur --------------------------------------------------- #
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.chart_head("Pengaruh HVAC dan lampu", "Rata-rata konsumsi untuk tiap kombinasi.")
        combo = df.groupby(["HVACUsage", "LightingUsage"])[T].mean()
        fig = ui.base_fig(330)
        top = 1.0
        for light, color, name in (("Off", MUTED, "Lampu mati"), ("On", AMBER, "Lampu menyala")):
            ys = [combo.get((h, light), np.nan) for h in ("Off", "On")]
            top = max([top] + [v for v in ys if pd.notna(v)])
            fig.add_trace(go.Bar(x=["HVAC mati", "HVAC menyala"], y=ys, name=name, marker_color=color,
                                 text=[ec.num_id(v, 1) if pd.notna(v) else "" for v in ys], textposition="outside",
                                 cliponaxis=False, hovertemplate="%{x}, " + name + ": %{y:.2f} kWh<extra></extra>"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="kWh per jam", range=[0, top * 1.15])
        ui.show(fig, "dash_combo")
    with c2:
        ui.chart_head("Hari libur dan hari biasa", "Sebaran konsumsi (kotak menunjukkan 50% data di tengah).")
        fig = ui.base_fig(330)
        for flag, name, color in (("No", "Bukan libur", BLUE), ("Yes", "Libur", AMBER)):
            sub = df.loc[df["Holiday"] == flag, T]
            if len(sub):
                fig.add_trace(go.Box(y=sub, name=name, marker_color=color, boxmean=True))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="kWh per jam")
        ui.show(fig, "dash_holiday")

    ui.chart_head("Peta panas jam dan hari", "Rata-rata konsumsi tiap jam pada tiap hari. Tiap kotak hanya merangkum beberapa pengamatan.")
    pivot = df.pivot_table(index="Hari", columns="Hour", values=T, aggfunc="mean").reindex(ec.DAY_NAMES_ID).dropna(how="all")
    fig = ui.base_fig(330)
    fig.add_trace(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=[[0, "#EEF3FA"], [0.5, "#FFD27A"], [1, AMBER]],
                             colorbar=dict(title="kWh"), hovertemplate="%{y}, pukul %{x}:00: %{z:.1f} kWh<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Jam", dtick=2)
    ui.show(fig, "dash_heat")


# --------------------------------------------------------------------------- #
# Analisis
# --------------------------------------------------------------------------- #
SCATTER_X = {
    "Suhu (°C)": "Temperature", "Kelembapan (%)": "Humidity", "Jumlah penghuni": "Occupancy",
    "Luas bangunan (kaki²)": "SquareFootage", "Energi terbarukan (%)": "RenewableEnergy", "Jam": "Hour",
}
SCATTER_COLOR = {
    "Tanpa warna": None,
    "Status HVAC": ("HVACUsage", {"Off": "HVAC mati", "On": "HVAC menyala"}),
    "Status lampu": ("LightingUsage", {"Off": "Lampu mati", "On": "Lampu menyala"}),
    "Hari libur": ("Holiday", {"No": "Bukan libur", "Yes": "Libur"}),
}


def analysis(ctx: ui.AppContext) -> None:
    df = ctx.df
    ui.hero("Analisis faktor konsumsi", "Faktor mana yang paling berkaitan dengan konsumsi energi? Semua grafik mengikuti filter di sidebar.")

    # -- korelasi ------------------------------------------------------------- #
    ui.chart_head("Korelasi antarvariabel", "Biru berarti berlawanan arah, oranye searah. Semakin pekat, semakin kuat.")
    cols = list(ec.CORR_COLUMNS)
    corr = df[cols].corr()
    labels = [ec.CORR_COLUMNS[c] for c in cols]
    fig = ui.base_fig(520)
    fig.add_trace(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
                             colorscale=[[0, BLUE], [0.5, "#FFFFFF"], [1, AMBER]], texttemplate="%{z:.2f}",
                             colorbar=dict(title="r"), hovertemplate="%{y} dan %{x}: r = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None, side="bottom")
    ui.show(fig, "ana_corr")

    target_corr = corr[T].drop(T).dropna()
    if len(target_corr):
        strongest = target_corr.abs().idxmax()
        st.info(f"Faktor dengan korelasi terkuat terhadap konsumsi adalah **{ec.CORR_COLUMNS[strongest]}** "
                f"(r = {ec.num_id(target_corr[strongest], 2)}). Korelasi hanya mengukur hubungan garis lurus, jadi hubungan "
                "yang melengkung atau saling memengaruhi tidak terlihat di sini.")

    # -- penjelajah sebar ------------------------------------------------------- #
    ui.chart_head("Hubungan satu variabel dengan konsumsi")
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        xlabel = st.selectbox("Variabel X", list(SCATTER_X), key="ana_x")
        clabel = st.selectbox("Warna berdasarkan", list(SCATTER_COLOR), key="ana_color")
        trend = st.checkbox("Tampilkan garis tren", value=True, key="ana_trend")
    xcol = SCATTER_X[xlabel]
    with c2:
        fig = ui.base_fig(430)
        spec = SCATTER_COLOR[clabel]
        if spec is None:
            fig.add_trace(go.Scatter(x=df[xcol], y=df[T], mode="markers", name="Pengamatan",
                                     marker=dict(color=BLUE, size=6, opacity=.55)))
        else:
            col, names = spec
            for value, color in (("Off" if col != "Holiday" else "No", BLUE), ("On" if col != "Holiday" else "Yes", AMBER)):
                sub = df[df[col] == value]
                fig.add_trace(go.Scatter(x=sub[xcol], y=sub[T], mode="markers", name=names[value],
                                         marker=dict(color=color, size=6, opacity=.55)))
        note = ""
        if trend and len(df) > 2 and df[xcol].nunique() > 1:
            slope, intercept = np.polyfit(df[xcol], df[T], 1)
            xs = np.array([df[xcol].min(), df[xcol].max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Tren linear",
                                     line=dict(color=NAVY, width=2.4, dash="dash"), hoverinfo="skip"))
            r = float(np.corrcoef(df[xcol], df[T])[0, 1])
            note = f"r = {ec.num_id(r, 2)}. Rata-rata, tiap kenaikan 1 satuan {xlabel.split(' (')[0].lower()} bergeser {ec.num_id(slope, 3, signed=True)} kWh."
        fig.update_xaxes(title=xlabel)
        fig.update_yaxes(title="kWh per jam")
        ui.show(fig, "ana_scatter")
        if note:
            st.caption(note)

    # -- distribusi ------------------------------------------------------------- #
    ui.chart_head("Sebaran konsumsi energi", "Kotak di atas meringkas median dan sebaran, histogram di bawah menunjukkan frekuensinya.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8], vertical_spacing=0.03)
    fig.add_trace(go.Box(x=df[T], marker_color=BLUE, name="", boxmean=True, showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[T], nbinsx=30, marker_color=BLUE, showlegend=False,
                               hovertemplate="%{x:.1f} kWh: %{y} jam<extra></extra>"), row=2, col=1)
    ui.style_fig(fig, 400)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title="kWh per jam", row=2, col=1)
    fig.update_yaxes(title="Jumlah jam", row=2, col=1)
    ui.show(fig, "ana_dist")

    # -- statistik -------------------------------------------------------------- #
    ui.chart_head("Statistik deskriptif")
    stats = df[["Temperature", "Humidity", "SquareFootage", "Occupancy", "RenewableEnergy", T]].describe().T
    stats.columns = ["Jumlah", "Rata-rata", "Simpangan baku", "Min", "Q1", "Median", "Q3", "Maks"]
    st.dataframe(stats.round(2), **ui.STRETCH_DF)

    ui.chart_head("Pencilan (metode IQR)",
                  "Nilai di luar Q1 − 1,5 × IQR atau Q3 + 1,5 × IQR ditandai sebagai kandidat pencilan. Belum tentu salah catat.")
    outliers = ec.iqr_summary(df, ["Temperature", "Humidity", "SquareFootage", "RenewableEnergy", T, "Occupancy"])
    view = outliers.copy()
    for col in ["Q1", "Q3", "Batas bawah", "Batas atas", "Persen"]:
        view[col] = outliers[col].map(lambda v: ec.num_id(v, 2))
    st.dataframe(view, hide_index=True, **ui.STRETCH_DF)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def data_page(ctx: ui.AppContext) -> None:
    df, raw = ctx.df, ctx.raw
    ui.hero("Data konsumsi energi", f"{ec.num_id(len(df), 0)} baris sesuai filter dari total {ec.num_id(len(raw), 0)} baris. Sumber data: {ctx.source}.")

    shown = [c for c in ["Timestamp", *ec.REQUIRED_BASE, T, "DayOfWeek"] if c in df.columns]
    c1, c2 = st.columns([3, 1])
    sort_col = c1.selectbox("Urutkan berdasarkan", shown, key="data_sort")
    desc = c2.toggle("Urutan menurun", value=False, key="data_desc")
    table = df[shown].sort_values(sort_col, ascending=not desc)
    st.dataframe(table, hide_index=True, height=460,
                 column_config={"Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm")},
                 **ui.STRETCH_DF)
    st.download_button("Unduh data terfilter (CSV)", data=table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="energi_terfilter.csv", mime="text/csv", type="primary")

    ui.chart_head("Kualitas data", "Dihitung pada seluruh data, bukan hanya yang lolos filter.")
    q1, q2, q3 = st.columns(3)
    q1.metric("Nilai kosong", ec.num_id(int(raw.isna().sum().sum()), 0))
    q2.metric("Baris duplikat", ec.num_id(int(raw.duplicated().sum()), 0))
    if "DayOfWeek" in raw.columns:
        match = float((raw["DayOfWeek"].astype(str).str.strip().str.lower() == raw["Timestamp"].dt.day_name().str.lower()).mean() * 100)
        q3.metric("DayOfWeek sesuai tanggal (%)", ec.num_id(match, 0))
        if match < 90:
            st.warning("Kolom DayOfWeek sering tidak cocok dengan hari pada Timestamp. Karena itu aplikasi memakai hari yang "
                       "dihitung dari Timestamp (IsWeekend), dan DayOfWeek tidak dipakai oleh model.")
    else:
        q3.metric("Kolom DayOfWeek", "Tidak ada")

    ui.chart_head("Format data")
    st.write("Untuk mengganti data latih, unggah file lewat bagian Data di sidebar. Kolom wajib: Timestamp, Temperature, Humidity, "
             "SquareFootage, Occupancy, HVACUsage, LightingUsage, RenewableEnergy, Holiday, dan EnergyConsumption (minimal 100 baris). "
             "Kolom DayOfWeek boleh ada tetapi tidak dipakai.")
    st.download_button("Unduh data bawaan (CSV)", data=ctx.raw.to_csv(index=False).encode("utf-8-sig"),
                       file_name="energy_consumption.csv", mime="text/csv")
