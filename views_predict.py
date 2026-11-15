"""Halaman prediksi: Prediksi (manual) dan Prediksi massal (file)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import energy_core as ec
import ui
from ui import AMBER, BLUE, GREEN, MUTED, NAVY, VOLT

T = ec.TARGET
HOURS_30_DAYS = 24 * 30


def _signed_rp(value: float) -> str:
    if abs(value) < 0.5:
        return ec.rupiah(0)
    return ("+" if value > 0 else "-") + ec.rupiah(abs(value))


def _slider(container, label: str, feature: str, step: float, ctx: ui.AppContext, extend: bool, integer: bool = False):
    b = ctx.bundle
    lo, hi = ec.slider_bounds(b.train_range, feature, step, extend)
    default = ec.slider_default(float(b.train_X[feature].mean()), step, lo, hi)
    if integer:
        return container.slider(label, int(lo), int(hi), int(round(default)), int(step))
    return container.slider(label, float(lo), float(hi), float(default), float(step))


# --------------------------------------------------------------------------- #
# Grafik untuk halaman Prediksi
# --------------------------------------------------------------------------- #
def _contribution_chart(contrib: pd.Series) -> go.Figure:
    contrib = contrib.reindex(contrib.abs().sort_values().index)  # terbesar di atas
    values = contrib.to_numpy()
    span = max(float(np.abs(values).max()), 1.0)
    fig = ui.base_fig(370)
    fig.add_trace(go.Bar(
        x=values, y=[ec.LABELS[f] for f in contrib.index], orientation="h",
        marker_color=[AMBER if v > 0 else BLUE for v in values],
        text=[ec.num_id(v, 2, signed=True) for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} kWh<extra></extra>",
    ))
    fig.update_xaxes(range=[-span * 1.55, span * 1.4], zeroline=True, zerolinecolor=NAVY, zerolinewidth=1.5,
                     title="kWh dibanding rata-rata")
    fig.update_yaxes(showgrid=False)
    return fig


SWEEPS = {
    "Suhu": ("Temperature", "Suhu", "°C"),
    "Jumlah penghuni": ("Occupancy", "Jumlah penghuni", "orang"),
    "Kelembapan": ("Humidity", "Kelembapan", "%"),
    "Energi terbarukan": ("RenewableEnergy", "Energi terbarukan", "%"),
    "Luas bangunan": ("SquareFootage", "Luas bangunan", "kaki²"),
    "Jam": ("Hour", "Jam", "pukul"),
}


def _sweep_chart(ctx: ui.AppContext, row: pd.DataFrame, pred: float, feature: str, title: str, unit: str) -> go.Figure:
    b = ctx.bundle
    if feature == "Hour":
        xs = np.arange(24)
    elif feature == "Occupancy":
        lo, hi = ec.slider_bounds(b.train_range, feature, 1)
        xs = np.arange(int(lo), int(hi) + 1)
    else:
        xs = np.linspace(b.train_range.loc[feature, "min"], b.train_range.loc[feature, "max"], 41)
    frame = pd.concat([row] * len(xs), ignore_index=True)
    frame[feature] = xs
    ys = b.predict(ctx.model_name, frame)
    mid, half = (ys.max() + ys.min()) / 2, max((ys.max() - ys.min()) / 2 * 1.3, 3.0)
    fig = ui.base_fig(370)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=3, shape="linear"), name="Prediksi",
                             hovertemplate=f"{title} %{{x:.1f}} {unit}: %{{y:.2f}} kWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=[float(row[feature].iloc[0])], y=[pred], mode="markers", name="Kondisi saat ini",
                             marker=dict(size=15, color=VOLT, line=dict(color=NAVY, width=2.5)),
                             hovertemplate="Saat ini: %{y:.2f} kWh<extra></extra>"))
    fig.update_xaxes(title=f"{title} ({unit})")
    fig.update_yaxes(title="kWh per jam", range=[mid - half, mid + half])
    fig.update_layout(showlegend=False)
    return fig


def _save_scenario(record: dict) -> None:
    scenarios = st.session_state.setdefault("scenarios", [])
    name = (st.session_state.get("scen_name") or "").strip() or f"Skenario {len(scenarios) + 1}"
    scenarios.append({"Nama": name, **record})
    st.session_state["scen_name"] = f"Skenario {len(scenarios) + 1}"


def _clear_scenarios() -> None:
    st.session_state["scenarios"] = []
    st.session_state["scen_name"] = "Skenario 1"


# --------------------------------------------------------------------------- #
# Prediksi manual
# --------------------------------------------------------------------------- #
def predict_page(ctx: ui.AppContext) -> None:
    b, name, tariff = ctx.bundle, ctx.model_name, ctx.tariff
    ui.hero("Prediksi konsumsi energi", f"Isi kondisi bangunan, lihat perkiraan kWh per jam dan biayanya. Model aktif: {name}.")

    extend = st.toggle(
        "Izinkan nilai di luar rentang data latih", value=False, key="extend_ranges",
        help="Melebarkan slider 50% ke tiap sisi. Prediksi di luar rentang yang pernah dilihat model kurang bisa diandalkan.",
    )

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Kondisi bangunan")
        with st.form("form_manual"):
            c1, c2 = st.columns(2)
            temperature = _slider(c1, "Suhu (°C)", "Temperature", 0.1, ctx, extend)
            humidity = _slider(c2, "Kelembapan (%)", "Humidity", 0.1, ctx, extend)
            square = _slider(c1, "Luas bangunan (kaki²)", "SquareFootage", 10, ctx, extend, integer=True)
            occupancy = _slider(c2, "Jumlah penghuni", "Occupancy", 1, ctx, extend, integer=True)
            renewable = _slider(c1, "Energi terbarukan (%)", "RenewableEnergy", 0.5, ctx, extend)
            hour = c2.slider("Jam", 0, 23, 12)
            t1, t2 = st.columns(2)
            hvac = t1.toggle("HVAC menyala", value=False)
            lighting = t2.toggle("Lampu menyala", value=False)
            holiday = t1.toggle("Hari libur", value=False)
            weekend = t2.toggle("Akhir pekan", value=False)
            submitted = st.form_submit_button("Prediksi konsumsi energi", type="primary", **ui.STRETCH_BTN)
        st.caption("Ubah nilai, lalu tekan tombol prediksi. Rentang slider mengikuti rentang data latih.")
        if submitted:
            st.toast("Prediksi diperbarui")

    values = {
        "Temperature": temperature, "Humidity": humidity, "SquareFootage": square, "Occupancy": occupancy,
        "HVACUsage": int(hvac), "LightingUsage": int(lighting), "RenewableEnergy": renewable,
        "Holiday": int(holiday), "Hour": hour, "IsWeekend": int(weekend),
    }
    row = pd.DataFrame([values])[ec.FEATURES]
    pred = float(b.predict(name, row)[0])

    day_frame = pd.concat([row] * 24, ignore_index=True)
    day_frame["Hour"] = np.arange(24)
    day_kwh = float(b.predict(name, day_frame).sum())

    diff = pred - b.baseline
    direction = "lebih tinggi" if diff >= 0 else "lebih rendah"
    foot = (f"Rata-rata data latih {ec.num_id(b.baseline, 2)} kWh per jam. "
            f"Kondisi ini <b>{ec.num_id(abs(diff), 2)} kWh {direction}</b>.")
    kategori, kelas = b.category(pred)
    rmse = b.metrics_for(name)["RMSE"]
    grid_kwh = pred * (1 - renewable / 100)

    with right:
        st.markdown(ui.meter_html(pred, foot), unsafe_allow_html=True)
        st.markdown(
            ui.stat_cards_html([
                ("Kategori", f'<span class="dot {kelas}"></span>{kategori}',
                 f"dibanding kuartil data ({ec.num_id(b.stats['q1'], 1)} dan {ec.num_id(b.stats['q3'], 1)} kWh)"),
                ("Dari jaringan", f"{ec.num_id(grid_kwh, 1)} kWh", f"{ec.rupiah(grid_kwh * tariff)} per jam"),
                ("Rentang perkiraan", f"{ec.num_id(pred - rmse, 1)}–{ec.num_id(pred + rmse, 1)}",
                 f"± RMSE model ({ec.num_id(rmse, 2)} kWh)"),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.receipt_html([("1 jam", pred, pred * tariff), ("24 jam", day_kwh, day_kwh * tariff),
                             ("30 hari", day_kwh * 30, day_kwh * 30 * tariff)], tariff),
            unsafe_allow_html=True,
        )

    # -- penjelasan ----------------------------------------------------------- #
    contrib, base, exact = b.explain(name, row)
    a, c = st.columns(2, gap="large")
    with a:
        if exact:
            sub = (f"Dibanding kondisi rata-rata data latih ({ec.num_id(base, 2)} kWh), tiap faktor menambah (oranye) "
                   "atau mengurangi (biru) prediksi.")
        else:
            sub = ("Perkiraan: selisih prediksi bila faktor ini diganti nilai-nilai dari data latih. Jumlahnya tidak selalu "
                   "persis sama dengan prediksi, karena model pohon tidak bersifat aditif.")
        ui.chart_head("Apa yang mendorong angka ini", sub)
        ui.show(_contribution_chart(contrib), "pred_contrib")
    with c:
        ui.chart_head("Jika satu faktor berubah", "Faktor lain tetap seperti isian. Titik kuning adalah kondisi saat ini.")
        choice = st.selectbox("Faktor yang divariasikan", list(SWEEPS), key="sweep_choice")
        feature, title, unit = SWEEPS[choice]
        ui.show(_sweep_chart(ctx, row, pred, feature, title, unit), "pred_sweep")

    # -- peluang hemat dan rekomendasi ------------------------------------------ #
    d1, d2 = st.columns(2, gap="large")
    with d1:
        ui.chart_head("Jika peralatan diubah",
                      f"Selisih terhadap prediksi saat ini. Biaya 30 hari mengasumsikan perubahan berlaku selama {HOURS_30_DAYS} jam.")
        levers = b.levers(name, row)
        st.dataframe(
            pd.DataFrame({
                "Perubahan": [lv[0] for lv in levers],
                "Selisih (kWh per jam)": [ec.num_id(lv[1], 2, signed=True) for lv in levers],
                "Selisih biaya 30 hari": [_signed_rp(lv[1] * HOURS_30_DAYS * tariff) for lv in levers],
            }),
            hide_index=True, **ui.STRETCH_DF,
        )
        ui.chart_head("Rekomendasi efisiensi")
        tips = []
        temp_q25 = float(b.train_X["Temperature"].quantile(0.25))
        ren_q25 = float(b.train_X["RenewableEnergy"].quantile(0.25))
        if hvac and temperature <= temp_q25:
            tips.append(f"Suhu {ec.num_id(temperature, 1)} °C tergolong sejuk dibanding data historis. Mematikan HVAC dapat menurunkan konsumsi.")
        if lighting and occupancy == 0:
            tips.append("Tidak ada penghuni tetapi lampu menyala. Sebaiknya dimatikan.")
        if kategori == "Tinggi":
            tips.append("Konsumsi tergolong tinggi (di atas 75% data historis). Pertimbangkan menaikkan setelan suhu HVAC.")
        if renewable <= ren_q25:
            tips.append(f"Kontribusi energi terbarukan {ec.num_id(renewable, 1)}% termasuk rendah. Cek kinerja panel surya.")
        if tips:
            for tip in tips:
                st.write("- " + tip)
        else:
            st.write("Tidak ada rekomendasi khusus. Kondisi ini tampak efisien dibanding data historis.")
    with d2:
        ui.chart_head("Prediksi tiap model", "Jika keempat model sepakat, titiknya berdekatan. Titik biru adalah model yang aktif.")
        preds = {n: float(b.predict(n, row)[0]) for n in ec.MODEL_NAMES}
        fig = ui.base_fig(290)
        fig.add_trace(go.Scatter(
            x=list(preds.values()), y=list(preds.keys()), mode="markers+text",
            text=[ec.num_id(v, 2) for v in preds.values()], textposition="top center",
            marker=dict(size=[18 if n == name else 13 for n in preds], color=[BLUE if n == name else MUTED for n in preds],
                        line=dict(color=NAVY, width=1.5)),
            hovertemplate="%{y}: %{x:.2f} kWh<extra></extra>",
        ))
        lo, hi = min(preds.values()), max(preds.values())
        pad = max((hi - lo) * 0.3, 3.0)
        fig.update_xaxes(title="kWh per jam", range=[lo - pad, hi + pad])
        fig.update_yaxes(range=[len(preds) - 0.5, -0.9], showgrid=True)
        ui.show(fig, "pred_models")

    # -- skenario --------------------------------------------------------------- #
    ui.chart_head("Bandingkan skenario",
                  "Simpan kondisi di atas, ubah isian, simpan lagi. Cocok untuk melihat berapa rupiah yang dihemat bila HVAC atau jam operasional berubah.")
    st.session_state.setdefault("scen_name", "Skenario 1")
    n1, n2 = st.columns([3, 1], vertical_alignment="bottom")
    n1.text_input("Nama skenario", key="scen_name")
    record = {
        "Model": name, "Suhu (°C)": temperature, "Penghuni": occupancy, "HVAC": "Nyala" if hvac else "Mati",
        "Lampu": "Nyala" if lighting else "Mati", "kWh per jam": pred, "kWh per hari": day_kwh,
    }
    n2.button("Simpan skenario", on_click=_save_scenario, args=(record,), **ui.STRETCH_BTN)

    scenarios = st.session_state.get("scenarios", [])
    if not scenarios:
        st.caption("Belum ada skenario tersimpan.")
        return
    sc = pd.DataFrame(scenarios)
    sc["Biaya 30 hari"] = sc["kWh per hari"] * 30 * tariff
    sc["Selisih vs skenario pertama"] = sc["Biaya 30 hari"] - sc["Biaya 30 hari"].iloc[0]
    shown = sc[["Nama", "Model", "Suhu (°C)", "Penghuni", "HVAC", "Lampu", "kWh per jam"]].copy()
    shown["kWh per jam"] = shown["kWh per jam"].map(lambda v: ec.num_id(v, 2))
    shown["Biaya 30 hari"] = sc["Biaya 30 hari"].map(ec.rupiah)
    shown["Selisih vs skenario pertama"] = sc["Selisih vs skenario pertama"].map(_signed_rp)
    t1, t2 = st.columns([3, 2], gap="large")
    with t1:
        st.dataframe(shown, hide_index=True, **ui.STRETCH_DF)
        st.button("Hapus semua skenario", on_click=_clear_scenarios)
    with t2:
        fig = ui.base_fig(300)
        fig.add_trace(go.Bar(x=sc["Nama"], y=sc["Biaya 30 hari"], marker_color=GREEN,
                             text=[ec.rupiah_short(v) for v in sc["Biaya 30 hari"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}: Rp %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Rupiah per 30 hari", tickformat=",.0f", range=[0, float(sc["Biaya 30 hari"].max()) * 1.2])
        ui.show(fig, "pred_scenarios")


# --------------------------------------------------------------------------- #
# Prediksi massal
# --------------------------------------------------------------------------- #
COLUMN_SPEC = pd.DataFrame(
    [
        ("Timestamp", "Waktu pengamatan. Jika tidak ada, isi Hour dan IsWeekend.", "2026-11-16 08:00:00"),
        ("Temperature", "Suhu (°C)", "25.4"),
        ("Humidity", "Kelembapan (%)", "45.0"),
        ("SquareFootage", "Luas bangunan (kaki²)", "1500"),
        ("Occupancy", "Jumlah penghuni", "5"),
        ("HVACUsage", "Status HVAC: On atau Off", "On"),
        ("LightingUsage", "Status lampu: On atau Off", "Off"),
        ("RenewableEnergy", "Kontribusi energi terbarukan (%)", "15.0"),
        ("Holiday", "Hari libur: Yes atau No", "No"),
        ("EnergyConsumption", "Opsional. Konsumsi aktual, dipakai untuk mengukur akurasi.", "75.2"),
    ],
    columns=["Kolom", "Isi", "Contoh"],
)


def batch_page(ctx: ui.AppContext) -> None:
    b, name, tariff = ctx.bundle, ctx.model_name, ctx.tariff
    ui.hero("Prediksi massal", f"Unggah tabel berisi banyak kondisi bangunan. Tiap baris diprediksi kWh per jam-nya dan biayanya. Model aktif: {name}.")

    c_up, c_act = st.columns([3, 2], gap="large")
    with c_up:
        uploaded = st.file_uploader("Unggah file CSV atau Excel (.xlsx)", type=["csv", "xlsx"], key="batch_upload")
    with c_act:
        st.download_button("Unduh template CSV", data=ec.template_frame().to_csv(index=False).encode("utf-8"),
                           file_name="template_prediksi_energi.csv", mime="text/csv", **ui.STRETCH_BTN)
        if st.session_state.get("use_sample"):
            if st.button("Hapus data contoh", **ui.STRETCH_BTN):
                st.session_state["use_sample"] = False
                st.rerun()
        elif st.button("Coba dengan data contoh", **ui.STRETCH_BTN):
            st.session_state["use_sample"] = True
            st.rerun()
        st.caption("Data contoh adalah jam-jam terakhir dari data latih (periode uji) beserta konsumsi aktualnya, jadi akurasi model langsung terlihat.")

    with st.expander("Kolom yang dibutuhkan"):
        st.dataframe(COLUMN_SPEC, hide_index=True, **ui.STRETCH_DF)
        st.caption("Nama kolom tidak peka huruf besar-kecil. Nilai On/Off dan Yes/No juga bisa ditulis Ya/Tidak atau 1/0. "
                   "CSV dengan pemisah titik koma ikut terbaca.")

    if uploaded is not None:
        st.session_state["use_sample"] = False
        try:
            df_in = ec.read_table(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"File tidak bisa dibaca: {exc}. Pastikan formatnya CSV atau .xlsx yang valid.")
            return
    elif st.session_state.get("use_sample"):
        df_in = ec.sample_upload_frame(ctx.raw)
    else:
        st.info("Belum ada data. Unggah file atau tekan Coba dengan data contoh.")
        return

    check = ec.check_upload(df_in, b.train_range)
    for message in check.errors:
        st.error(message)
    if not check.ok:
        return
    for message in check.warnings:
        st.warning(message)

    result = ec.predict_upload(check, lambda X: b.predict(name, X))
    result["Biaya_Rp"] = result["Prediksi_kWh"] * tariff
    result["Kategori"] = b.categorize(result["Prediksi_kWh"])
    lead = (["Timestamp"] if check.has_timestamp else []) + ["Prediksi_kWh", "Biaya_Rp", "Kategori"]
    result = result[lead + [c for c in result.columns if c not in lead]]
    ok = result[check.valid].copy()
    if check.has_timestamp:
        ok = ok.sort_values("Timestamp")

    total_kwh = float(ok["Prediksi_kWh"].sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Baris diprediksi", ec.num_id(len(ok), 0))
    k2.metric("Total (kWh)", ec.num_id(total_kwh, 1))
    k3.metric("Rata-rata/jam (kWh)", ec.num_id(ok["Prediksi_kWh"].mean(), 2))
    k4.metric("Biaya (Rp)", ec.rupiah_short(total_kwh * tariff), help=f"Tepatnya {ec.rupiah(total_kwh * tariff)}.")
    k5.metric("Kategori tinggi", ec.num_id(int((ok["Kategori"] == "Tinggi").sum()), 0),
              help="Prediksi di atas kuartil ketiga data historis.")

    x = ok["Timestamp"] if check.has_timestamp else pd.Series(np.arange(1, len(ok) + 1), index=ok.index)
    x_title = "Waktu" if check.has_timestamp else "Urutan baris"

    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        ui.chart_head("Konsumsi per jam",
                      "Garis biru adalah prediksi" + (", garis navy putus-putus adalah konsumsi aktual." if check.has_actual else "."))
        fig = ui.base_fig(340)
        if check.has_actual:
            fig.add_trace(go.Scatter(x=x, y=ok[T], mode="lines", name="Aktual", line=dict(color=NAVY, width=1.6, dash="dot"),
                                     hovertemplate="Aktual: %{y:.2f} kWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=ok["Prediksi_kWh"], mode="lines", name="Prediksi", line=dict(color=BLUE, width=2.4),
                                 hovertemplate="Prediksi: %{y:.2f} kWh<extra></extra>"))
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="kWh per jam")
        ui.show(fig, "batch_line")
    with g2:
        ui.chart_head("Biaya kumulatif", f"Dihitung dengan tarif {ec.rupiah(tariff, 2)} per kWh.")
        fig = ui.base_fig(340)
        fig.add_trace(go.Scatter(x=x, y=(ok["Prediksi_kWh"] * tariff).cumsum(), mode="lines", line=dict(color=GREEN, width=2.6),
                                 fill="tozeroy", fillcolor="rgba(27,122,87,.14)", hovertemplate="Rp %{y:,.0f}<extra></extra>"))
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="Rupiah", tickformat=",.0f")
        ui.show(fig, "batch_cost")

    if check.has_actual:
        ev = ok.dropna(subset=[T])
        if len(ev) >= 2:
            ui.chart_head("Seberapa akurat prediksinya", "Dibandingkan dengan konsumsi aktual pada file yang sama.")
            mae = mean_absolute_error(ev[T], ev["Prediksi_kWh"])
            rmse = float(np.sqrt(mean_squared_error(ev[T], ev["Prediksi_kWh"])))
            r2 = r2_score(ev[T], ev["Prediksi_kWh"])
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{ec.num_id(mae, 4)} kWh", help="Rata-rata selisih mutlak antara prediksi dan aktual.")
            m2.metric("RMSE", f"{ec.num_id(rmse, 4)} kWh", help="Seperti MAE, tetapi selisih besar dihukum lebih berat.")
            m3.metric("R²", ec.num_id(r2, 4), help="Bagian variasi konsumsi yang berhasil dijelaskan model. 1 berarti sempurna.")
            e1, e2 = st.columns(2, gap="large")
            with e1:
                lo, hi = float(min(ev[T].min(), ev["Prediksi_kWh"].min())), float(max(ev[T].max(), ev["Prediksi_kWh"].max()))
                fig = ui.base_fig(340)
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NAVY, width=1.5, dash="dash"),
                                         hoverinfo="skip", name="Prediksi sempurna"))
                fig.add_trace(go.Scatter(x=ev[T], y=ev["Prediksi_kWh"], mode="markers", name="Jam pengamatan",
                                         marker=dict(color=BLUE, size=7, opacity=.6),
                                         hovertemplate="Aktual %{x:.2f}, prediksi %{y:.2f} kWh<extra></extra>"))
                fig.update_xaxes(title="Aktual (kWh)")
                fig.update_yaxes(title="Prediksi (kWh)")
                ui.show(fig, "batch_avp")
            with e2:
                fig = ui.base_fig(340)
                fig.add_trace(go.Histogram(x=ev[T] - ev["Prediksi_kWh"], nbinsx=24, marker_color=AMBER, opacity=.9,
                                           hovertemplate="Selisih %{x:.1f} kWh: %{y} jam<extra></extra>"))
                fig.add_vline(x=0, line=dict(color=NAVY, width=1.5, dash="dash"))
                fig.update_xaxes(title="Selisih aktual dikurangi prediksi (kWh)")
                fig.update_yaxes(title="Jumlah jam")
                ui.show(fig, "batch_resid")

    ui.chart_head("Tabel hasil", "Kolom Prediksi_kWh, Biaya_Rp, dan Kategori ditempatkan di depan, diikuti data asli Anda.")
    config = {
        "Prediksi_kWh": st.column_config.NumberColumn("Prediksi (kWh)", format="%.2f"),
        "Biaya_Rp": st.column_config.NumberColumn("Biaya (Rp)", format="%.0f"),
    }
    if check.has_timestamp:
        config["Timestamp"] = st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm")
    st.dataframe(result, hide_index=True, column_config=config, height=320, **ui.STRETCH_DF)
    st.download_button("Unduh hasil prediksi (CSV)", data=result.to_csv(index=False).encode("utf-8-sig"),
                       file_name="hasil_prediksi_energi.csv", mime="text/csv", type="primary")
