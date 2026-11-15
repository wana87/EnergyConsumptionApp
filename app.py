"""Meteran prediksi energi: monitoring, analisis, dan prediksi konsumsi listrik per jam.

Jalankan lokal:  streamlit run app.py
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import energy_core as ec
import ui
import views_explore as vx
import views_info as vi
import views_predict as vp

st.set_page_config(
    page_title="Meteran Prediksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()

PAGES = {
    "Dashboard": vx.dashboard,
    "Analisis": vx.analysis,
    "Prediksi": vp.predict_page,
    "Prediksi massal": vp.batch_page,
    "Data": vx.data_page,
    "Model": vi.model_page,
    "Tentang": vi.about_page,
}
FILTERED_PAGES = {"Dashboard", "Analisis", "Data"}

TARIFFS = {
    "R-1 1.300–2.200 VA": 1444.70,
    "R-1 900 VA (RTM)": 1352.00,
    "R-2/R-3 3.500 VA ke atas": 1699.53,
    "R-1 900 VA bersubsidi": 605.00,
    "R-1 450 VA bersubsidi": 415.00,
}
CUSTOM_TARIFF = "Tarif sendiri"


# --------------------------------------------------------------------------- #
# Data dan model (disimpan di cache supaya tidak dihitung ulang tiap interaksi)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return ec.load_raw()


@st.cache_data(show_spinner=False)
def load_uploaded(content: bytes, name: str) -> ec.DatasetCheck:
    buffer = io.BytesIO(content)
    buffer.name = name
    return ec.check_dataset(ec.read_table(buffer))


@st.cache_data(show_spinner=False)
def enriched(raw: pd.DataFrame) -> pd.DataFrame:
    return ec.enrich(raw)


@st.cache_resource(show_spinner="Melatih keempat model…")
def get_bundle(raw: pd.DataFrame) -> ec.ModelBundle:
    return ec.train_models(raw)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
raw, source, is_default = load_default(), "data bawaan (Energy_consumption.csv)", True

with st.sidebar:
    st.markdown(
        ui.one_line('<div class="side-brand"><div class="hero-mark"><div class="hero-bolt"></div></div>'
                    "<span>Meteran prediksi energi</span></div>"),
        unsafe_allow_html=True,
    )
    menu = st.radio("Menu", list(PAGES), label_visibility="collapsed", key="menu")

    st.markdown("#### Data")
    with st.expander("Ganti data latih"):
        upload = st.file_uploader(
            "Unggah CSV atau Excel", type=["csv", "xlsx"], key="dataset_upload",
            help="Kolom wajib: Timestamp, Temperature, Humidity, SquareFootage, Occupancy, HVACUsage, LightingUsage, "
                 "RenewableEnergy, Holiday, EnergyConsumption. Minimal 100 baris. Semua model dilatih ulang dari data ini.",
        )
        if upload is not None:
            try:
                check = load_uploaded(upload.getvalue(), upload.name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"File tidak bisa dibaca: {exc}")
            else:
                for message in check.errors:
                    st.error(message)
                for message in check.warnings:
                    st.warning(message)
                if check.ok:
                    raw, source, is_default = check.raw, upload.name, False
                    st.success(f"Data {upload.name} dipakai ({ec.num_id(len(raw), 0)} baris).")
        if is_default:
            st.caption("Memakai data bawaan. Hapus file di atas untuk kembali ke data bawaan.")

    data = enriched(raw)
    bundle = get_bundle(raw)

    st.markdown("#### Model prediksi")
    best = bundle.best_model()
    model_name = st.selectbox(
        "Model", ec.MODEL_NAMES, index=ec.MODEL_NAMES.index(best), key="model_name",
        format_func=lambda n: f"{n} (terbaik)" if n == best else n,
        help="Dipakai di halaman Prediksi, Prediksi massal, dan Model. Penanda terbaik berdasarkan RMSE terendah pada data uji.",
    )
    st.markdown(
        ui.one_line(f'<div class="side-note">R² {ec.num_id(bundle.metrics_for(model_name)["R2"], 3)} dan MAE '
                    f'{ec.num_id(bundle.metrics_for(model_name)["MAE"], 2)} kWh pada {ec.num_id(bundle.n_test, 0)} jam data uji.</div>'),
        unsafe_allow_html=True,
    )

    st.markdown("#### Tarif listrik")
    choice = st.selectbox(
        "Golongan tarif PLN", [*TARIFFS, CUSTOM_TARIFF], key="tariff_choice",
        help="RTM adalah Rumah Tangga Mampu. R-1 untuk daya sampai 2.200 VA, R-2 untuk 3.500–5.500 VA, R-3 untuk 6.600 VA ke atas.",
    )
    if choice == CUSTOM_TARIFF:
        tariff = float(st.number_input("Tarif (Rp per kWh)", min_value=0.0, value=1444.70, step=10.0, format="%.2f"))
    else:
        tariff = TARIFFS[choice]
    st.markdown(f'<div class="tarif-chip">{ec.rupiah(tariff, 2)} per kWh</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note">Tarif non-subsidi periode Oktober sampai Desember 2026. '
        "Ganti dengan tarif di tagihan Anda jika berbeda.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Filter")
    with st.expander("Filter data"):
        t_min, t_max = data["Timestamp"].min().date(), data["Timestamp"].max().date()
        date_range = st.date_input("Rentang tanggal", value=(t_min, t_max), min_value=t_min, max_value=t_max,
                                   key=f"date_range_{t_min}_{t_max}")
        f_hvac = st.multiselect("HVAC", ["On", "Off"], default=["On", "Off"], key="f_hvac")
        f_light = st.multiselect("Lampu", ["On", "Off"], default=["On", "Off"], key="f_light")
        f_holiday = st.multiselect("Hari libur", ["Yes", "No"], default=["Yes", "No"], key="f_holiday")
        st.caption("Filter berlaku untuk Dashboard, Analisis, dan Data.")


# --------------------------------------------------------------------------- #
# Halaman
# --------------------------------------------------------------------------- #
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = t_min, t_max

mask = (
    (data["Timestamp"] >= pd.Timestamp(start))
    & (data["Timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    & data["HVACUsage"].isin(f_hvac)
    & data["LightingUsage"].isin(f_light)
    & data["Holiday"].isin(f_holiday)
)
filtered = data[mask]

if filtered.empty and menu in FILTERED_PAGES:
    st.warning("Tidak ada data yang cocok dengan filter. Ubah filter di sidebar.")
    st.stop()

context = ui.AppContext(
    raw=raw, data=data, df=filtered, bundle=bundle, tariff=tariff,
    model_name=model_name, source=source, is_default=is_default,
)
PAGES[menu](context)
