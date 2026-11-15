"""Halaman informasi: Model dan Tentang."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import energy_core as ec
import ui
from ui import AMBER, BLUE, GRID, MUTED, NAVY

T = ec.TARGET


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def model_page(ctx: ui.AppContext) -> None:
    b, name = ctx.bundle, ctx.model_name
    ui.hero("Model prediksi", f"Empat model regresi dibandingkan pada {ec.num_id(b.n_test, 0)} jam data uji. Model aktif: {name}.")

    m = b.metrics_for(name)
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{ec.num_id(m['MAE'], 4)} kWh", help="Rata-rata selisih mutlak prediksi dan aktual pada data uji.")
    k2.metric("RMSE", f"{ec.num_id(m['RMSE'], 4)} kWh", help="Seperti MAE, tetapi selisih besar dihukum lebih berat.")
    k3.metric("R²", ec.num_id(m["R2"], 4),
              help=f"Model menjelaskan sekitar {ec.num_id(m['R2'] * 100, 0)}% variasi konsumsi pada data uji.")
    if ctx.is_default:
        st.caption("Angka Linear Regression dan Gradient Boosting sama dengan laporan. Decision Tree dan Random Forest dapat "
                   "berbeda sedikit dari laporan, misalnya karena versi scikit-learn.")

    # -- perbandingan ---------------------------------------------------------- #
    ui.chart_head("Perbandingan empat model", "Model yang aktif ditandai biru. Pindahkan model aktif lewat sidebar.")
    comp = b.comparison
    colors = [BLUE if n == name else MUTED for n in comp["Model"]]
    titles = ("MAE, makin kecil makin baik", "RMSE, makin kecil makin baik", "R², makin besar makin baik")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.05)
    for i, col in enumerate(["MAE", "RMSE", "R2"], start=1):
        fig.add_trace(go.Bar(x=comp[col], y=comp["Model"], orientation="h", marker_color=colors, showlegend=False,
                             text=[ec.num_id(v, 4) for v in comp[col]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"), row=1, col=i)
        top = max(float(comp[col].max()), 0.1)
        low = min(float(comp[col].min()), 0.0)
        fig.update_xaxes(range=[low * 1.2, top * 1.35], gridcolor=GRID, row=1, col=i)
    ui.style_fig(fig, 290)
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    ui.style_subplot_titles(fig)
    ui.show(fig, "model_compare")

    table = comp.sort_values("R2", ascending=False).reset_index(drop=True)
    table.insert(0, "Peringkat", range(1, len(table) + 1))
    shown = pd.DataFrame({
        "Peringkat": table["Peringkat"], "Model": table["Model"],
        "MAE": table["MAE"].map(lambda v: ec.num_id(v, 4)), "RMSE": table["RMSE"].map(lambda v: ec.num_id(v, 4)),
        "R²": table["R2"].map(lambda v: ec.num_id(v, 4)),
    })
    st.dataframe(shown, hide_index=True, **ui.STRETCH_DF)

    # -- diagnosis model aktif ----------------------------------------------------- #
    test = b.test_frame
    a1, a2 = st.columns(2, gap="large")
    with a1:
        ui.chart_head(f"Aktual dan prediksi, {name}", "Makin dekat ke garis putus-putus, makin akurat.")
        lo, hi = float(min(test["Aktual"].min(), test[name].min())), float(max(test["Aktual"].max(), test[name].max()))
        fig = ui.base_fig(360)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NAVY, width=1.5, dash="dash"),
                                 hoverinfo="skip", name="Prediksi sempurna"))
        fig.add_trace(go.Scatter(x=test["Aktual"], y=test[name], mode="markers", name="Jam uji",
                                 marker=dict(color=BLUE, size=7, opacity=.6),
                                 hovertemplate="Aktual %{x:.2f}, prediksi %{y:.2f} kWh<extra></extra>"))
        fig.update_xaxes(title="Aktual (kWh)")
        fig.update_yaxes(title="Prediksi (kWh)")
        ui.show(fig, "model_avp")
    with a2:
        ui.chart_head("Sebaran selisih", "Selisih aktual dikurangi prediksi. Idealnya berpusat di nol tanpa pola.")
        fig = ui.base_fig(360)
        fig.add_trace(go.Histogram(x=test["Aktual"] - test[name], nbinsx=24, marker_color=AMBER, opacity=.9,
                                   hovertemplate="Selisih %{x:.1f} kWh: %{y} jam<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=NAVY, width=1.5, dash="dash"))
        fig.update_xaxes(title="Selisih (kWh)")
        fig.update_yaxes(title="Jumlah jam")
        ui.show(fig, "model_resid")

    ui.chart_head("Periode uji sepanjang waktu", "Konsumsi aktual dan prediksi model aktif pada data yang tidak dipakai melatih.")
    fig = ui.base_fig(320)
    fig.add_trace(go.Scatter(x=test["Timestamp"], y=test["Aktual"], mode="lines", name="Aktual",
                             line=dict(color=NAVY, width=1.6, dash="dot"), hovertemplate="Aktual: %{y:.2f} kWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=test["Timestamp"], y=test[name], mode="lines", name=name, line=dict(color=BLUE, width=2.2),
                             hovertemplate="Prediksi: %{y:.2f} kWh<extra></extra>"))
    fig.update_yaxes(title="kWh per jam")
    ui.show(fig, "model_time")

    # -- faktor berpengaruh -------------------------------------------------------- #
    imp = b.importance(name)
    imp = imp.reindex(imp.sort_values().index)
    if name == ec.LINEAR:
        sub = "Besar pengaruh bila faktor berubah satu simpangan baku (kWh), sehingga faktor berskala berbeda bisa dibandingkan."
        axis = "kWh per 1 simpangan baku"
    else:
        sub = "Kepentingan fitur dari model pohon: seberapa sering dan seberapa besar fitur dipakai untuk membagi data (jumlahnya 1)."
        axis = "Kepentingan fitur"
    ui.chart_head(f"Faktor yang paling berpengaruh, {name}", sub)
    fig = ui.base_fig(360)
    fig.add_trace(go.Bar(x=imp.values, y=[ec.LABELS[f] for f in imp.index], orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title=axis)
    fig.update_yaxes(showgrid=False)
    ui.show(fig, "model_importance")

    effects = b.linear.effects_table()
    temp_effect = float(effects.loc[effects["Feature"] == "Temperature", "Efek"].iloc[0])
    hvac_effect = float(effects.loc[effects["Feature"] == "HVACUsage", "Efek"].iloc[0])
    ui.chart_head("Pengaruh tiap faktor dalam satuan asli (Linear Regression)",
                  f"Pada data ini, kenaikan suhu 1 °C menambah prediksi sekitar {ec.num_id(temp_effect, 2)} kWh per jam, "
                  f"dan HVAC menyala sekitar {ec.num_id(hvac_effect, 2)} kWh.")
    st.dataframe(
        pd.DataFrame({"Faktor": effects["Faktor"], "Perubahan": effects["Perubahan"],
                      "Efek (kWh per jam)": [ec.num_id(v, 3, signed=True) for v in effects["Efek"]]}),
        hide_index=True, **ui.STRETCH_DF,
    )

    with st.expander("Cara model dibuat"):
        tr, te = b.periods["train"], b.periods["test"]
        st.markdown(
            f"""
1. **Data.** {ec.num_id(b.n_train + b.n_test, 0)} observasi per jam. Kolom DayOfWeek tidak dipakai.
2. **Fitur waktu.** Hour dan IsWeekend diturunkan dari Timestamp.
3. **Encoding.** HVACUsage, LightingUsage, dan Holiday diubah menjadi 0 (Off/No) dan 1 (On/Yes).
4. **Pembagian data kronologis.** Latih {b.n_train} jam ({tr[0]:%d %b} sampai {tr[1]:%d %b %Y}),
   uji {b.n_test} jam ({te[0]:%d %b} sampai {te[1]:%d %b %Y}).
5. **Standardisasi (Linear Regression).** Lima fitur numerik distandardisasi dengan parameter dari data latih saja,
   supaya data uji tidak bocor ke pelatihan. Model pohon tidak membutuhkannya.
6. **Model.** Linear Regression, Decision Tree, Random Forest (200 pohon), dan Gradient Boosting dilatih pada data latih,
   lalu dibandingkan pada data uji dengan MAE, RMSE, dan R².

Saat Anda memasukkan data, aplikasi memakai langkah yang sama persis sebelum memanggil model.
"""
        )

    with st.expander("Batasan yang perlu diingat"):
        st.markdown(
            f"""
- **Data sintetis.** Dataset bawaan meniru skenario nyata, jadi polanya belum tentu sama dengan bangunan sebenarnya.
- **Akurasi sedang.** R² {ec.num_id(m['R2'], 3)} pada model aktif berarti sekitar {ec.num_id((1 - m['R2']) * 100, 0)}% variasi konsumsi
  belum dijelaskan fitur yang ada. Pakai hasilnya sebagai perkiraan, bukan angka pasti.
- **Rentang terbatas.** Model hanya melihat nilai dalam rentang data latih. Di luar itu, model linear meneruskan garisnya begitu saja,
  sedangkan model pohon mendatar di nilai tepi. Keduanya kurang bisa diandalkan.
- **Satuan.** Hasil ditampilkan dalam kWh per jam, mengikuti antarmuka prediksi pada laporan. Kolom RenewableEnergy dibaca sebagai persen.
"""
        )


# --------------------------------------------------------------------------- #
# Tentang
# --------------------------------------------------------------------------- #
def about_page(ctx: ui.AppContext) -> None:
    raw, b = ctx.raw, ctx.bundle
    ui.hero("Tentang sistem", "Meteran prediksi energi memantau, menganalisis, dan memprediksi konsumsi listrik per jam beserta biayanya.")

    start, end = raw["Timestamp"].min(), raw["Timestamp"].max()
    st.markdown(
        f"""
### Tujuan
Memperkirakan besar konsumsi energi listrik per jam dari kondisi lingkungan, bangunan, operasional, dan waktu, untuk mendukung
efisiensi dan perencanaan energi. Targetnya angka kontinu (kWh), sehingga pendekatannya regresi, bukan klasifikasi.

### Mengapa topik ini
1. Konsumsi energi penting untuk efisiensi operasional dan pengambilan keputusan.
2. Target bersifat numerik, cocok untuk kasus regresi.
3. Fitur yang tersedia relevan: lingkungan, bangunan, operasional, dan waktu.
4. Mendukung alur analisis data end-to-end: pemahaman masalah, EDA, pemodelan, evaluasi, hingga prototipe prediksi.

### Dataset
- Sumber saat ini: **{ctx.source}**. {ec.num_id(len(raw), 0)} observasi per jam, {start:%d %b %Y} sampai {end:%d %b %Y}.
- Data bawaan: dataset sintetis Kaggle *Energy-consumption-prediction* yang meniru skenario konsumsi energi dunia nyata.
- Struktur: 1 kolom waktu (Timestamp), 5 prediktor numerik (Temperature, Humidity, SquareFootage, Occupancy, RenewableEnergy),
  4 kategorikal (HVACUsage, LightingUsage, DayOfWeek, Holiday), dan 1 target (EnergyConsumption).
- Saat persiapan data, Timestamp diturunkan menjadi Hour dan IsWeekend, sedangkan DayOfWeek tidak dipakai.

### Alur proyek dan letaknya di aplikasi
1. **Pemahaman masalah.** Halaman ini.
2. **Pemahaman data dan EDA.** Dashboard, Analisis, dan Data.
3. **Persiapan data.** Bagian Cara model dibuat di halaman Model.
4. **Pemodelan.** Halaman Model: empat model regresi dilatih pada data latih.
5. **Evaluasi.** Halaman Model: MAE, RMSE, R², grafik aktual dan prediksi, serta sebaran selisih.
6. **Prototipe.** Prediksi (satu per satu) dan Prediksi massal (dari file).

### Fitur
| Halaman | Isi |
|---|---|
| Dashboard | KPI, tren konsumsi (kWh atau rupiah), sumber energi, pola jam dan hari, pengaruh HVAC dan lampu, hari libur, peta panas |
| Analisis | Korelasi, penjelajah sebar dengan garis tren, sebaran konsumsi, statistik, pencilan IQR |
| Prediksi | Meteran kWh, kategori konsumsi, struk biaya, faktor penyumbang, simulasi satu faktor, peluang hemat, rekomendasi, bandingkan skenario |
| Prediksi massal | Unggah CSV atau Excel, validasi, biaya kumulatif, akurasi bila ada data aktual, unduh hasil |
| Data | Tabel terfilter, unduh, pemeriksaan kualitas data |
| Model | Perbandingan model, diagnosis, faktor berpengaruh, batasan |

### Cara memakai data sendiri
Buka bagian **Data** di sidebar dan unggah CSV atau Excel dengan kolom yang sama (termasuk EnergyConsumption, minimal 100 baris).
Semua halaman dan keempat model akan dilatih ulang dari data itu.

### Tarif listrik
Biaya dihitung dari tarif yang dipilih di sidebar. Tarif non-subsidi periode Oktober sampai Desember 2026 memakai ketetapan PLN
(rumah tangga 1.300–2.200 VA Rp 1.444,70 per kWh). Perkiraan belum termasuk pajak, biaya beban, dan biaya lain di tagihan.

**Teknologi:** Python, Streamlit, pandas, scikit-learn, Plotly.
"""
    )
