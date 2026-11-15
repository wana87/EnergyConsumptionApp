# Meteran prediksi energi

Aplikasi Streamlit untuk memprediksi konsumsi listrik per jam (kWh) dan biayanya, berdasarkan
model Linear Regression dari notebook `Oktriza01.ipynb`.

## Fitur

| Tab | Fungsi |
|---|---|
| Prediksi manual | Isi kondisi bangunan lewat slider dan toggle. Hasil tampil di meteran kWh, struk biaya (1 jam, 24 jam, 30 hari), grafik faktor penyumbang, dan grafik "bagaimana jika satu faktor berubah". |
| Prediksi dari file | Unggah CSV atau Excel (.xlsx) berisi banyak baris. Ada validasi kolom dan nilai, peringatan jika di luar rentang data latih, grafik konsumsi dan biaya kumulatif, tabel hasil yang bisa diunduh, dan metrik akurasi otomatis bila file memuat kolom `EnergyConsumption`. |
| Grafik pemakaian | Eksplorasi data historis: deret waktu (per jam, harian, mingguan), pola per jam, suhu vs konsumsi, dan perbandingan menurut kondisi. |
| Tentang model | Metrik MAE, RMSE, R², perbandingan empat model, pengaruh tiap faktor, langkah pembuatan model, dan batasannya. |

Tarif listrik dipilih di sidebar (default Rp 1.444,70 per kWh, golongan R-1 1.300–2.200 VA, periode Oktober–Desember 2026) dan bisa diganti dengan tarif sendiri.

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub dan unggah seluruh isi folder ini, termasuk `.streamlit/` dan `data/`.
2. Buka <https://share.streamlit.io>, pilih **Create app**, lalu pilih repository dan branch tersebut.
3. Isi **Main file path** dengan `app.py` dan tekan **Deploy**.

Model tidak disimpan sebagai file. Aplikasi melatih Linear Regression dari `data/Energy_consumption.csv` saat pertama dibuka
(kurang dari satu detik), sehingga tidak ada masalah kompatibilitas versi scikit-learn.

## Format file unggahan

Lihat `data/template_upload.csv`. Kolom wajib: `Temperature`, `Humidity`, `SquareFootage`, `Occupancy`,
`HVACUsage`, `LightingUsage`, `RenewableEnergy`, `Holiday`, ditambah `Timestamp` (atau `Hour` dan `IsWeekend`).
`EnergyConsumption` opsional, dipakai untuk mengukur akurasi.

- Nama kolom tidak peka huruf besar-kecil.
- Nilai On/Off dan Yes/No juga bisa ditulis Ya/Tidak, 1/0, atau True/False.
- CSV dengan pemisah koma, titik koma, atau tab terbaca otomatis.

## Struktur proyek

```
app.py                  tampilan Streamlit
energy_core.py          data, fitur, model, validasi file (tanpa kode Streamlit)
data/                   dataset latih, template unggahan
tests/test_core.py      tes: metrik sama dengan notebook, contoh 69,44 kWh, validasi file
.streamlit/config.toml  tema
```

Jalankan tes dengan `python tests/test_core.py` atau `pytest tests`.

## Catatan teknis

- Pipeline identik dengan notebook: `Hour` dan `IsWeekend` dari `Timestamp`, `DayOfWeek` dibuang, encoding 0/1,
  pembagian kronologis 80/20, `StandardScaler` dilatih hanya pada data latih.
- Hasil pada data uji: MAE 4,1363, RMSE 4,9940, R² 0,5930.
- Gaya tampilan memuat font Barlow Condensed dan Public Sans dari Google Fonts. Tanpa internet aplikasi tetap jalan dengan font bawaan.
- Versi Streamlit dikunci (`1.67.0`) karena sebagian gaya CSS bergantung pada struktur HTML Streamlit.
- Diuji dengan pandas 2.2 dan 3.0, scikit-learn 1.9, plotly 7.1.
