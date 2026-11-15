# Meteran prediksi energi

Aplikasi Streamlit untuk memantau, menganalisis, dan memprediksi konsumsi listrik per jam (kWh) beserta biayanya.
Memakai pipeline dari notebook `Oktriza01.ipynb` dan menambahkan fungsi monitoring seperti aplikasi contoh.

## Halaman

| Halaman | Isi |
|---|---|
| Dashboard | KPI, tren konsumsi (per jam, harian, mingguan, dalam kWh atau rupiah), sumber energi jaringan vs terbarukan, pola jam dan hari, pengaruh HVAC dan lampu, hari libur, peta panas jam x hari |
| Analisis | Heatmap korelasi dengan kalimat temuan, penjelajah sebar (pilih X, warna, garis tren), sebaran konsumsi, statistik deskriptif, pencilan metode IQR |
| Prediksi | Meteran kWh, kategori konsumsi (rendah/normal/tinggi), energi dari jaringan, rentang perkiraan, struk biaya (1 jam, 24 jam, 30 hari), faktor penyumbang, simulasi satu faktor, selisih bila HVAC/lampu diubah, rekomendasi efisiensi, prediksi keempat model, bandingkan skenario |
| Prediksi massal | Unggah CSV/Excel, validasi, biaya kumulatif, kategori per baris, akurasi otomatis bila ada `EnergyConsumption`, unduh hasil |
| Data | Tabel terfilter dan unduh, pemeriksaan kualitas data (nilai kosong, duplikat, kecocokan DayOfWeek) |
| Model | Perbandingan Linear Regression, Decision Tree, Random Forest, Gradient Boosting; aktual vs prediksi, sebaran selisih, faktor berpengaruh, cara model dibuat, batasan |
| Tentang | Konteks proyek, struktur dataset, alur analisis dan letaknya di aplikasi |

Sidebar: menu, **ganti data latih** (semua halaman dan keempat model dilatih ulang), pilihan **model prediksi**,
**tarif listrik** (golongan PLN atau tarif sendiri), dan **filter data** (tanggal, HVAC, lampu, hari libur).

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub dan unggah seluruh isi folder ini, termasuk `.streamlit/` dan `data/`.
2. Buka <https://share.streamlit.io>, pilih **Create app**, lalu pilih repository dan branch tersebut.
3. Isi **Main file path** dengan `app.py` dan tekan **Deploy**.

Model tidak disimpan sebagai file. Aplikasi melatih keempat model dari `data/Energy_consumption.csv` saat pertama dibuka
(beberapa detik), jadi tidak ada masalah kompatibilitas versi scikit-learn.

## Format data

Untuk **prediksi massal** lihat `data/template_upload.csv`. Kolom wajib: `Temperature`, `Humidity`, `SquareFootage`,
`Occupancy`, `HVACUsage`, `LightingUsage`, `RenewableEnergy`, `Holiday`, ditambah `Timestamp` (atau `Hour` dan `IsWeekend`).

Untuk **mengganti data latih** kolom wajibnya sama, ditambah `Timestamp` dan `EnergyConsumption`, minimal 100 baris.
Kolom `DayOfWeek` boleh ada tetapi tidak dipakai.

- Nama kolom tidak peka huruf besar-kecil.
- Nilai On/Off dan Yes/No juga bisa ditulis Ya/Tidak, 1/0, atau True/False.
- CSV dengan pemisah koma, titik koma, atau tab terbaca otomatis.

## Struktur proyek

```
app.py              sidebar, sumber data, routing halaman
energy_core.py      data, fitur, empat model, penjelasan, validasi file (tanpa kode Streamlit)
ui.py               tema CSS, komponen meteran dan struk, gaya grafik
views_explore.py    Dashboard, Analisis, Data
views_predict.py    Prediksi, Prediksi massal
views_info.py       Model, Tentang
data/               dataset bawaan dan template unggahan
tests/test_core.py  22 tes inti
.streamlit/         tema
```

Jalankan tes dengan `python tests/test_core.py` atau `pytest tests`.

## Catatan teknis

- Pipeline identik dengan notebook: `Hour` dan `IsWeekend` dari `Timestamp`, `DayOfWeek` dibuang, encoding 0/1, pembagian
  kronologis 80/20, `StandardScaler` (hanya Linear Regression) dilatih pada data latih saja.
- Pada data bawaan: Linear Regression MAE 4,1363, RMSE 4,9940, R² 0,5930; Gradient Boosting MAE 4,1481, RMSE 5,0562, R² 0,5828
  (sama dengan laporan). Decision Tree dan Random Forest dapat berbeda sedikit dari laporan, misalnya karena versi scikit-learn.
- `RenewableEnergy` dibaca sebagai persen (sesuai legenda notebook), jadi energi dari jaringan = prediksi x (1 - persen/100).
- Penjelasan faktor penyumbang untuk Linear Regression bersifat tepat. Untuk model pohon bersifat perkiraan.
- Tampilan memuat font Barlow Condensed dan Public Sans dari Google Fonts. Tanpa internet aplikasi tetap jalan dengan font bawaan.
- Versi Streamlit dikunci (`1.67.0`) karena sebagian gaya CSS bergantung pada struktur HTML Streamlit.
- Diuji dengan pandas 2.2 dan 3.0, scikit-learn 1.9, plotly 7.1.
