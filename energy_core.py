"""Inti perhitungan aplikasi: data, fitur, model, dan validasi file unggahan.

Seluruh langkah meniru notebook Oktriza01.ipynb:
  - fitur waktu (Hour, IsWeekend) diturunkan dari Timestamp, DayOfWeek tidak dipakai
  - HVACUsage, LightingUsage, Holiday dikodekan 0/1
  - pembagian data kronologis 80% latih / 20% uji
  - StandardScaler dilatih hanya pada data latih (lima fitur numerik)
  - model akhir: Linear Regression
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Energy_consumption.csv"

TARGET = "EnergyConsumption"

# Urutan ini sama dengan X_train.columns pada notebook.
FEATURES = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "HVACUsage",
    "LightingUsage",
    "RenewableEnergy",
    "Holiday",
    "Hour",
    "IsWeekend",
]
NUMERIC_SCALED = ["Temperature", "Humidity", "SquareFootage", "Occupancy", "RenewableEnergy"]
BINARY_INPUT = ["HVACUsage", "LightingUsage", "Holiday"]
REQUIRED_BASE = NUMERIC_SCALED + BINARY_INPUT

LABELS = {
    "Temperature": "Suhu (°C)",
    "Humidity": "Kelembapan (%)",
    "SquareFootage": "Luas bangunan (kaki²)",
    "Occupancy": "Jumlah penghuni",
    "HVACUsage": "HVAC menyala",
    "LightingUsage": "Lampu menyala",
    "RenewableEnergy": "Energi terbarukan (%)",
    "Holiday": "Hari libur",
    "Hour": "Jam",
    "IsWeekend": "Akhir pekan",
}

# Perubahan yang dipakai untuk menjelaskan koefisien model dalam satuan asli.
STEP = {
    "Temperature": (1.0, "Suhu naik 1 °C"),
    "Humidity": (1.0, "Kelembapan naik 1 %"),
    "SquareFootage": (100.0, "Luas naik 100 kaki²"),
    "Occupancy": (1.0, "Penghuni bertambah 1 orang"),
    "HVACUsage": (1.0, "HVAC menyala"),
    "LightingUsage": (1.0, "Lampu menyala"),
    "RenewableEnergy": (1.0, "Energi terbarukan naik 1 %"),
    "Holiday": (1.0, "Hari libur"),
    "Hour": (1.0, "Jam maju 1 jam"),
    "IsWeekend": (1.0, "Akhir pekan"),
}

# Hasil perbandingan empat model pada notebook (data uji 200 baris terakhir).
REPORTED_COMPARISON = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
        "MAE": [4.1363, 5.7689, 4.2064, 4.1481],
        "RMSE": [4.9940, 7.2267, 5.1921, 5.0562],
        "R2": [0.5930, 0.1478, 0.5601, 0.5828],
    }
)

# Batas wajar untuk menolak data yang jelas salah ketik.
SANITY = {
    "Humidity": (0, 100),
    "SquareFootage": (1, None),
    "Occupancy": (0, None),
    "RenewableEnergy": (0, 100),
    "Hour": (0, 23),
}


# --------------------------------------------------------------------------- #
# Format angka dan rupiah gaya Indonesia
# --------------------------------------------------------------------------- #
def num_id(value: float, decimals: int = 2, signed: bool = False) -> str:
    """12345.678 -> '12.345,68' (titik ribuan, koma desimal)."""
    sign = "+" if signed and value > 0 else ""
    text = f"{value:,.{decimals}f}"
    text = text.replace(",", "_").replace(".", ",").replace("_", ".")
    return sign + text


def rupiah(value: float, decimals: int = 0) -> str:
    return "Rp " + num_id(value, decimals)


def rupiah_short(value: float) -> str:
    """Ringkas untuk kartu angka: 22246732 -> '22,2 juta' (tanpa awalan Rp, taruh di label)."""
    a = abs(value)
    if a >= 1e9:
        return f"{num_id(value / 1e9, 2)} miliar"
    if a >= 1e6:
        return f"{num_id(value / 1e6, 1)} juta"
    return num_id(value, 0)


# --------------------------------------------------------------------------- #
# Data bawaan
# --------------------------------------------------------------------------- #
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering dan encoding, sama seperti bagian Persiapan Data notebook."""
    df = raw.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["IsWeekend"] = (df["Timestamp"].dt.dayofweek >= 5).astype(int)
    df["HVACUsage"] = df["HVACUsage"].map({"Off": 0, "On": 1})
    df["LightingUsage"] = df["LightingUsage"].map({"Off": 0, "On": 1})
    df["Holiday"] = df["Holiday"].map({"No": 0, "Yes": 1})
    return df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class EnergyModel:
    scaler: StandardScaler
    regressor: LinearRegression
    design_mean: np.ndarray
    design_std: np.ndarray
    baseline: float
    train_range: pd.DataFrame
    metrics: dict
    periods: dict

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Susun matriks fitur persis seperti saat pelatihan (lima fitur numerik distandardisasi)."""
        D = X[FEATURES].astype("float64").copy()
        D[NUMERIC_SCALED] = self.scaler.transform(D[NUMERIC_SCALED])
        return D

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(self.design(X))

    def contributions(self, row: pd.DataFrame) -> pd.Series:
        """Selisih prediksi (kWh) tiap fitur terhadap kondisi rata-rata data latih.

        baseline + jumlah kontribusi = prediksi.
        """
        d = self.design(row).iloc[0].to_numpy()
        return pd.Series(self.regressor.coef_ * (d - self.design_mean), index=FEATURES)

    def effects_table(self) -> pd.DataFrame:
        scale = dict(zip(NUMERIC_SCALED, self.scaler.scale_))
        rows = []
        for name, coef, std in zip(FEATURES, self.regressor.coef_, self.design_std):
            per_unit = coef / scale[name] if name in scale else coef
            step, text = STEP[name]
            rows.append(
                {
                    "Feature": name,
                    "Faktor": LABELS[name],
                    "Perubahan": text,
                    "Efek": per_unit * step,
                    "Efek1SD": coef * std,
                }
            )
        return pd.DataFrame(rows)


def train_model(raw: pd.DataFrame | None = None) -> EnergyModel:
    raw = load_raw() if raw is None else raw
    df = engineer(raw)
    X, y = df[FEATURES], df[TARGET]

    split = int(len(df) * 0.8)  # pembagian kronologis
    X_tr, X_te = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(X_tr[NUMERIC_SCALED])
    design_tr = X_tr.astype("float64").copy()
    design_tr[NUMERIC_SCALED] = scaler.transform(X_tr[NUMERIC_SCALED])
    regressor = LinearRegression().fit(design_tr, y_tr)

    design_te = X_te.astype("float64").copy()
    design_te[NUMERIC_SCALED] = scaler.transform(X_te[NUMERIC_SCALED])
    y_hat = regressor.predict(design_te)

    design_mean = design_tr.mean().to_numpy()
    baseline = float(
        regressor.predict(pd.DataFrame([design_mean], columns=FEATURES))[0]
    )

    ts = df["Timestamp"]
    return EnergyModel(
        scaler=scaler,
        regressor=regressor,
        design_mean=design_mean,
        design_std=design_tr.std(ddof=0).to_numpy(),
        baseline=baseline,
        train_range=X_tr.agg(["min", "max"]).T.rename(columns={"min": "min", "max": "max"}),
        metrics={
            "MAE": float(mean_absolute_error(y_te, y_hat)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_hat))),
            "R2": float(r2_score(y_te, y_hat)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        },
        periods={
            "train": (ts.iloc[0], ts.iloc[split - 1]),
            "test": (ts.iloc[split], ts.iloc[-1]),
        },
    )


# --------------------------------------------------------------------------- #
# File unggahan
# --------------------------------------------------------------------------- #
def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANON = {_norm(c): c for c in FEATURES + ["Timestamp", TARGET]}
_CANON.update({"datetime": "Timestamp", "daytype": "IsWeekend", "weekend": "IsWeekend"})

_TRUE = {"on", "yes", "y", "ya", "true", "t", "weekend", "akhirpekan", "libur", "menyala", "nyala"}
_FALSE = {"off", "no", "n", "tidak", "false", "f", "weekday", "harikerja", "mati"}


def _to_binary(series: pd.Series) -> pd.Series:
    """On/Off, Yes/No, Ya/Tidak, 1/0, True/False -> 1.0 / 0.0 (selain itu NaN)."""
    num = pd.to_numeric(series, errors="coerce").astype("float64")
    out = num.where(num.isin([0, 1]))
    tokens = series.astype(str).str.strip().str.lower().str.replace(r"[\s_\-]", "", regex=True)
    from_text = tokens.map(lambda v: 1.0 if v in _TRUE else (0.0 if v in _FALSE else np.nan))
    return out.fillna(from_text.where(num.isna())).astype("float64")


def read_table(uploaded) -> pd.DataFrame:
    """Baca CSV (pemisah koma, titik koma, atau tab terdeteksi otomatis) atau Excel .xlsx."""
    name = getattr(uploaded, "name", "").lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    try:
        return pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, sep=None, engine="python", encoding="latin-1")


@dataclass
class UploadCheck:
    ok: bool
    errors: list[str]
    warnings: list[str]
    original: pd.DataFrame | None = None  # tabel pengguna dengan nama kolom baku
    work: pd.DataFrame | None = None  # fitur hasil parsing (sebelum disaring)
    valid: pd.Series | None = None
    has_actual: bool = False
    has_timestamp: bool = False


def check_upload(df_in: pd.DataFrame, model: EnergyModel) -> UploadCheck:
    errors: list[str] = []
    warnings: list[str] = []

    df = df_in.dropna(how="all").dropna(axis=1, how="all").copy()
    df = df.rename(columns={c: _CANON[_norm(c)] for c in df.columns if _norm(c) in _CANON})
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty:
        return UploadCheck(False, ["File tidak berisi baris data."], [])

    has_ts = "Timestamp" in df.columns
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    if not has_ts:
        missing += [c for c in ("Hour", "IsWeekend") if c not in df.columns]
    if missing:
        hint = "" if has_ts else " (atau sertakan kolom Timestamp agar Hour dan IsWeekend dihitung otomatis)"
        return UploadCheck(False, [f"Kolom wajib belum ada: {', '.join(missing)}{hint}."], [])

    work = pd.DataFrame(index=df.index)
    for c in NUMERIC_SCALED:
        work[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BINARY_INPUT:
        work[c] = _to_binary(df[c])

    if has_ts:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        work["Timestamp"] = ts
        work["Hour"] = ts.dt.hour.astype("float64")
        work["IsWeekend"] = (ts.dt.dayofweek >= 5).astype("float64").where(ts.notna())
    else:
        work["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
        work["IsWeekend"] = _to_binary(df["IsWeekend"])

    for col, (lo, hi) in SANITY.items():
        if lo is not None:
            work.loc[work[col] < lo, col] = np.nan
        if hi is not None:
            work.loc[work[col] > hi, col] = np.nan

    has_actual = TARGET in df.columns
    if has_actual:
        work[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    valid = work[FEATURES].notna().all(axis=1)
    if not valid.any():
        return UploadCheck(
            False,
            ["Tidak ada baris yang valid. Periksa isi kolom, misalnya HVACUsage berisi On/Off dan angka memakai titik desimal."],
            [],
        )

    n_bad = int((~valid).sum())
    if n_bad:
        rows = (df.index[~valid] + 2).tolist()  # +2: baris 1 adalah judul kolom
        shown = ", ".join(str(r) for r in rows[:8]) + (f", dan {len(rows) - 8} lainnya" if len(rows) > 8 else "")
        cols = [LABELS[c] for c in FEATURES if work.loc[:, c].isna().any()]
        warnings.append(
            f"{n_bad} baris dilewati karena ada nilai kosong atau tidak valid (baris {shown}). "
            f"Periksa kolom: {', '.join(cols)}."
        )

    for c in NUMERIC_SCALED:
        lo, hi = model.train_range.loc[c, "min"], model.train_range.loc[c, "max"]
        v = work.loc[valid, c]
        tol = 0.05 * (hi - lo)  # selisih kecil di tepi rentang tidak perlu diperingatkan
        n_out = int(((v < lo - tol) | (v > hi + tol)).sum())
        if n_out:
            warnings.append(
                f"{LABELS[c]}: {n_out} baris di luar rentang data latih ({num_id(lo, 1)} sampai {num_id(hi, 1)}). "
                "Prediksi di luar rentang ini kurang bisa diandalkan."
            )

    return UploadCheck(True, errors, warnings, df, work, valid, has_actual, has_ts)


def predict_upload(check: UploadCheck, model: EnergyModel) -> pd.DataFrame:
    """Tabel hasil: kolom pengguna + Prediksi_kWh (kosong untuk baris yang dilewati)."""
    out = check.original.copy()
    if check.has_timestamp:
        out["Timestamp"] = check.work["Timestamp"]
    pred = pd.Series(np.nan, index=out.index, dtype="float64")
    pred.loc[check.valid] = model.predict(check.work.loc[check.valid])
    out["Prediksi_kWh"] = pred
    return out


# --------------------------------------------------------------------------- #
# Contoh dan template
# --------------------------------------------------------------------------- #
def sample_upload_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """200 baris terakhir (periode uji) lengkap dengan konsumsi aktual."""
    cols = ["Timestamp"] + REQUIRED_BASE + [TARGET]
    return raw[cols].iloc[-200:].reset_index(drop=True)


def template_frame() -> pd.DataFrame:
    """Contoh 24 jam dengan pola harian sederhana, nilai berada dalam rentang data latih."""
    hours = np.arange(24)
    temp = 25 - 3.2 * np.cos((hours - 3) / 24 * 2 * np.pi)
    solar = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) * 28
    occupancy = np.array([0, 0, 0, 0, 0, 1, 3, 6, 8, 9, 9, 8, 7, 8, 9, 9, 8, 7, 6, 5, 3, 2, 1, 0])
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2026-11-16 00:00:00", periods=24, freq="h"),
            "Temperature": temp.round(1),
            "Humidity": (50 - 5 * np.sin((hours - 6) / 24 * 2 * np.pi)).round(1),
            "SquareFootage": 1500,
            "Occupancy": occupancy,
            "HVACUsage": np.where(temp > 25.5, "On", "Off"),
            "LightingUsage": np.where((hours < 7) | (hours >= 17), "On", "Off"),
            "RenewableEnergy": solar.round(1),
            "Holiday": "No",
        }
    )
