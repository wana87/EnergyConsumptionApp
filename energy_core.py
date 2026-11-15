"""Inti perhitungan aplikasi (tanpa kode Streamlit): data, fitur, model, validasi file.

Alur pelatihan meniru notebook Oktriza01.ipynb:
  - fitur waktu (Hour, IsWeekend) diturunkan dari Timestamp, DayOfWeek tidak dipakai
  - HVACUsage, LightingUsage, Holiday dikodekan 0/1
  - pembagian data kronologis 80% latih / 20% uji
  - Linear Regression memakai StandardScaler yang dilatih hanya pada data latih
  - Decision Tree, Random Forest, Gradient Boosting dilatih pada fitur tanpa standardisasi
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

CORE_VERSION = 2  # dinaikkan bila antarmuka modul ini berubah; dicek oleh app.py

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Energy_consumption.csv"

TARGET = "EnergyConsumption"

# Urutan ini sama dengan X_train.columns pada notebook.
FEATURES = [
    "Temperature", "Humidity", "SquareFootage", "Occupancy", "HVACUsage",
    "LightingUsage", "RenewableEnergy", "Holiday", "Hour", "IsWeekend",
]
NUMERIC_SCALED = ["Temperature", "Humidity", "SquareFootage", "Occupancy", "RenewableEnergy"]
BINARY_INPUT = ["HVACUsage", "LightingUsage", "Holiday"]
REQUIRED_BASE = NUMERIC_SCALED + BINARY_INPUT
MIN_DATASET_ROWS = 100

LINEAR = "Linear Regression"
MODEL_NAMES = [LINEAR, "Decision Tree", "Random Forest", "Gradient Boosting"]

DAY_NAMES_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

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

# Perubahan yang dipakai untuk menjelaskan koefisien model linear dalam satuan asli.
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

# Kolom untuk analisis korelasi (nama pendek untuk sumbu heatmap).
CORR_COLUMNS = {
    "Temperature": "Suhu", "Humidity": "Kelembapan", "SquareFootage": "Luas", "Occupancy": "Penghuni",
    "RenewableEnergy": "Terbarukan", "HVAC": "HVAC", "Lighting": "Lampu", "IsHoliday": "Libur",
    "IsWeekend": "Akhir pekan", "Hour": "Jam", TARGET: "Konsumsi",
}

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
# Data bawaan dan turunannya
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


def enrich(raw: pd.DataFrame) -> pd.DataFrame:
    """Kolom tambahan untuk dashboard dan analisis. Kolom teks asli dipertahankan.

    RenewableEnergy dibaca sebagai persen (sesuai legenda notebook), sehingga porsi terbarukan
    dari konsumsi = konsumsi x persen / 100.
    """
    df = raw.copy()
    ts = df["Timestamp"]
    df["Tanggal"] = ts.dt.normalize()
    df["Hour"] = ts.dt.hour
    df["IsWeekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["Hari"] = ts.dt.dayofweek.map(dict(enumerate(DAY_NAMES_ID)))
    df["HVAC"] = (df["HVACUsage"] == "On").astype(int)
    df["Lighting"] = (df["LightingUsage"] == "On").astype(int)
    df["IsHoliday"] = (df["Holiday"] == "Yes").astype(int)
    share = (df["RenewableEnergy"] / 100).clip(0, 1)
    df["Terbarukan_kWh"] = df[TARGET] * share
    df["Jaringan_kWh"] = df[TARGET] - df["Terbarukan_kWh"]
    return df


def iqr_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ringkasan outlier metode IQR (Q1 - 1,5 IQR sampai Q3 + 1,5 IQR), seperti di notebook."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({"Variabel": col, "Q1": q1, "Q3": q3, "Batas bawah": lo, "Batas atas": hi,
                     "Jumlah outlier": n, "Persen": n / len(df) * 100 if len(df) else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class EnergyModel:
    """Linear Regression dengan standardisasi, persis seperti notebook."""

    scaler: StandardScaler
    regressor: LinearRegression
    design_mean: np.ndarray
    design_std: np.ndarray
    baseline: float

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
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
            rows.append({"Feature": name, "Faktor": LABELS[name], "Perubahan": text,
                         "Efek": per_unit * step, "Efek1SD": coef * std})
        return pd.DataFrame(rows)


def _fit_linear(X_tr: pd.DataFrame, y_tr: pd.Series) -> EnergyModel:
    scaler = StandardScaler().fit(X_tr[NUMERIC_SCALED])
    design_tr = X_tr.astype("float64").copy()
    design_tr[NUMERIC_SCALED] = scaler.transform(X_tr[NUMERIC_SCALED])
    regressor = LinearRegression().fit(design_tr, y_tr)
    design_mean = design_tr.mean().to_numpy()
    baseline = float(regressor.predict(pd.DataFrame([design_mean], columns=FEATURES))[0])
    return EnergyModel(scaler, regressor, design_mean, design_tr.std(ddof=0).to_numpy(), baseline)


@dataclass
class ModelBundle:
    """Keempat model beserta metrik, data uji, dan penjelasan."""

    linear: EnergyModel
    trees: dict
    comparison: pd.DataFrame  # Model, MAE, RMSE, R2
    test_frame: pd.DataFrame  # Timestamp, Aktual, satu kolom prediksi per model
    train_X: pd.DataFrame
    train_range: pd.DataFrame
    periods: dict
    n_train: int
    n_test: int
    baseline: float  # rata-rata konsumsi pada data latih
    stats: dict  # mean, q1, q3 seluruh data (untuk kategori)
    explain_sample: pd.DataFrame

    # -- prediksi ----------------------------------------------------------- #
    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURES]
        if name == LINEAR:
            return self.linear.predict(X)
        return self.trees[name].predict(X)

    def metrics_for(self, name: str) -> dict:
        row = self.comparison.set_index("Model").loc[name]
        return {"MAE": float(row["MAE"]), "RMSE": float(row["RMSE"]), "R2": float(row["R2"])}

    def best_model(self) -> str:
        return str(self.comparison.sort_values("RMSE").iloc[0]["Model"])

    # -- penjelasan --------------------------------------------------------- #
    def explain(self, name: str, row: pd.DataFrame) -> tuple[pd.Series, float, bool]:
        """(kontribusi per fitur dalam kWh, titik acuan, apakah penjumlahan tepat).

        Linear Regression: tepat (kontribusi = koefisien x selisih dari rata-rata).
        Model pohon: perkiraan, yaitu selisih prediksi dengan rata-rata prediksi bila fitur itu
        diganti nilai-nilai dari data latih.
        """
        if name == LINEAR:
            return self.linear.contributions(row), self.linear.baseline, True
        pred = float(self.predict(name, row)[0])
        sample = self.explain_sample
        n = len(sample)
        out = {}
        for f in FEATURES:
            frame = pd.concat([row[FEATURES]] * n, ignore_index=True)
            frame[f] = sample[f].to_numpy()
            out[f] = pred - float(self.predict(name, frame).mean())
        return pd.Series(out), float(self.predict(name, sample).mean()), False

    def importance(self, name: str) -> pd.Series:
        if name == LINEAR:
            values = np.abs(self.linear.regressor.coef_ * self.linear.design_std)
        else:
            values = self.trees[name].feature_importances_
        return pd.Series(values, index=FEATURES)

    def levers(self, name: str, row: pd.DataFrame) -> list[tuple[str, float]]:
        """Selisih prediksi (kWh per jam) bila HVAC atau lampu dibalik statusnya."""
        base = float(self.predict(name, row)[0])
        out = []
        for col, on_text, off_text in (
            ("HVACUsage", "HVAC dimatikan", "HVAC dinyalakan"),
            ("LightingUsage", "Lampu dimatikan", "Lampu dinyalakan"),
        ):
            alt = row.copy()
            was_on = int(alt[col].iloc[0]) == 1
            alt[col] = 0 if was_on else 1
            out.append((on_text if was_on else off_text, float(self.predict(name, alt)[0]) - base))
        return out

    # -- kategori ----------------------------------------------------------- #
    def category(self, value: float) -> tuple[str, str]:
        if value < self.stats["q1"]:
            return "Rendah", "low"
        if value > self.stats["q3"]:
            return "Tinggi", "high"
        return "Normal", "mid"

    def categorize(self, values: pd.Series) -> pd.Series:
        out = np.select([values < self.stats["q1"], values > self.stats["q3"]], ["Rendah", "Tinggi"], "Normal")
        return pd.Series(out, index=values.index).where(values.notna())


def train_models(raw: pd.DataFrame) -> ModelBundle:
    df = engineer(raw)
    X, y = df[FEATURES], df[TARGET]
    split = int(len(df) * 0.8)  # pembagian kronologis
    X_tr, X_te = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    linear = _fit_linear(X_tr, y_tr)
    trees = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    preds = {LINEAR: linear.predict(X_te)}
    for name, model in trees.items():
        model.fit(X_tr, y_tr)
        preds[name] = model.predict(X_te)

    comparison = pd.DataFrame(
        [{"Model": n, "MAE": float(mean_absolute_error(y_te, preds[n])),
          "RMSE": float(np.sqrt(mean_squared_error(y_te, preds[n]))),
          "R2": float(r2_score(y_te, preds[n]))} for n in MODEL_NAMES]
    )
    test_frame = pd.DataFrame({"Timestamp": df["Timestamp"].iloc[split:].to_numpy(), "Aktual": y_te.to_numpy()})
    for n in MODEL_NAMES:
        test_frame[n] = preds[n]

    ts = df["Timestamp"]
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    return ModelBundle(
        linear=linear, trees=trees, comparison=comparison, test_frame=test_frame,
        train_X=X_tr, train_range=X_tr.agg(["min", "max"]).T,
        periods={"train": (ts.iloc[0], ts.iloc[split - 1]), "test": (ts.iloc[split], ts.iloc[-1])},
        n_train=int(len(X_tr)), n_test=int(len(X_te)), baseline=float(y_tr.mean()),
        stats={"mean": float(y.mean()), "q1": float(q1), "q3": float(q3)},
        explain_sample=X_tr.sample(min(150, len(X_tr)), random_state=0).reset_index(drop=True),
    )


def slider_bounds(ranges: pd.DataFrame, feature: str, step: float, extend: bool = False) -> tuple[float, float]:
    """Batas slider dari rentang data latih (dibulatkan ke langkah). `extend` melebarkan 50% ke tiap sisi."""
    lo, hi = float(ranges.loc[feature, "min"]), float(ranges.loc[feature, "max"])
    if extend:
        span = hi - lo
        lo, hi = lo - span * 0.5, hi + span * 0.5
    s_lo, s_hi = SANITY.get(feature, (None, None))
    if s_lo is not None:
        lo = max(lo, s_lo)
    if s_hi is not None:
        hi = min(hi, s_hi)
    lo, hi = round(math.floor(lo / step) * step, 6), round(math.ceil(hi / step) * step, 6)
    if hi <= lo:
        hi = lo + step * 10
    return lo, hi


def slider_default(mean_value: float, step: float, lo: float, hi: float) -> float:
    return round(min(max(round(mean_value / step) * step, lo), hi), 6)


# --------------------------------------------------------------------------- #
# Membaca dan memeriksa file
# --------------------------------------------------------------------------- #
def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANON = {_norm(c): c for c in FEATURES + ["Timestamp", TARGET]}
_CANON.update({"datetime": "Timestamp", "daytype": "IsWeekend", "weekend": "IsWeekend", "dayofweek": "DayOfWeek"})

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
class _Parsed:
    error: str | None
    df: pd.DataFrame | None = None
    work: pd.DataFrame | None = None
    has_ts: bool = False
    has_actual: bool = False


def _parse(df_in: pd.DataFrame, require_ts: bool = False, require_actual: bool = False) -> _Parsed:
    df = df_in.dropna(how="all").dropna(axis=1, how="all").copy()
    df = df.rename(columns={c: _CANON[_norm(c)] for c in df.columns if _norm(c) in _CANON})
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty:
        return _Parsed("File tidak berisi baris data.")

    has_ts = "Timestamp" in df.columns
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    hint = ""
    if not has_ts:
        if require_ts:
            missing.append("Timestamp")
        else:
            missing += [c for c in ("Hour", "IsWeekend") if c not in df.columns]
            hint = " (atau sertakan kolom Timestamp agar Hour dan IsWeekend dihitung otomatis)"
    if require_actual and TARGET not in df.columns:
        missing.append(TARGET)
    if missing:
        return _Parsed(f"Kolom wajib belum ada: {', '.join(missing)}{hint}.")

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
    return _Parsed(None, df, work, has_ts, has_actual)


def _bad_rows_message(index: pd.Index, valid: pd.Series, work: pd.DataFrame, columns: list[str]) -> str:
    rows = (index[~valid] + 2).tolist()  # +2: baris 1 adalah judul kolom
    shown = ", ".join(str(r) for r in rows[:8]) + (f", dan {len(rows) - 8} lainnya" if len(rows) > 8 else "")
    cols = [LABELS.get(c, c) for c in columns if work[c].isna().any()]
    return (f"{len(rows)} baris dilewati karena ada nilai kosong atau tidak valid (baris {shown}). "
            f"Periksa kolom: {', '.join(cols)}.")


# -- file untuk prediksi ------------------------------------------------------ #
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


def check_upload(df_in: pd.DataFrame, ranges: pd.DataFrame) -> UploadCheck:
    p = _parse(df_in)
    if p.error:
        return UploadCheck(False, [p.error], [])
    df, work = p.df, p.work
    valid = work[FEATURES].notna().all(axis=1)
    if not valid.any():
        return UploadCheck(False, ["Tidak ada baris yang valid. Periksa isi kolom, misalnya HVACUsage berisi On/Off "
                                   "dan angka memakai titik desimal."], [])
    warnings: list[str] = []
    if (~valid).any():
        warnings.append(_bad_rows_message(df.index, valid, work, FEATURES))
    for c in NUMERIC_SCALED:
        lo, hi = ranges.loc[c, "min"], ranges.loc[c, "max"]
        tol = 0.05 * (hi - lo)  # selisih kecil di tepi rentang tidak perlu diperingatkan
        v = work.loc[valid, c]
        n_out = int(((v < lo - tol) | (v > hi + tol)).sum())
        if n_out:
            warnings.append(f"{LABELS[c]}: {n_out} baris di luar rentang data latih ({num_id(lo, 1)} sampai "
                            f"{num_id(hi, 1)}). Prediksi di luar rentang ini kurang bisa diandalkan.")
    return UploadCheck(True, [], warnings, df, work, valid, p.has_actual, p.has_ts)


def predict_upload(check: UploadCheck, predict_fn) -> pd.DataFrame:
    """Tabel hasil: kolom pengguna + Prediksi_kWh (kosong untuk baris yang dilewati)."""
    out = check.original.copy()
    if check.has_timestamp:
        out["Timestamp"] = check.work["Timestamp"]
    pred = pd.Series(np.nan, index=out.index, dtype="float64")
    pred.loc[check.valid] = predict_fn(check.work.loc[check.valid])
    out["Prediksi_kWh"] = pred
    return out


# -- file untuk mengganti data latih ----------------------------------------- #
@dataclass
class DatasetCheck:
    ok: bool
    errors: list[str]
    warnings: list[str]
    raw: pd.DataFrame | None = None


def check_dataset(df_in: pd.DataFrame) -> DatasetCheck:
    """Validasi dataset pengganti. Hasilnya berskema sama dengan data bawaan."""
    p = _parse(df_in, require_ts=True, require_actual=True)
    if p.error:
        return DatasetCheck(False, [p.error], [])
    df, work = p.df, p.work
    valid = work[FEATURES].notna().all(axis=1) & work[TARGET].notna()
    warnings: list[str] = []
    if (~valid).any():
        warnings.append(_bad_rows_message(df.index, valid, work, FEATURES + [TARGET]))
    if int(valid.sum()) < MIN_DATASET_ROWS:
        return DatasetCheck(False, [f"Hanya {int(valid.sum())} baris valid. Minimal {MIN_DATASET_ROWS} baris "
                                    "dibutuhkan untuk melatih dan menguji model."], warnings)

    raw = pd.DataFrame({
        "Timestamp": work["Timestamp"], "Temperature": work["Temperature"], "Humidity": work["Humidity"],
        "SquareFootage": work["SquareFootage"], "Occupancy": work["Occupancy"],
        "HVACUsage": np.where(work["HVACUsage"] == 1, "On", "Off"),
        "LightingUsage": np.where(work["LightingUsage"] == 1, "On", "Off"),
        "RenewableEnergy": work["RenewableEnergy"],
        "Holiday": np.where(work["Holiday"] == 1, "Yes", "No"),
        TARGET: work[TARGET],
    }).loc[valid]
    if "DayOfWeek" in df.columns:
        raw["DayOfWeek"] = df.loc[valid, "DayOfWeek"]
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    return DatasetCheck(True, [], warnings, raw)


# --------------------------------------------------------------------------- #
# Contoh dan template
# --------------------------------------------------------------------------- #
def sample_upload_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Hingga 200 baris terakhir (periode uji) lengkap dengan konsumsi aktual."""
    cols = ["Timestamp"] + REQUIRED_BASE + [TARGET]
    return raw[cols].iloc[-min(200, len(raw)):].reset_index(drop=True)


def template_frame() -> pd.DataFrame:
    """Contoh 24 jam dengan pola harian sederhana, nilai berada dalam rentang data latih."""
    hours = np.arange(24)
    temp = 25 - 3.2 * np.cos((hours - 3) / 24 * 2 * np.pi)
    solar = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) * 28
    occupancy = np.array([0, 0, 0, 0, 0, 1, 3, 6, 8, 9, 9, 8, 7, 8, 9, 9, 8, 7, 6, 5, 3, 2, 1, 0])
    return pd.DataFrame({
        "Timestamp": pd.date_range("2026-11-16 00:00:00", periods=24, freq="h"),
        "Temperature": temp.round(1),
        "Humidity": (50 - 5 * np.sin((hours - 6) / 24 * 2 * np.pi)).round(1),
        "SquareFootage": 1500,
        "Occupancy": occupancy,
        "HVACUsage": np.where(temp > 25.5, "On", "Off"),
        "LightingUsage": np.where((hours < 7) | (hours >= 17), "On", "Off"),
        "RenewableEnergy": solar.round(1),
        "Holiday": "No",
    })
