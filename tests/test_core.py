"""Jalankan dengan:  python -m pytest tests  (atau  python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import energy_core as ec  # noqa: E402

RAW = ec.load_raw()
BUNDLE = ec.train_models(RAW)
RANGES = BUNDLE.train_range

REPORT_ROW = pd.DataFrame([{"Temperature": 21.0, "Humidity": 38.4, "SquareFootage": 1500, "Occupancy": 7,
                            "HVACUsage": 0, "LightingUsage": 1, "RenewableEnergy": 10.0, "Holiday": 0,
                            "Hour": 18, "IsWeekend": 1}])


def test_linear_metrics_match_notebook():
    m = BUNDLE.metrics_for(ec.LINEAR)
    assert abs(m["MAE"] - 4.1363) < 1e-4 and abs(m["RMSE"] - 4.9940) < 1e-4 and abs(m["R2"] - 0.5930) < 1e-4
    assert (BUNDLE.n_train, BUNDLE.n_test) == (800, 200)


def test_gradient_boosting_matches_report():
    m = BUNDLE.metrics_for("Gradient Boosting")
    assert abs(m["MAE"] - 4.1481) < 1e-4 and abs(m["RMSE"] - 5.0562) < 1e-4 and abs(m["R2"] - 0.5828) < 1e-4


def test_linear_is_best_model_on_default_data():
    assert BUNDLE.best_model() == ec.LINEAR


def test_report_example_is_69_44():
    assert round(float(BUNDLE.predict(ec.LINEAR, REPORT_ROW)[0]), 2) == 69.44


def test_all_models_predict_and_explain():
    for name in ec.MODEL_NAMES:
        assert np.isfinite(BUNDLE.predict(name, REPORT_ROW)[0])
        contrib, base, exact = BUNDLE.explain(name, REPORT_ROW)
        assert list(contrib.index) == ec.FEATURES and np.isfinite(base)
        assert exact == (name == ec.LINEAR)
        imp = BUNDLE.importance(name)
        assert (imp >= 0).all() and imp.sum() > 0


def test_linear_contributions_add_up_to_prediction():
    row = REPORT_ROW.assign(Temperature=28.0, Occupancy=9, HVACUsage=1)
    contrib, base, exact = BUNDLE.explain(ec.LINEAR, row)
    assert exact and abs(base + contrib.sum() - float(BUNDLE.predict(ec.LINEAR, row)[0])) < 1e-9


def test_levers_flip_status_and_sign():
    on = REPORT_ROW.assign(HVACUsage=1, LightingUsage=1)
    labels = dict(BUNDLE.levers(ec.LINEAR, on))
    assert labels["HVAC dimatikan"] < 0 and labels["Lampu dimatikan"] < 0
    off = REPORT_ROW.assign(HVACUsage=0, LightingUsage=0)
    labels = dict(BUNDLE.levers(ec.LINEAR, off))
    assert labels["HVAC dinyalakan"] > 0 and labels["Lampu dinyalakan"] > 0


def test_categories():
    q1, q3 = BUNDLE.stats["q1"], BUNDLE.stats["q3"]
    assert BUNDLE.category(q1 - 1)[0] == "Rendah" and BUNDLE.category(q3 + 1)[0] == "Tinggi"
    assert BUNDLE.category((q1 + q3) / 2)[0] == "Normal"
    s = pd.Series([q1 - 1, (q1 + q3) / 2, q3 + 1, np.nan])
    out = BUNDLE.categorize(s)
    assert out.tolist()[:3] == ["Rendah", "Normal", "Tinggi"] and pd.isna(out.iloc[3])


def test_slider_bounds_and_defaults_for_default_data():
    expect = {"Temperature": (0.1, 20.0, 30.0), "Humidity": (0.1, 30.0, 60.0), "SquareFootage": (10, 1000, 2000),
              "Occupancy": (1, 0, 9), "RenewableEnergy": (0.5, 0.0, 30.0)}
    for feat, (step, lo, hi) in expect.items():
        got = ec.slider_bounds(RANGES, feat, step)
        assert got == (lo, hi), (feat, got)
        d = ec.slider_default(float(BUNDLE.train_X[feat].mean()), step, *got)
        assert lo <= d <= hi
        wide = ec.slider_bounds(RANGES, feat, step, extend=True)
        assert wide[0] <= lo and wide[1] >= hi
    assert ec.slider_bounds(RANGES, "Humidity", 0.1, extend=True)[1] <= 100  # batas wajar dihormati
    assert ec.slider_bounds(RANGES, "Occupancy", 1, extend=True)[0] >= 0


def test_enrich_renewable_share():
    e = ec.enrich(RAW)
    assert np.allclose(e["Terbarukan_kWh"] + e["Jaringan_kWh"], e[ec.TARGET])
    assert (e["Terbarukan_kWh"] <= e[ec.TARGET]).all() and set(e["Hari"].unique()) <= set(ec.DAY_NAMES_ID)


def test_iqr_summary_shape():
    s = ec.iqr_summary(RAW, ["Temperature", ec.TARGET])
    assert list(s["Variabel"]) == ["Temperature", ec.TARGET] and (s["Jumlah outlier"] >= 0).all()


# ---------------- file untuk prediksi ----------------
def _base_frame(n=5):
    return pd.DataFrame({
        "Timestamp": pd.date_range("2026-01-05 08:00", periods=n, freq="h"),
        "Temperature": [25.0] * n, "Humidity": [45.0] * n, "SquareFootage": [1500] * n, "Occupancy": [5] * n,
        "HVACUsage": ["On"] * n, "LightingUsage": ["Off"] * n, "RenewableEnergy": [15.0] * n, "Holiday": ["No"] * n,
    })


def test_upload_accepts_template_and_sample():
    for frame in (ec.template_frame(), ec.sample_upload_frame(RAW)):
        chk = ec.check_upload(frame, RANGES)
        assert chk.ok and chk.valid.all() and not chk.warnings
        out = ec.predict_upload(chk, lambda X: BUNDLE.predict("Random Forest", X))
        assert out["Prediksi_kWh"].notna().all()


def test_upload_flexible_names_and_values():
    df = _base_frame().rename(columns={"Temperature": "temperature ", "HVACUsage": "hvac_usage"})
    df["hvac_usage"] = ["Ya", "tidak", 1, 0, "TRUE"]
    chk = ec.check_upload(df, RANGES)
    assert chk.ok and chk.valid.all() and chk.work["HVACUsage"].tolist() == [1, 0, 1, 0, 1]


def test_upload_without_timestamp_needs_hour_and_weekend():
    df = _base_frame().drop(columns="Timestamp")
    chk = ec.check_upload(df, RANGES)
    assert not chk.ok and "Hour" in chk.errors[0]
    df["Hour"], df["IsWeekend"] = [8, 9, 10, 11, 12], ["Weekday", "Weekday", "Weekend", "Weekend", "Weekday"]
    assert ec.check_upload(df, RANGES).ok


def test_upload_missing_columns_message():
    chk = ec.check_upload(_base_frame().drop(columns=["Humidity", "Holiday"]), RANGES)
    assert not chk.ok and "Humidity" in chk.errors[0] and "Holiday" in chk.errors[0]


def test_upload_skips_bad_rows_and_warns_about_range():
    df = _base_frame(6).astype({"Temperature": "object"})
    df.loc[1, "Temperature"] = "abc"
    df.loc[2, "HVACUsage"] = "mungkin"
    df.loc[3, "Humidity"] = 140
    df.loc[4, "Temperature"] = 45.0
    chk = ec.check_upload(df, RANGES)
    assert chk.ok and int(chk.valid.sum()) == 3
    assert any("3 baris dilewati" in w for w in chk.warnings) and any("di luar rentang" in w for w in chk.warnings)
    out = ec.predict_upload(chk, lambda X: BUNDLE.predict(ec.LINEAR, X))
    assert out["Prediksi_kWh"].isna().sum() == 3


def test_upload_all_invalid_is_error():
    df = _base_frame(3)
    df["Temperature"] = "x"
    assert not ec.check_upload(df, RANGES).ok


def test_tiny_overshoot_at_range_edge_is_not_warned():
    df = _base_frame(3)
    df.loc[0, "Temperature"] = RANGES.loc["Temperature", "min"] - 0.003
    chk = ec.check_upload(df, RANGES)
    assert chk.ok and not chk.warnings


# ---------------- file untuk mengganti data latih ----------------
def test_dataset_roundtrip_gives_same_model():
    chk = ec.check_dataset(RAW.copy())
    assert chk.ok and not chk.warnings and len(chk.raw) == len(RAW)
    assert np.allclose(chk.raw[ec.TARGET], RAW[ec.TARGET]) and "DayOfWeek" in chk.raw.columns
    again = ec.train_models(chk.raw)
    assert abs(again.metrics_for(ec.LINEAR)["R2"] - BUNDLE.metrics_for(ec.LINEAR)["R2"]) < 1e-9


def test_dataset_is_sorted_and_cleaned():
    shuffled = RAW.sample(frac=1, random_state=1).astype({"Temperature": "object"}).reset_index(drop=True)
    shuffled.loc[0, "Temperature"] = "rusak"
    chk = ec.check_dataset(shuffled)
    assert chk.ok and len(chk.raw) == len(RAW) - 1 and chk.raw["Timestamp"].is_monotonic_increasing
    assert any("1 baris dilewati" in w for w in chk.warnings)


def test_dataset_rejections():
    assert not ec.check_dataset(RAW.drop(columns=ec.TARGET)).ok
    assert "Timestamp" in ec.check_dataset(RAW.drop(columns="Timestamp")).errors[0]
    small = ec.check_dataset(RAW.head(50))
    assert not small.ok and "Minimal 100" in small.errors[0]


def test_currency_format():
    assert ec.rupiah(1444.7, 2) == "Rp 1.444,70" and ec.rupiah(111234567) == "Rp 111.234.567"
    assert ec.num_id(77.0558, 2) == "77,06"
    assert ec.rupiah_short(22_246_732) == "22,2 juta" and ec.rupiah_short(2_100_000_000) == "2,10 miliar"
    assert ec.rupiah_short(950_000) == "950.000"


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tes lulus")
