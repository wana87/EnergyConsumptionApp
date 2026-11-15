"""Jalankan dengan:  python -m pytest tests  (atau  python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import energy_core as ec  # noqa: E402

MODEL = ec.train_model()
RAW = ec.load_raw()


def test_metrics_match_notebook():
    m = MODEL.metrics
    assert abs(m["MAE"] - 4.1363) < 1e-4
    assert abs(m["RMSE"] - 4.9940) < 1e-4
    assert abs(m["R2"] - 0.5930) < 1e-4
    assert (m["n_train"], m["n_test"]) == (800, 200)


def test_report_example_is_69_44():
    row = pd.DataFrame(
        [{"Temperature": 21.0, "Humidity": 38.4, "SquareFootage": 1500, "Occupancy": 7,
          "HVACUsage": 0, "LightingUsage": 1, "RenewableEnergy": 10.0, "Holiday": 0,
          "Hour": 18, "IsWeekend": 1}]
    )
    assert round(float(MODEL.predict(row)[0]), 2) == 69.44


def test_contributions_add_up_to_prediction():
    row = pd.DataFrame(
        [{"Temperature": 28.0, "Humidity": 50.0, "SquareFootage": 1800, "Occupancy": 9,
          "HVACUsage": 1, "LightingUsage": 1, "RenewableEnergy": 3.0, "Holiday": 1,
          "Hour": 14, "IsWeekend": 0}]
    )
    pred = float(MODEL.predict(row)[0])
    assert abs(MODEL.baseline + MODEL.contributions(row).sum() - pred) < 1e-9


def _base_frame(n=5):
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2026-01-05 08:00", periods=n, freq="h"),
            "Temperature": [25.0] * n, "Humidity": [45.0] * n, "SquareFootage": [1500] * n,
            "Occupancy": [5] * n, "HVACUsage": ["On"] * n, "LightingUsage": ["Off"] * n,
            "RenewableEnergy": [15.0] * n, "Holiday": ["No"] * n,
        }
    )


def test_upload_accepts_template_and_sample():
    for frame in (ec.template_frame(), ec.sample_upload_frame(RAW)):
        chk = ec.check_upload(frame, MODEL)
        assert chk.ok and chk.valid.all()
        out = ec.predict_upload(chk, MODEL)
        assert out["Prediksi_kWh"].notna().all()


def test_upload_flexible_names_and_values():
    df = _base_frame().rename(columns={"Temperature": "temperature ", "HVACUsage": "hvac_usage"})
    df["hvac_usage"] = ["Ya", "tidak", 1, 0, "TRUE"]
    chk = ec.check_upload(df, MODEL)
    assert chk.ok and chk.valid.all()
    assert chk.work["HVACUsage"].tolist() == [1, 0, 1, 0, 1]


def test_upload_without_timestamp_needs_hour_and_weekend():
    df = _base_frame().drop(columns="Timestamp")
    chk = ec.check_upload(df, MODEL)
    assert not chk.ok and "Hour" in chk.errors[0]
    df["Hour"], df["IsWeekend"] = [8, 9, 10, 11, 12], ["Weekday", "Weekday", "Weekend", "Weekend", "Weekday"]
    assert ec.check_upload(df, MODEL).ok


def test_upload_missing_columns_message():
    chk = ec.check_upload(_base_frame().drop(columns=["Humidity", "Holiday"]), MODEL)
    assert not chk.ok and "Humidity" in chk.errors[0] and "Holiday" in chk.errors[0]


def test_upload_skips_bad_rows_and_warns_about_range():
    df = _base_frame(6).astype({"Temperature": "object"})
    df.loc[1, "Temperature"] = "abc"      # tidak bisa dibaca
    df.loc[2, "HVACUsage"] = "mungkin"    # kategori tidak dikenal
    df.loc[3, "Humidity"] = 140           # tidak masuk akal
    df.loc[4, "Temperature"] = 45.0       # valid tetapi di luar rentang latih
    chk = ec.check_upload(df, MODEL)
    assert chk.ok and int(chk.valid.sum()) == 3
    assert any("3 baris dilewati" in w for w in chk.warnings)
    assert any("di luar rentang" in w for w in chk.warnings)
    out = ec.predict_upload(chk, MODEL)
    assert out["Prediksi_kWh"].isna().sum() == 3


def test_upload_all_invalid_is_error():
    df = _base_frame(3)
    df["Temperature"] = "x"
    assert not ec.check_upload(df, MODEL).ok


def test_currency_format():
    assert ec.rupiah(1444.7, 2) == "Rp 1.444,70"
    assert ec.rupiah(111234567) == "Rp 111.234.567"
    assert ec.num_id(77.0558, 2) == "77,06"
    assert ec.rupiah_short(22_246_732) == "22,2 juta"
    assert ec.rupiah_short(2_100_000_000) == "2,10 miliar"
    assert ec.rupiah_short(950_000) == "950.000"


def test_tiny_overshoot_at_range_edge_is_not_warned():
    df = _base_frame(3)
    lo = MODEL.train_range.loc["Temperature", "min"]
    df.loc[0, "Temperature"] = lo - 0.003
    chk = ec.check_upload(df, MODEL)
    assert chk.ok and not chk.warnings


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tes lulus")
