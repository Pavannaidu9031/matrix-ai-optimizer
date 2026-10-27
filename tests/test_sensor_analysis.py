"""Response-curve analysis on synthetic H2 exposures with known answers."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sensor_analysis as sa  # noqa: E402

TAU_ON, TAU_OFF = 12.0, 30.0          # s
ON, OFF, END = 60.0, 300.0, 700.0     # s
BASE, DROP = 50.0, 8.0                # uW; WO3 darkens in H2, so intensity falls


def synthetic(noise=0.05, dt=0.5, seed=1):
    t = np.arange(0, END, dt)
    y = np.full_like(t, BASE)
    on = (t >= ON) & (t < OFF)
    y[on] = BASE - DROP * (1 - np.exp(-(t[on] - ON) / TAU_ON))
    level_at_off = BASE - DROP * (1 - math.exp(-(OFF - ON) / TAU_ON))
    off = t >= OFF
    y[off] = BASE - (BASE - level_at_off) * np.exp(-(t[off] - OFF) / TAU_OFF)
    return t, y + np.random.default_rng(seed).normal(0, noise, len(t))


def to_csv(t, y, header="Time (s),Power (uW)", sep=",", preamble=("Thorlabs PM100D", "Wavelength: 1550 nm", "")):
    lines = list(preamble) + [header] + [f"{a}{sep}{b}" for a, b in zip(t, y)]
    return "\n".join(lines).encode()


def test_known_times_with_switch_points():
    t, y = synthetic()
    r = sa.analyze_response(t, y, h2_on_s=ON, h2_off_s=OFF)
    assert r["direction"] == "decrease"
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)
    assert r["response_time_s"] == pytest.approx(TAU_ON * math.log(10), abs=1.5)
    assert r["recovery_time_s"] == pytest.approx(TAU_OFF * math.log(10), abs=3.0)
    assert r["snr"] > 50


def test_detects_switch_points_automatically():
    t, y = synthetic()
    r = sa.analyze_response(t, y)
    assert r["h2_on_detected"] and r["h2_off_detected"]
    assert r["h2_on_s"] == pytest.approx(ON, abs=3)
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.05)
    assert r["response_time_s"] == pytest.approx(TAU_ON * math.log(10), abs=4)
    assert r["recovery_time_s"] is not None


def test_increasing_signal():
    t, y = synthetic()
    r = sa.analyze_response(t, 2 * BASE - y, h2_on_s=ON, h2_off_s=OFF)
    assert r["direction"] == "increase"
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)


def test_no_recovery_is_reported_not_invented():
    t, y = synthetic()
    keep = t < OFF + 20
    r = sa.analyze_response(t[keep], y[keep], h2_on_s=ON, h2_off_s=OFF)
    assert r["recovery_time_s"] is None
    assert any("recovery not measured" in w for w in r["warnings"])


def test_flat_signal_is_rejected():
    t = np.arange(0, 100, 0.5)
    with pytest.raises(sa.CurveError):
        sa.analyze_response(t, np.full_like(t, 50.0) + np.random.default_rng(0).normal(0, 0.01, len(t)))


def test_csv_with_instrument_preamble_and_mw_units():
    t, y = synthetic()
    raw = to_csv(t, y / 1000.0, header="Time (s),Power (mW)")
    r = sa.analyze_file(raw, h2_on_s=ON, h2_off_s=OFF)
    assert r["unit"] == "mw"
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)


def test_semicolon_decimal_comma_and_milliseconds():
    t, y = synthetic()
    lines = ["Zeit [ms];Leistung [µW]"] + [f"{int(a * 1000)};{b:.4f}".replace(".", ",") for a, b in zip(t, y)]
    r = sa.analyze_file("\n".join(lines).encode("utf-8"), time_unit="ms", h2_on_s=ON, h2_off_s=OFF)
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)
    assert r["duration_s"] == pytest.approx(END - 0.5, abs=1)


def test_clock_time_and_date_column_are_handled():
    t, y = synthetic()
    lines = ["Date\tTime\tPower (W)"]
    for a, b in zip(t, y):
        s = 10 * 3600 + a
        lines.append(f"2026-10-27\t{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}\t{b * 1e-6:.6e}")
    r = sa.analyze_file("\n".join(lines).encode(), h2_on_s=ON, h2_off_s=OFF)
    assert r["columns"] == ["Time", "Power (W)"]   # date column dropped
    assert r["unit"] == "w"
    assert r["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)


def test_dbm_conversion():
    assert sa.to_microwatts(np.array([0.0, -30.0]), "dbm") == pytest.approx([1000.0, 1.0])


def test_curve_is_downsampled():
    t, y = synthetic(dt=0.05)
    r = sa.analyze_response(t, y, h2_on_s=ON, h2_off_s=OFF)
    assert len(r["curve"]["t"]) <= sa.MAX_CURVE_POINTS


def test_garbage_file_gives_readable_error():
    with pytest.raises(sa.CurveError, match="numeric"):
        sa.analyze_file(b"hello\nworld\n")
